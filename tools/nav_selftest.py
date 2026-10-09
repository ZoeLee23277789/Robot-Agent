#!/usr/bin/env python
"""
導航層的回歸測試：不用 LLM，只用 robot_server 的動作 + CoppeliaSim 的真值。

    .venv/bin/python tools/nav_selftest.py                      # 預設規模（約 15 分鐘）
    .venv/bin/python tools/nav_selftest.py --goals 30 --objects 10   # 大一點
    .venv/bin/python tools/nav_selftest.py --only rotate,goals   # 只跑部分

測什麼：
    rotate    各角度原地旋轉的精度（世界朝向由輪子算，跟 agent 無關）
    goals     隨機抽地圖上可走的目標點做 navigate_to：到達率、終點誤差、時間、碰撞次數
    landmarks 12 個固定地標各 go_to 一次（隧道/平台應停在入口，其他停在前緣）
    passages  drive_through 隧道與平台各一次，用軌跡判定是否真的穿過
    objects   對每個零散物件（積木/球/泡棉）用真值座標做 navigate_to(stop_m=0.4, face)：
              終點距離要在合理範圍、朝向要對準——這是 approach 的幾何部分，去掉 VLM

碰撞：測試期間另開一條 zmq 連線每 0.2 秒查一次機器人零件跟「非機器人、非地面」物件的接觸，
任何一次接觸算一次碰撞事件（同一物件連續接觸只算一次）。

結果印成表格並存 robot_eval/results/nav_selftest_<時間>.json。每次改導航層就跑一次，比較數字。
"""
import argparse
import asyncio
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Test_Robot"))

from coppeliasim_zmqremoteapi_client import RemoteAPIClient  # noqa: E402

from robot_agent.client import RobotClient  # noqa: E402

SERVER = "http://127.0.0.1:8765"
TURN_SIGN = -1.0  # turn_left 為正時世界朝向減少（跟 robot_server 一致）


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def state():
    return json.load(urllib.request.urlopen(SERVER + "/state", timeout=3))


class Truth:
    """CoppeliaSim 真值：機器人姿態、物件位置、接觸。"""

    def __init__(self):
        self.sim = RemoteAPIClient().require("sim")
        self.robot = self.sim.getObject("/RoboMaster")
        self.robot_tree = set(self.sim.getObjectsInTree(self.robot, self.sim.handle_all, 0)) | {self.robot}
        self.robot_shapes = [s for s in self.sim.getObjectsInTree(self.robot, self.sim.object_shape_type, 0)
                             if self.sim.getObjectInt32Param(s, self.sim.shapeintparam_respondable)]
        wheels = {}
        for o in self.sim.getObjectsInTree(self.robot, self.sim.object_joint_type, 0):
            a = self.sim.getObjectAlias(o)
            if a.endswith("_wheel_joint"):
                wheels[a] = o
        self.wheels = wheels

    def moved_since(self, snap, min_m=0.05):
        """跟 snap 比，被推動 ≥ min_m 的物件 [(name, m)...]。接觸監看 0.2 s 輪詢一次會漏掉短暫的推擠，位移不會漏。"""
        now = self.objects()
        rows = []
        for n, (x, y) in snap.items():
            if n in now:
                d = math.hypot(now[n][0] - x, now[n][1] - y)
                if d >= min_m:
                    rows.append((n, d))
        return sorted(rows, key=lambda t: -t[1])

    def pose(self):
        p = self.sim.getObjectPosition(self.robot, self.sim.handle_world)
        w = self.wheels
        fl, fr = (self.sim.getObjectPosition(w[k], self.sim.handle_world) for k in ("front_left_wheel_joint", "front_right_wheel_joint"))
        rl, rr = (self.sim.getObjectPosition(w[k], self.sim.handle_world) for k in ("rear_left_wheel_joint", "rear_right_wheel_joint"))
        hd = math.degrees(math.atan2((fl[1] + fr[1]) / 2 - (rl[1] + rr[1]) / 2, (fl[0] + fr[0]) / 2 - (rl[0] + rr[0]) / 2))
        return p[0], p[1], hd

    def objects(self, prefixes=("small_", "ball_orange", "ball_purple", "ball_pink", "ball_cyan", "foam_")):
        out = {}
        for o in self.sim.getObjectsInTree(self.sim.handle_scene, sim_shape(self.sim), 0):
            a = self.sim.getObjectAlias(o)
            if a.startswith(prefixes) and o not in self.robot_tree:
                p = self.sim.getObjectPosition(o, self.sim.handle_world)
                out[a] = (p[0], p[1])
        return out


def sim_shape(sim):
    return sim.object_shape_type


class ContactMonitor(threading.Thread):
    """背景輪詢接觸：另開一條 zmq 連線，不跟主測試搶。"""

    def __init__(self):
        super().__init__(daemon=True)
        self.sim = RemoteAPIClient().require("sim")
        self.robot = self.sim.getObject("/RoboMaster")
        self.robot_tree = set(self.sim.getObjectsInTree(self.robot, self.sim.handle_all, 0)) | {self.robot}
        self.shapes = [s for s in self.sim.getObjectsInTree(self.robot, self.sim.object_shape_type, 0)
                       if self.sim.getObjectInt32Param(s, self.sim.shapeintparam_respondable)]
        self.ignore = {"ground_slab"}
        self.events = []       # (time, other_alias)
        self._active = set()
        self.running = True
        self.section = "idle"
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            touching = set()
            try:
                for s in self.shapes:
                    for i in range(8):
                        r = self.sim.getContactInfo(self.sim.handle_all, s, i)
                        if not r or not r[0]:
                            break
                        other = r[0][1] if r[0][0] == s else r[0][0]
                        if other in self.robot_tree or other < 0:
                            continue
                        alias = self.sim.getObjectAlias(other)
                        if alias in self.ignore or alias.startswith("tile_") or alias.startswith("mat_") or alias == "person_spot":
                            continue
                        touching.add(alias)
            except Exception:
                pass
            with self.lock:
                for a in touching - self._active:
                    try:
                        rp = self.sim.getObjectPosition(self.robot, self.sim.handle_world)
                    except Exception:
                        rp = (float("nan"), float("nan"), 0.0)
                    self.events.append((time.time(), self.section, a, (rp[0], rp[1])))
                self._active = touching
            time.sleep(0.2)

    def take(self):
        with self.lock:
            ev, self._active = self.events, set()
            self.events = []
        return ev


def coppelia_rss_mb():
    try:
        pid = subprocess.run(["pgrep", "-x", "coppeliaSim"], capture_output=True, text=True).stdout.split()
        if not pid:
            return None
        rss = subprocess.run(["ps", "-o", "rss=", "-p", pid[0]], capture_output=True, text=True).stdout.strip()
        return int(rss) // 1024
    except Exception:
        return None


async def test_rotate(rc, truth, angles):
    rows = []
    for a in angles:
        _, _, h0 = truth.pose()
        t0 = time.time()
        r = await rc.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": a})
        _, _, h1 = truth.pose()
        expected = TURN_SIGN * a
        err = wrap(wrap(h1 - h0) - expected)
        rows.append({"turn_left": a, "measured": round(wrap(h1 - h0), 1), "error_deg": round(err, 1),
                     "ok": bool(r.get("ok")) and abs(err) <= 10, "s": round(time.time() - t0, 1)})
        print(f"  turn {a:+6.0f}: measured {wrap(h1 - h0):+7.1f}  error {err:+5.1f}  {'ok' if rows[-1]['ok'] else 'FAIL'}")
    return rows


def sample_free_goals(n, seed):
    from nav_map import NavMap, world_to_cell, cell_to_world  # noqa: E402  機器人端的地圖，用快取很快
    sim = RemoteAPIClient().require("sim")
    nm = NavMap(sim, sim.getObject("/RoboMaster"))
    nm.refresh_dynamic()
    import cv2
    import numpy as np
    from nav_map import CELL as _CELL
    clearance = cv2.distanceTransform((~nm.occ).astype(np.uint8), cv2.DIST_L2, 3) * _CELL
    rng = random.Random(seed)
    goals = []
    tries = 0
    while len(goals) < n and tries < 5000:
        tries += 1
        x, y = rng.uniform(-2.7, 2.7), rng.uniform(-2.7, 2.7)
        ix, iy = world_to_cell(x, y)
        if not nm.is_free(ix, iy):
            continue
        # 離「真實障礙」至少 0.35 m：通道挖空區裡緊貼牆的格子也算可走，抽到那種點會製造假的失敗
        if clearance[iy, ix] < 0.35:
            continue
        if any(math.hypot(x - gx, y - gy) < 0.6 for gx, gy in goals):
            continue
        goals.append(cell_to_world(ix, iy))
    return goals


async def test_goals(rc, truth, mon, goals, tol=0.15):
    rows = []
    for gx, gy in goals:
        mon.section = f"goal({gx:.2f},{gy:.2f})"
        mon.take()
        t0 = time.time()
        r = await rc.act("navigate_to", {"x": gx, "y": gy, "tolerance_m": tol}, timeout=320)
        x, y, _ = truth.pose()
        d = math.hypot(x - gx, y - gy)
        hits = mon.take()
        rows.append({"goal": [round(gx, 2), round(gy, 2)], "server_ok": bool(r.get("ok")), "final_err_m": round(d, 3),
                     "ok": bool(r.get("ok")) and d <= tol + 0.05, "s": round(time.time() - t0, 1),
                     "collisions": [h[2] for h in hits], "collision_at": [[h[2], round(h[3][0], 2), round(h[3][1], 2)] for h in hits], "msg": str(r.get("message", ""))[:160]})
        print(f"  goal ({gx:+.2f},{gy:+.2f}): {'ok  ' if rows[-1]['ok'] else 'FAIL'} err={d:.2f}m {time.time() - t0:5.0f}s"
              f"{'  collisions=' + ','.join(sorted(set(h[2] for h in hits))) if hits else ''}")
    return rows


def footprint_dist(sim, name, x, y):
    """機器人到地標群組地面投影（軸對齊外框）的距離；在外框內為 0。讀不到就 None。
    「到中心的距離」對長椅、平台這種大結構不公平（平台 2.5 m 長，停在入口外 0.65 m 離中心就 2 m）。"""
    try:
        h = sim.getObject("/" + name, {"noError": True})
        if h == -1:
            return None
        xs, ys = [], []
        for sh in sim.getObjectsInTree(h, sim.object_shape_type, 0):
            lo = [sim.getObjectFloatParam(sh, q) for q in (sim.objfloatparam_objbbox_min_x, sim.objfloatparam_objbbox_min_y, sim.objfloatparam_objbbox_min_z)]
            hi = [sim.getObjectFloatParam(sh, q) for q in (sim.objfloatparam_objbbox_max_x, sim.objfloatparam_objbbox_max_y, sim.objfloatparam_objbbox_max_z)]
            m = sim.getObjectMatrix(sh, sim.handle_world)
            for cx in (lo[0], hi[0]):
                for cy in (lo[1], hi[1]):
                    pt = sim.multiplyVector(m, [cx, cy, lo[2]])
                    xs.append(pt[0])
                    ys.append(pt[1])
        if not xs:
            return None
        dx = max(min(xs) - x, 0.0, x - max(xs))
        dy = max(min(ys) - y, 0.0, y - max(ys))
        return math.hypot(dx, dy)
    except Exception:
        return None


async def test_landmarks(rc, truth, mon):
    sim = getattr(truth, "sim", None) or RemoteAPIClient().require("sim")
    lm = state().get("static_landmarks") or {}
    rows = []
    for name in sorted(lm):
        mon.section = f"landmark {name}"
        mon.take()
        t0 = time.time()
        r = await rc.act("go_to_landmark", {"name": name, "stop_m": 0.45}, timeout=320)
        x, y, hd = truth.pose()
        lx, ly = lm[name]["world_xy"]
        d = math.hypot(x - lx, y - ly)
        de = footprint_dist(sim, name, x, y)
        hits = mon.take()
        # 停太遠（回歸測試出現過 4.79 m 卻回報成功）不算到：離中心 ≤ 2.0 m，或離結構外框 ≤ 1.0 m
        ok = bool(r.get("ok")) and (d <= 2.0 or (de is not None and de <= 1.0))
        rows.append({"landmark": name, "ok": ok, "dist_to_centre_m": round(d, 2),
                     "dist_to_edge_m": None if de is None else round(de, 2), "s": round(time.time() - t0, 1),
                     "collisions": [h[2] for h in hits], "collision_at": [[h[2], round(h[3][0], 2), round(h[3][1], 2)] for h in hits], "msg": str(r.get("message", ""))[:160]})
        print(f"  {name:<14}: {'ok  ' if ok else 'FAIL'} dist={d:.2f}m edge={'?' if de is None else f'{de:.2f}m'} {time.time() - t0:5.0f}s"
              f"{'  collisions=' + ','.join(sorted(set(h[2] for h in hits))) if hits else ''}")
    return rows


async def test_passages(rc, truth, mon):
    rows = []
    for name in ("tunnel", "platform"):
        mon.section = f"drive_through {name}"
        mon.take()
        t0 = time.time()
        r = await rc.act("drive_through", {"name": name}, timeout=400)
        hits = mon.take()
        rows.append({"structure": name, "ok": bool(r.get("ok")), "passed": r.get("passed"), "s": round(time.time() - t0, 1),
                     "collisions": [h[2] for h in hits], "collision_at": [[h[2], round(h[3][0], 2), round(h[3][1], 2)] for h in hits], "msg": str(r.get("message", ""))[:200]})
        print(f"  {name:<9}: {'ok  ' if r.get('ok') else 'FAIL'} passed={r.get('passed')} {time.time() - t0:5.0f}s"
              f"{'  collisions=' + ','.join(sorted(set(h[2] for h in hits))) if hits else ''}")
    return rows


async def test_objects(rc, truth, mon, n, seed):
    objs = truth.objects()
    names = sorted(objs)
    random.Random(seed).shuffle(names)
    rows = []
    for name in names[:n]:
        ox, oy = objs[name]
        mon.section = f"object {name}"
        mon.take()
        t0 = time.time()
        r = await rc.act("navigate_to", {"x": ox, "y": oy, "stop_m": 0.4, "face": True}, timeout=320)
        x, y, hd = truth.pose()
        nx, ny = objs.get(name, (ox, oy))     # 物件可能被推動，用當下位置算
        d = math.hypot(x - nx, y - ny)
        bearing_err = wrap(math.degrees(math.atan2(ny - y, nx - x)) - hd)
        moved = math.hypot(nx - ox, ny - oy)
        hits = mon.take()
        half = 0.125 if name.startswith("foam_") else (0.04 if name.startswith("ball_") else 0.035)
        front = d - half - 0.166   # 車頭前緣到物件表面（stop_m 的定義）
        ok = bool(r.get("ok")) and 0.2 <= front <= 0.6 and abs(bearing_err) <= 12 and moved < 0.05
        rows.append({"object": name, "ok": ok, "server_ok": bool(r.get("ok")), "final_dist_m": round(d, 2),
                     "bearing_err_deg": round(bearing_err, 1), "object_moved_m": round(moved, 3),
                     "s": round(time.time() - t0, 1), "collisions": [h[2] for h in hits], "collision_at": [[h[2], round(h[3][0], 2), round(h[3][1], 2)] for h in hits]})
        print(f"  {name:<16}: {'ok  ' if ok else 'FAIL'} dist={d:.2f}m bearing_err={bearing_err:+.0f}° moved={moved:.2f}m {time.time() - t0:5.0f}s"
              f"{'  collisions=' + ','.join(sorted(set(h[2] for h in hits))) if hits else ''}")
    return rows


async def main():
    ap = argparse.ArgumentParser(description="LLM-free regression test for the navigation layer")
    ap.add_argument("--goals", type=int, default=12)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default="rotate,goals,landmarks,passages,objects")
    args = ap.parse_args()
    only = set(args.only.split(","))

    rc = RobotClient(SERVER, "")
    h = await rc.health()
    if not h.get("ok") or h.get("robot_link") != "ok":
        raise SystemExit(f"robot server not ready: {h}  （先跑 ./start_all.sh）")
    truth = Truth()
    mon = ContactMonitor()
    mon.start()

    async def run_section(key, coro):
        snap = truth.objects()
        rows = await coro
        pushed = truth.moved_since(snap)
        if pushed:
            print("  !! 物件被推動：" + ", ".join("%s %.2f m" % t for t in pushed))
        report.setdefault("pushed", {})[key] = [{"object": n, "moved_m": round(d, 2)} for n, d in pushed]
        return rows
    t_all = time.time()
    rss0 = coppelia_rss_mb()
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "coppelia_rss_mb_start": rss0, "sections": {}}
    try:
        if "rotate" in only:
            print("\n== rotate ==")
            report["sections"]["rotate"] = await run_section("rotate", test_rotate(rc, truth, [15, 45, 90, 135, 180, -15, -45, -90, -135, -180]))
        if "goals" in only:
            print("\n== random goals ==")
            goals = sample_free_goals(args.goals, args.seed)
            report["sections"]["goals"] = await run_section("goals", test_goals(rc, truth, mon, goals))
        if "landmarks" in only:
            print("\n== landmarks ==")
            report["sections"]["landmarks"] = await run_section("landmarks", test_landmarks(rc, truth, mon))
        if "passages" in only:
            print("\n== passages ==")
            report["sections"]["passages"] = await run_section("passages", test_passages(rc, truth, mon))
        if "objects" in only:
            print("\n== objects (approach geometry, no VLM) ==")
            report["sections"]["objects"] = await run_section("objects", test_objects(rc, truth, mon, args.objects, args.seed))
    finally:
        await rc.stop()
        mon.running = False
    report["coppelia_rss_mb_end"] = coppelia_rss_mb()
    report["duration_s"] = round(time.time() - t_all)

    print("\n" + "=" * 72)
    pushed_all = [(sec, r["object"], r["moved_m"]) for sec, rows in report.get("pushed", {}).items() for r in rows]
    if pushed_all:
        print("物件被推動 %d 次：" % len(pushed_all) + ", ".join("%s/%s %.2f m" % t for t in pushed_all))
    summary = {}
    for sec, rows in report["sections"].items():
        n = len(rows)
        ok = sum(1 for r in rows if r.get("ok"))
        coll = sum(len(r.get("collisions", [])) for r in rows)
        extra = ""
        if sec == "rotate":
            extra = f"  max|err|={max(abs(r['error_deg']) for r in rows):.1f}°"
        if sec == "goals":
            extra = f"  mean err={sum(r['final_err_m'] for r in rows) / max(n, 1):.2f}m"
        summary[sec] = {"pass": ok, "total": n, "collisions": coll}
        print(f"{sec:<10} {ok}/{n} pass  collisions={coll}{extra}")
    report["summary"] = summary
    print(f"CoppeliaSim RSS: {rss0} -> {report['coppelia_rss_mb_end']} MB   總時間 {report['duration_s']}s")
    out = Path("robot_eval/results") / f"nav_selftest_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("報告：", out)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已中止")
