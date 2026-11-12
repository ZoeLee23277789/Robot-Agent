"""
模擬器裡的 NexArm（Env/add_nexarm.py 建的模型）後端：方法、回傳值跟 nexarm.py 的 NexArm 一模一樣，
robot_server.py 的 _nex_* 處理函式兩邊共用，agent 看不出差別。

跟實體一樣的地方（刻意對齊，模擬練的動作才能直接搬到實體）：
  - 座標：x 前伸、y 側向（左正）、z 高度，單位 mm，相對安裝面；pitch 度，0 水平、−90 朝下。
  - 可達範圍查同一張實測表 nexarm.REACH（模型在數學上到得了更多地方，但實體到不了的這裡也不去）。
  - 逆運動學用實體韌體同一組解（手肘朝上朝後：五組 CMD 7 樣本全部如此）。
  - 狀態 dict 一樣有 servos（用擬合的 o1/o2/c 反推出伺服值），夾爪 claw 一樣是度：−60 全開、50 全合。
  - 夾到東西的判據一樣是「手指停在離目標幾度」：有東西在指間時手指停在物件寬度，算回 claw 的差。

跟實體不一樣、要知道的：
  - 關節是運動學模式（直接設角度），沒有伺服下垂、沒有負載停短，所以 corrected 永遠 False、誤差接近 0。
  - 抓取不是靠物理摩擦：合爪時指間有動態小物件就把它掛到 nexarm_tcp 底下（設成靜態跟著走），開爪時放回原父物件、恢復動態。
    舊手臂的 Lua 夾爪腳本也是用掛接（attachPoint）做的，agent 端沒有差別。
  - zmq 的 REQ socket 不能跨執行緒共用（robot_server 踩過）：動作執行緒跟狀態輪詢執行緒各自開一個 RemoteAPIClient。
"""

import math
import threading
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

import nexarm
from nexarm import (CLAW_CLOSE, CLAW_OPEN, COUNTS_PER_DEG, HOLD_SHORTFALL_DEG, HOME, REACH, REACH_TOL_DEG, REACH_TOL_MM,
                    SERVO_CENTER, WORKSPACE, _clamp_to)

# 擬合的運動學（mm / 度），跟 nexarm_workspace.py、Env/add_nexarm.py 同一組
AX, AZ, L1, L2, L3 = -0.12, 110.35, 227.45, 145.31, 129.79
O1, O2, C = 80.79, -80.39, -0.04
GAP_OPEN_MM, FINGER_T_MM = 35.0, 8.0     # Env/add_nexarm.py 的 GAP_OPEN / FINGER 厚度
GRASP_MAX_DIM_M = 0.12                   # 比這個大的東西不當作可抓物件（桌子、桶子）
STEP_HZ = 25                             # 關節插值頻率


def ik_planar(r, z, pitch):
    """(水平距離, 高度, 俯仰) → (p1, p2)：上臂、前臂相對水平面的角度（度）。到不了回 None。手肘朝上朝後那一組解。"""
    wx = r - AX - L3 * math.cos(math.radians(pitch))
    wz = z - AZ - L3 * math.sin(math.radians(pitch))
    d = math.hypot(wx, wz)
    if d > L1 + L2 - 1e-6 or d < abs(L1 - L2) + 1e-6:
        return None
    a = math.atan2(wz, wx)
    b = math.acos(max(-1.0, min(1.0, (L1 ** 2 + d ** 2 - L2 ** 2) / (2 * L1 * d))))
    g = math.acos(max(-1.0, min(1.0, (L1 ** 2 + L2 ** 2 - d ** 2) / (2 * L1 * L2))))
    p1 = math.degrees(a + b)
    p2 = p1 - (180.0 - math.degrees(g))
    return p1, p2


def servos_to_joints(servos):
    """實體回報的六顆伺服位置 → (q1, p1, p2, pitch, claw) 度，跟 nexarm_workspace.py 的 FK 同一組換算。"""
    d = [(s - SERVO_CENTER) / COUNTS_PER_DEG for s in servos]
    q1 = -d[0]          # 實測：底座往 +y（韌體正向）轉時伺服 1 的數值變小（2026-11-10 nexarm_rotate_test.py）
    p1 = O1 + d[1]
    p2 = p1 + O2 + d[2]
    pitch = C + d[1] + d[2] - d[3]
    return q1, p1, p2, pitch, d[4], d[5]


def set_model_joints(sim, h, q1, p1, p2, pitch, roll, gap_mm):
    """把關節角套到 Env/add_nexarm.py 的模型上（j2 = p1、j3 = p2 − p1、j4 = pitch − p2、j5 = roll；手指關節 = 半開度 + 半指厚）。"""
    sim.setJointPosition(h["j1"], math.radians(q1))
    sim.setJointPosition(h["j2"], math.radians(p1))
    sim.setJointPosition(h["j3"], math.radians(p2 - p1))
    sim.setJointPosition(h["j4"], math.radians(pitch - p2))
    sim.setJointPosition(h["j5"], math.radians(-roll))    # 模型 j5 正向跟韌體 roll 正向相反（2026-11-10 使用者看分身確認）
    half = (gap_mm / 2 + FINGER_T_MM / 2) / 1000.0
    sim.setJointPosition(h["fj_l"], half)
    sim.setJointPosition(h["fj_r"], half)


class SimMirror(object):
    """數位分身：把實體 NexArm 的伺服位置即時套到模擬器裡的模型上，實體怎麼動模擬就怎麼動（只讀實體、不送指令）。
    status_fn 回傳實體的狀態 dict（nexarm.NexArm.status()[0]）。zmq 連線在自己的執行緒裡建，不跟別人共用。"""

    def __init__(self, status_fn, port=23001, rate_hz=10.0):
        self.status_fn, self.port, self.rate_hz = status_fn, port, rate_hz
        self.ok = False
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        try:
            sim = RemoteAPIClient(port=self.port).require("sim")
            h = NexArmSim._find_handles(sim)
        except Exception as e:
            self.error = str(e)
            print("[nexarm_sim] 數位分身連不上模擬器 %d：%s" % (self.port, e))
            return
        self.ok = True
        print("[nexarm_sim] 數位分身：實體 NexArm 的動作會即時顯示在模擬器（zmq %d）" % self.port)
        last = None
        while not self._stop.is_set():
            try:
                st = self.status_fn()
                if st and st.get("servos") and st["servos"] != last:
                    q1, p1, p2, pitch, roll, claw = servos_to_joints(st["servos"])
                    set_model_joints(sim, h, q1, p1, p2, pitch, roll, claw_to_gap_mm(claw))
                    last = list(st["servos"])
            except Exception as e:
                self.error = str(e)
                print("[nexarm_sim] 數位分身更新失敗：%s（2 秒後重試）" % e)
                time.sleep(2.0)
            time.sleep(1.0 / self.rate_hz)

    def close(self):
        self._stop.set()


def claw_to_gap_mm(claw):
    """夾爪角度 → 兩指內緣距離。實體 −60 全開、50 全合，中間當線性（TODO 實體量幾個點）。"""
    return max(0.0, min(GAP_OPEN_MM, GAP_OPEN_MM * (CLAW_CLOSE - claw) / (CLAW_CLOSE - CLAW_OPEN)))


def gap_mm_to_claw(gap):
    return CLAW_CLOSE - (CLAW_CLOSE - CLAW_OPEN) * max(0.0, min(GAP_OPEN_MM, gap)) / GAP_OPEN_MM


class NexArmSim(object):
    def __init__(self, port=23000, on_status=None, rate_hz=5.0):
        self.port = port
        self.on_status = on_status
        self.sim = RemoteAPIClient(port=port).require("sim")      # 動作執行緒專用
        self._h = self._find_handles(self.sim)
        self._lock = threading.Lock()
        self._target = None
        self._held = None            # (handle, 原本的父物件, 原本是不是靜態)
        self._claw = self._read_claw(self.sim)
        self._status = None
        self._status_ts = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poller, args=(rate_hz,), daemon=True)
        self._thread.start()
        if not self.wait_status(3.0):
            raise RuntimeError("模擬器裡的 NexArm 3 秒內讀不到狀態")
        t0 = time.time()
        n = len(self._grasp_candidates())    # 先建好可抓物件清單，第一次合爪才不用等 4 秒
        print("[nexarm_sim] 場景裡 %d 個可抓的動態小物件（%.1fs）" % (n, time.time() - t0))

    # ---- 場景 ----
    @staticmethod
    def _find_handles(sim):
        roots = [o for o in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2) if sim.getObjectAlias(o) == "RoboMaster"]
        good = [h for h in roots if any(sim.getObjectAlias(o) == "GyroSensor" for o in sim.getObjectsInTree(h, sim.handle_all, 0))]
        if not roots:
            raise RuntimeError("場景裡沒有 /RoboMaster")
        root = (good or roots)[0]
        names = ["mount", "j1", "j2", "j3", "j4", "j5", "fj_l", "fj_r", "tcp", "gripper"]
        h = {"root": root}
        for n in names:
            obj = sim.getObject("./nexarm_" + n, {"proxy": root, "noError": True})
            if obj == -1:
                raise RuntimeError("場景裡沒有 nexarm_%s：先跑 Env/add_nexarm.py，並用 Capstone_Playground_nexarm.ttt" % n)
            h[n] = obj
        # 手腕相機 2026-11-11 已從場景拆掉（前鏡頭改回車頭的 EP 原廠相機），有就記、沒有也不是錯
        h["wrist_cam"] = sim.getObject("./nexarm_wrist_cam", {"proxy": root, "noError": True})
        return h

    def _read_joints(self, sim):
        """(q1, p1, p2, pitch) 度：模型的 j2 = p1、j3 = p2 − p1、j4 = pitch − p2。"""
        q1 = math.degrees(sim.getJointPosition(self._h["j1"]))
        p1 = math.degrees(sim.getJointPosition(self._h["j2"]))
        p2 = p1 + math.degrees(sim.getJointPosition(self._h["j3"]))
        pitch = p2 + math.degrees(sim.getJointPosition(self._h["j4"]))
        roll = -math.degrees(sim.getJointPosition(self._h["j5"]))
        return q1, p1, p2, pitch, roll

    def _read_claw(self, sim):
        gap = 2 * sim.getJointPosition(self._h["fj_l"]) * 1000.0 - FINGER_T_MM
        return gap_mm_to_claw(gap)

    def _make_status(self, sim):
        q1, p1, p2, pitch, roll = self._read_joints(sim)
        pos = sim.getObjectPosition(self._h["tcp"], self._h["mount"])
        claw = self._read_claw(sim)
        d2, d3 = p1 - O1, p2 - p1 - O2
        d4 = C + d2 + d3 - pitch
        servos = [int(round(SERVO_CENTER + v * COUNTS_PER_DEG)) for v in (-q1, d2, d3, d4, roll, claw)]
        return {"x": int(round(pos[0] * 1000)), "y": int(round(pos[1] * 1000)), "z": int(round(pos[2] * 1000)),
                "pitch": round(pitch, 1), "roll": int(round(roll)), "claw": int(round(claw)), "servos": servos}

    def _poller(self, rate_hz):
        sim = RemoteAPIClient(port=self.port).require("sim")     # 這個執行緒自己的連線
        period = 1.0 / rate_hz
        while not self._stop.is_set():
            try:
                st = self._make_status(sim)
                self._status, self._status_ts = st, time.time()
                if self.on_status is not None:
                    try:
                        self.on_status(st)
                    except Exception:
                        pass
            except Exception as e:
                print("[nexarm_sim] 讀狀態失敗：%s" % e)
                time.sleep(1.0)
            time.sleep(period)

    # ---- 跟 nexarm.NexArm 相同的介面 ----
    def status(self):
        st = self._status
        return (dict(st) if st else None), (time.time() - self._status_ts if st else None)

    def wait_status(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._status is not None:
                return True
            time.sleep(0.05)
        return False

    def link_ok(self, max_age=1.5):
        return self._status is not None and time.time() - self._status_ts < max_age

    limits_here = nexarm.NexArm.limits_here     # 只用 self._status 跟 REACH，兩邊共用

    def _base_pose(self):
        if self._target is not None:
            return list(self._target)
        st = self._status
        if st is None:
            return None
        return [st["x"], st["y"], st["z"], st["pitch"], st["roll"], st["claw"]]

    def _set_joints(self, q1, p1, p2, pitch, roll, gap_mm):
        set_model_joints(self.sim, self._h, q1, p1, p2, pitch, roll, gap_mm)

    def _interpolate(self, start, goal, ms):
        """關節空間線性插值，牆鐘 ms 毫秒走完（最少 4 步），讓相機、距離感測器看得到過程。"""
        steps = max(4, int(ms / 1000.0 * STEP_HZ))
        for i in range(1, steps + 1):
            a = i / steps
            cur = [s + (g - s) * a for s, g in zip(start, goal)]
            self._set_joints(*cur)
            time.sleep(ms / 1000.0 / steps)

    def move_to(self, x=None, y=None, z=None, pitch=None, roll=None, claw=None, ms=None, correct=True):
        base = self._base_pose()
        if base is None:
            return {"ok": False, "message": "no status from the arm", "pose": None}
        names = ["x", "y", "z", "pitch", "roll", "claw"]
        want = [x, y, z, pitch, roll, claw]
        tgt, clamped = [], []
        for name, b, w in zip(names, base, want):
            v = float(b if w is None else w)
            c = _clamp_to(name, v)
            if abs(c - v) > 1e-6:
                clamped.append("%s %.0f->%.0f" % (name, v, c))
            tgt.append(c)
        tgt[0], tgt[1], note = nexarm.clamp_yaw(tgt[0], tgt[1])
        if note:
            clamped.append(note)
        r = math.hypot(tgt[0], tgt[1])
        before = dict(self._status) if self._status else None
        if REACH is not None and not REACH.reachable(r, tgt[2], tgt[3]):
            span = REACH.z_span(r, tgt[3])
            if span is None:
                rs = REACH.r_span(tgt[2], tgt[3])
                return {"ok": False, "pose": before, "unreachable": True,
                        "message": ("target out of reach: at pitch %.0f deg the gripper cannot get to forward %.0f mm at any "
                                    "height%s" % (tgt[3], r, ("; at height %.0f mm the forward reach is %d to %d mm" % (tgt[2], rs[0], rs[1]))
                                                  if rs else ""))}
            zc = min(max(tgt[2], span[0]), span[1])
            clamped.append("z %.0f->%.0f (reachable height at forward %.0f mm, pitch %.0f deg: %d to %d mm)"
                           % (tgt[2], zc, r, tgt[3], span[0], span[1]))
            tgt[2] = float(zc)

        sol = ik_planar(r, tgt[2], tgt[3])
        cur = self._read_joints(self.sim)
        q1 = math.degrees(math.atan2(tgt[1], tgt[0])) if r > 1e-6 else cur[0]
        if sol is None:
            # 表說到得了但解析 IK 沒解（表格點是 10 mm、容許 3 mm 誤差）：不動，老實回沒到
            return {"ok": False, "pose": before, "before": before, "target": dict(zip(names, tgt)), "unreachable": True,
                    "message": "no inverse-kinematics solution for forward %.0f mm, height %.0f mm, pitch %.0f deg" % (r, tgt[2], tgt[3])}
        dist = math.hypot(math.hypot(before["x"] - tgt[0], before["y"] - tgt[1]), before["z"] - tgt[2]) if before else 0.0
        if ms is None:
            ms = int(min(3000, max(500, 500 + dist * 5)))
        gap_goal = claw_to_gap_mm(tgt[5])
        with self._lock:
            # 合爪：指間有東西就只合到物件寬度並掛接；開爪：放掉
            grasp_note = ""
            if tgt[5] > self._claw + 1e-6 and self._held is None:
                obj, width = self._object_between_fingers()
                # 比夾爪全開還寬的東西夾不住（實體會被手指推開），當作指間沒東西；GAP_OPEN_MM 要等實體量過才準
                if obj is not None and gap_goal <= width <= GAP_OPEN_MM + 2.0:
                    gap_goal = width
            elif tgt[5] < self._claw - 1e-6 and self._held is not None:
                self._release()
            start = list(cur) + [claw_to_gap_mm(self._claw)]
            self._interpolate(start, [q1, sol[0], sol[1], tgt[3], tgt[4], gap_goal], ms)
            if tgt[5] > self._claw + 1e-6 and self._held is None:
                obj, width = self._object_between_fingers()
                if obj is not None and gap_goal <= width + 1.0 and width <= GAP_OPEN_MM + 2.0:
                    self._attach(obj)
            self._claw = gap_mm_to_claw(gap_goal)
            after = self._make_status(self.sim)
            self._status, self._status_ts = after, time.time()
            if self.on_status is not None:   # 輪詢執行緒可能剛讀到插值中途的值，馬上用最終狀態蓋掉（實測 /state 讀到 claw −54）
                try:
                    self.on_status(after)
                except Exception:
                    pass
        err = math.hypot(math.hypot(after["x"] - tgt[0], after["y"] - tgt[1]), after["z"] - tgt[2])
        perr = abs(after["pitch"] - tgt[3])
        ok = err <= REACH_TOL_MM and perr <= REACH_TOL_DEG
        self._target = tuple(tgt) if ok else None
        return {"ok": ok, "pose": dict(after), "before": before, "target": dict(zip(names, tgt)),
                "error_mm": round(err, 1), "pitch_error_deg": round(perr, 1), "clamped": ", ".join(clamped),
                "ms": ms, "corrected": False}

    def move_by(self, dx=0.0, dy=0.0, dz=0.0, dpitch=0.0, dyaw=0.0, droll=0.0):
        base = self._base_pose()
        if base is None:
            return {"ok": False, "message": "no status from the arm", "pose": None}
        return self.move_to(*nexarm.relative_target(base, dx, dy, dz, dpitch, dyaw, droll))

    def home(self):
        return self.move_to(*HOME, roll=0.0)

    def stow(self):
        r = self.move_to(x=120.0, y=0.0, z=200.0, pitch=-90.0, roll=0.0)
        if not r.get("ok"):
            return r
        return self.move_to(*nexarm.STOW, roll=0.0)

    def set_claw(self, claw, ms=800):
        r = self.move_to(claw=claw, ms=ms, correct=False)
        st = r.get("pose")
        if st:
            expected = SERVO_CENTER + claw * COUNTS_PER_DEG
            shortfall = (expected - st["servos"][5]) / COUNTS_PER_DEG
            r["shortfall_deg"] = round(shortfall, 1)
            r["holding"] = bool(claw > 0 and shortfall >= HOLD_SHORTFALL_DEG and self._held is not None)
        else:
            r["shortfall_deg"] = None
            r["holding"] = False
        return r

    # ---- 抓取：掛接 ----
    def _grasp_candidates(self):
        """場景裡可抓的東西：動態、會碰撞、最長邊 ≤ 12 cm、不在機器人身上。結果快取 20 秒：
        604 個形狀每個問 9 次要 9 秒（第一版實測），物件清單不常變，每次只需要重新問位置。回 {handle: 最短邊 mm}。"""
        now = time.time()
        cache = getattr(self, "_cand_cache", None)
        if cache is not None and now - cache[0] < 20.0:
            return cache[1]
        sim, h = self.sim, self._h
        robot_tree = set(sim.getObjectsInTree(h["root"], sim.handle_all, 0))
        out = {}
        for sh in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0):
            if sh in robot_tree or sim.getObjectInt32Param(sh, sim.shapeintparam_static):
                continue
            if not sim.getObjectInt32Param(sh, sim.shapeintparam_respondable):
                continue
            try:
                mn = [sim.getObjectFloatParam(sh, p) for p in (sim.objfloatparam_objbbox_min_x, sim.objfloatparam_objbbox_min_y, sim.objfloatparam_objbbox_min_z)]
                mx = [sim.getObjectFloatParam(sh, p) for p in (sim.objfloatparam_objbbox_max_x, sim.objfloatparam_objbbox_max_y, sim.objfloatparam_objbbox_max_z)]
            except Exception:
                continue
            dims = [b - a for a, b in zip(mn, mx)]
            if max(dims) <= GRASP_MAX_DIM_M:
                out[sh] = min(dims) * 1000.0
        self._cand_cache = (now, out)
        return out

    def _object_between_fingers(self):
        """指間有沒有可抓的動態小物件：回 (handle, 物件沿側向的寬度 mm)；沒有回 (None, None)。"""
        sim, h = self.sim, self._h
        tcp_x = L3 / 1000.0 - 0.05
        best = None
        for sh, width in self._grasp_candidates().items():
            try:
                p = sim.getObjectPosition(sh, h["gripper"])      # 夾爪座標：x 前、y 上、z 側向
            except Exception:
                self._cand_cache = None                            # 物件被刪了，下次重建清單
                continue
            if abs(p[0] - tcp_x) <= 0.035 and abs(p[1]) <= 0.03 and abs(p[2]) <= GAP_OPEN_MM / 2000.0:
                d = abs(p[0] - tcp_x) + abs(p[2])
                if best is None or d < best[0]:
                    best = (d, sh, width)
        return (best[1], best[2]) if best else (None, None)

    def _attach(self, obj):
        sim = self.sim
        parent = sim.getObjectParent(obj)
        was_static = sim.getObjectInt32Param(obj, sim.shapeintparam_static)
        sim.setObjectInt32Param(obj, sim.shapeintparam_static, 1)
        sim.setObjectParent(obj, self._h["tcp"], True)
        self._held = (obj, parent, was_static)
        print("[nexarm_sim] 夾到 %s，掛在夾爪上" % sim.getObjectAlias(obj))

    def _release(self):
        sim = self.sim
        obj, parent, was_static = self._held
        self._held = None
        try:
            sim.setObjectParent(obj, parent, True)
            sim.setObjectInt32Param(obj, sim.shapeintparam_static, was_static)
            sim.resetDynamicObject(obj)
            print("[nexarm_sim] 放開 %s" % sim.getObjectAlias(obj))
        except Exception as e:
            print("[nexarm_sim] 放開物件失敗：%s" % e)

    def held_object(self):
        return self._held[0] if self._held else None

    def close(self):
        self._stop.set()
        try:
            self._thread.join(timeout=0.5)
        except Exception:
            pass


if __name__ == "__main__":
    # 工具用法：把 NexArm 場景載進某個 CoppeliaSim（通常是視窗版 23001）給數位分身顯示用
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23001)
    ap.add_argument("--load-nexarm-scene", action="store_true", help="載入 scenes/Capstone_Playground_nexarm.ttt（會關掉目前開著的場景）")
    a = ap.parse_args()
    sim = RemoteAPIClient(port=a.port).require("sim")
    if a.load_nexarm_scene:
        if sim.getSimulationState() != sim.simulation_stopped:
            sim.stopSimulation()
            while sim.getSimulationState() != sim.simulation_stopped:
                time.sleep(0.1)
        sim.loadScene(os.path.expanduser("~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04/scenes/Capstone_Playground_nexarm.ttt"))
    print("scene:", sim.getStringParam(sim.stringparam_scene_path_and_name), "| nexarm:", NexArmSim._find_handles(sim)["mount"])
