#!/usr/bin/env python
"""模擬分身：把無視窗模擬器（agent 真正在用的那台，zmq 23000）裡 RoboMaster 的底盤位姿與升降柱高度，
即時複製到視窗版 CoppeliaSim（zmq 23001），讓人在視窗裡看得到底盤在跑。

為什麼要這樣：2026-11-10 起 agent 用無視窗那台（視窗版不穩、模擬速度只有 0.55 倍），視窗版只當實體 NexArm 的數位分身
（start_arm.sh --mirror-sim 23001 只同步手臂關節），使用者：「以前跑的時候視窗裡的底盤會動，現在不動了」。
手臂關節不在這裡複製——實體分身已經在同步，而實體手臂本來就是照模擬做的。

用法：
    ~/rmenv/bin/python tools/sim_twin.py            # 前景跑，Ctrl+C 結束
    nohup ~/rmenv/bin/python tools/sim_twin.py > /tmp/sim_twin.log 2>&1 &
    ~/rmenv/bin/python tools/sim_twin.py --real http://127.0.0.1:8767
        # explore 模式（2026-11-12）：來源不是無視窗模擬器而是實體 server 的里程計（/state 的 world_xy / world_heading_deg），
        # 並把它用 ToF 建出來的占據格（/map.json）畫成灰色方塊：視窗版就是「機器人自己建出來的模擬環境」。
        # 視窗版要載空場景 Capstone_Explore_nexarm.ttt（run_explore.sh 會載）。
視窗那台要保持「停止」狀態（分身只是搬物件，不跑物理）。批次每題重開模擬時來源的 handle 會變，這裡遇到錯誤就重新找。
"""
import argparse
import json
import math
import sys
import time
import urllib.request

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def find_robot(sim):
    """跟 robot_server.py / run_batch.py 同一套：多台 RoboMaster 時挑有 GyroSensor 的完整模型。"""
    roots = [o for o in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2) if sim.getObjectAlias(o) == "RoboMaster"]
    if not roots:
        raise RuntimeError("找不到 /RoboMaster")
    good = [h for h in roots if any(sim.getObjectAlias(o) == "GyroSensor" for o in sim.getObjectsInTree(h, sim.handle_all, 0))]
    return (good or roots)[0]


def find_lift(sim, root):
    try:
        return sim.getObject("./arm_lift_joint", {"proxy": root})
    except Exception:
        return -1


def connect(port):
    # 2026-11-17：視窗版關掉再開，舊連線的 recv() 永遠等不到回覆，整支程式從前一天卡到隔天、視窗裡的車不再動
    # （start_office.sh 看到行程還在就說「已在跑」）。先用 TCP 探一下有沒有開，連線加 3 秒逾時：逾時就丟例外 → 主迴圈重連。
    import socket
    import zmq
    socket.create_connection(("127.0.0.1", int(port)), timeout=0.5).close()
    client = RemoteAPIClient(port=port)
    client.socket.setsockopt(zmq.RCVTIMEO, 3000)
    client.socket.setsockopt(zmq.SNDTIMEO, 3000)
    client.socket.setsockopt(zmq.LINGER, 0)
    sim = client.require("sim")
    root = find_robot(sim)
    return sim, root, find_lift(sim, root)


class FovOverlay:
    """在視窗版畫出前鏡頭（Vision_sensor，掛在夾爪下）的視野錐：8 條洋紅線，每次同步都照視窗裡感測器當下的位姿重畫。
    用 drawing object 畫而不是放真的形狀：形狀會被模擬裡的視覺感測器（前鏡頭、俯視相機）渲染到，俯視深度圖還會把它當障礙
    （2026-11-11 實測 special property 0 也擋不掉）。drawing object 只是畫面上的線，不存檔、不被感測器看到。"""

    def __init__(self, sim, root, length_m=0.8):
        self.sim = sim
        self.cam = sim.getObject("./Vision_sensor", {"proxy": root, "noError": True})
        self.h = -1
        if self.cam == -1:
            return
        res = sim.getVisionSensorRes(self.cam)
        ang = sim.getObjectFloatParam(self.cam, sim.visionfloatparam_perspective_angle)
        th = math.tan(ang / 2.0) * length_m
        tv = th * res[1] / float(res[0])   # 視角套在長邊（寬），垂直依長寬比
        c = [(th, tv, length_m), (-th, tv, length_m), (-th, -tv, length_m), (th, -tv, length_m)]
        o = (0.0, 0.0, 0.0)
        self.segs = [(o, c[i]) for i in range(4)] + [(c[i], c[(i + 1) % 4]) for i in range(4)]   # 感測器座標
        self.h = sim.addDrawingObject(sim.drawing_lines, 2.0, 0.0, -1, 8, [1.0, 0.0, 1.0])

    def update(self):
        if self.h == -1:
            return
        m = self.sim.getObjectMatrix(self.cam, self.sim.handle_world)
        def w(p):
            return [m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
                    m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
                    m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11]]
        self.sim.addDrawingObjectItem(self.h, None)   # 清掉上一輪
        for a, b in self.segs:
            self.sim.addDrawingObjectItem(self.h, w(a) + w(b))


class TofMapView:
    """把實體 server 的占據格（/map.json）畫進視窗版：每一列相鄰的占據格合併成一根灰色長方體（靜態、不碰撞）。
    版本號沒變就不重畫；路徑用綠線、目標用洋紅線框、出發點用藍線框（drawing object，不存檔）。"""

    def __init__(self, sim):
        self.sim = sim
        self.shapes = []
        self.version = None
        self.lines = sim.addDrawingObject(sim.drawing_lines, 3.0, 0.0, -1, 400, [0.1, 0.7, 0.1])
        self.marks = sim.addDrawingObject(sim.drawing_lines, 3.0, 0.0, -1, 64, [0.2, 0.4, 1.0])

    def _clear_shapes(self):
        if self.shapes:
            try:
                self.sim.removeObjects(self.shapes)
            except Exception:
                pass
        self.shapes = []

    def update(self, d):
        sim = self.sim
        if d.get("version") != self.version:
            self.version = d.get("version")
            cell = float(d.get("cell_m", 0.05))
            cells = sorted((round(y, 3), round(x, 3)) for x, y in d.get("occupied", []))
            runs = []   # (y, x_start, n)
            for y, x in cells:
                if runs and runs[-1][0] == y and abs(runs[-1][1] + runs[-1][2] * cell - x) < cell * 0.5:
                    runs[-1][2] += 1
                else:
                    runs.append([y, x, 1])
            self._clear_shapes()
            for y, x0, n in runs:
                h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [n * cell, cell, 0.25], 0)
                sim.setObjectPosition(h, sim.handle_world, [x0 + (n - 1) * cell / 2.0, y, 0.125])
                sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
                sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
                sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [0.45, 0.45, 0.45])
                sim.setObjectAlias(h, "tof_map")
                self.shapes.append(h)
        # 路徑與標記每次都重畫（便宜）
        sim.addDrawingObjectItem(self.lines, None)
        path = d.get("path") or []
        rb = (d.get("robot") or {}).get("world_xy")
        pts = ([rb] if rb else []) + [list(p) for p in path]
        for p0, p1 in zip(pts, pts[1:]):
            sim.addDrawingObjectItem(self.lines, [p0[0], p0[1], 0.02, p1[0], p1[1], 0.02])
        sim.addDrawingObjectItem(self.marks, None)
        def box(cx, cy, r, z=0.02):
            c = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
            for i in range(4):
                a, b = c[i], c[(i + 1) % 4]
                sim.addDrawingObjectItem(self.marks, [a[0], a[1], z, b[0], b[1], z])
        home = d.get("home")
        if home:
            box(home[0], home[1], 0.12)
        b = d.get("bounds")
        if b:   # 探索範圍：藍線框
            c = [(b[0], b[2]), (b[1], b[2]), (b[1], b[3]), (b[0], b[3])]
            for i in range(4):
                p0, p1 = c[i], c[(i + 1) % 4]
                sim.addDrawingObjectItem(self.marks, [p0[0], p0[1], 0.03, p1[0], p1[1], 0.03])
        tgt = d.get("target")
        if tgt and tgt.get("xy"):
            box(tgt["xy"][0], tgt["xy"][1], 0.08, 0.05)


def upright_frame(sim, root):
    """RoboMaster 模型根物件的座標系不是「x 前、z 上」：2026-11-12 實測前後輪差在根物件的 z 軸、左右輪差在 y 軸
    （用 setObjectOrientation(0,0,yaw) 擺上去車頭會朝地）。從四個輪子的位置算出根物件座標系裡的「前」「左」「上」，
    之後要擺成世界朝向 heading 就用 pose_matrix()。"""
    def P(name):
        h = sim.getObject("./" + name, {"proxy": root})
        return sim.getObjectPosition(h, root)
    fl, fr, rl, rr = P("front_left_wheel_joint"), P("front_right_wheel_joint"), P("rear_left_wheel_joint"), P("rear_right_wheel_joint")
    def sub(a, b): return [a[i] - b[i] for i in range(3)]
    def mid(a, b): return [(a[i] + b[i]) / 2.0 for i in range(3)]
    def unit(v):
        n = math.sqrt(sum(x * x for x in v)); return [x / n for x in v]
    def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    fwd = unit(sub(mid(fl, fr), mid(rl, rr)))
    left = unit(sub(mid(fl, rl), mid(fr, rr)))
    up = unit(cross(fwd, left))
    left = unit(cross(up, fwd))   # 正交化
    return fwd, left, up


def pose_matrix(frame, x, y, z, heading_deg):
    """世界矩陣（CoppeliaSim 的 12 個數，列優先）：把根物件的 fwd/left/up 對到世界的 (cos,sin,0)/(-sin,cos,0)/(0,0,1)。"""
    fwd, left, up = frame
    a = math.radians(heading_deg)
    F, L, U = [math.cos(a), math.sin(a), 0.0], [-math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]
    # R = F·fwdᵀ + L·leftᵀ + U·upᵀ
    R = [[F[i] * fwd[j] + L[i] * left[j] + U[i] * up[j] for j in range(3)] for i in range(3)]
    return [R[0][0], R[0][1], R[0][2], x, R[1][0], R[1][1], R[1][2], y, R[2][0], R[2][1], R[2][2], z]


def _http_json(url, timeout=2.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main_real(a):
    """explore 模式：位姿從實體 server 的里程計來，障礙從它的 /map.json 來。"""
    base = a.real.rstrip("/")
    dst = None
    fov = view = None
    z_rest = None
    last_map = 0.0
    last_ok = 0.0
    while True:
        try:
            if dst is None:
                dst = connect(a.dst)
                dsim, droot, _ = dst
                z_rest = dsim.getObjectPosition(droot, dsim.handle_world)[2]
                if not (-0.5 < z_rest < 0.5):   # 之前被別的分身程式搬到掉出世界的高度（實測 z=-1002）：拉回這個模型的靜止高度
                    print("[sim_twin] 視窗版機器人高度 %.1f 不合理，改用 0.066" % z_rest, flush=True)
                    z_rest = 0.066
                print("[sim_twin] 目標 %d：RoboMaster handle %d（靜止高度 z=%.3f）" % (a.dst, droot, z_rest), flush=True)
                frame = upright_frame(dsim, droot)
                fov = None   # explore 模式不畫前鏡頭視野錐：地圖方塊已經夠多，洋紅線只會把畫面弄亂（2026-11-12 使用者截圖）
                view = None if a.no_map else TofMapView(dsim)
                print("[sim_twin] 實體來源 %s：位姿走里程計，%s" % (base, "不畫 server 的占據格（場景本身就是掃描來的場地）" if a.no_map else "占據格畫成灰方塊"), flush=True)
            dsim, droot, _ = dst
            if dsim.getSimulationState() != 0:
                print("[sim_twin] 視窗版模擬正在播放，請按停止", flush=True)
                time.sleep(1.0)
                continue
            st = _http_json(base + "/state")
            xy, hd = st.get("world_xy"), st.get("world_heading_deg")
            if xy and hd is not None:
                dsim.setObjectMatrix(droot, dsim.handle_world, pose_matrix(frame, float(xy[0]), float(xy[1]), z_rest, float(hd)))
            if fov is not None:
                fov.update()
            if view is not None and time.time() - last_map > 1.0:
                last_map = time.time()
                try:
                    view.update(_http_json(base + "/map.json"))
                except Exception as e:
                    if time.time() - last_ok > 30:
                        print("[sim_twin] /map.json 讀不到：%s" % str(e)[:80], flush=True)
            if time.time() - last_ok > 30:
                print("[sim_twin] 同步中：底盤 (%.2f, %.2f) 朝向 %s°，方塊 %d 根" % (
                    (xy or [0, 0])[0], (xy or [0, 0])[1], hd, len(view.shapes) if view is not None else 0), flush=True)
                last_ok = time.time()
            time.sleep(1.0 / a.hz)
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            print("[sim_twin] %s：%s，2 秒後重連" % (type(e).__name__, str(e)[:120]), flush=True)
            dst = None
            time.sleep(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=int, default=23000, help="agent 用的模擬器（無視窗）")
    ap.add_argument("--dst", type=int, default=23001, help="要顯示的模擬器（視窗版）")
    ap.add_argument("--real", default=None, metavar="URL",
                    help="explore 模式：來源改成實體 server（例如 http://127.0.0.1:8767）的里程計與 ToF 地圖，不連 --src")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--no-map", action="store_true", help="--real 時不要把 server 的占據格畫成方塊（場景已是手機掃描建的 Scan_*.ttt 時用）")
    a = ap.parse_args()
    if a.real:
        return main_real(a)
    src = dst = None
    fov = None
    warned_running = False
    last_ok = 0.0
    while True:
        try:
            if src is None:
                src = connect(a.src)
                print("[sim_twin] 來源 %d：RoboMaster handle %d" % (a.src, src[1]), flush=True)
            if dst is None:
                dst = connect(a.dst)
                print("[sim_twin] 目標 %d：RoboMaster handle %d" % (a.dst, dst[1]), flush=True)
                fov = FovOverlay(dst[0], dst[1])
                print("[sim_twin] 前鏡頭視野錐：%s" % ("有畫" if fov.h != -1 else "找不到 Vision_sensor，不畫"), flush=True)
            ssim, sroot, slift = src
            dsim, droot, dlift = dst
            if dsim.getSimulationState() != 0:
                if not warned_running:
                    print("[sim_twin] 視窗版模擬正在播放，分身只在停止狀態下搬物件；請按停止", flush=True)
                    warned_running = True
                time.sleep(1.0)
                continue
            warned_running = False
            m = ssim.getObjectMatrix(sroot, ssim.handle_world)
            dsim.setObjectMatrix(droot, dsim.handle_world, m)
            if slift != -1 and dlift != -1:
                dsim.setJointPosition(dlift, ssim.getJointPosition(slift))
            fov.update()
            if time.time() - last_ok > 30:
                print("[sim_twin] 同步中：底盤 (%.2f, %.2f)" % (m[3], m[7]), flush=True)
                last_ok = time.time()
            time.sleep(1.0 / a.hz)
        except KeyboardInterrupt:
            return 0
        except Exception as e:  # 模擬重開（handle 變了）或某一台還沒起來：重新連
            print("[sim_twin] %s：%s，2 秒後重連" % (type(e).__name__, str(e)[:120]), flush=True)
            src = dst = None
            fov = None
            time.sleep(2.0)


if __name__ == "__main__":
    sys.exit(main())
