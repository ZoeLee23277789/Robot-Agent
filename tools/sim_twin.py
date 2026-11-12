#!/usr/bin/env python
"""模擬分身：把無視窗模擬器（agent 真正在用的那台，zmq 23000）裡 RoboMaster 的底盤位姿與升降柱高度，
即時複製到視窗版 CoppeliaSim（zmq 23001），讓人在視窗裡看得到底盤在跑。

為什麼要這樣：2026-11-10 起 agent 用無視窗那台（視窗版不穩、模擬速度只有 0.55 倍），視窗版只當實體 NexArm 的數位分身
（start_arm.sh --mirror-sim 23001 只同步手臂關節），使用者：「以前跑的時候視窗裡的底盤會動，現在不動了」。
手臂關節不在這裡複製——實體分身已經在同步，而實體手臂本來就是照模擬做的。

用法：
    ~/rmenv/bin/python tools/sim_twin.py            # 前景跑，Ctrl+C 結束
    nohup ~/rmenv/bin/python tools/sim_twin.py > /tmp/sim_twin.log 2>&1 &
視窗那台要保持「停止」狀態（分身只是搬物件，不跑物理）。批次每題重開模擬時來源的 handle 會變，這裡遇到錯誤就重新找。
"""
import argparse
import math
import sys
import time

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
    sim = RemoteAPIClient(port=port).require("sim")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=int, default=23000, help="agent 用的模擬器（無視窗）")
    ap.add_argument("--dst", type=int, default=23001, help="要顯示的模擬器（視窗版）")
    ap.add_argument("--hz", type=float, default=10.0)
    a = ap.parse_args()
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
