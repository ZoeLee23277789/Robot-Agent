"""
把 /RoboMaster 移回 build_playground_new.py 定義的起點 (ROBOT_START)，方向轉正對著 +y、貼地，
不會動到場景裡任何其他物件。

sim/real 模式下，robot_server.py 的 /reset 只會收手臂、開夾爪，位置不會動（README 就是這樣寫的，
mock 模式才會整個重置）——手動測試、把機器人搬到各種角落之後，這支腳本就是拿來把它放回乾淨的起點，
不用重新跑一次 build_playground_new.py 整個重建場景。

跟 build_playground_new.py 的 prepare_robot() 用同一套「量相機朝向、轉正、移到起點、貼地」邏輯，
但故意不呼叫它最後 sim.resetDynamicObject() 那段：那段是設計給模擬「停止」狀態、場景剛蓋好、
機器人控制腳本還沒開始跑的時候用的。實測發現如果在模擬「播放中」、RoboMaster 控制腳本已經在跑
的時候呼叫 resetDynamicObject()，會讓底盤整個沒反應（呼叫 move_chassis 都回 timeout、travelled
永遠是 0），要停止再重新播放模擬才能救回來。所以這支腳本只搬位置、不動態力學重置，可以在模擬
播放中安全執行。

使用方式：
    python reset_robot.py
"""
import argparse
import math

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from build_playground_new import ROBOT_HEADING, ROBOT_START, world_aabb


def reset_robot(sim, path, fixed_scale):
    h = sim.getObject(path, {"noError": True})
    if h == -1:
        raise SystemExit(f"找不到 {path}")
    rp = sim.getObjectPosition(h, sim.handle_world)

    cam = -1
    for o in sim.getObjectsInTree(h, sim.object_visionsensor_type, 0):
        cam = o
        break
    if cam != -1:
        cm = sim.getObjectMatrix(cam, sim.handle_world)
        heading = math.atan2(cm[6], cm[2])
    else:
        m = sim.getObjectMatrix(h, sim.handle_world)
        heading = math.atan2(m[4], m[0])
        print("找不到相機，改用 local +x 判斷車頭")

    m = sim.getObjectMatrix(h, sim.handle_world)
    sim.setObjectMatrix(h, sim.rotateAroundAxis(m, [0, 0, 1], rp, ROBOT_HEADING - heading), sim.handle_world)
    if cam != -1:
        cm = sim.getObjectMatrix(cam, sim.handle_world)
        print(f"轉向後相機朝向 {math.degrees(math.atan2(cm[6], cm[2])):.0f} 度（+y 應為 90）")

    lo, hi = world_aabb(sim, h)
    width = hi[0] - lo[0]
    scale = fixed_scale or (width / 0.24 if width > 0 else 1.0)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    p = sim.getObjectPosition(h, sim.handle_world)
    sim.setObjectPosition(h, [p[0] + ROBOT_START[0] * scale - cx, p[1] + ROBOT_START[1] * scale - cy,
                              p[2] - lo[2] + 0.01], sim.handle_world)
    return scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="/RoboMaster")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="要跟建場景那次用的 --scale 一致，預設 1.0（build_playground_new.py 的預設值）")
    args = ap.parse_args()

    sim = RemoteAPIClient().require("sim")
    scale = reset_robot(sim, args.robot, args.scale or None)
    print(f"機器人已經移回起點、面向 +y，場景比例 = {scale:.3f}")


if __name__ == "__main__":
    main()
