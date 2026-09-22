"""
把 RoboMaster 放到幾個位置與朝向，擷取它的相機（vision sensor）畫面存成 PNG，
用來檢查視覺模型實際會看到什麼。

使用方式（CoppeliaSim 開著、模擬停止狀態）：
    pip install coppeliasim-zmqremoteapi-client pillow
    python camera_check.py                       # 預設在起點轉一圈 + 走到幾個地標前
    python camera_check.py --sensor /RoboMaster/Vision_sensor
    python camera_check.py --poses "0,-1,90" "1,0,0" "-1,1,180"   # x,y,yaw度

輸出在 camera_shots/ 資料夾，檔名含位置與朝向。
腳本會啟動模擬（讓 vision sensor 更新）、擷取完後停止，並把機器人放回原位。
"""

import argparse
import math
import os
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def find_vision_sensor(sim, robot):
    for h in sim.getObjectsInTree(robot, sim.object_visionsensor_type, 0):
        return h
    return -1


def robot_yaw_from(m):
    return math.atan2(m[4], m[0])


def robot_yaw(sim, h):
    return robot_yaw_from(sim.getObjectMatrix(h, sim.handle_world))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="/RoboMaster")
    ap.add_argument("--sensor", default=None, help="vision sensor 路徑，不給就自動找機器人底下第一個")
    ap.add_argument("--poses", nargs="*", default=None, help='每個為 "x,y,yaw度"')
    ap.add_argument("--yaw-offset", type=float, default=0.0,
                    help="模型的車頭方向與 local +x 的夾角（度），和 build_playroom 的 --extra-yaw 相同")
    ap.add_argument("--out", default="camera_shots")
    args = ap.parse_args()

    from PIL import Image

    sim = RemoteAPIClient().require("sim")
    robot = sim.getObject(args.robot)
    sensor = sim.getObject(args.sensor) if args.sensor else find_vision_sensor(sim, robot)
    if sensor == -1:
        raise SystemExit("找不到 vision sensor，請用 --sensor 指定路徑（在場景樹裡看名稱）")
    print("使用 vision sensor:", sim.getObjectAlias(sensor, 2))
    res = sim.getVisionSensorRes(sensor)
    fov = sim.getObjectFloatParam(sensor, sim.visionfloatparam_perspective_angle)
    print(f"解析度 {res[0]}x{res[1]}，視角 {math.degrees(fov):.1f} 度")

    if args.poses:
        poses = [tuple(float(v) for v in p.split(",")) for p in args.poses]
    else:
        p0 = sim.getObjectPosition(robot, sim.handle_world)
        yaw0 = math.degrees(robot_yaw(sim, robot)) - args.yaw_offset
        poses = [(p0[0], p0[1], yaw0 + d) for d in (0, 45, 90, 135, 180, 225, 270, 315)]

    os.makedirs(args.out, exist_ok=True)
    orig_pos = sim.getObjectPosition(robot, sim.handle_world)
    orig_mat = sim.getObjectMatrix(robot, sim.handle_world)

    def place(x, y, yaw_deg):
        """繞世界 z 軸把機器人轉到指定朝向，再平移到 (x, y)；不動模型本身的傾斜。"""
        target = math.radians(yaw_deg + args.yaw_offset)
        m = sim.rotateAroundAxis(orig_mat, [0, 0, 1], orig_pos, target - robot_yaw_from(orig_mat))
        m = list(m)
        m[3], m[7] = x, y
        sim.setObjectMatrix(robot, m, sim.handle_world)

    sim.startSimulation()
    try:
        time.sleep(0.5)
        for x, y, yaw in poses:
            place(x, y, yaw)
            time.sleep(0.3)
            sim.handleVisionSensor(sensor)
            img, res = sim.getVisionSensorImg(sensor)
            im = Image.frombytes("RGB", (res[0], res[1]), img).transpose(Image.FLIP_TOP_BOTTOM)
            name = f"{args.out}/x{x:+.2f}_y{y:+.2f}_yaw{yaw:+.0f}.png"
            im.save(name)
            print("已存", name)
    finally:
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)
        sim.setObjectMatrix(robot, orig_mat, sim.handle_world)
    print("完成。把幾張圖丟給你的視覺模型問「你看到什麼」，就知道環境設計是否可行。")


if __name__ == "__main__":
    main()
