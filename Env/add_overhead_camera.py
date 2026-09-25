"""
在房間正中央上方加一個朝下看的正交（orthographic）攝影機，給整個 6x6m 場地一張俯視圖。

用途：機器人自己的攝影機裝在手臂上、只能看前方，找東西要靠自轉搜索，很容易找一圈都找不到、
或轉過頭跟丟。這個俯視攝影機讓 agent 多一個 locate_overhead(object) 動作，一次看到整個場地，
直接算出目標在世界座標系裡相對於機器人的方位角跟距離，不用再靠一步一步轉、一步一步猜。

只在乎 build_playground_new.py 蓋出來的場景：房間中心是世界座標原點，這個假設寫死在
robot_agent/perception.py 的 OVERHEAD_ORTHO_SIZE / 相機世界座標常數裡，跟這裡的
ORTHO_SIZE、相機 (0,0) 的位置要保持一致，改這裡的話那邊也要一起改。

使用方式（模擬播放中或停止都可以）：
    python add_overhead_camera.py
    python add_overhead_camera.py --remove
"""
import argparse

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

NAME = "overhead_cam"
HEIGHT = 4.5          # 攝影機離地高度 (m)，隨便多高都行，只要看得到整個房間，跟世界座標換算無關
ORTHO_SIZE = 7.0       # 正交視野寬度 (m)，要蓋過 build_playground_new.py 的 ROOM=6.0 留一點邊界
RESOLUTION = 512


def remove_old(sim):
    for h in sim.getObjectsInTree(sim.handle_scene, sim.object_visionsensor_type, 0):
        if sim.getObjectAlias(h) == NAME:
            sim.removeObjects([h])
            print(f"刪除舊的 {NAME}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()

    sim = RemoteAPIClient().require("sim")
    remove_old(sim)
    if args.remove:
        print("只刪除，未新增。")
        return

    # options=4：不畫出視野椎體、不需要自轉；不設 perspective 那個 bit，走正交投影。
    h = sim.createVisionSensor(
        4, [RESOLUTION, RESOLUTION, 0, 0],
        [0.05, HEIGHT + 2.0, ORTHO_SIZE, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sim.setObjectAlias(h, NAME)
    sim.setObjectPosition(h, [0.0, 0.0, HEIGHT], sim.handle_world)
    # 預設朝向的 local -Z 是往上看（實測驗證過），繞世界 X 轉 180 度才會朝下看地板。
    import math
    sim.setObjectOrientation(h, [math.pi, 0.0, 0.0], sim.handle_world)

    print(f"建立 {NAME}，高度 {HEIGHT}m、正交視野 {ORTHO_SIZE}m、解析度 {RESOLUTION}x{RESOLUTION}。")
    print("機器人端 robot_server.py 之後會透過 zmqRemoteApi 讀這個攝影機。")


if __name__ == "__main__":
    main()
