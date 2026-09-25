"""
在 /RoboMaster 上加三個真的距離感測器（前、左、右），取代 RoboMaster SDK 在這個模擬器裡
永遠回報 0 的 ToF（sensor.sub_distance 在這個 CoppeliaSim 外掛裡沒有實作，SDK 讀不到真數值，
所有歷史紀錄裡 tof_distance_mm 一次都沒變過，證實是死的）。

做法：用 CoppeliaSim 自己的 proximity sensor（sim.createProximitySensor），直接掛在機器人上，
繞過 RoboMaster SDK，機器人端 (robot_server.py) 再用 zmqRemoteApi 讀這三個感測器的即時距離。

方向校正：不能直接假設 /RoboMaster 這個物件的 local 座標軸哪個是「前」，量出來的 Euler 角
很怪、不是乾淨的 90 度倍數（可能是這個模型匯入時的固定座標重映射，不代表機器人真的歪了）。
所以改成量測「真的移動」：把當下 RoboMaster SDK 回報的 yaw_deg 讀進來，配合我們從歷史紀錄
回歸出來的關係──世界座標下的前進方向角度 = yaw_deg（跟 chassis.move 的 forward_m 實際位移
方向對過，8 個不同 yaw 值誤差都在 1 度內）──算出當下前/左/右三個世界方向，用世界座標直接
擺好感測器方向，再用 keepInPlace=True 掛到 /RoboMaster 底下，之後機器人怎麼轉都會跟著轉對。

使用方式：
    python add_proximity_sensors.py                          # 用預設站台的 /state 讀 yaw
    python add_proximity_sensors.py --yaw-deg -120.2          # 手動指定 yaw（不想連 robot_server 時用）
    python add_proximity_sensors.py --remove                  # 只刪除舊的三個感測器
"""

import argparse
import json
import math
import sys
import urllib.request

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

NAMES = ("prox_front", "prox_left", "prox_right")

# (方向, 沿前方偏移, 沿右方偏移, 高度偏移, 偵測距離, 全開角度)
# 角度刻意留窄：量測高度 0.12m 時，角度太寬會讓偵測椎體在還沒到 range 之前就先掃到地板本身，
# 開闊地板也會一直報「有東西」。20 度全開角、0.12m 高，錐體要到 0.68m 才會碰到地板，
# 比 front 的 range(0.6) 遠，side 的 range(0.35) 也還在安全範圍內。
SPECS = [
    ("prox_front", "fwd", 0.18, 0.00, 0.12, 0.60, 20),
    ("prox_left", "left", 0.05, -0.14, 0.12, 0.35, 20),
    ("prox_right", "right", 0.05, 0.14, 0.12, 0.35, 20),
]


def fetch_yaw_deg(url):
    with urllib.request.urlopen(url + "/state", timeout=3) as r:
        return json.load(r)["yaw_deg"]


def remove_old(sim, root):
    for name in NAMES:
        for h in sim.getObjectsInTree(root, sim.handle_all, 0):
            if sim.getObjectAlias(h) == name:
                sim.removeObjects([h])
                print(f"刪除舊的 {name}")


def make_sensor(sim, root, root_pos, fwd, right, up, name, dir_kind, along_fwd, along_right, height, rng, angle_deg):
    direction = {"fwd": fwd, "left": tuple(-v for v in right), "right": right}[dir_kind]
    pos = [root_pos[i] + fwd[i] * along_fwd + right[i] * along_right + up[i] * height for i in range(3)]

    z_axis = direction
    x_axis = _norm(_cross(up, z_axis))
    y_axis = _cross(z_axis, x_axis)
    m = [x_axis[0], y_axis[0], z_axis[0], pos[0],
         x_axis[1], y_axis[1], z_axis[1], pos[1],
         x_axis[2], y_axis[2], z_axis[2], pos[2]]

    angle = math.radians(angle_deg)
    far_size = 2 * rng * math.tan(angle / 2)
    h = sim.createProximitySensor(
        sim.proximitysensor_pyramid, 16, 4,
        [4, 4, 4, 4, 0, 0, 0, 0],
        [0.0, rng, 0.02, 0.02, far_size, far_size, 0.0, 0.0, 0.0, angle, 0.0, 0.0, 0.005, 0.0, 0.0])
    sim.setObjectMatrix(h, m, sim.handle_world)
    sim.setObjectAlias(h, name)
    sim.setObjectParent(h, root, True)
    return h


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / n for c in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="/RoboMaster")
    ap.add_argument("--server-url", default="http://127.0.0.1:8765")
    ap.add_argument("--yaw-deg", type=float, default=None, help="不想連 robot_server 時手動指定當下 yaw")
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()

    client = RemoteAPIClient()
    sim = client.require("sim")
    root = sim.getObject(args.robot, {"noError": True})
    if root == -1:
        sys.exit(f"找不到 {args.robot}")

    remove_old(sim, root)
    if args.remove:
        print("只刪除，未新增。")
        return

    if args.yaw_deg is not None:
        yaw = args.yaw_deg
    else:
        try:
            yaw = fetch_yaw_deg(args.server_url)
        except Exception as e:
            sys.exit(f"讀不到 {args.server_url}/state 的 yaw_deg（{e}），改用 --yaw-deg 手動指定。")
    print(f"校正用 yaw_deg = {yaw:.1f}")

    rad = math.radians(yaw)
    fwd = (math.cos(rad), math.sin(rad), 0.0)
    right = (math.sin(rad), -math.cos(rad), 0.0)
    up = (0.0, 0.0, 1.0)
    root_pos = sim.getObjectPosition(root, sim.handle_world)

    for name, dir_kind, along_fwd, along_right, height, rng, angle_deg in SPECS:
        h = make_sensor(sim, root, root_pos, fwd, right, up, name, dir_kind,
                         along_fwd, along_right, height, rng, angle_deg)
        res, dist = sim.readProximitySensor(h)[:2]
        print(f"建立 {name} (handle={h})，目前讀值：{'%.2fm' % dist if res else '沒偵測到東西'}")

    print("完成。機器人端 robot_server.py 之後會透過 zmqRemoteApi 讀這三個感測器。")


if __name__ == "__main__":
    main()
