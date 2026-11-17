#!/usr/bin/env python
"""手機 LiDAR 底面（tools/scan_floorplan.py 的 floorplan.json）→ CoppeliaSim 場景 + 機器人座標系的占據圖（2026-11-13）。

座標：掃描模型 y 朝上、地面是 (x, z)；機器人世界座標以出發點為原點、車頭 +x、左手 +y。
    --start X,Z   出發點在模型裡的位置（看 scans/.../topdown.png 的格線）
    --heading D   車頭在模型裡的方向（度）：0 = 模型 +x（圖的右）、90 = 模型 +z（圖的下）、180 = −x、270 = −z
每個占據格變成 cell×cell×height 的靜態方塊（同列相鄰的先合併成長條），機器人擺在原點朝 +x，存成 scenes/Scan_<名字>.ttt；
同時寫 scenes/Scan_<名字>.json（robot_server 的 /map.json 格式，占據格已轉成機器人座標）跟 .png。

用法：~/rmenv/bin/python tools/scan_to_scene.py scans/2026_11_12/floorplan.json --start -1.0,0.0 --heading 0 [--dst 23002] [--name 2026_11_12]
    建議用無視窗的 CoppeliaSim 存檔（視窗版存 35 MB 場景會當）：
    cd ~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04 && ./coppeliaSim -h -GzmqRemoteApi.rpcPort=23002 scenes/Capstone_Explore_nexarm.ttt
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Test_Robot"))
from export_explored_scene import merge_rects  # noqa: E402


def to_robot_frame(cells, start, heading_deg):
    """模型 (x, z) → 機器人世界 (x 前, y 左)。模型俯視圖 x 向右、z 向下，所以車頭 h=(cos, sin) 的左邊是 (hz, -hx)。"""
    ph = math.radians(heading_deg)
    h = np.array([math.cos(ph), math.sin(ph)])
    left = np.array([h[1], -h[0]])
    P = np.asarray(cells, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    return np.stack([P @ h, P @ left], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("floorplan")
    ap.add_argument("--start", required=True, help="模型座標 X,Z")
    ap.add_argument("--heading", type=float, default=0.0, help="模型裡的車頭方向（度），0=+x 90=+z")
    ap.add_argument("--height", type=float, default=0.30)
    ap.add_argument("--radius", type=float, default=0.0, help="只保留離出發點這麼遠以內的占據格；0 = 整張掃描都留（使用者 11-13：「我給的圖就是所有的障礙物」）")
    ap.add_argument("--dst", type=int, default=23002)
    ap.add_argument("--name", default=None)
    ap.add_argument("--wall", action="append", default=[], metavar="X0,Z0,X1,Z1",
                    help="補一段直線牆（模型座標，可重複）：LiDAR 穿過玻璃會把牆掃成缺口（使用者 11-13：「右側是一直線的牆，開口是玻璃」）")
    ap.add_argument("--no-sim", action="store_true", help="只寫 json/png，不建場景")
    ap.add_argument("--sim-dir", default=os.path.expanduser("~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04"))
    a = ap.parse_args()
    fp = json.load(open(a.floorplan, encoding="utf-8"))
    cell = float(fp["cell_m"])
    start = [float(v) for v in a.start.split(",")]
    occupied = [list(c) for c in fp["occupied"]]
    for w in a.wall:   # 直線牆：沿著線每 cell/2 補一格
        x0, z0, x1, z1 = [float(v) for v in w.split(",")]
        n = max(1, int(math.hypot(x1 - x0, z1 - z0) / (cell / 2)))
        occupied += [[x0 + (x1 - x0) * i / n, z0 + (z1 - z0) * i / n] for i in range(n + 1)]
        print("[scan_to_scene] 補牆 (%.2f,%.2f)→(%.2f,%.2f)" % (x0, z0, x1, z1))
    R = to_robot_frame(occupied, start, a.heading)
    # 轉完再對回 cell 格（旋轉後格子中心不一定對齊），去重
    snapped = np.round(np.floor(R / cell) * cell + cell / 2, 3)
    keep = np.hypot(snapped[:, 0], snapped[:, 1]) <= a.radius if a.radius > 0 else np.ones(len(snapped), bool)
    cells = sorted({(float(x), float(y)) for x, y in snapped[keep]})
    near = [c for c in cells if math.hypot(*c) < 0.25]
    if near:
        print("[scan_to_scene] 警告：出發點 0.25 m 內有 %d 個占據格——出發點可能設在牆/桌子裡" % len(near))
    name = a.name or os.path.basename(os.path.dirname(os.path.abspath(a.floorplan))) or time.strftime("%Y%m%d_%H%M")
    base = os.path.join(a.sim_dir, "scenes", "Scan_" + name)
    xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
    bounds = [round(min(xs) - 0.1, 2), round(max(xs) + 0.1, 2), round(min(ys) - 0.1, 2), round(max(ys) + 0.1, 2)]
    ext = a.radius if a.radius > 0 else math.ceil(max(abs(v) for v in bounds) + 0.5)
    out = {"source": a.floorplan, "cell_m": cell, "extent_m": ext, "occupied": [list(c) for c in cells],
           "bounds": bounds, "home": [0.0, 0.0, 0.0], "robot": {"world_xy": [0.0, 0.0], "heading_deg": 0.0},
           "scan_start": start, "scan_heading_deg": a.heading, "walls": a.wall}
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(out, f)
    try:
        import cv2
        S = max(2, int(900 / (2 * ext / cell)))
        n = int(2 * ext / cell)
        img = np.full((n * S, n * S, 3), 255, np.uint8)
        for x, y in cells:   # 圖：+x 朝上、+y 朝左（跟 /map.png 一樣）
            px = int((ext - y) / cell) * S; py = int((ext - x) / cell) * S
            cv2.rectangle(img, (px, py), (px + S - 1, py + S - 1), (40, 40, 40), -1)
        for g in range(-int(ext), int(ext) + 1):
            p = int((ext - g) / cell) * S
            cv2.line(img, (p, 0), (p, n * S), (200, 200, 255), 1); cv2.line(img, (0, p), (n * S, p), (200, 200, 255), 1)
        c = int(ext / cell) * S
        cv2.circle(img, (c, c), 8, (0, 0, 255), 2); cv2.arrowedLine(img, (c, c), (c, c - 30), (0, 0, 255), 2)
        cv2.putText(img, "+x up (robot heading)  +y left  1 m grid", (5, n * S - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1)
        cv2.imwrite(base + ".png", img)
    except Exception as e:
        print("[scan_to_scene] png 略過：%s" % e)
    print("[scan_to_scene] 占據格 %d（出發點模型 %s、朝向 %.0f°），範圍 %s → %s.json/.png" % (len(cells), start, a.heading, bounds, base))
    if a.no_sim:
        return 0
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    import sim_twin
    rects = merge_rects(cells, cell)
    sim = RemoteAPIClient(port=a.dst).require("sim")
    if sim.getSimulationState() != 0:
        sim.stopSimulation(); time.sleep(1.0)
    if "Explore" not in sim.getStringParam(sim.stringparam_scene_name):
        sim.loadScene(os.path.join(a.sim_dir, "scenes", "Capstone_Explore_nexarm.ttt"))
    objs = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2)
    old = [o for o in objs if sim.getObjectAlias(o).split("_")[0] in ("tof", "obstacle", "explored", "scan")]
    if old:
        sim.removeObjects(old)
    group = sim.createDummy(0.01); sim.setObjectAlias(group, "scan_obstacles")
    t0 = time.time()
    for i, (x0, y0, x1, y1) in enumerate(rects):
        hnd = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [x1 - x0, y1 - y0, a.height], 0)
        sim.setObjectPosition(hnd, sim.handle_world, [(x0 + x1) / 2, (y0 + y1) / 2, a.height / 2])
        sim.setObjectInt32Param(hnd, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(hnd, sim.shapeintparam_respondable, 1)
        sim.setShapeColor(hnd, None, sim.colorcomponent_ambient_diffuse, [0.55, 0.55, 0.55])
        sim.setObjectAlias(hnd, "obstacle_%03d" % i)
        sim.setObjectParent(hnd, group, True)
    # 地板：空場景的地板是 10×10 m 在原點，走道可能長 11 m 又偏一邊 → 把 Playground 搬到場地中心、地板放大到蓋住範圍 +0.5 m
    try:
        pg = sim.getObject("/Playground"); slab = sim.getObject("/Playground/ground/ground_slab")
        sim.setObjectPosition(pg, sim.handle_world, [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, 0.0])
        bx = sim.getObjectFloatParam(slab, sim.objfloatparam_modelbbox_max_x) - sim.getObjectFloatParam(slab, sim.objfloatparam_modelbbox_min_x)
        by = sim.getObjectFloatParam(slab, sim.objfloatparam_modelbbox_max_y) - sim.getObjectFloatParam(slab, sim.objfloatparam_modelbbox_min_y)
        fx = max(1.0, (bounds[1] - bounds[0] + 1.0) / bx); fy = max(1.0, (bounds[3] - bounds[2] + 1.0) / by)
        if fx > 1.0 or fy > 1.0:
            sim.scaleObject(slab, fx, fy, 1.0, 0)
        print("[scan_to_scene] 地板移到場地中心，放大 x%.2f y%.2f" % (fx, fy))
    except Exception as e:
        print("[scan_to_scene] 地板沒調整：%s" % e)
    root = sim.getObject("/RoboMaster")
    frame = sim_twin.upright_frame(sim, root)
    sim.setObjectMatrix(root, sim.handle_world, sim_twin.pose_matrix(frame, 0.0, 0.0, 0.066, 0.0))
    sim.saveScene(base + ".ttt")
    print("已存：%s.ttt（%d 個方塊，%.1fs）；開啟：cd %s && ./coppeliaSim.sh scenes/Scan_%s.ttt" % (base, len(rects), time.time() - t0, a.sim_dir, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
