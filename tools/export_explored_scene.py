#!/usr/bin/env python
"""把實體 explore 模式建出來的占據格存成一個 CoppeliaSim 場景（使用者 2026-11-12：「你沒有給我那個模擬環境」）。

來源：實體 server 的 /map.json（障礙格 + 探索範圍 + 出發點）。
做法：在視窗版（zmq 23001，已載空場景 Capstone_Explore_nexarm.ttt）裡把舊的 tof_map 方塊清掉，障礙格合併成長方體
（同一列相鄰格先合併，再把上下相同的列合併成矩形），機器人擺回原點朝 +x，存成 scenes/Explored_<時間>.ttt；
同時把 /map.png 跟 /map.json 存到同名的 .png / .json。存完視窗版繼續給 sim_twin 用。

用法：~/rmenv/bin/python tools/export_explored_scene.py [--real http://127.0.0.1:8767] [--dst 23001] [--height 0.3] [--name 名字]
之後要用：cd ~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04 && ./coppeliaSim.sh scenes/Explored_xxx.ttt
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Test_Robot"))
from coppeliasim_zmqremoteapi_client import RemoteAPIClient  # noqa: E402
import sim_twin  # noqa: E402  （tools/ 下）


def merge_rects(cells, cell):
    """占據格中心座標 → 矩形 [(x0, y0, x1, y1)]：先同列合併，再把寬度相同、x 相同的相鄰列合併。"""
    rows = {}
    for x, y in cells:
        rows.setdefault(round(y, 3), []).append(round(x, 3))
    runs = []   # (y, x_start, n)
    for y in sorted(rows):
        xs = sorted(rows[y])
        cur = None
        for x in xs:
            if cur and abs(cur[1] + cur[2] * cell - x) < cell * 0.5:
                cur[2] += 1
            else:
                cur = [y, x, 1]
                runs.append(cur)
    rects = []   # [y0, x0, n, rows]
    for y, x0, n in runs:
        merged = False
        for r in rects:
            if r[1] == x0 and r[2] == n and abs(r[0] + r[3] * cell - y) < cell * 0.5:
                r[3] += 1
                merged = True
                break
        if not merged:
            rects.append([y, x0, n, 1])
    out = []
    for y0, x0, n, m in rects:
        out.append((x0 - cell / 2, y0 - cell / 2, x0 + (n - 0.5) * cell, y0 + (m - 0.5) * cell))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="http://127.0.0.1:8767")
    ap.add_argument("--dst", type=int, default=23001)
    ap.add_argument("--height", type=float, default=0.30, help="障礙方塊高度（ToF/相機不知道真實高度）")
    ap.add_argument("--name", default=None)
    ap.add_argument("--sim-dir", default=os.path.expanduser("~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04"))
    a = ap.parse_args()
    with urllib.request.urlopen(a.real.rstrip("/") + "/map.json", timeout=5) as r:
        d = json.loads(r.read().decode("utf-8"))
    with urllib.request.urlopen(a.real.rstrip("/") + "/map.png", timeout=5) as r:
        png = r.read()
    cell = float(d.get("cell_m", 0.05))
    rects = merge_rects([tuple(c) for c in d.get("occupied", [])], cell)
    name = a.name or ("Explored_" + time.strftime("%Y%m%d_%H%M"))
    out = os.path.join(a.sim_dir, "scenes", name + ".ttt")

    print("[export] %d 個占據格 → %d 個矩形" % (len(d.get("occupied", [])), len(rects)), flush=True)
    sim = RemoteAPIClient(port=a.dst).require("sim")
    print("[export] 已連上視窗版 %d" % a.dst, flush=True)
    if sim.getSimulationState() != 0:
        sim.stopSimulation()
        time.sleep(1.0)
    scene = sim.getStringParam(sim.stringparam_scene_name)
    if "Explore" not in scene:
        sim.loadScene(os.path.join(a.sim_dir, "scenes", "Capstone_Explore_nexarm.ttt"))
    objs = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2)
    old = [o for o in objs if sim.getObjectAlias(o).startswith("tof_map") or sim.getObjectAlias(o).startswith("obstacle_")]
    if old:
        sim.removeObjects(old)
    print("[export] 舊方塊清掉 %d 個，開始建方塊..." % len(old), flush=True)
    t0 = time.time()
    group = sim.createDummy(0.01)
    sim.setObjectAlias(group, "explored_obstacles")
    for i, (x0, y0, x1, y1) in enumerate(rects):
        h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [x1 - x0, y1 - y0, a.height], 0)
        sim.setObjectPosition(h, sim.handle_world, [(x0 + x1) / 2, (y0 + y1) / 2, a.height / 2])
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)   # 存檔的版本要能碰撞：之後模擬跑任務用
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [0.5, 0.5, 0.5])
        sim.setObjectAlias(h, "obstacle_%03d" % i)
        sim.setObjectParent(h, group, True)
    print("[export] 方塊建好 %.1fs，存檔中..." % (time.time() - t0), flush=True)
    root = sim.getObject("/RoboMaster")
    frame = sim_twin.upright_frame(sim, root)
    home = d.get("home") or [0.0, 0.0, 0.0]
    sim.setObjectMatrix(root, sim.handle_world, sim_twin.pose_matrix(frame, home[0], home[1], 0.066, home[2]))
    sim.saveScene(out)
    with open(out[:-4] + ".png", "wb") as f:
        f.write(png)
    with open(out[:-4] + ".json", "w", encoding="utf-8") as f:
        json.dump(d, f)
    print("已存：%s（%d 個障礙方塊，來自 %d 個占據格；範圍 %s）" % (out, len(rects), len(d.get("occupied", [])), d.get("bounds")))
    print("地圖圖檔：%s.png / .json" % out[:-4])
    print("開啟：cd %s && ./coppeliaSim.sh scenes/%s.ttt" % (a.sim_dir, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
