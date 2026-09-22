"""
在辦公室中央空地建立一個照 RoboMaster EP 尺度設計的任務區 /TaskArena。

內容：
  * 三塊貼地的彩色區域（紅、綠、黃），當作「送到哪裡」的目標
  * 一個矮的開口收納盒，當作「放進盒子裡」的目標
  * 幾個小方塊（可被推動、可被夾取），顏色彼此不同
  * 兩個矮障礙物，用來練習繞路

所有東西都在地面或很低的高度，機器人不需要把手臂抬高。
可重複執行，每次會先刪掉舊的 /TaskArena。可與 build_office.py 共存。

使用方式（模擬停止狀態）：
    python build_task_arena.py
    python build_task_arena.py --cube 0.04 --bin-wall 0.05
    python build_task_arena.py --remove-office      # 同時拿掉辦公室，只留任務區
"""

import argparse
import sys

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

RED = [0.85, 0.15, 0.15]
GREEN = [0.20, 0.65, 0.25]
YELLOW = [0.95, 0.80, 0.15]
BLUE = [0.15, 0.35, 0.85]
ORANGE = [0.95, 0.50, 0.10]
PURPLE = [0.55, 0.25, 0.70]
GRAY = [0.45, 0.45, 0.48]
BIN = [0.30, 0.30, 0.33]


def remove_tree(sim, path):
    h = sim.getObject(path, {"noError": True})
    if h != -1:
        objs = sim.getObjectsInTree(h, sim.handle_all, 0)
        sim.removeObjects(objs)
        print(f"已刪除 {path}（{len(objs)} 個物件）")


def set_visible_props(sim, h):
    try:
        sp = (sim.objectspecialproperty_collidable
              | sim.objectspecialproperty_measurable
              | sim.objectspecialproperty_detectable
              | sim.objectspecialproperty_renderable)
        sim.setObjectSpecialProperty(h, sp)
    except Exception:
        pass


def make_box(sim, parent, name, size, pos, color, static=True, respondable=True, mass=None):
    opts = (8 if respondable else 0) | (16 if static else 0)
    h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, opts)
    sim.setObjectAlias(h, name)
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
    sim.setObjectPosition(h, pos, sim.handle_world)
    set_visible_props(sim, h)
    if mass is not None:
        sim.setShapeMass(h, mass)
    sim.setObjectParent(h, parent, True)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=float, default=0.045, help="方塊邊長 (m)")
    ap.add_argument("--zone", type=float, default=0.40, help="彩色區域邊長 (m)")
    ap.add_argument("--bin-inner", type=float, default=0.22, help="收納盒內寬 (m)")
    ap.add_argument("--bin-wall", type=float, default=0.05, help="收納盒牆高 (m)")
    ap.add_argument("--remove-office", action="store_true")
    args = ap.parse_args()

    client = RemoteAPIClient()
    sim = client.require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sys.exit("請先停止模擬再執行此腳本。")

    remove_tree(sim, "/TaskArena")
    if args.remove_office:
        remove_tree(sim, "/Office")

    root = sim.createDummy(0.05)
    sim.setObjectAlias(root, "TaskArena")
    sim.setObjectPosition(root, [0, 0, 0], sim.handle_world)

    # 1. 目標區域：很薄、不參與碰撞，只是地上的色塊，機器人可以直接開上去
    zones = sim.createDummy(0.02)
    sim.setObjectAlias(zones, "zones")
    sim.setObjectParent(zones, root, True)
    z = args.zone
    for name, xy, color in [("zone_red", (1.4, -0.9), RED),
                            ("zone_green", (1.4, 0.6), GREEN),
                            ("zone_yellow", (-0.2, -1.6), YELLOW)]:
        make_box(sim, zones, name, [z, z, 0.002], [xy[0], xy[1], 0.001], color,
                 static=True, respondable=False)

    # 2. 開口收納盒：底板加四面矮牆
    bin_g = sim.createDummy(0.02)
    sim.setObjectAlias(bin_g, "bin")
    sim.setObjectParent(bin_g, root, True)
    bx, by = -0.9, 0.4
    w, t, hw = args.bin_inner, 0.01, args.bin_wall
    make_box(sim, bin_g, "bin_base", [w + 2 * t, w + 2 * t, t], [bx, by, t / 2], BIN)
    for name, dx, dy, sx, sy in [("bin_n", 0, (w + t) / 2, w + 2 * t, t),
                                 ("bin_s", 0, -(w + t) / 2, w + 2 * t, t),
                                 ("bin_e", (w + t) / 2, 0, t, w),
                                 ("bin_w", -(w + t) / 2, 0, t, w)]:
        make_box(sim, bin_g, name, [sx, sy, hw], [bx + dx, by + dy, t + hw / 2], BIN)

    # 3. 可操作的小方塊：動態、輕、放在開闊處
    cubes = sim.createDummy(0.02)
    sim.setObjectAlias(cubes, "cubes")
    sim.setObjectParent(cubes, root, True)
    c = args.cube
    for name, xy, color in [("cube_blue", (0.7, 0.3), BLUE),
                            ("cube_orange", (0.5, -0.6), ORANGE),
                            ("cube_purple", (-0.5, -0.7), PURPLE)]:
        make_box(sim, cubes, name, [c, c, c], [xy[0], xy[1], c / 2 + 0.001], color,
                 static=False, respondable=True, mass=0.03)

    # 4. 矮障礙物：練習繞路，不擋相機視線
    obs = sim.createDummy(0.02)
    sim.setObjectAlias(obs, "obstacles")
    sim.setObjectParent(obs, root, True)
    make_box(sim, obs, "obstacle_1", [0.6, 0.08, 0.10], [0.9, -0.15, 0.05], GRAY)
    make_box(sim, obs, "obstacle_2", [0.08, 0.5, 0.10], [-0.15, -1.0, 0.05], GRAY)

    n = len(sim.getObjectsInTree(root, sim.handle_all, 0))
    print(f"任務區建立完成，共 {n} 個物件。記得 File > Save scene 存檔。")
    print("若與既有的紅、藍方塊或家具重疊，可直接在 GUI 拖動，或修改腳本內座標。")


if __name__ == "__main__":
    main()
