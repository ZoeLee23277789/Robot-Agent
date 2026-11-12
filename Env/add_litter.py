#!/usr/bin/env python
"""在遊戲場放垃圾與兩個垃圾桶，讓機器人有「收拾」類的任務可做。可重複執行。

物件尺寸全部配合夾爪的物理限制：手指放到最低只佔離地 0.136-0.152 m，所以每件垃圾至少 16 cm 高、不超過 7 cm 寬，
夾的是那一段（跟 15 cm 的小積木同一套 pick 流程，感測器送到 95/80 mm 再關）。扁的紙屑、袋子抓不起來，不放。
  trash_bottle_1/2  淡藍透明感的寶特瓶：圓柱 6 cm × 20 cm，30 g
  trash_can_1/2     桃紅色汽水罐：圓柱 6.5 cm × 16 cm，40 g
  trash_cup_1       白色紙杯：圓柱 7 cm × 16 cm，20 g
  trash_carton_1    米白色紙盒：7 × 7 × 18 cm，40 g
  bin_trash         黑色一般垃圾桶（紙杯、紙盒）；bin_recycle 白色回收桶（瓶、罐）。0.35 m 見方、8 cm 高，跟玩具桶同構造。
位置：場景很擠（0.45 m 的桶子整個房間只有一個位置放得下），桶子縮成 0.35 m 並排在東南角隧道南口與藍柱之間；
垃圾散在各區，其中一瓶一罐放在平台底下——俯視相機被平台甲板擋住看不到，要開進去用前相機找（題目 t07/t08/n09）。

    .venv/bin/python Env/add_litter.py            # 修改並存檔（會先備份 .ttt）
    .venv/bin/python Env/add_litter.py --dry-run  # 只改執行中的場景
    .venv/bin/python Env/add_litter.py --remove   # 全部拆掉

改完要 ./start_all.sh --restart（伺服器的靜態地標要重新建）。
"""
import argparse
import math
import os
import shutil
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

SCENE = os.path.expanduser("~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04/scenes/Capstone_Playground.ttt")

BIN_INNER, BIN_WALL_H, BIN_T = 0.35, 0.08, 0.02
BINS = {"bin_trash": ((1.85, -1.95), [0.10, 0.10, 0.10]), "bin_recycle": ((2.45, -1.95), [0.95, 0.95, 0.95])}
# name -> (kind, size[x,y,z], colour, mass, (x, y))
LITTER = {
    # 全部用方柱（box）不用圓柱：夾爪腳本只在「兩根手指都碰到」時才把物件掛接到夾爪上，圓柱只會有一邊的指墊接觸、
    # 掛不上，開車一晃就掉（2026-10-21 瓶子實測三次都掉）；方柱兩邊平面接觸，跟小積木一樣可靠。
    # 瓶 24 cm、罐 22 cm：比柱子上的球（頂點 0.174）高一截，俯視深度量到 ≥ 0.19 就能跟同色的青球/粉紅球分開
    # 瓶子用飽和的天藍 (0.30,0.80,0.95)：淡藍 (0.55,0.85,0.95) 的飽和度 107 過不了俯視色塊檢測的門檻 150
    # 高度 21/20 cm、80 g：24 cm 的 50 g 盒子重心太高，送料時手指一碰就倒（2026-10-21 t01）；仍比柱子上的球（頂 0.174）高
    # 瓶子 7 cm 寬：6 cm 的太細夾不穩（罐 6.5、杯/盒 7 都成功，6 cm 的瓶子三次都被撞倒）
    "trash_bottle_1": ("box", [0.07, 0.07, 0.20], [0.30, 0.80, 0.95], 0.08, (0.10, 2.10)),    # 平台底下（俯視看不到）
    "trash_bottle_2": ("box", [0.07, 0.07, 0.20], [0.30, 0.80, 0.95], 0.08, (-0.30, -1.15)),  # 房間中央、紫球西邊（跟罐子同尺寸，罐子實測可夾）
    "trash_can_1":    ("box", [0.065, 0.065, 0.20], [0.85, 0.15, 0.55], 0.08, (0.60, -2.45)), # 南邊綠墊旁
    "trash_can_2":    ("box", [0.065, 0.065, 0.20], [0.85, 0.15, 0.55], 0.08, (1.20, 2.20)),  # 平台底下（俯視看不到）
    # 杯子淺灰、紙盒深棕：原本的白/米白跟米色地磚（0.93,0.91,0.86）幾乎同色，俯視完全分不出來
    "trash_cup_1":    ("box", [0.07, 0.07, 0.16], [0.78, 0.78, 0.78], 0.04, (-1.70, -2.45)),  # 入口西邊、球池南牆外
    "trash_carton_1": ("box", [0.07, 0.07, 0.18], [0.35, 0.22, 0.12], 0.05, (1.60, -2.55)),   # 垃圾桶旁
}
FRICTION = 0.5   # 跟小積木一樣（球才需要 2.0）


def get(sim, alias):
    h = sim.getObject("/" + alias, {"noError": True})
    return None if h == -1 else h


def set_shape(sim, h, static, respondable, color):
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1 if static else 0)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1 if respondable else 0)
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
    sim.setObjectSpecialProperty(h, sim.objectspecialproperty_collidable | sim.objectspecialproperty_measurable
                                 | sim.objectspecialproperty_detectable | sim.objectspecialproperty_renderable)


def build_bin(sim, pg, name, xy, color):
    g = get(sim, name)
    if g is None:
        g = sim.createDummy(0.03)
        sim.setObjectAlias(g, name)
        if pg is not None:
            sim.setObjectParent(g, pg, True)
    sim.setObjectPosition(g, sim.handle_world, [xy[0], xy[1], 0.0])
    o, t, h = BIN_INNER + 2 * BIN_T, BIN_T, BIN_WALL_H
    zc = t + h / 2
    parts = {f"{name}_base": ([o, o, t], (0, 0, t / 2)), f"{name}_n": ([o, t, h], (0, (BIN_INNER + t) / 2, zc)),
             f"{name}_s": ([o, t, h], (0, -(BIN_INNER + t) / 2, zc)), f"{name}_e": ([t, BIN_INNER, h], ((BIN_INNER + t) / 2, 0, zc)),
             f"{name}_w": ([t, BIN_INNER, h], (-(BIN_INNER + t) / 2, 0, zc))}
    for pn, (size, local) in parts.items():
        s = get(sim, pn)
        if s is None:
            s = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
            sim.setObjectAlias(s, pn)
            sim.setObjectParent(s, g, True)
        sim.setObjectOrientation(s, sim.handle_world, [0, 0, 0])
        sim.setObjectPosition(s, sim.handle_world, [xy[0] + local[0], xy[1] + local[1], local[2]])
        set_shape(sim, s, True, True, color)
    print("桶子 %-12s 放在 (%+.2f, %+.2f)，內徑 %.2f m、牆高 %.2f m" % (name, xy[0], xy[1], BIN_INNER, BIN_WALL_H))
    return g


def build_litter(sim, pg):
    for name, (kind, size, color, mass, xy) in LITTER.items():
        h = get(sim, name)
        if h is not None:
            _, pure_type, dims = sim.getShapeGeomInfo(h)
            want_type = sim.pure_primitive_cylinder if kind == "cyl" else sim.pure_primitive_cuboid
            if pure_type != want_type or any(abs(a - b) > 0.002 for a, b in zip(dims, size)):
                sim.removeObjects([h])   # 形狀或尺寸改了就重建
                h = None
        if h is None:
            prim = sim.primitiveshape_cylinder if kind == "cyl" else sim.primitiveshape_cuboid
            h = sim.createPrimitiveShape(prim, size, 0)
            sim.setObjectAlias(h, name)
            if pg is not None:
                sim.setObjectParent(h, pg, True)
        set_shape(sim, h, False, True, color)
        sim.setShapeMass(h, mass)
        sim.setEngineFloatParam(sim.bullet_body_friction, h, FRICTION)
        sim.setEngineFloatParam(sim.ode_body_friction, h, FRICTION)
        sim.setObjectOrientation(h, sim.handle_world, [0, 0, 0])
        sim.setObjectPosition(h, sim.handle_world, [xy[0], xy[1], size[2] / 2 + 0.002])
        sim.resetDynamicObject(h)
        print("垃圾 %-15s %s %s 放在 (%+.2f, %+.2f)" % (name, kind, [round(v, 3) for v in size], xy[0], xy[1]))


def remove_all(sim):
    gone = []
    for name in list(LITTER) + list(BINS):
        h = get(sim, name)
        if h is not None:
            gone += sim.getObjectsInTree(h, sim.handle_all, 0) + [h]
    if gone:
        sim.removeObjects(gone)
    print("已刪除 %d 個物件" % len(gone))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--port", type=int, default=23000)
    args = ap.parse_args()
    sim = RemoteAPIClient(port=args.port).require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)
        print("模擬已停止")
    pg = get(sim, "Playground")
    if args.remove:
        remove_all(sim)
    else:
        for name, (xy, color) in BINS.items():
            build_bin(sim, pg, name, xy, color)
        build_litter(sim, pg)
    if args.dry_run:
        print("--dry-run：沒有存檔")
        return
    backup = SCENE.replace(".ttt", time.strftime("_backup_%Y%m%d_%H%M%S.ttt"))
    shutil.copy2(SCENE, backup)
    print("備份：", backup)
    sim.saveScene(SCENE)
    print("已存檔：", SCENE)


if __name__ == "__main__":
    main()
