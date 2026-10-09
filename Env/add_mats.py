#!/usr/bin/env python
"""
把三張地墊（mat_red / mat_green / mat_blue）真的放進場景。可重複執行。

為什麼要補（2026-10-08）：存檔的 Capstone_Playground.ttt 裡這三個名字只是 dummy（座標標記），底下沒有任何形狀，
建場景腳本原本要放的彩色墊子不見了。h02／h05／h06 要把東西放到墊子上，agent 照座標放對了，畫面裡卻只有灰色地板，
judge 用畫面判斷時不管成功或失敗都不可信（h05 夾起橘球、放到紅墊座標，被判「球在地板上」）。

- 墊子 0.6 m 見方、1.2 cm 厚，掛在原本的 dummy 底下，顏色跟同色小積木一樣。
- 不參與碰撞、距離感測器偵測不到：只是看得見，物理跟導航地圖都不受影響（nav_map 不把非碰撞形狀當障礙）。
- 用 0.6 m 而不是建場景腳本的 1.0 m：房間太擠，1.0 m 的紅墊會蓋住橘球的柱子、藍墊壓到藍色收納箱。
- 兩個東西順便挪開：橘球柱子 (0.10,-0.20) 在紅墊上 → (0.15, 0.00)；黃積木原本就坐在綠墊正中央
  （h06「把黃積木放到綠墊上」一開始就成立了）→ (0.00, -2.20)。

    .venv/bin/python Env/add_mats.py            # 修改並存檔（會先備份 .ttt）
    .venv/bin/python Env/add_mats.py --dry-run  # 只改執行中的場景
"""

import argparse
import os
import shutil
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

SCENE = os.path.expanduser("~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04/scenes/Capstone_Playground.ttt")
MAT_SIZE, MAT_H = 0.6, 0.012
MATS = {"mat_red": [0.85, 0.12, 0.12], "mat_green": [0.20, 0.65, 0.25], "mat_blue": [0.15, 0.35, 0.85]}
ORANGE_XY = (0.15, 0.00)
YELLOW_XY = (0.00, -2.20)
BALL_CENTER_Z = 0.144


def get(sim, alias):
    try:
        return sim.getObject("/" + alias)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--port", type=int, default=23000)
    args = ap.parse_args()
    sim = RemoteAPIClient(port=args.port).require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)
        print("模擬已停止")

    for name, rgb in MATS.items():
        d = get(sim, name)
        if d is None:
            raise SystemExit("找不到 /" + name)
        x, y, _ = sim.getObjectPosition(d, sim.handle_world)
        surf = get(sim, name + "_surface")
        if surf is None:
            surf = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [MAT_SIZE, MAT_SIZE, MAT_H], 0)
            sim.setObjectAlias(surf, name + "_surface")
            sim.setObjectParent(surf, d, True)
        sim.setObjectPosition(surf, sim.handle_world, [x, y, MAT_H / 2])
        sim.setObjectOrientation(surf, sim.handle_world, [0, 0, 0])
        sim.setObjectInt32Param(surf, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(surf, sim.shapeintparam_respondable, 0)
        sim.setObjectSpecialProperty(surf, sim.objectspecialproperty_renderable)   # 看得到，但不可碰撞/量測/偵測
        sim.setShapeColor(surf, None, sim.colorcomponent_ambient_diffuse, rgb)
        print(f"{name}: {MAT_SIZE} m 見方的墊子放在 ({x:+.2f}, {y:+.2f})")

    tee, ball = get(sim, "tee_orange"), get(sim, "ball_orange")
    sim.setObjectPosition(tee, sim.handle_world, [ORANGE_XY[0], ORANGE_XY[1], sim.getObjectPosition(tee, sim.handle_world)[2]])
    sim.setObjectOrientation(ball, sim.handle_world, [0, 0, 0])
    sim.setObjectPosition(ball, sim.handle_world, [ORANGE_XY[0], ORANGE_XY[1], BALL_CENTER_Z + 0.001])
    sim.resetDynamicObject(ball)
    print(f"ball_orange + tee_orange → ({ORANGE_XY[0]:+.2f}, {ORANGE_XY[1]:+.2f})")

    yb = get(sim, "small_yellow")
    sim.setObjectOrientation(yb, sim.handle_world, [0, 0, 0])
    sim.setObjectPosition(yb, sim.handle_world, [YELLOW_XY[0], YELLOW_XY[1], sim.getObjectPosition(yb, sim.handle_world)[2]])
    sim.resetDynamicObject(yb)
    print(f"small_yellow → ({YELLOW_XY[0]:+.2f}, {YELLOW_XY[1]:+.2f})")

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
