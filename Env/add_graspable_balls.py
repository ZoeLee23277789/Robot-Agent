#!/usr/bin/env python
"""
把場景裡的球改成機器人夾得起來的樣子，並把藍色收納箱搬到開放空間。可重複執行。

為什麼要放在柱子上（2026-10-07 實測）：
- 夾爪的手指是兩片很薄的水平板，手臂放到最低時也只佔離地 0.136-0.152 m 這一段；原本 8 cm 的球
  放在地上（球心 0.04 m）手指根本碰不到，球再小只會更碰不到。
- 球心要對準手指板中央 z = 0.144 m，所以 6 cm 的球放在 0.114 m 高、3 cm 粗的柱子上。
- 6 cm、摩擦係數 2.0、50 g：用 skills.pick 同一套步驟（導航到 0.30 m → 手臂 (180,30) → 送入到
  感測器讀柱子 120 mm → 關 → 抬 6 cm → 倒退 0.25 m）連續 8 次全部夾起帶走；5 cm 或摩擦 0.5 時
  關夾爪會把球擠飛。

藍色收納箱（bin_balls）原本卡在長椅、泡棉樑、隧道圍起來的角落，機器人轉身半徑停不到「前方 30 cm」
（m01 一直失敗的原因），搬到西北邊的空地（離最近的固定障礙 0.96 m、最近的零散物件 0.73 m）。

    .venv/bin/python Env/add_graspable_balls.py            # 修改並存檔（會先備份 .ttt）
    .venv/bin/python Env/add_graspable_balls.py --dry-run  # 只修改執行中的場景，不存檔

需要 CoppeliaSim 開著並載入 Capstone_Playground.ttt（headless 或 GUI 都可以）。
"""

import argparse
import os
import shutil
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

SCENE = os.path.expanduser("~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04/scenes/Capstone_Playground.ttt")

BALL_D = 0.06
BALL_CENTER_Z = 0.144          # 手指板中央
POST_D = 0.03
POST_TOP = BALL_CENTER_Z - BALL_D / 2
BALL_MASS = 0.05
BALL_FRICTION = 2.0

# 名稱 -> (x, y, 顏色)。位置都挑在導航地圖上離固定障礙 >= 0.79 m 的地方，柱子不會把路堵死。
BALLS = {
    "ball_purple": (0.36, -1.35, [0.60, 0.25, 0.75]),   # 原位置，h09 的目標
    "ball_orange": (0.10, -0.20, [0.98, 0.50, 0.10]),   # 原本緊貼滑梯（淨空 0.1 m），搬到中央走道
    "ball_pink": (1.30, -2.00, [0.95, 0.45, 0.70]),     # 新增：起點往東的主要通道上（離黃積木 0.6 m、隧道南口站位 0.9 m）
    "ball_cyan": (-0.80, 0.30, [0.20, 0.80, 0.85]),     # 新增：中間偏西的空地（離藍色泡棉方塊 0.85 m；(0, 2) 會落在平台底下，俯視相機看不到）
}
BIN_BALLS_XY = (-1.40, 0.70)


def get(sim, alias):
    try:
        return sim.getObject("/" + alias)
    except Exception:
        return None


def make_post(sim, name, x, y):
    h = get(sim, name)
    if h is None:
        h = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [POST_D, POST_D, POST_TOP], 0)
        sim.setObjectAlias(h, name)
    sim.setObjectPosition(h, sim.handle_world, [x, y, POST_TOP / 2])
    sim.setObjectOrientation(h, sim.handle_world, [0, 0, 0])
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [0.15, 0.15, 0.15])
    return h


def make_ball(sim, name, x, y, rgb):
    h = get(sim, name)
    parent = -1
    if h is not None:
        # 一定要是「純」球體（pure spheroid）而且直徑正確。2026-10-07 第一版用 scaleObject 把原本 8 cm 的球
        # 縮成 6 cm，結果變成凸多面體網格：靜置沒問題，但夾的時候接觸跟摩擦完全不同，紫色/橘色球
        # 連續失敗、新建的純球體卻一夾就起來。不是純球體就刪掉重建，保留名稱跟父物件。
        _, pure_type, dims = sim.getShapeGeomInfo(h)
        if pure_type != sim.pure_primitive_spheroid or abs(dims[0] - BALL_D) > 0.002:
            parent = sim.getObjectParent(h)
            sim.removeObjects([h])
            h = None
    if h is None:
        h = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [BALL_D] * 3, 0)
        sim.setObjectAlias(h, name)
        if parent != -1:
            sim.setObjectParent(h, parent, True)
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
    sim.setShapeMass(h, BALL_MASS)
    sim.setEngineFloatParam(sim.bullet_body_friction, h, BALL_FRICTION)
    sim.setEngineFloatParam(sim.ode_body_friction, h, BALL_FRICTION)
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, rgb)
    sim.setObjectOrientation(h, sim.handle_world, [0, 0, 0])
    sim.setObjectPosition(h, sim.handle_world, [x, y, BALL_CENTER_Z + 0.001])
    sim.resetDynamicObject(h)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只改執行中的場景，不存檔")
    ap.add_argument("--port", type=int, default=23000)
    args = ap.parse_args()

    sim = RemoteAPIClient(port=args.port).require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)
        print("模擬已停止")

    for name, (x, y, rgb) in BALLS.items():
        make_post(sim, "tee_" + name.split("_", 1)[1], x, y)
        make_ball(sim, name, x, y, rgb)
        print(f"{name}: 直徑 {BALL_D*100:.0f} cm，放在 ({x:+.2f}, {y:+.2f}) 的柱子上（柱頂 {POST_TOP:.3f} m）")

    b = get(sim, "bin_balls")
    if b is None:
        raise SystemExit("找不到 /bin_balls")
    p = sim.getObjectPosition(b, sim.handle_world)
    sim.setObjectPosition(b, sim.handle_world, [BIN_BALLS_XY[0], BIN_BALLS_XY[1], p[2]])
    print(f"bin_balls: ({p[0]:+.2f}, {p[1]:+.2f}) → ({BIN_BALLS_XY[0]:+.2f}, {BIN_BALLS_XY[1]:+.2f})")

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
