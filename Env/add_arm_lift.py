#!/usr/bin/env python
"""給 RoboMaster 加一根可升降的手臂升降柱，並在場景放一張矮桌 + 桌上的柱子與萊姆綠球當測試目標。可重複執行。

為什麼（2026-10-20）：夾爪手指在手臂放到最低時離地 0.136-0.152 m、最高也只有約 0.25 m，桌面高度的東西完全夠不到。
使用者要能抓到約 40-50 cm 高的物件，所以在底盤跟手臂底座之間插一個直線關節（prismatic），整組手臂（連同手臂上的相機、
以及 robot_server.py 建在升降座上的前方距離感測器）可以升高 0-0.40 m；手臂本身的 (x,y) 座標、外掛的逆運動學都不變。
整台等比例放大或加長連桿都不行：外掛的里程計、手臂逆運動學用的是真車的尺寸，模型變大之後回報的距離/姿態全部會錯。

模型結構（只動這一層，其餘零件原封不動）：
    RoboMaster (root, 底盤)
      └ arm_lift_joint (prismatic，軸 = 世界 +z，行程 0-0.40 m，位置控制，伺服器用 zmq 設定目標)
          └ arm_lift_carriage (動態、不碰撞的小方塊，當作升降座)
              ├ arm_lift_rod (純視覺的方柱，升起時看得到，平常藏在底盤裡)
              └ arm_base_attachment_joint (原本掛在 root 下的力感測器，整個手臂子樹都在它下面)

桌子：0.26 m 見方、0.33 m 高的實心木色方塊 table_body（底面貼地，導航地圖會把它當障礙，不會想從底下鑽過），
放在 (0.00, 0.80)（場景裡唯一放得下、四周淨空 ≥ 0.55 m 的空地）。桌面刻意小：夾爪中心只在車頭前約 0.15 m，
球在桌子正中央時車頭要停到離桌緣 2 cm 才送得進指間，桌面再大球就夠不到（0.40 m 的第一版就是這樣失敗的）。桌上一根 0.114 m 的柱子 tee_lime，
上面一顆 6 cm 的萊姆綠球 ball_lime，球心離桌面 0.144 m —— 跟地上四顆球完全同一套幾何，升降座升到桌面高度 0.33 m 之後，
pick 的流程一步都不用改（球心離地 0.474 m）。萊姆綠是場景裡沒有的色相，pick 用顏色對準車頭時不會被別的東西拉走。

    .venv/bin/python Env/add_arm_lift.py            # 修改並存檔（會先備份 .ttt）
    .venv/bin/python Env/add_arm_lift.py --dry-run  # 只改執行中的場景，不存檔
    .venv/bin/python Env/add_arm_lift.py --remove   # 拆掉升降柱跟桌子（手臂掛回底盤），一樣會存檔

改完要 ./start_all.sh --restart：robot_server.py 會重新建立距離感測器並把前方那顆掛到升降座上。
"""

import argparse
import math
import os
import shutil
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

SCENE = os.path.expanduser("~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04/scenes/Capstone_Playground.ttt")

LIFT_MAX_M = 0.40          # 升降行程
LIFT_MAX_VEL = 0.25        # m/s
LIFT_FORCE_N = 300.0       # 位置控制的最大推力（手臂子樹約 0.67 kg + 升降座 0.3 kg）
CARRIAGE_SIZE = (0.02, 0.09, 0.11)   # 升降座尺寸，沿 root 的 x(上)、y(側向)、z(前後)；root 的 x 軸朝上
ROD_SIDE = 0.03

# 桌面只能 0.26 m 見方：夾爪中心只在車頭前約 0.15 m，球放桌子正中央時車頭要能停到離桌緣 2 cm 處才送得進指間；
# 第一版 0.40 m 的桌子球離桌緣 0.20 m，車頭頂到桌緣時感測器還讀 213 mm，怎麼樣都夾不到。高度 0.33 m 才是重點。
TABLE_XY, TABLE_SIZE, TABLE_H = (0.00, 0.80), 0.26, 0.33
POST_D, POST_H = 0.03, 0.114
BALL_D, BALL_MASS, BALL_FRICTION = 0.06, 0.05, 2.0
BALL_ABOVE_SURFACE = 0.144           # 球心離桌面：手指板中央的高度，跟地上的球一樣
WOOD, DARK, LIME = [0.55, 0.40, 0.28], [0.15, 0.15, 0.15], [0.62, 0.95, 0.12]


def find_robot(sim):
    """跟 robot_server.py 的 _find_robot_root 同一套規則：有 GyroSensor 的那台才是外掛驅動的完整模型。"""
    roots = [o for o in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2) if sim.getObjectAlias(o) == "RoboMaster"]
    if not roots:
        raise SystemExit("找不到 /RoboMaster")
    good = [h for h in roots if any(sim.getObjectAlias(o) == "GyroSensor" for o in sim.getObjectsInTree(h, sim.handle_all, 0))]
    if len(roots) > 1:
        print(f"!! 場景裡有 {len(roots)} 台 RoboMaster {roots}，完整的是 {good}，只改 {(good or roots)[0]}")
    return (good or roots)[0]


def child(sim, root, alias):
    return sim.getObject("./" + alias, {"proxy": root, "noError": True})


def get(sim, alias):
    h = sim.getObject("/" + alias, {"noError": True})
    return None if h == -1 else h


def set_props(sim, h, detectable):
    flags = sim.objectspecialproperty_renderable
    if detectable:
        flags |= (sim.objectspecialproperty_collidable | sim.objectspecialproperty_measurable
                  | sim.objectspecialproperty_detectable)
    sim.setObjectSpecialProperty(h, flags)


def set_shape(sim, h, static, respondable, color, detectable):
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1 if static else 0)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1 if respondable else 0)
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
    set_props(sim, h, detectable)


def build_lift(sim, root):
    fs = child(sim, root, "arm_base_attachment_joint")
    if fs == -1:
        raise SystemExit("找不到 arm_base_attachment_joint（手臂底座的力感測器），這不是預期的 RoboMasterEP 模型")
    fs_pos = sim.getObjectPosition(fs, sim.handle_world)

    j = child(sim, root, "arm_lift_joint")
    if j == -1:
        j = sim.createJoint(sim.joint_prismatic_subtype, sim.jointmode_dynamic, 0, [0.06, 0.02])
        sim.setObjectAlias(j, "arm_lift_joint")
        sim.setObjectPosition(j, sim.handle_world, fs_pos)
        sim.setObjectParent(j, root, True)
        sim.setObjectOrientation(j, root, [0.0, math.pi / 2, 0.0])   # 繞 y 轉 90°：關節 z 軸 = root 的 x 軸 = 世界 +z
    car = child(sim, root, "arm_lift_carriage")
    if car == -1:
        car = sim.createPrimitiveShape(sim.primitiveshape_cuboid, list(CARRIAGE_SIZE), 0)
        sim.setObjectAlias(car, "arm_lift_carriage")
        sim.setObjectPosition(car, sim.handle_world, fs_pos)
        sim.setObjectParent(car, j, True)
        sim.setObjectOrientation(car, root, [0.0, 0.0, 0.0])
    set_shape(sim, car, static=False, respondable=False, color=DARK, detectable=False)
    sim.computeMassAndInertia(car, 1500)   # 0.02*0.09*0.11 m³ * 1500 kg/m³ ≈ 0.3 kg
    rod = child(sim, root, "arm_lift_rod")
    if rod == -1:
        length = LIFT_MAX_M + 0.05
        rod = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [length, ROD_SIDE, ROD_SIDE], 0)
        sim.setObjectAlias(rod, "arm_lift_rod")
        sim.setObjectParent(rod, car, False)
        sim.setObjectOrientation(rod, car, [0.0, 0.0, 0.0])
        # 頂端貼著升降座底面、往下（root 的 -x）延伸：升降座在最低點時整根藏在底盤/地板裡，升起來才露出
        sim.setObjectPosition(rod, car, [-(CARRIAGE_SIZE[0] / 2 + length / 2), 0.0, 0.0])
    set_shape(sim, rod, static=True, respondable=False, color=DARK, detectable=False)

    sim.setJointInterval(j, False, [0.0, LIFT_MAX_M])
    try:
        sim.setObjectInt32Param(j, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
    except Exception:   # 舊版 API
        sim.setObjectInt32Param(j, sim.jointintparam_motor_enabled, 1)
        sim.setObjectInt32Param(j, sim.jointintparam_ctrl_enabled, 1)
    sim.setJointTargetPosition(j, 0.0)
    sim.setJointTargetForce(j, LIFT_FORCE_N, False)
    sim.setObjectFloatParam(j, sim.jointfloatparam_maxvel, LIFT_MAX_VEL)

    if sim.getObjectParent(fs) != car:
        sim.setObjectParent(fs, car, True)      # 整個手臂子樹掛到升降座上，世界姿態不變
    after = sim.getObjectPosition(fs, sim.handle_world)
    print("升降柱：joint %d / carriage %d / rod %d，手臂底座世界座標 %s → %s（應相同）"
          % (j, car, rod, [round(v, 4) for v in fs_pos], [round(v, 4) for v in after]))
    return j, car


def remove_lift(sim, root):
    fs = child(sim, root, "arm_base_attachment_joint")
    car = child(sim, root, "arm_lift_carriage")
    if fs != -1 and car != -1 and sim.getObjectParent(fs) == car:
        sim.setObjectParent(fs, root, True)
    gone = [h for h in (child(sim, root, a) for a in ("arm_lift_rod", "arm_lift_carriage", "arm_lift_joint")) if h != -1]
    if gone:
        sim.removeObjects(gone)
    print("升降柱已拆掉（%d 個物件），手臂掛回底盤" % len(gone))


def make_ball(sim, name, parent, x, y, z, rgb):
    """跟 Env/add_graspable_balls.py 一樣：一定要是純球體（縮放過的網格球夾不起來），不是就重建。"""
    h = get(sim, name)
    if h is not None:
        _, pure_type, dims = sim.getShapeGeomInfo(h)
        if pure_type != sim.pure_primitive_spheroid or abs(dims[0] - BALL_D) > 0.002:
            sim.removeObjects([h])
            h = None
    if h is None:
        h = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [BALL_D] * 3, 0)
        sim.setObjectAlias(h, name)
        if parent is not None:
            sim.setObjectParent(h, parent, True)
    set_shape(sim, h, static=False, respondable=True, color=rgb, detectable=True)
    sim.setShapeMass(h, BALL_MASS)
    sim.setEngineFloatParam(sim.bullet_body_friction, h, BALL_FRICTION)
    sim.setEngineFloatParam(sim.ode_body_friction, h, BALL_FRICTION)
    sim.setObjectOrientation(h, sim.handle_world, [0, 0, 0])
    sim.setObjectPosition(h, sim.handle_world, [x, y, z])
    sim.resetDynamicObject(h)
    return h


def build_table(sim):
    pg = get(sim, "Playground")
    x, y = TABLE_XY
    g = get(sim, "table")
    if g is None:
        g = sim.createDummy(0.03)
        sim.setObjectAlias(g, "table")
        if pg is not None:
            sim.setObjectParent(g, pg, True)
    sim.setObjectPosition(g, sim.handle_world, [x, y, 0.0])
    body = get(sim, "table_body")
    if body is not None:
        _, _, dims = sim.getShapeGeomInfo(body)
        if any(abs(a - b) > 0.002 for a, b in zip(dims, [TABLE_SIZE, TABLE_SIZE, TABLE_H])):
            sim.removeObjects([body])   # 尺寸改了就重建（實測第一版 0.40 m 太大）
            body = None
    if body is None:
        body = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [TABLE_SIZE, TABLE_SIZE, TABLE_H], 0)
        sim.setObjectAlias(body, "table_body")
        sim.setObjectParent(body, g, True)
    sim.setObjectOrientation(body, sim.handle_world, [0, 0, 0])
    sim.setObjectPosition(body, sim.handle_world, [x, y, TABLE_H / 2])
    set_shape(sim, body, static=True, respondable=True, color=WOOD, detectable=True)

    tee = get(sim, "tee_lime")
    if tee is None:
        tee = sim.createPrimitiveShape(sim.primitiveshape_cylinder, [POST_D, POST_D, POST_H], 0)
        sim.setObjectAlias(tee, "tee_lime")
    # 柱子跟地上的 tee_* 一樣直接掛在 Playground 下，不放進 table 群組：伺服器算地標 table 的頂面高度 top_m 時
    # 才不會把柱頂 0.444 m 當成桌面（實測第一版就這樣，手臂會升到 444 mm、夾在球上方）
    if sim.getObjectParent(tee) != (pg if pg is not None else -1):
        sim.setObjectParent(tee, pg if pg is not None else -1, True)
    sim.setObjectOrientation(tee, sim.handle_world, [0, 0, 0])
    sim.setObjectPosition(tee, sim.handle_world, [x, y, TABLE_H + POST_H / 2])
    set_shape(sim, tee, static=True, respondable=True, color=DARK, detectable=True)

    make_ball(sim, "ball_lime", pg, x, y, TABLE_H + BALL_ABOVE_SURFACE + 0.001, LIME)
    print("桌子 table_body %.2f m 見方、%.2f m 高放在 (%+.2f, %+.2f)；桌上柱子 tee_lime 頂 %.3f m，球 ball_lime 球心 %.3f m"
          % (TABLE_SIZE, TABLE_H, x, y, TABLE_H + POST_H, TABLE_H + BALL_ABOVE_SURFACE))


def remove_table(sim):
    for alias in ("ball_lime", "table"):
        h = get(sim, alias)
        if h is not None:
            sim.removeObjects(sim.getObjectsInTree(h, sim.handle_all, 0) + [h] if alias == "table" else [h])
    print("桌子、柱子、球已刪除")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只改執行中的場景，不存檔")
    ap.add_argument("--remove", action="store_true", help="拆掉升降柱跟桌子")
    ap.add_argument("--port", type=int, default=23000)
    args = ap.parse_args()

    sim = RemoteAPIClient(port=args.port).require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)
        print("模擬已停止")

    root = find_robot(sim)
    if args.remove:
        remove_lift(sim, root)
        remove_table(sim)
    else:
        build_lift(sim, root)
        build_table(sim)

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
