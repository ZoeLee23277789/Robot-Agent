"""
把 /RoboMaster 移回 build_playground_new.py 定義的起點 (ROBOT_START)，方向轉正對著 ROBOT_HEADING（目前是 +x）、貼地，
不會動到場景裡任何其他物件。

sim/real 模式下，robot_server.py 的 /reset 只會收手臂、開夾爪，位置不會動（README 就是這樣寫的，
mock 模式才會整個重置）——手動測試、把機器人搬到各種角落之後，這支腳本就是拿來把它放回乾淨的起點，
不用重新跑一次 build_playground_new.py 整個重建場景。

跟 build_playground_new.py 的 prepare_robot() 用同一套「量相機朝向、轉正、移到起點、貼地」邏輯，
但故意不呼叫它最後 sim.resetDynamicObject() 那段：那段是設計給模擬「停止」狀態、場景剛蓋好、
機器人控制腳本還沒開始跑的時候用的。實測發現如果在模擬「播放中」、RoboMaster 控制腳本已經在跑
的時候呼叫 resetDynamicObject()，會讓底盤整個沒反應（呼叫 move_chassis 都回 timeout、travelled
永遠是 0），要停止再重新播放模擬才能救回來。所以這支腳本只搬位置、不動態力學重置，可以在模擬
播放中安全執行。

使用方式：
    python reset_robot.py
"""
import argparse
import math
import os
import sys

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# build_playground_new.py 已經搬到 Env/，這支腳本還留在 Test_Robot/，把 Env/ 加進路徑才 import 得到
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Env"))
from build_playground_new import POS, ROBOT_HEADING, ROBOT_START, world_aabb  # noqa: E402

MOVABLE = ("foam_cube_red", "foam_cube_blue", "foam_cyl_yellow", "foam_beam_green",
           "small_red", "small_blue_1", "small_blue_2", "small_green", "small_yellow", "ball_orange", "ball_purple")


def reset_objects(sim, scale=1.0):
    """零散物件放回 build_playground_new.py 的位置、擺正。回傳 [(name, 移動距離)]。
    實測存檔的場景裡 foam_beam_green 已經被推到 1 m 外還轉了 90°、small_yellow 偏了 0.44 m，
    每次模擬重播都從這個歪掉的狀態開始，回歸測試跟跑分的條件才會跟建場景時不一樣。"""
    shapes = {sim.getObjectAlias(o): o for o in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0)}
    moved = []
    for name in MOVABLE:
        h = shapes.get(name)
        if h is None or name not in POS:
            continue
        x, y = POS[name][0] * scale, POS[name][1] * scale
        p = sim.getObjectPosition(h, sim.handle_world)
        zmin = sim.getObjectFloatParam(h, sim.objfloatparam_objbbox_min_z)
        zmax = sim.getObjectFloatParam(h, sim.objfloatparam_objbbox_max_z)
        d = math.hypot(p[0] - x, p[1] - y)
        sim.setObjectOrientation(h, [0.0, 0.0, 0.0], sim.handle_world)
        sim.setObjectPosition(h, [x, y, (zmax - zmin) / 2 + 0.002], sim.handle_world)
        try:
            sim.resetDynamicObject(h)
        except Exception:
            pass
        moved.append((name, d))
    return moved


def find_robot(sim, path):
    """場景若有多台 alias 相同的 RoboMaster（實測踩過：一台是被拆過、外掛不收的複製品），
    挑有 GyroSensor 的完整模型，跟 robot_server.py 的 _find_robot_root 同一套規則。"""
    roots = [o for o in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2)
             if sim.getObjectAlias(o) == path.strip("/")]
    if not roots:
        raise SystemExit(f"找不到 {path}")
    good = [h for h in roots if any(sim.getObjectAlias(o) == "GyroSensor" for o in sim.getObjectsInTree(h, sim.handle_all, 0))]
    if len(roots) > 1:
        print(f"!! 場景裡有 {len(roots)} 台 {path} 模型 {roots}，完整的是 {good}，只搬 {(good or roots)[0]}；請刪掉多餘的並存檔")
    return (good or roots)[0]


def reset_robot(sim, path, fixed_scale):
    h = find_robot(sim, path)
    rp = sim.getObjectPosition(h, sim.handle_world)

    cam = -1
    for o in sim.getObjectsInTree(h, sim.object_visionsensor_type, 0):
        cam = o
        break
    if cam != -1:
        cm = sim.getObjectMatrix(cam, sim.handle_world)
        heading = math.atan2(cm[6], cm[2])
    else:
        m = sim.getObjectMatrix(h, sim.handle_world)
        heading = math.atan2(m[4], m[0])
        print("找不到相機，改用 local +x 判斷車頭")

    m = sim.getObjectMatrix(h, sim.handle_world)
    sim.setObjectMatrix(h, sim.rotateAroundAxis(m, [0, 0, 1], rp, ROBOT_HEADING - heading), sim.handle_world)
    if cam != -1:
        cm = sim.getObjectMatrix(cam, sim.handle_world)
        print(f"轉向後相機朝向 {math.degrees(math.atan2(cm[6], cm[2])):.0f} 度（目標 {math.degrees(ROBOT_HEADING):.0f}）")

    lo, hi = world_aabb(sim, h)
    width = hi[0] - lo[0]
    scale = fixed_scale or (width / 0.24 if width > 0 else 1.0)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    p = sim.getObjectPosition(h, sim.handle_world)
    sim.setObjectPosition(h, [p[0] + ROBOT_START[0] * scale - cx, p[1] + ROBOT_START[1] * scale - cy,
                              p[2] - lo[2] + 0.01], sim.handle_world)
    return scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="/RoboMaster")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="要跟建場景那次用的 --scale 一致，預設 1.0（build_playground_new.py 的預設值）")
    ap.add_argument("--objects", action="store_true", help="也把零散物件（泡棉、積木、球）放回建場景時的位置並擺正")
    ap.add_argument("--no-robot", action="store_true",
                    help="不動機器人（server 連著的時候把車傳送走會讓底盤控制失效，要重開 server 才會恢復）")
    args = ap.parse_args()

    sim = RemoteAPIClient().require("sim")
    if args.no_robot:
        scale = args.scale or 1.0
    else:
        scale = reset_robot(sim, args.robot, args.scale or None)
        print(f"機器人已經移回起點 {ROBOT_START}、面向 {math.degrees(ROBOT_HEADING):.0f} 度，場景比例 = {scale:.3f}")
    if args.objects:
        moved = reset_objects(sim, scale)
        big = [(n, d) for n, d in moved if d >= 0.05]
        print("零散物件已放回建場景位置：%d 個；原本偏離 ≥ 5 cm 的：%s" % (
            len(moved), ", ".join("%s %.2f m" % t for t in big) if big else "無"))


if __name__ == "__main__":
    main()
