"""
在 CoppeliaSim 目前開啟的場景中建立「室內遊戲廣場」/Playground，以 RoboMaster EP 的尺寸為基準。

場景內容（5 x 5 m，牆高 0.5 m，門在南牆）：
  * 軟墊積木區（西側）：25 到 30 cm 的大型軟墊積木（方塊、圓柱、長條），機器人只能推
  * 球池（東北角）：1.4 x 1.4 m、圍欄 6 cm，池底鋪滿 8 cm 彩色球（約 170 顆，物理計算會變重，--pit-size 1.0 約 80 顆）；另外有 2 顆球散落在外面
  * 自己鋪一塊 9 x 9 m 的地板取代預設的 5 x 5 Floor，牆外不再是黑的
  * 三塊分區地墊（紅、綠、藍）和兩個收納箱（藍色裝球、橘色裝小積木）
  * 5 根可夾取的小積木（7 x 7 x 15 cm：紅、藍 x2、綠、黃）
  * 北牆一張長椅，門口附近一個「人」的地面標記

使用方式（模擬停止狀態）：
    pip install coppeliasim-zmqremoteapi-client
    python build_playground.py
    python build_playground.py --variant hidden_red      # H07：紅色小積木藏在球池後面
    python build_playground.py --variant ball_block      # M18：球擋在機器人和綠色小積木之間
    python build_playground.py --pit-wall 0.04 --pit-size 1.6   # 更矮的圍欄、更大的球池
    python build_playground.py --remove-objects /Cuboid  # 順便刪掉舊物件
    python build_playground.py --remove                  # 只刪除 /Playground

腳本會：刪掉舊的 /Playground、/Playroom、/Office、/TaskArena；量測機器人寬度換算比例；
用相機視線方向把機器人轉成面向 +y、放到起點並貼地。物件路徑如 /Playground/small_red。
"""

import argparse
import math
import sys

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

EP_WIDTH = 0.24
ROOM = 5.0
WALL_T = 0.05
WALL_H = 0.50
DOOR_X, DOOR_W = -1.5, 0.7
ROBOT_START = (-1.0, -1.7)
ROBOT_HEADING = math.pi / 2
PIT_INNER = 1.4          # 球池內寬 (m)
PIT_WALL = 0.06          # 球池圍欄高 (m)，比 8 cm 的球矮，機器人看得到球、也搆得到
GROUND = 9.0             # 自鋪地板邊長 (m)

C = {
    "wall": [0.93, 0.90, 0.84], "bench": [0.55, 0.40, 0.28], "bench_dk": [0.40, 0.28, 0.18],
    "pit": [0.20, 0.55, 0.80], "bin_ball": [0.15, 0.40, 0.85], "bin_block": [0.95, 0.55, 0.20],
    "mat_red": [0.90, 0.35, 0.35], "mat_green": [0.40, 0.75, 0.45], "mat_blue": [0.45, 0.65, 0.90],
    "person": [0.60, 0.40, 0.80],
    "red": [0.85, 0.12, 0.12], "blue": [0.15, 0.35, 0.85], "green": [0.20, 0.65, 0.25],
    "yellow": [0.97, 0.82, 0.10], "orange": [0.98, 0.50, 0.10], "purple": [0.60, 0.25, 0.75],
    "pink": [0.95, 0.45, 0.70], "cyan": [0.20, 0.80, 0.85],
}

# 世界座標 (x, y)，單位 m，房間中心為原點
POS = {
    "bench": (0.0, 2.2), "ball_pit": (1.65, 1.65), "bin_balls": (2.1, -0.3), "bin_blocks": (-2.1, 1.6),
    "mat_red": (-1.6, -0.2), "mat_green": (0.6, -0.5), "mat_blue": (0.3, 1.2), "person": (1.6, -1.8),
    "foam_cube_red": (-1.7, 0.9), "foam_cube_blue": (-0.9, 1.5), "foam_cyl_yellow": (-1.1, 0.3),
    "foam_beam_green": (-0.3, 0.4),
    "small_red": (-0.5, -1.0), "small_blue_1": (-1.9, -1.1), "small_blue_2": (0.9, 0.3),
    "small_green": (-1.5, 1.6), "small_yellow": (0.2, -0.1),
    "ball_orange": (1.3, 0.2), "ball_purple": (0.4, -1.4),
}
VARIANTS = {
    "default": {},
    "hidden_red": {"small_red": (2.2, 0.6)},        # 球池南側、東牆邊，從起點看不到
    "ball_block": {"ball_orange": (-1.3, 0.1)},      # 機器人起點與 small_green 之間
}


class Builder:
    def __init__(self, sim, root, s):
        self.sim, self.root, self.s = sim, root, s

    def _props(self, h):
        sim = self.sim
        try:
            sim.setObjectSpecialProperty(
                h, sim.objectspecialproperty_collidable | sim.objectspecialproperty_measurable
                | sim.objectspecialproperty_detectable | sim.objectspecialproperty_renderable)
        except Exception:
            pass

    def shape(self, kind, size, local, color, origin, static=True, respondable=True):
        sim, s = self.sim, self.s
        prim = {"box": sim.primitiveshape_cuboid, "cyl": sim.primitiveshape_cylinder,
                "sph": sim.primitiveshape_spheroid}[kind]
        opts = 4 | (8 if respondable else 0) | (16 if static else 0)
        h = sim.createPrimitiveShape(prim, [v * s for v in size], opts)
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
        sim.setObjectPosition(h, [(origin[0] + local[0]) * s, (origin[1] + local[1]) * s, local[2] * s],
                              sim.handle_world)
        self._props(h)
        return h

    def rotate(self, handles, yaw, origin):
        if not yaw:
            return
        sim = self.sim
        pivot = [origin[0] * self.s, origin[1] * self.s, 0.0]
        for h in handles:
            m = sim.getObjectMatrix(h, sim.handle_world)
            sim.setObjectMatrix(h, sim.rotateAroundAxis(m, [0, 0, 1], pivot, yaw), sim.handle_world)

    def attach(self, h, name, parent=None):
        self.sim.setObjectAlias(h, name)
        self.sim.setObjectParent(h, parent if parent is not None else self.root, True)
        return h

    def group(self, name, origin):
        h = self.sim.createDummy(0.03 * self.s)
        self.sim.setObjectPosition(h, [origin[0] * self.s, origin[1] * self.s, 0], self.sim.handle_world)
        return self.attach(h, name)

    def static_object(self, name, origin, parts, yaw=0.0):
        g = self.group(name, origin)
        hs = [self.attach(self.shape(k, sz, lc, col, origin, True, resp), pn, g)
              for pn, k, sz, lc, col, resp in parts]
        self.rotate(hs, yaw, origin)
        return g

    def dynamic_object(self, name, origin, kind, size, color, mass, z=None):
        """單一凸形狀的動態物件（避免 non-convex 警告與不穩定）。"""
        sim = self.sim
        z = size[2] / 2 + 0.002 if z is None else z
        h = self.shape(kind, size, (0, 0, z), color, origin, static=False)
        for param, val in (("shapeintparam_static", 0), ("shapeintparam_respondable", 1)):
            try:
                sim.setObjectInt32Param(h, getattr(sim, param), val)
            except Exception:
                pass
        try:
            sim.setShapeMass(h, mass)
            sim.resetDynamicObject(h)
        except Exception:
            pass
        return self.attach(h, name)


# ------------------------------------------------------------------ 場景

def open_box(b, name, origin, inner, wall_h, color, t=0.02):
    """底板加四面矮牆的開口箱子（球池、收納箱共用）。"""
    o = inner + 2 * t
    zc = t + wall_h / 2
    b.static_object(name, origin, [
        (f"{name}_base", "box", [o, o, t], (0, 0, t / 2), color, True),
        (f"{name}_n", "box", [o, t, wall_h], (0, (inner + t) / 2, zc), color, True),
        (f"{name}_s", "box", [o, t, wall_h], (0, -(inner + t) / 2, zc), color, True),
        (f"{name}_e", "box", [t, inner, wall_h], ((inner + t) / 2, 0, zc), color, True),
        (f"{name}_w", "box", [t, inner, wall_h], (-(inner + t) / 2, 0, zc), color, True),
    ])


def build_ground(b, size):
    """自己鋪一塊比房間大的地板，讓牆外不是黑的。頂面在 z = 0。"""
    b.static_object("ground", (0, 0), [
        ("ground_slab", "box", [size, size, 0.10], (0, 0, -0.05), [0.82, 0.80, 0.76], True)])


def build_scene(b, pos):
    hx, z = ROOM / 2, WALL_H / 2
    L = (DOOR_X - DOOR_W / 2) + hx
    R = hx - (DOOR_X + DOOR_W / 2)
    b.static_object("walls", (0, 0), [
        ("wall_north", "box", [ROOM, WALL_T, WALL_H], (0, hx - WALL_T / 2, z), C["wall"], True),
        ("wall_east", "box", [WALL_T, ROOM - 2 * WALL_T, WALL_H], (hx - WALL_T / 2, 0, z), C["wall"], True),
        ("wall_west", "box", [WALL_T, ROOM - 2 * WALL_T, WALL_H], (-hx + WALL_T / 2, 0, z), C["wall"], True),
        ("wall_south_L", "box", [L, WALL_T, WALL_H], (-hx + L / 2, -hx + WALL_T / 2, z), C["wall"], True),
        ("wall_south_R", "box", [R, WALL_T, WALL_H], (hx - R / 2, -hx + WALL_T / 2, z), C["wall"], True),
    ])
    b.group("door", (DOOR_X, -hx))

    # 長椅：1.2 x 0.4、座高 0.35，靠北牆
    b.static_object("bench", pos["bench"], [
        ("bench_seat", "box", [1.2, 0.4, 0.06], (0, 0, 0.32), C["bench"], True),
        ("bench_back", "box", [1.2, 0.05, 0.30], (0, 0.175, 0.50), C["bench"], True),
        ("bench_leg_L", "box", [0.06, 0.36, 0.29], (-0.55, 0, 0.145), C["bench_dk"], True),
        ("bench_leg_R", "box", [0.06, 0.36, 0.29], (0.55, 0, 0.145), C["bench_dk"], True),
    ])

    open_box(b, "ball_pit", pos["ball_pit"], PIT_INNER, PIT_WALL, C["pit"])
    open_box(b, "bin_balls", pos["bin_balls"], 0.45, 0.08, C["bin_ball"])
    open_box(b, "bin_blocks", pos["bin_blocks"], 0.45, 0.08, C["bin_block"])

    for name in ("mat_red", "mat_green", "mat_blue"):
        b.static_object(name, pos[name], [
            (f"{name}_surface", "box", [1.0, 1.0, 0.004], (0, 0, 0.002), C[name], False)])

    b.static_object("person", pos["person"], [
        ("person_spot", "cyl", [0.6, 0.6, 0.003], (0, 0, 0.0015), C["person"], False)])

    # 大型軟墊積木：只能推
    b.dynamic_object("foam_cube_red", pos["foam_cube_red"], "box", [0.25, 0.25, 0.25], C["red"], 0.15)
    b.dynamic_object("foam_cube_blue", pos["foam_cube_blue"], "box", [0.25, 0.25, 0.25], C["blue"], 0.15)
    b.dynamic_object("foam_cyl_yellow", pos["foam_cyl_yellow"], "cyl", [0.25, 0.25, 0.30], C["yellow"], 0.15)
    b.dynamic_object("foam_beam_green", pos["foam_beam_green"], "box", [0.60, 0.20, 0.20], C["green"], 0.20)

    # 小積木：7 x 7 x 15，可夾
    for name, color in (("small_red", C["red"]), ("small_blue_1", C["blue"]), ("small_blue_2", C["blue"]),
                        ("small_green", C["green"]), ("small_yellow", C["yellow"])):
        b.dynamic_object(name, pos[name], "box", [0.07, 0.07, 0.18], color, 0.05)

    # 散落在外面的球
    b.dynamic_object("ball_orange", pos["ball_orange"], "sph", [0.08] * 3, C["orange"], 0.03)
    b.dynamic_object("ball_purple", pos["ball_purple"], "sph", [0.08] * 3, C["purple"], 0.03)

    # 球池裡的球：n x n 排列鋪滿池底，顏色交錯；球比圍欄高，相機看得到
    colors = [C["red"], C["yellow"], C["pink"], C["green"], C["blue"], C["purple"]]
    px, py = pos["ball_pit"]
    n = max(1, int(PIT_INNER // 0.10))
    pitch = PIT_INNER / n
    k = 0
    for i in range(n):
        for j in range(n):
            x = px - PIT_INNER / 2 + pitch * (i + 0.5)
            y = py - PIT_INNER / 2 + pitch * (j + 0.5)
            b.dynamic_object(f"pit_ball_{k}", (x, y), "sph", [0.08] * 3, colors[(k * 3 + i) % 8], 0.03, z=0.07)
            k += 1


# ------------------------------------------------------------------ 機器人

def world_aabb(sim, root):
    lo, hi = [1e9] * 3, [-1e9] * 3
    P = [sim.objfloatparam_objbbox_min_x, sim.objfloatparam_objbbox_min_y, sim.objfloatparam_objbbox_min_z,
         sim.objfloatparam_objbbox_max_x, sim.objfloatparam_objbbox_max_y, sim.objfloatparam_objbbox_max_z]
    for h in sim.getObjectsInTree(root, sim.object_shape_type, 0):
        v = [sim.getObjectFloatParam(h, p) for p in P]
        m = sim.getObjectMatrix(h, sim.handle_world)
        for cx in (v[0], v[3]):
            for cy in (v[1], v[4]):
                for cz in (v[2], v[5]):
                    w = [m[4 * r] * cx + m[4 * r + 1] * cy + m[4 * r + 2] * cz + m[4 * r + 3] for r in range(3)]
                    lo = [min(a, c) for a, c in zip(lo, w)]
                    hi = [max(a, c) for a, c in zip(hi, w)]
    return lo, hi


def prepare_robot(sim, path, fixed_scale, extra_yaw=0.0):
    h = sim.getObject(path, {"noError": True})
    if h == -1:
        print(f"找不到 {path}，比例使用 1.0，機器人位置不變（可用 --robot 指定路徑）")
        return fixed_scale or 1.0
    rp = sim.getObjectPosition(h, sim.handle_world)

    cam = -1
    for o in sim.getObjectsInTree(h, sim.object_visionsensor_type, 0):
        cam = o
        break
    if cam != -1:
        cm = sim.getObjectMatrix(cam, sim.handle_world)
        heading = math.atan2(cm[6], cm[2])
    else:
        heading = None
        for o in sim.getObjectsInTree(h, sim.handle_all, 0):
            if "gripper" in sim.getObjectAlias(o).lower():
                gp = sim.getObjectPosition(o, sim.handle_world)
                heading = math.atan2(gp[1] - rp[1], gp[0] - rp[0])
                break
        if heading is None:
            m = sim.getObjectMatrix(h, sim.handle_world)
            heading = math.atan2(m[4], m[0])
        print("找不到相機，改用夾爪或 local +x 判斷車頭")
    m = sim.getObjectMatrix(h, sim.handle_world)
    sim.setObjectMatrix(h, sim.rotateAroundAxis(m, [0, 0, 1], rp, ROBOT_HEADING - heading + extra_yaw),
                        sim.handle_world)
    if cam != -1:
        cm = sim.getObjectMatrix(cam, sim.handle_world)
        print(f"轉向後相機朝向 {math.degrees(math.atan2(cm[6], cm[2])):.0f} 度（+y 應為 90）")

    lo, hi = world_aabb(sim, h)
    width = hi[0] - lo[0]
    scale = fixed_scale or (width / EP_WIDTH if width > 0 else 1.0)
    print(f"機器人寬 {width:.3f} m、長 {hi[1] - lo[1]:.3f} m，比例 = {scale:.3f}")

    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    p = sim.getObjectPosition(h, sim.handle_world)
    sim.setObjectPosition(h, [p[0] + ROBOT_START[0] * scale - cx, p[1] + ROBOT_START[1] * scale - cy,
                              p[2] - lo[2] + 0.002], sim.handle_world)
    for o in sim.getObjectsInTree(h, sim.handle_all, 0):
        try:
            sim.resetDynamicObject(o)
        except Exception:
            pass
    return scale


# ------------------------------------------------------------------ 主程式

def remove_tree(sim, path):
    h = sim.getObject(path, {"noError": True})
    if h != -1:
        objs = sim.getObjectsInTree(h, sim.handle_all, 0)
        sim.removeObjects(objs)
        print(f"已刪除 {path}（{len(objs)} 個物件）")


def list_strays(sim, keep):
    names = ["/" + sim.getObjectAlias(h) for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 1)
             if sim.getObjectAlias(h) not in keep]
    if names:
        print("場景最上層還有這些 shape，若是舊物件請用 --remove-objects 刪除：", " ".join(names))


def main():
    global PIT_WALL, PIT_INNER
    ap = argparse.ArgumentParser(description="以 RoboMaster EP 尺寸為基準建立室內遊戲廣場")
    ap.add_argument("--variant", choices=list(VARIANTS), default="default")
    ap.add_argument("--robot", default="/RoboMaster")
    ap.add_argument("--scale", type=float, default=None, help="強制比例，不量測機器人")
    ap.add_argument("--extra-yaw", type=float, default=0.0, help="車頭方向補正角（度），通常不需要")
    ap.add_argument("--pit-wall", type=float, default=PIT_WALL, help="球池圍欄高 (m)")
    ap.add_argument("--pit-size", type=float, default=PIT_INNER, help="球池內寬 (m)，球數會自動鋪滿")
    ap.add_argument("--ground", type=float, default=GROUND, help="自鋪地板邊長 (m)")
    ap.add_argument("--keep-floor", action="store_true", help="保留 CoppeliaSim 預設的 5x5 Floor（會和自鋪地板重疊）")
    ap.add_argument("--remove-objects", nargs="*", default=[])
    ap.add_argument("--remove", action="store_true", help="只刪除 /Playground")
    args = ap.parse_args()

    PIT_WALL, PIT_INNER = args.pit_wall, args.pit_size

    sim = RemoteAPIClient().require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sys.exit("請先停止模擬再執行此腳本。")

    for p in ["/Playground", "/Playroom", "/Office", "/TaskArena"] + args.remove_objects:
        remove_tree(sim, p)
    if args.remove:
        return
    if not args.keep_floor:
        remove_tree(sim, "/Floor")     # 用自鋪的大地板取代預設 5x5 Floor

    scale = prepare_robot(sim, args.robot, args.scale, math.radians(args.extra_yaw))

    root = sim.createDummy(0.05)
    sim.setObjectAlias(root, "Playground")
    sim.setObjectPosition(root, [0, 0, 0], sim.handle_world)
    b = Builder(sim, root, scale)

    pos = dict(POS)
    pos.update(VARIANTS[args.variant])
    build_ground(b, args.ground)
    build_scene(b, pos)

    list_strays(sim, {"Floor", args.robot.strip("/"), "Playground"})
    n = len(sim.getObjectsInTree(root, sim.handle_all, 0))
    print(f"遊戲廣場建立完成（variant = {args.variant}，scale = {scale:.3f}），共 {n} 個物件。記得存檔。")


if __name__ == "__main__":
    main()
