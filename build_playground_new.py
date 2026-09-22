"""
在 CoppeliaSim 目前開啟的場景中建立「室內兒童遊戲場」/Playground（仿商用室內遊樂場的樣子），
以 RoboMaster EP 的尺寸為基準。

場景（6 x 6 m，牆高 0.6 m，門在南牆）：
  * 彩色拼接軟墊地板（0.75 m 一格，灰／米色交錯）
  * 遊具主結構（北側）：黑色鋼架撐起 0.8 m 高的平台，機器人可以從平台底下開過去；
    平台前方一道藍色溜滑梯（紅色護欄）滑到地面；平台西側一組紅黃相間的軟墊階梯；平台四周有彩色護欄
  * 球池（西南）：1.6 x 1.2 m，黑白格紋的厚軟墊圍牆 14 cm；底層固定的球鋪滿、上層約 60 顆可動的球
  * 橘色方形隧道（東側），內寬 0.5 m，機器人可以開穿過去
  * 四個角落各一根彩色軟包立柱（紅、藍、黃、綠）當地標
  * 三塊分區地墊（紅、綠、藍）、兩個收納箱（藍色裝球、橘色裝小積木）、大型軟墊積木（只能推）
  * 5 根可夾取的小積木（7 x 7 x 15 cm）、2 顆散落的球、東北角一張長椅、門口附近「人」的地面標記

使用方式（模擬停止狀態）：
    pip install coppeliasim-zmqremoteapi-client
    python build_playground.py
    python build_playground.py --variant hidden_red | ball_block
    python build_playground.py --no-tiles           # 不鋪拼接地板（物件數少一點）
    python build_playground.py --remove-objects /Cuboid
    python build_playground.py --remove

腳本會刪掉舊的 /Playground、/Playroom、/Office、/TaskArena 和預設 Floor，量測機器人寬度換算比例，
用相機視線方向把機器人轉成面向 +y、放到起點並貼地。物件路徑如 /Playground/small_red。
"""

import argparse
import math
import random
import sys

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

EP_WIDTH = 0.24
ROOM = 6.0
WALL_T = 0.05
WALL_H = 0.60
DOOR_X, DOOR_W = -1.5, 0.8
ROBOT_START = (-0.6, -2.3)
ROBOT_HEADING = math.pi / 2
TILE = 0.75
GROUND = 10.0

# 精簡配色：只留 12 個色號，結構件（平台護欄、階梯、球池、隧道、收納箱、地墊、立柱…）
# 全部重複使用同一組紅／黃／藍／綠／橘，不再各自配一個「很接近但不同」的色調。
# 紫色只留給 ball_purple 一個物件用（多個任務靠「紫色球」辨識），場景裡不會再有別的東西是紫色。
C = {
    "wall": [0.93, 0.91, 0.86], "tile_a": [0.60, 0.60, 0.62], "tile_b": [0.93, 0.91, 0.86],
    "black": [0.08, 0.08, 0.08], "white": [0.95, 0.95, 0.95], "wood": [0.55, 0.40, 0.28],
    "red": [0.85, 0.16, 0.16], "blue": [0.16, 0.42, 0.85], "green": [0.22, 0.60, 0.30],
    "yellow": [0.97, 0.78, 0.12], "orange": [0.95, 0.50, 0.12], "purple": [0.55, 0.25, 0.75],
}
BALL_COLORS = [C["red"], C["yellow"], C["blue"], C["green"]]

# 世界座標 (x, y)，單位 m，房間中心為原點
POS = {
    "platform": (0.6, 2.1), "slide_x": 1.3, "stairs_x": -0.6,
    "ball_pit": (-1.7, -1.1), "tunnel": (2.2, -0.4),
    "bench": (2.6, 1.6), "bin_balls": (2.6, 0.5), "bin_blocks": (-2.6, 0.9),
    "mat_red": (0.0, -0.5), "mat_green": (0.7, -2.0), "mat_blue": (-0.7, 0.8), "person": (1.7, -2.4),
    "pillar_red": (-2.65, -2.65), "pillar_blue": (2.65, -2.65), "pillar_yellow": (2.65, 2.65),
    "pillar_green": (-2.65, 2.65),
    "foam_cube_red": (-2.2, 2.0), "foam_cube_blue": (-1.6, 0.0), "foam_cyl_yellow": (0.2, 0.6),
    "foam_beam_green": (1.9, 1.0),
    "small_red": (-0.1, -1.6), "small_blue_1": (-2.4, 0.0), "small_blue_2": (0.9, -1.1),
    "small_green": (-2.2, 1.4), "small_yellow": (0.7, -2.0),
    "ball_orange": (0.9, 0.2), "ball_purple": (0.3, -1.3),
}
VARIANTS = {
    "default": {},
    "hidden_red": {"small_red": (2.1, 2.3)},         # 長椅北側、平台東邊，從起點看不到
    "ball_block": {"ball_orange": (-1.4, 0.7)},       # 起點到 small_green 的路上
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

    def shape(self, kind, size, local, color, origin, static=True, respondable=True, tilt=None):
        """tilt = (alpha, beta, gamma) 世界座標 Euler 角，用於斜的滑梯。"""
        sim, s = self.sim, self.s
        prim = {"box": sim.primitiveshape_cuboid, "cyl": sim.primitiveshape_cylinder,
                "sph": sim.primitiveshape_spheroid}[kind]
        opts = 4 | (8 if respondable else 0) | (16 if static else 0)
        h = sim.createPrimitiveShape(prim, [v * s for v in size], opts)
        # 選項位元在某些版本不會真的設定物理屬性，這裡明確設定一次
        try:
            sim.setObjectInt32Param(h, sim.shapeintparam_static, 1 if static else 0)
            sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1 if respondable else 0)
        except Exception:
            pass
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
        if tilt:
            sim.setObjectOrientation(h, list(tilt), sim.handle_world)
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

    def group(self, name, origin, parent=None):
        h = self.sim.createDummy(0.03 * self.s)
        self.sim.setObjectPosition(h, [origin[0] * self.s, origin[1] * self.s, 0], self.sim.handle_world)
        return self.attach(h, name, parent)

    def static_object(self, name, origin, parts, yaw=0.0, parent=None):
        """parts: (name, kind, size, local, color, respondable[, tilt])"""
        g = self.group(name, origin, parent)
        hs = []
        for p in parts:
            pn, k, sz, lc, col, resp = p[:6]
            tilt = p[6] if len(p) > 6 else None
            hs.append(self.attach(self.shape(k, sz, lc, col, origin, True, resp, tilt), pn, g))
        self.rotate(hs, yaw, origin)
        return g

    def dynamic_object(self, name, origin, kind, size, color, mass, z=None, parent=None):
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
        return self.attach(h, name, parent)


# ------------------------------------------------------------------ 場景元件

def build_ground_and_tiles(b, ground, tiles):
    b.static_object("ground", (0, 0), [
        ("ground_slab", "box", [ground, ground, 0.50], (0, 0, -0.25), C["tile_b"], True)])
    if not tiles:
        return
    n = int(round(ROOM / TILE))
    parts = []
    for i in range(n):
        for j in range(n):
            x = -ROOM / 2 + TILE * (i + 0.5)
            y = -ROOM / 2 + TILE * (j + 0.5)
            col = C["tile_a"] if (i + j) % 2 == 0 else C["tile_b"]
            parts.append((f"tile_{i}_{j}", "box", [TILE - 0.01, TILE - 0.01, 0.006], (x, y, 0.003), col, False))
    b.static_object("floor_tiles", (0, 0), parts)


def build_walls(b):
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


def build_play_structure(b):
    """平台 + 溜滑梯 + 階梯 + 護欄，全部靜態。"""
    px, py = POS["platform"]
    W, D, H, deck = 2.4, 1.2, 0.80, 0.06       # 平台寬(x)、深(y)、頂面高
    post = 0.08
    parts = [("deck", "box", [W, D, deck], (0, 0, H - deck / 2), C["green"], True)]
    pads = [C["red"], C["yellow"], C["blue"], C["green"], C["red"], C["blue"]]
    k = 0
    for sx in (-1, 0, 1):
        for sy in (-1, 1):
            x, y = sx * (W / 2 - post / 2), sy * (D / 2 - post / 2)
            parts.append((f"post_{k}", "box", [post, post, H - deck], (x, y, (H - deck) / 2), C["black"], True))
            parts.append((f"post_pad_{k}", "box", [post + 0.06, post + 0.06, 0.45], (x, y, 0.30), pads[k], True))
            k += 1
    rail_h = 0.35
    parts.append(("rail_back", "box", [W, 0.04, rail_h], (0, D / 2 - 0.02, H + rail_h / 2), C["yellow"], True))
    parts.append(("rail_left", "box", [0.04, D, rail_h], (-W / 2 + 0.02, 0, H + rail_h / 2), C["red"], True))
    parts.append(("rail_right", "box", [0.04, D, rail_h], (W / 2 - 0.02, 0, H + rail_h / 2), C["blue"], True))
    sx_local = POS["slide_x"] - px
    front_w = (sx_local - 0.32) - (-W / 2)
    parts.append(("rail_front", "box", [front_w, 0.04, rail_h],
                  (-W / 2 + front_w / 2, -D / 2 + 0.02, H + rail_h / 2), C["green"], True))
    b.static_object("platform", (px, py), parts)

    # 溜滑梯：從平台前緣 (y = py - D/2, z = H) 往南滑到地面
    run, width = 2.0, 0.6
    length = math.hypot(run, H)
    ang = math.atan2(H, run)
    cy = py - D / 2 - run / 2
    cz = H / 2
    tilt = (ang, 0.0, 0.0)                      # 繞 x 軸傾斜，+y 方向往下
    b.static_object("slide", (POS["slide_x"], cy), [
        ("slide_bed", "box", [width, length, 0.05], (0, 0, cz), C["blue"], True, tilt),
        ("slide_rail_L", "box", [0.05, length, 0.20], (-(width / 2 - 0.025), 0, cz + 0.09), C["red"], True, tilt),
        ("slide_rail_R", "box", [0.05, length, 0.20], ((width / 2 - 0.025), 0, cz + 0.09), C["red"], True, tilt),
        ("slide_landing", "box", [width, 0.5, 0.05], (0, -run / 2 - 0.25, 0.025), C["blue"], True),
    ])

    # 軟墊階梯：四階，紅黃相間，從西側上平台
    sx0 = POS["stairs_x"]
    parts = []
    for i in range(4):
        h = 0.2 * (i + 1)
        dx = -0.25 * (3 - i) - 0.125
        col = C["red"] if i % 2 == 0 else C["yellow"]
        parts.append((f"step_{i}", "box", [0.25, 0.8, h], (dx, 0, h / 2), col, True))
    b.static_object("stairs", (sx0, py), parts)


def build_ball_pit(b, inner_x, inner_y, wall_h, wall_t, seed):
    """黑白格紋厚軟墊圍牆、藍色池底、底層固定球 + 上層可動球。"""
    cx, cy = POS["ball_pit"]
    parts = [("ball_pit_base", "box", [inner_x + 2 * wall_t, inner_y + 2 * wall_t, 0.02],
              (0, 0, 0.01), C["blue"], True)]
    seg = 0.2
    zc = 0.02 + wall_h / 2
    for side, horiz, off in (("n", True, inner_y / 2 + wall_t / 2), ("s", True, -(inner_y / 2 + wall_t / 2)),
                             ("e", False, inner_x / 2 + wall_t / 2), ("w", False, -(inner_x / 2 + wall_t / 2))):
        L = inner_x + 2 * wall_t if horiz else inner_y
        n = int(round(L / seg))
        for i in range(n):
            t = -L / 2 + (i + 0.5) * L / n
            col = C["black"] if i % 2 == 0 else C["white"]
            size = [L / n, wall_t, wall_h] if horiz else [wall_t, L / n, wall_h]
            loc = (t, off, zc) if horiz else (off, t, zc)
            parts.append((f"pit_wall_{side}_{i}", "box", size, loc, col, True))
    rng = random.Random(seed)
    d = 0.08
    nx, ny = int(inner_x // (d * 1.05)), int(inner_y // (d * 1.05))
    px, py = inner_x / nx, inner_y / ny
    for i in range(nx):
        for j in range(ny):
            x = -inner_x / 2 + px * (i + 0.5) + rng.uniform(-0.008, 0.008)
            y = -inner_y / 2 + py * (j + 0.5) + rng.uniform(-0.008, 0.008)
            parts.append((f"pit_fixed_{i}_{j}", "sph", [d] * 3, (x, y, 0.02 + d / 2), rng.choice(BALL_COLORS), True))
    b.static_object("ball_pit", (cx, cy), parts)
    g = b.group("pit_balls", (cx, cy))
    for k in range(60):
        x = rng.uniform(-inner_x / 2 + d, inner_x / 2 - d)
        y = rng.uniform(-inner_y / 2 + d, inner_y / 2 - d)
        b.dynamic_object(f"pit_ball_{k}", (cx + x, cy + y), "sph", [d] * 3, rng.choice(BALL_COLORS), 0.03,
                         z=0.02 + d * 1.5 + rng.uniform(0, 0.05), parent=g)


def build_tunnel(b):
    cx, cy = POS["tunnel"]
    L, inner, t = 1.2, 0.5, 0.05
    o = inner + 2 * t
    b.static_object("tunnel", (cx, cy), [
        ("tunnel_left", "box", [t, L, o], (-(inner + t) / 2, 0, o / 2), C["orange"], True),
        ("tunnel_right", "box", [t, L, o], ((inner + t) / 2, 0, o / 2), C["orange"], True),
        ("tunnel_top", "box", [o, L, t], (0, 0, o - t / 2), C["orange"], True),
        ("tunnel_ring_n", "box", [o + 0.10, 0.06, 0.06], (0, L / 2, o + 0.01), C["yellow"], False),
        ("tunnel_ring_s", "box", [o + 0.10, 0.06, 0.06], (0, -L / 2, o + 0.01), C["yellow"], False),
    ])


def open_box(b, name, origin, inner, wall_h, color, t=0.02):
    o = inner + 2 * t
    zc = t + wall_h / 2
    b.static_object(name, origin, [
        (f"{name}_base", "box", [o, o, t], (0, 0, t / 2), color, True),
        (f"{name}_n", "box", [o, t, wall_h], (0, (inner + t) / 2, zc), color, True),
        (f"{name}_s", "box", [o, t, wall_h], (0, -(inner + t) / 2, zc), color, True),
        (f"{name}_e", "box", [t, inner, wall_h], ((inner + t) / 2, 0, zc), color, True),
        (f"{name}_w", "box", [t, inner, wall_h], (-(inner + t) / 2, 0, zc), color, True),
    ])


def build_furniture(b, pos):
    b.static_object("bench", pos["bench"], [
        ("bench_seat", "box", [1.2, 0.4, 0.06], (0, 0, 0.32), C["wood"], True),
        ("bench_back", "box", [1.2, 0.05, 0.30], (0, 0.175, 0.50), C["wood"], True),
        ("bench_leg_L", "box", [0.06, 0.36, 0.29], (-0.55, 0, 0.145), C["black"], True),
        ("bench_leg_R", "box", [0.06, 0.36, 0.29], (0.55, 0, 0.145), C["black"], True),
    ], yaw=math.pi / 2)
    open_box(b, "bin_balls", pos["bin_balls"], 0.45, 0.08, C["blue"])
    open_box(b, "bin_blocks", pos["bin_blocks"], 0.45, 0.08, C["orange"])
    for name in ("mat_red", "mat_green", "mat_blue"):
        base = name.split("_", 1)[1]  # "red" / "green" / "blue"：跟同色小積木共用色號
        b.static_object(name, pos[name], [
            (f"{name}_surface", "box", [1.0, 1.0, 0.012], (0, 0, 0.006), C[base], False)])
    # 人形標記改用木色（原本用紫色，會跟 ball_purple 撞色，任務也不靠顏色辨識這個記號）
    b.static_object("person", pos["person"], [
        ("person_spot", "cyl", [0.6, 0.6, 0.012], (0, 0, 0.006), C["wood"], False)])
    for name, col in (("pillar_red", C["red"]), ("pillar_blue", C["blue"]),
                      ("pillar_yellow", C["yellow"]), ("pillar_green", C["green"])):
        b.static_object(name, pos[name], [
            (f"{name}_body", "cyl", [0.30, 0.30, 1.0], (0, 0, 0.5), col, True),
            (f"{name}_cap", "cyl", [0.34, 0.34, 0.06], (0, 0, 1.03), C["black"], True)])


def build_movables(b, pos):
    b.dynamic_object("foam_cube_red", pos["foam_cube_red"], "box", [0.25, 0.25, 0.25], C["red"], 0.15)
    b.dynamic_object("foam_cube_blue", pos["foam_cube_blue"], "box", [0.25, 0.25, 0.25], C["blue"], 0.15)
    b.dynamic_object("foam_cyl_yellow", pos["foam_cyl_yellow"], "cyl", [0.25, 0.25, 0.30], C["yellow"], 0.15)
    b.dynamic_object("foam_beam_green", pos["foam_beam_green"], "box", [0.60, 0.20, 0.20], C["green"], 0.20)
    for name, color in (("small_red", C["red"]), ("small_blue_1", C["blue"]), ("small_blue_2", C["blue"]),
                        ("small_green", C["green"]), ("small_yellow", C["yellow"])):
        b.dynamic_object(name, pos[name], "box", [0.07, 0.07, 0.15], color, 0.05, z=0.09)
    b.dynamic_object("ball_orange", pos["ball_orange"], "sph", [0.08] * 3, C["orange"], 0.03, z=0.06)
    b.dynamic_object("ball_purple", pos["ball_purple"], "sph", [0.08] * 3, C["purple"], 0.03, z=0.06)


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
    print(f"機器人外框寬 {width:.3f} m、長 {hi[1] - lo[1]:.3f} m（EP 車體 0.24 x 0.32），場景比例 = {scale:.3f}")
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    p = sim.getObjectPosition(h, sim.handle_world)
    sim.setObjectPosition(h, [p[0] + ROBOT_START[0] * scale - cx, p[1] + ROBOT_START[1] * scale - cy,
                              p[2] - lo[2] + 0.01], sim.handle_world)
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
    """列出真正位於場景最上層（沒有父物件）的 shape，通常是使用者之前手動加的方塊。"""
    names = []
    for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0):
        if sim.getObjectParent(h) == -1 and sim.getObjectAlias(h) not in keep:
            names.append("/" + sim.getObjectAlias(h))
    if names:
        print("場景最上層還有這些 shape，若是舊物件請用 --remove-objects 刪除：", " ".join(names))


def main():
    ap = argparse.ArgumentParser(description="以 RoboMaster EP 尺寸為基準建立室內兒童遊戲場")
    ap.add_argument("--variant", choices=list(VARIANTS), default="default")
    ap.add_argument("--robot", default="/RoboMaster")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="場景比例，預設 1.0（模型為真實尺寸時用這個）；給 0 則依量到的機器人寬度自動換算")
    ap.add_argument("--extra-yaw", type=float, default=0.0, help="車頭方向補正角（度），通常不需要")
    ap.add_argument("--no-tiles", action="store_true", help="不鋪拼接地板")
    ap.add_argument("--pit-wall", type=float, default=0.14, help="球池圍牆高 (m)")
    ap.add_argument("--seed", type=int, default=7, help="球池球的隨機配置種子")
    ap.add_argument("--keep-floor", action="store_true", help="保留預設 Floor")
    ap.add_argument("--remove-objects", nargs="*", default=[])
    ap.add_argument("--remove", action="store_true", help="只刪除 /Playground")
    args = ap.parse_args()

    sim = RemoteAPIClient().require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sys.exit("請先停止模擬再執行此腳本。")

    for p in ["/Playground", "/Playroom", "/Office", "/TaskArena"] + args.remove_objects:
        remove_tree(sim, p)
    if args.remove:
        return
    if not args.keep_floor:
        remove_tree(sim, "/Floor")

    scale = prepare_robot(sim, args.robot, args.scale or None, math.radians(args.extra_yaw))

    root = sim.createDummy(0.05)
    sim.setObjectAlias(root, "Playground")
    sim.setObjectPosition(root, [0, 0, 0], sim.handle_world)
    b = Builder(sim, root, scale)

    pos = dict(POS)
    pos.update(VARIANTS[args.variant])
    build_ground_and_tiles(b, GROUND, not args.no_tiles)
    build_walls(b)
    build_play_structure(b)
    build_ball_pit(b, 1.6, 1.2, args.pit_wall, 0.20, args.seed)
    build_tunnel(b)
    build_furniture(b, pos)
    build_movables(b, pos)

    list_strays(sim, {"Floor", args.robot.strip("/"), "Playground"})
    n = len(sim.getObjectsInTree(root, sim.handle_all, 0))
    print(f"遊戲場建立完成（variant = {args.variant}，scale = {scale:.3f}），共 {n} 個物件。記得存檔。")


if __name__ == "__main__":
    main()