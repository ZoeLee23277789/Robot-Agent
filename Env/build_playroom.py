"""
在 CoppeliaSim 目前開啟的場景中，以 RoboMaster EP 的尺寸為基準建立遊戲房 /Playroom。

和前一版的差異：
  * 執行時先量測機器人模型的實際寬度，換算比例 scale = 量到的寬度 / 0.24 m（EP 實車寬），
    房間、家具、玩具的尺寸與位置全部乘上這個比例，所以不管模型本身的單位是否正確，比例都一致。
  * 用機器人相機的視線方向判斷車頭，把機器人轉成面向沙發 (+y)，並貼齊地面。
  * 房間 4 x 4 m、牆高 0.5 m（--room 可調）；沙發、櫃子為兒童尺寸；人改成地上的圓形標記。
  * 玩具全部做成約 15 cm 高讓相機看得到：積木 6 x 6 x 15 cm 長柱、玩具卡車 15 cm 高、小鴨 15 cm 高、
    泰迪熊 34 cm 高；可夾的玩具水平截面維持 6 到 7 cm（EP 夾爪開口約 10 cm）。球是例外，做成 9 cm。
    --toy-scale 可再整體縮放，超過 1.3 積木就會超出夾爪開口。
  * 家具正面朝向房間中央，玩具正面朝向機器人。

使用方式（模擬停止狀態）：
    pip install coppeliasim-zmqremoteapi-client
    python build_playroom.py
    python build_playroom.py --variant hidden_red      # H07：紅積木藏在沙發後面
    python build_playroom.py --variant ball_block      # M18：球擋在機器人和綠積木之間
    python build_playroom.py --remove-objects /Cuboid /Cuboid0
    python build_playroom.py --room 5 --toy-scale 1.2  # 更大的房間、更大的玩具
    python build_playroom.py --scale 1.0               # 不量測，強制使用指定比例
    python build_playroom.py --extra-yaw 90            # 車頭方向自動判斷錯時的補正（度）
    python build_playroom.py --remove                  # 只刪除 /Playroom

座標：房間中心為原點，單位公尺，+y 為北牆。POS 表以 3 m 房間定義，
實際位置 = POS × (ROOM / 3) × 機器人比例。預設 4 m 房間時門在南牆 x ≈ -1.07，機器人起點 ≈ (-0.27, -1.07)，面向 +y。
"""

import argparse
import math
import sys

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

EP_WIDTH = 0.24          # RoboMaster EP 實車寬度 (m)

ROOM = 4.0               # 房間邊長 (m)，可用 --room 覆寫
BASE_ROOM = 3.0          # POS 表是以 3 m 房間定義的，實際位置乘上 ROOM / BASE_ROOM
TOY = 1.0                # 玩具額外倍率，可用 --toy-scale 覆寫（1.0 = 玩具約 15 cm 高）
WALL_T = 0.05
WALL_H = 0.50
DOOR_X, DOOR_W = -0.8, 0.6
ROBOT_START = (-0.2, -0.8)
ROBOT_HEADING = math.pi / 2          # 面向 +y

C = {
    "wall": [0.93, 0.90, 0.84], "mat": [0.55, 0.80, 0.90], "mat_rim": [0.30, 0.55, 0.75],
    "box": [0.95, 0.55, 0.20], "sofa": [0.45, 0.58, 0.45], "sofa_dk": [0.35, 0.47, 0.35],
    "shelf": [0.80, 0.68, 0.50], "person": [0.60, 0.40, 0.80],
    "red": [0.85, 0.12, 0.12], "blue": [0.15, 0.35, 0.85], "green": [0.20, 0.65, 0.25],
    "yellow": [0.97, 0.82, 0.10], "orange": [0.98, 0.50, 0.10], "car": [0.20, 0.45, 0.85],
    "tire": [0.10, 0.10, 0.10], "glass": [0.75, 0.88, 0.95], "bear": [0.62, 0.42, 0.25],
    "bear_lt": [0.80, 0.62, 0.42], "black": [0.05, 0.05, 0.05],
}

# 位置（scale = 1）。與 playroom_tasks.xlsx 的 Scene 分頁一致。
POS = {
    "sofa": (0.2, 0.875), "toy_box": (-1.05, 1.05), "toy_shelf": (1.325, 0.8),
    "play_mat": (0.85, -0.25), "person": (1.05, -1.05),
    "red_block": (0.3, -0.45), "blue_block_1": (-0.85, -0.35), "blue_block_2": (0.35, 0.35),
    "green_block": (-0.6, 0.6), "yellow_block": (-0.3, 0.15), "ball_orange": (0.6, 0.45),
    "toy_car": (0.85, -0.25), "rubber_duck": (-0.9, -1.1), "teddy_bear": (-1.15, 0.2),
}
VARIANTS = {
    "default": {},
    "hidden_red": {"red_block": (0.2, 1.28)},     # 沙發和北牆之間的通道裡
    "ball_block": {"ball_orange": (-0.4, -0.1)},  # 機器人起點和綠積木之間
}
# 正面朝向（家具的 local +x 為正面）
FACING = {"sofa": -math.pi / 2, "toy_shelf": math.pi, "teddy_bear": 0.0}


class Builder:
    def __init__(self, sim, root, s):
        self.sim, self.root, self.s = sim, root, s
        self.L = ROOM / BASE_ROOM      # 位置放大倍率

    def at(self, xy):
        """把 3 m 房間定義的位置換算成目前房間大小。"""
        return (xy[0] * self.L, xy[1] * self.L)

    def _props(self, h):
        sim = self.sim
        try:
            sim.setObjectSpecialProperty(
                h, sim.objectspecialproperty_collidable | sim.objectspecialproperty_measurable
                | sim.objectspecialproperty_detectable | sim.objectspecialproperty_renderable)
        except Exception:
            pass

    def shape(self, kind, size, local, color, origin, static=True, respondable=True, roll=0.0):
        """size 與 local 都是 scale = 1 的數值；local 是相對物件原點 origin 的偏移（未旋轉）。"""
        sim, s = self.sim, self.s
        prim = {"box": sim.primitiveshape_cuboid, "cyl": sim.primitiveshape_cylinder,
                "sph": sim.primitiveshape_spheroid}[kind]
        opts = 4 | (8 if respondable else 0) | (16 if static else 0)
        h = sim.createPrimitiveShape(prim, [v * s for v in size], opts)
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
        if roll:
            sim.setObjectOrientation(h, [roll, 0, 0], sim.handle_world)
        sim.setObjectPosition(h, [(origin[0] + local[0]) * s, (origin[1] + local[1]) * s,
                                  local[2] * s], sim.handle_world)
        self._props(h)
        return h

    def rotate(self, handles, yaw, origin):
        """以物件原點為中心、繞世界 z 軸旋轉，與各 shape 本身的座標框無關。"""
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
        """parts: [(part_name, kind, size, local, color, respondable)]"""
        origin = self.at(origin)
        g = self.group(name, origin)
        hs = []
        for pname, kind, size, local, color, resp in parts:
            h = self.shape(kind, size, local, color, origin, static=True, respondable=resp)
            hs.append(self.attach(h, pname, g))
        self.rotate(hs, yaw, origin)
        return g

    def dynamic_object(self, name, origin, parts, mass, yaw=0.0):
        """parts: [(kind, size, local, color, roll)]，多個 part 會合併成單一 compound。
        玩具的尺寸與各 part 的相對位置都乘上 TOY 倍率。"""
        sim = self.sim
        origin = self.at(origin)
        hs = [self.shape(k, [v * TOY for v in sz], [v * TOY for v in lc], col, origin, static=False, roll=r)
              for k, sz, lc, col, r in parts]
        self.rotate(hs, yaw, origin)
        h = hs[0] if len(hs) == 1 else sim.groupShapes(hs)
        for param, val in (("shapeintparam_static", 0), ("shapeintparam_respondable", 1)):
            try:
                sim.setObjectInt32Param(h, getattr(sim, param), val)
            except Exception:
                pass
        self._props(h)
        try:
            sim.setShapeMass(h, mass)
        except Exception:
            pass
        try:
            sim.resetDynamicObject(h)
        except Exception:
            pass
        return self.attach(h, name)


def yaw_toward(src, dst):
    return math.atan2(dst[1] - src[1], dst[0] - src[0])


# ------------------------------------------------------------------ 場景內容

def build_room(b):
    hx = ROOM / 2
    z = WALL_H / 2
    dx = DOOR_X * b.L
    L = (dx - DOOR_W / 2) + hx
    R = hx - (dx + DOOR_W / 2)
    b.static_object("walls", (0, 0), [
        ("wall_north", "box", [ROOM, WALL_T, WALL_H], (0, hx - WALL_T / 2, z), C["wall"], True),
        ("wall_east", "box", [WALL_T, ROOM - 2 * WALL_T, WALL_H], (hx - WALL_T / 2, 0, z), C["wall"], True),
        ("wall_west", "box", [WALL_T, ROOM - 2 * WALL_T, WALL_H], (-hx + WALL_T / 2, 0, z), C["wall"], True),
        ("wall_south_L", "box", [L, WALL_T, WALL_H], (-hx + L / 2, -hx + WALL_T / 2, z), C["wall"], True),
        ("wall_south_R", "box", [R, WALL_T, WALL_H], (hx - R / 2, -hx + WALL_T / 2, z), C["wall"], True),
    ])
    b.group("door", (DOOR_X * b.L, -hx))


def build_furniture(b):
    # 兒童沙發：寬 1.0、深 0.45、座高 0.26；正面（local +x）朝南
    b.static_object("sofa", POS["sofa"], [
        ("sofa_seat", "box", [0.45, 1.0, 0.26], (0, 0, 0.13), C["sofa"], True),
        ("sofa_back", "box", [0.12, 1.0, 0.20], (-0.165, 0, 0.36), C["sofa_dk"], True),
        ("sofa_arm_L", "box", [0.45, 0.10, 0.10], (0, 0.45, 0.31), C["sofa_dk"], True),
        ("sofa_arm_R", "box", [0.45, 0.10, 0.10], (0, -0.45, 0.31), C["sofa_dk"], True),
    ], yaw=FACING["sofa"])

    # 玩具箱：內寬 0.45、牆高 0.06
    i, t, h = 0.45 * TOY, 0.015, 0.06 * TOY
    o = i + 2 * t
    b.static_object("toy_box", POS["toy_box"], [
        ("toy_box_base", "box", [o, o, t], (0, 0, t / 2), C["box"], True),
        ("toy_box_n", "box", [o, t, h], (0, (i + t) / 2, t + h / 2), C["box"], True),
        ("toy_box_s", "box", [o, t, h], (0, -(i + t) / 2, t + h / 2), C["box"], True),
        ("toy_box_e", "box", [t, i, h], ((i + t) / 2, 0, t + h / 2), C["box"], True),
        ("toy_box_w", "box", [t, i, h], (-(i + t) / 2, 0, t + h / 2), C["box"], True),
    ])

    # 矮玩具櫃：寬 0.7、深 0.25、高 0.35；開口（local +x）朝西
    w, d, hh, tt = 0.7, 0.25, 0.35, 0.02
    b.static_object("toy_shelf", POS["toy_shelf"], [
        ("toy_shelf_back", "box", [tt, w, hh], (-d / 2 + tt / 2, 0, hh / 2), C["shelf"], True),
        ("toy_shelf_sideN", "box", [d, tt, hh], (0, w / 2 - tt / 2, hh / 2), C["shelf"], True),
        ("toy_shelf_sideS", "box", [d, tt, hh], (0, -w / 2 + tt / 2, hh / 2), C["shelf"], True),
        ("toy_shelf_bottom", "box", [d, w, tt], (0, 0, tt / 2), C["shelf"], True),
        ("toy_shelf_mid", "box", [d, w, tt], (0, 0, hh / 2), C["shelf"], True),
        ("toy_shelf_top", "box", [d, w, tt], (0, 0, hh - tt / 2), C["shelf"], True),
    ], yaw=FACING["toy_shelf"])

    # 遊戲地墊 0.6 x 0.6（不參與碰撞）
    b.static_object("play_mat", POS["play_mat"], [
        ("play_mat_rim", "box", [0.94 * TOY, 0.94 * TOY, 0.002], (0, 0, 0.001), C["mat_rim"], False),
        ("play_mat_surface", "box", [0.90 * TOY, 0.90 * TOY, 0.002], (0, 0, 0.002), C["mat"], False),
    ])

    # 人的位置：地上直徑 0.4 的圓形標記，「給我」「過來」以 /Playroom/person 為準
    b.static_object("person", POS["person"], [
        ("person_spot", "cyl", [0.60 * TOY, 0.60 * TOY, 0.003], (0, 0, 0.0015), C["person"], False),
    ])


def build_toys(b, pos):
    """所有玩具都做成約 15 cm 高，讓機器人的相機看得到；可夾的玩具水平截面維持 6 到 7 cm，
    在 EP 夾爪約 10 cm 的開口範圍內。球是唯一例外，球不可能又高又窄，所以做成 9 cm。"""
    robot = ROBOT_START     # pos 與 ROBOT_START 同一座標系，方向不受縮放影響
    face = lambda k: yaw_toward(pos[k], robot)

    # 積木：6 x 6 x 15 cm 的長柱體
    for name, color in (("red_block", C["red"]), ("blue_block_1", C["blue"]),
                        ("blue_block_2", C["blue"]), ("green_block", C["green"]),
                        ("yellow_block", C["yellow"])):
        b.dynamic_object(name, pos[name], [("box", [0.06, 0.06, 0.15], (0, 0, 0.076), color, 0)], 0.06)

    # 球：直徑 9 cm
    b.dynamic_object("ball_orange", pos["ball_orange"],
                     [("sph", [0.09] * 3, (0, 0, 0.046), C["orange"], 0)], 0.04)

    # 玩具卡車：15 長 x 7 寬 x 15 高，車頭（local +x）朝機器人
    wheels = [("cyl", [0.04, 0.04, 0.012], (dx, dy, 0.02), C["tire"], math.pi / 2)
              for dx in (0.05, -0.05) for dy in (0.038, -0.038)]
    b.dynamic_object("toy_car", pos["toy_car"], [
        ("box", [0.15, 0.07, 0.06], (0, 0, 0.05), C["car"], 0),
        ("box", [0.07, 0.062, 0.07], (-0.03, 0, 0.115), C["glass"], 0),
    ] + wheels, 0.08, yaw=face("toy_car"))

    # 小鴨：約 15 cm 高、10 cm 長，頭和嘴朝機器人
    b.dynamic_object("rubber_duck", pos["rubber_duck"], [
        ("sph", [0.10, 0.075, 0.08], (0, 0, 0.041), C["yellow"], 0),
        ("sph", [0.06, 0.06, 0.06], (0.03, 0, 0.12), C["yellow"], 0),
        ("box", [0.03, 0.02, 0.012], (0.065, 0, 0.115), C["orange"], 0),
    ], 0.04, yaw=face("rubber_duck"))

    # 泰迪熊：約 34 cm 高、20 cm 寬（夾不起來，只能推），臉朝房間中央
    b.dynamic_object("teddy_bear", pos["teddy_bear"], [
        ("sph", [0.18, 0.20, 0.18], (0, 0, 0.091), C["bear"], 0),
        ("sph", [0.14, 0.14, 0.14], (0, 0, 0.25), C["bear"], 0),
        ("sph", [0.05, 0.05, 0.05], (0, 0.055, 0.31), C["bear"], 0),
        ("sph", [0.05, 0.05, 0.05], (0, -0.055, 0.31), C["bear"], 0),
        ("sph", [0.05, 0.055, 0.045], (0.065, 0, 0.24), C["bear_lt"], 0),
        ("sph", [0.016] * 3, (0.06, 0.028, 0.27), C["black"], 0),
        ("sph", [0.016] * 3, (0.06, -0.028, 0.27), C["black"], 0),
    ], 0.40, yaw=FACING["teddy_bear"])


# ------------------------------------------------------------------ 機器人

def world_aabb(sim, root):
    """機器人整棵樹所有 shape 的世界座標 AABB。"""
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


def find_gripper(sim, root):
    for h in sim.getObjectsInTree(root, sim.handle_all, 0):
        if "gripper" in sim.getObjectAlias(h).lower():
            return h
    return -1


def prepare_robot(sim, path, fixed_scale, extra_yaw=0.0):
    """把機器人轉成面向 +y，量測寬度換算比例，再放到起點並貼齊地面。回傳 scale。
    extra_yaw：自動判斷車頭方向錯了時的補正角（弧度，逆時針為正）。"""
    h = sim.getObject(path, {"noError": True})
    if h == -1:
        print(f"找不到 {path}，比例使用 1.0，機器人位置不變（可用 --robot 指定路徑）")
        return fixed_scale or 1.0

    # 1. 車頭方向：優先用相機（vision sensor）的視線軸，沒有相機才退回夾爪位置
    rp = sim.getObjectPosition(h, sim.handle_world)
    cam = -1
    for o in sim.getObjectsInTree(h, sim.object_visionsensor_type, 0):
        cam = o
        break
    if cam != -1:
        cm = sim.getObjectMatrix(cam, sim.handle_world)
        heading = math.atan2(cm[6], cm[2])          # 相機 local +z（視線方向）在世界 XY 的方向
        print(f"以相機 {sim.getObjectAlias(cam)} 的視線方向判斷車頭")
    else:
        g = find_gripper(sim, h)
        if g != -1:
            gp = sim.getObjectPosition(g, sim.handle_world)
            heading = math.atan2(gp[1] - rp[1], gp[0] - rp[0])
            print("找不到相機，以夾爪位置判斷車頭")
        else:
            m = sim.getObjectMatrix(h, sim.handle_world)
            heading = math.atan2(m[4], m[0])
            print("找不到相機與夾爪，假設車頭朝 local +x")
    m = sim.getObjectMatrix(h, sim.handle_world)
    sim.setObjectMatrix(h, sim.rotateAroundAxis(m, [0, 0, 1], rp, ROBOT_HEADING - heading + extra_yaw),
                        sim.handle_world)
    if cam != -1:
        cm = sim.getObjectMatrix(cam, sim.handle_world)
        print(f"轉向後相機朝向 {math.degrees(math.atan2(cm[6], cm[2])):.0f} 度（+y 應為 90）")

    # 2. 量測（此時車頭朝 +y，所以 x 方向的寬度就是車寬）
    lo, hi = world_aabb(sim, h)
    width, length = hi[0] - lo[0], hi[1] - lo[1]
    scale = fixed_scale or (width / EP_WIDTH if width > 0 else 1.0)
    print(f"機器人寬 {width:.3f} m、長 {length:.3f} m、高 {hi[2] - lo[2]:.3f} m，比例 = {scale:.3f}")
    if not fixed_scale and not 0.5 < scale < 2.0:
        print("  比例偏離 1 很多，代表模型單位可能和實車不同；環境會照這個比例縮放。")

    # 3. 移到起點（以 AABB 中心對齊）並讓最低點貼地
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    p = sim.getObjectPosition(h, sim.handle_world)
    L = ROOM / BASE_ROOM
    sim.setObjectPosition(h, [p[0] + ROBOT_START[0] * L * scale - cx, p[1] + ROBOT_START[1] * L * scale - cy,
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
    """列出場景最上層、不屬於本腳本也不是機器人或地板的物件，方便使用者清掉舊方塊。"""
    names = []
    for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 1):
        a = sim.getObjectAlias(h)
        if a not in keep:
            names.append("/" + a)
    if names:
        print("場景最上層還有這些 shape，若是舊的方塊請用 --remove-objects 刪除：", " ".join(names))


def main():
    global ROOM, TOY
    ap = argparse.ArgumentParser(description="以 RoboMaster EP 尺寸為基準建立遊戲房")
    ap.add_argument("--variant", choices=list(VARIANTS), default="default")
    ap.add_argument("--robot", default="/RoboMaster")
    ap.add_argument("--scale", type=float, default=None, help="強制比例，不量測機器人")
    ap.add_argument("--room", type=float, default=ROOM, help="房間邊長 (m)，預設 4")
    ap.add_argument("--toy-scale", type=float, default=TOY,
                    help="玩具額外倍率，預設 1.0（約 15 cm 高）。EP 夾爪最大開口約 10 cm，別超過 1.3")
    ap.add_argument("--extra-yaw", type=float, default=0.0,
                    help="車頭方向補正角（度，逆時針為正）。自動判斷後車頭朝東就用 90，朝西用 -90，朝南用 180")
    ap.add_argument("--keep-others", action="store_true", help="不要刪除 /Office 與 /TaskArena")
    ap.add_argument("--remove-objects", nargs="*", default=[])
    ap.add_argument("--remove", action="store_true", help="只刪除 /Playroom")
    args = ap.parse_args()

    ROOM, TOY = args.room, args.toy_scale

    sim = RemoteAPIClient().require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sys.exit("請先停止模擬再執行此腳本。")

    remove_tree(sim, "/Playroom")
    if args.remove:
        return
    if not args.keep_others:
        remove_tree(sim, "/Office")
        remove_tree(sim, "/TaskArena")
    for p in args.remove_objects:
        remove_tree(sim, p)

    scale = prepare_robot(sim, args.robot, args.scale, math.radians(args.extra_yaw))

    root = sim.createDummy(0.05)
    sim.setObjectAlias(root, "Playroom")
    sim.setObjectPosition(root, [0, 0, 0], sim.handle_world)
    b = Builder(sim, root, scale)

    pos = dict(POS)
    pos.update(VARIANTS[args.variant])
    build_room(b)
    build_furniture(b)
    build_toys(b, pos)

    list_strays(sim, {"Floor", args.robot.strip("/"), "Playroom"})
    n = len(sim.getObjectsInTree(root, sim.handle_all, 0))
    print(f"遊戲房建立完成（variant = {args.variant}，scale = {scale:.3f}），共 {n} 個物件。記得存檔。")


if __name__ == "__main__":
    main()
