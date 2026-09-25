"""
在 CoppeliaSim 目前開啟的場景中建立一間 5 m x 5 m 的小型辦公室。

使用方式（CoppeliaSim 開著、模擬為停止狀態）：
    pip install coppeliasim-zmqremoteapi-client
    python build_office.py
    python build_office.py --wall-height 1.5 --blocks /red_block /blue_block

腳本可重複執行：每次會先刪掉舊的 /Office 再重建，不會動到 RoboMaster 與 Floor。
建好後在 CoppeliaSim 裡 File > Save scene 存檔即可。

座標約定：房間中心在世界原點 (0, 0)，+y 為北牆，南牆有一個門口。
機器人所在的中央區域刻意留空，方便行駛。
"""

import argparse
import math
import sys

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOM_X = 5.0      # 東西向長度 (m)，配合預設 5x5 Floor
ROOM_Y = 5.0      # 南北向長度 (m)
WALL_T = 0.10     # 牆厚

COLOR = {
    "wall":   [0.90, 0.89, 0.86],
    "wood":   [0.62, 0.45, 0.30],
    "light_wood": [0.78, 0.64, 0.46],
    "dark":   [0.22, 0.22, 0.25],
    "metal":  [0.60, 0.62, 0.65],
    "fabric": [0.20, 0.30, 0.45],
    "green":  [0.25, 0.55, 0.25],
    "pot":    [0.55, 0.35, 0.25],
    "white":  [0.96, 0.96, 0.96],
    "frame":  [0.40, 0.40, 0.42],
}


class Builder:
    def __init__(self, sim, root):
        self.sim = sim
        self.root = root

    def group(self, name, x, y, yaw=0.0, parent=None):
        """建立一個 dummy 當作家具的原點，子物件都用相對座標擺放。"""
        sim = self.sim
        h = sim.createDummy(0.02)
        sim.setObjectAlias(h, name)
        sim.setObjectParent(h, parent if parent is not None else self.root, True)
        sim.setObjectPosition(h, [x, y, 0.0], sim.handle_world)
        sim.setObjectOrientation(h, [0.0, 0.0, yaw], sim.handle_world)
        return h

    def _finish(self, h, name, color, parent, pos, orient):
        sim = self.sim
        sim.setObjectAlias(h, name)
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
        sim.setObjectParent(h, parent, True)
        sim.setObjectPosition(h, pos, sim.handle_parent)
        sim.setObjectOrientation(h, orient or [0.0, 0.0, 0.0], sim.handle_parent)
        # 讓視覺感測器、距離感測器與碰撞檢查都看得到這些物件
        try:
            sp = (sim.objectspecialproperty_collidable
                  | sim.objectspecialproperty_measurable
                  | sim.objectspecialproperty_detectable
                  | sim.objectspecialproperty_renderable)
            sim.setObjectSpecialProperty(h, sp)
        except Exception:
            pass
        return h

    def _options(self, respondable):
        # bit3 = respondable, bit4 = static（不受重力、不會被機器人推走）
        return (8 if respondable else 0) | 16

    def box(self, name, size, pos, color, parent, orient=None, respondable=True):
        h = self.sim.createPrimitiveShape(
            self.sim.primitiveshape_cuboid, size, self._options(respondable))
        return self._finish(h, name, color, parent, pos, orient)

    def cylinder(self, name, diameter, height, pos, color, parent, respondable=True):
        h = self.sim.createPrimitiveShape(
            self.sim.primitiveshape_cylinder, [diameter, diameter, height],
            self._options(respondable))
        return self._finish(h, name, color, parent, pos, None)

    def sphere(self, name, diameter, pos, color, parent, respondable=True):
        h = self.sim.createPrimitiveShape(
            self.sim.primitiveshape_spheroid, [diameter, diameter, diameter],
            self._options(respondable))
        return self._finish(h, name, color, parent, pos, None)


# ---------------------------------------------------------------- 家具

def four_legs(b, g, prefix, w, d, leg_h, leg_t, color, inset=0.05):
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        b.box(f"{prefix}_leg{i}", [leg_t, leg_t, leg_h],
              [sx * (w / 2 - inset), sy * (d / 2 - inset), leg_h / 2], color, g)


def desk(b, name, x, y, yaw=0.0, w=1.4, d=0.7, h=0.75):
    g = b.group(name, x, y, yaw)
    top_t = 0.03
    b.box(f"{name}_top", [w, d, top_t], [0, 0, h - top_t / 2], COLOR["wood"], g)
    four_legs(b, g, name, w, d, h - top_t, 0.05, COLOR["metal"])
    # 桌子背板（靠牆那側），讓桌子看起來比較像辦公桌
    b.box(f"{name}_panel", [w - 0.1, 0.02, 0.4], [0, d / 2 - 0.06, h - top_t - 0.2],
          COLOR["metal"], g)
    return g


def chair(b, name, x, y, yaw=0.0):
    """椅子面向 local +y（坐的人面向桌子），椅背在 -y。"""
    g = b.group(name, x, y, yaw)
    seat_z, seat_t, s = 0.45, 0.06, 0.46
    b.box(f"{name}_seat", [s, s, seat_t], [0, 0, seat_z], COLOR["fabric"], g)
    b.box(f"{name}_back", [s, 0.05, 0.45],
          [0, -(s / 2 - 0.025), seat_z + seat_t / 2 + 0.225], COLOR["fabric"], g)
    four_legs(b, g, name, s, s, seat_z - seat_t / 2, 0.035, COLOR["dark"], inset=0.04)
    return g


def bookshelf(b, name, x, y, yaw=0.0, w=1.0, d=0.35, h=1.6, shelves=4):
    """書架開口朝 local +y。"""
    g = b.group(name, x, y, yaw)
    t = 0.02
    for sx in (1, -1):
        b.box(f"{name}_side{'R' if sx > 0 else 'L'}", [t, d, h],
              [sx * (w / 2 - t / 2), 0, h / 2], COLOR["light_wood"], g)
    b.box(f"{name}_back", [w, t, h], [0, -(d / 2 - t / 2), h / 2], COLOR["light_wood"], g)
    for i in range(shelves + 1):
        z = t / 2 + i * (h - t) / shelves
        b.box(f"{name}_shelf{i}", [w - 2 * t, d - t, t], [0, t / 2, z],
              COLOR["light_wood"], g)
    # 幾本書，讓相機畫面不會太空
    book_colors = [[0.7, 0.2, 0.2], [0.2, 0.4, 0.7], [0.8, 0.7, 0.2], [0.3, 0.6, 0.4]]
    for i in range(1, shelves):
        z0 = t + i * (h - t) / shelves
        for j in range(5):
            bh = 0.22 + 0.03 * ((i + j) % 3)
            b.box(f"{name}_book{i}_{j}", [0.05, 0.22, bh],
                  [-w / 2 + 0.1 + j * 0.07, 0.0, z0 + bh / 2],
                  book_colors[(i + j) % 4], g)
    return g


def cabinet(b, name, x, y, yaw=0.0, w=0.5, d=0.6, h=0.72, drawers=3):
    """檔案櫃，抽屜朝 local +y。"""
    g = b.group(name, x, y, yaw)
    b.box(f"{name}_body", [w, d, h], [0, 0, h / 2], COLOR["frame"], g)
    for i in range(drawers):
        z = (i + 0.5) * h / drawers
        b.box(f"{name}_handle{i}", [0.14, 0.02, 0.02], [0, d / 2 + 0.01, z],
              COLOR["metal"], g)
    return g


def low_table(b, name, x, y, yaw=0.0, w=0.8, d=0.5, h=0.18):
    """矮桌，高度可調，給機器人夾取任務用。"""
    g = b.group(name, x, y, yaw)
    t = 0.03
    b.box(f"{name}_top", [w, d, t], [0, 0, h - t / 2], COLOR["wood"], g)
    four_legs(b, g, name, w, d, h - t, 0.04, COLOR["dark"])
    return g


def trash_bin(b, name, x, y):
    g = b.group(name, x, y)
    b.cylinder(f"{name}_body", 0.28, 0.35, [0, 0, 0.175], COLOR["dark"], g)
    return g


def plant(b, name, x, y):
    g = b.group(name, x, y)
    b.cylinder(f"{name}_pot", 0.30, 0.30, [0, 0, 0.15], COLOR["pot"], g)
    b.cylinder(f"{name}_stem", 0.04, 0.25, [0, 0, 0.42], COLOR["wood"], g)
    b.sphere(f"{name}_leaves", 0.50, [0, 0, 0.75], COLOR["green"], g)
    return g


def whiteboard(b, name, x, y, yaw, wall_h):
    """掛在牆上的白板，板面朝 local +y。"""
    g = b.group(name, x, y, yaw)
    bw, bh = 1.2, min(0.7, wall_h - 0.5)
    zc = min(1.0, wall_h - bh / 2 - 0.05)
    b.box(f"{name}_frame", [bw + 0.04, 0.02, bh + 0.04], [0, 0, zc], COLOR["frame"], g)
    b.box(f"{name}_board", [bw, 0.01, bh], [0, 0.012, zc], COLOR["white"], g)
    return g


def walls(b, wall_h, door_x, door_w):
    g = b.group("Walls", 0, 0)
    hx, hy, t = ROOM_X / 2, ROOM_Y / 2, WALL_T
    z = wall_h / 2
    c = COLOR["wall"]
    b.box("wall_north", [ROOM_X, t, wall_h], [0, hy - t / 2, z], c, g)
    b.box("wall_east", [t, ROOM_Y - 2 * t, wall_h], [hx - t / 2, 0, z], c, g)
    b.box("wall_west", [t, ROOM_Y - 2 * t, wall_h], [-hx + t / 2, 0, z], c, g)
    # 南牆分兩段，中間留門口
    left_end, right_start = door_x - door_w / 2, door_x + door_w / 2
    seg_l = left_end - (-hx)
    seg_r = hx - right_start
    if seg_l > 0.01:
        b.box("wall_south_L", [seg_l, t, wall_h], [-hx + seg_l / 2, -hy + t / 2, z], c, g)
    if seg_r > 0.01:
        b.box("wall_south_R", [seg_r, t, wall_h], [hx - seg_r / 2, -hy + t / 2, z], c, g)
    return g


# ---------------------------------------------------------------- 主程式

def place_blocks(sim, paths, table_xy, table_h):
    """把既有的方塊（例如紅、藍方塊）擺到矮桌上。"""
    n = len(paths)
    for i, p in enumerate(paths):
        h = sim.getObject(p, {"noError": True})
        if h == -1:
            print(f"  找不到 {p}，略過")
            continue
        size = sim.getShapeBB(h)
        dx = (i - (n - 1) / 2) * 0.25
        sim.setObjectPosition(h, [table_xy[0] + dx, table_xy[1], table_h + size[2] / 2 + 0.002],
                              sim.handle_world)
        sim.setObjectOrientation(h, [0, 0, 0], sim.handle_world)
        try:
            sim.resetDynamicObject(h)
        except Exception:
            pass
        print(f"  {p} 已放到矮桌上")


def main():
    ap = argparse.ArgumentParser(description="在 CoppeliaSim 場景中建立小型辦公室")
    ap.add_argument("--wall-height", type=float, default=1.2,
                    help="牆高 (m)。真實辦公室約 2.5，模擬時矮一點比較好從上方觀察")
    ap.add_argument("--door-x", type=float, default=1.0, help="南牆門口中心的 x 座標")
    ap.add_argument("--door-w", type=float, default=1.0, help="門口寬度")
    ap.add_argument("--table-h", type=float, default=0.18,
                    help="任務矮桌高度 (m)，請依 RoboMaster 手臂可達高度調整")
    ap.add_argument("--blocks", nargs="*", default=[],
                    help="要搬到矮桌上的既有物件路徑，例如 /red_block /blue_block")
    ap.add_argument("--remove", action="store_true", help="只刪除現有的 /Office，不重建")
    args = ap.parse_args()

    client = RemoteAPIClient()
    sim = client.require("sim")

    if sim.getSimulationState() != sim.simulation_stopped:
        sys.exit("請先停止模擬再執行此腳本。")

    old = sim.getObject("/Office", {"noError": True})
    if old != -1:
        objs = sim.getObjectsInTree(old, sim.handle_all, 0)
        sim.removeObjects(objs)
        print(f"已刪除舊的 /Office（{len(objs)} 個物件）")
    if args.remove:
        return

    root = sim.createDummy(0.05)
    sim.setObjectAlias(root, "Office")
    sim.setObjectPosition(root, [0, 0, 0], sim.handle_world)
    b = Builder(sim, root)

    hx, hy, t = ROOM_X / 2, ROOM_Y / 2, WALL_T

    walls(b, args.wall_height, args.door_x, args.door_w)

    # 北牆：兩張辦公桌 + 椅子 + 檔案櫃 + 垃圾桶
    desk_y = hy - t - 0.35 - 0.05
    desk(b, "desk_1", -1.1, desk_y)
    chair(b, "chair_1", -1.1, desk_y - 0.6)
    desk(b, "desk_2", 0.6, desk_y)
    chair(b, "chair_2", 0.6, desk_y - 0.6)
    cabinet(b, "cabinet", hx - t - 0.3, hy - t - 0.35, yaw=math.pi)
    trash_bin(b, "trash_bin", -hx + t + 0.25, hy - t - 0.25)

    # 西牆：書架，開口朝東
    bookshelf(b, "bookshelf", -hx + t + 0.175 + 0.01, 0.3, yaw=-math.pi / 2)

    # 東牆：白板，板面朝西
    whiteboard(b, "whiteboard", hx - t - 0.012, 0.0, math.pi / 2, args.wall_height)

    # 西南角：任務區矮桌；東南角：盆栽
    table_xy = (-1.2, -1.3)
    low_table(b, "task_table", table_xy[0], table_xy[1], h=args.table_h)
    plant(b, "plant", hx - t - 0.3, -hy + t + 0.3)

    if args.blocks:
        place_blocks(sim, args.blocks, table_xy, args.table_h)

    n = len(sim.getObjectsInTree(root, sim.handle_all, 0))
    print(f"辦公室建立完成，共 {n} 個物件。記得 File > Save scene 存檔。")


if __name__ == "__main__":
    main()
