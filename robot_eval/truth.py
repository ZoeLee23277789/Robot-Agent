"""模擬器真值快照：任務前後各抓一次零散物件的位置、地標的地面範圍／頂面高度、機器人姿態，
給 judge 的 truth 型里程碑用（物件最後在不在桶子裡、有沒有被動過），不用再只靠畫面猜。

為什麼需要（2026-10-20）：agent 兩次把綠色小積木當成萊姆綠球丟進藍桶後回報成功，只看畫面的 judge 也可能被騙；
模擬器裡每個物件的真實座標是最便宜、最不會錯的證據。只有 sim 模式有，real/mock 回 None，里程碑就交回給 LLM judge。
"""
import math
import time
from typing import Optional

LOOSE_PREFIXES = ("small_", "ball_", "foam_", "trash_")
LANDMARKS = ["table", "bin_balls", "bin_blocks", "bin_trash", "bin_recycle", "mat_red", "mat_green", "mat_blue", "bench", "platform"]
_BB = ("objfloatparam_objbbox_min_x", "objfloatparam_objbbox_min_y", "objfloatparam_objbbox_min_z",
       "objfloatparam_objbbox_max_x", "objfloatparam_objbbox_max_y", "objfloatparam_objbbox_max_z")


def _extent(sim, h):
    lo, hi = [1e9] * 3, [-1e9] * 3
    best_area, top = -1.0, None
    shapes = sim.getObjectsInTree(h, sim.object_shape_type, 0)
    if len(shapes) > 80:
        shapes = shapes[:60]   # 平台底下 200 多顆固定池球，底板跟牆先建、取前 60 個就涵蓋外框（跟 robot_server 同一套）
    for sh in shapes:
        size = sim.getShapeBB(sh)[0]   # 回傳 (尺寸, 姿態)；基本形狀的 bbox 以原點為中心：2 次 zmq 來回就夠，不用 15 次
        m = sim.getObjectMatrix(sh, sim.handle_world)
        slo, shi = [1e9] * 3, [-1e9] * 3
        for cx in (-size[0] / 2, size[0] / 2):
            for cy in (-size[1] / 2, size[1] / 2):
                for cz in (-size[2] / 2, size[2] / 2):
                    w = [m[0] * cx + m[1] * cy + m[2] * cz + m[3], m[4] * cx + m[5] * cy + m[6] * cz + m[7],
                         m[8] * cx + m[9] * cy + m[10] * cz + m[11]]
                    slo = [min(a, c) for a, c in zip(slo, w)]
                    shi = [max(a, c) for a, c in zip(shi, w)]
        lo = [min(a, c) for a, c in zip(lo, slo)]
        hi = [max(a, c) for a, c in zip(hi, shi)]
        area = (shi[0] - slo[0]) * (shi[1] - slo[1])
        if area > best_area:
            best_area, top = area, shi[2]     # 面積最大的形狀的頂面 = 桌面／座面（不是椅背或放在上面的東西）
    if not shapes:
        return None
    return {"bbox": [round(lo[0], 3), round(hi[0], 3), round(lo[1], 3), round(hi[1], 3)],
            "top_m": round(top, 3), "zmax": round(hi[2], 3)}


def snapshot() -> Optional[dict]:
    """回傳 {"objects": {alias: [x, y, z]}, "landmarks": {name: {bbox, top_m}}, "robot_xy": [x, y], "time": ...}；連不上模擬器回 None。"""
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        sim = RemoteAPIClient().require("sim")
    except Exception:
        return None
    try:
        objects = {}
        for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0):
            alias = sim.getObjectAlias(h)
            if alias.startswith(LOOSE_PREFIXES) and not sim.getObjectInt32Param(h, sim.shapeintparam_static):
                # 只算會動的零散物件：ball_pit_base 這種別名撞到前綴的靜態底板不算
                objects[alias] = [round(v, 3) for v in sim.getObjectPosition(h, sim.handle_world)]
        landmarks = {}
        for name in LANDMARKS:
            h = sim.getObject("/" + name, {"noError": True})
            if h != -1:
                ext = _extent(sim, h)
                if ext:
                    landmarks[name] = ext
        robot_xy = None
        roots = [o for o in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2) if sim.getObjectAlias(o) == "RoboMaster"]
        good = [r for r in roots if any(sim.getObjectAlias(o) == "GyroSensor" for o in sim.getObjectsInTree(r, sim.handle_all, 0))]
        if roots:
            p = sim.getObjectPosition((good or roots)[0], sim.handle_world)
            robot_xy = [round(p[0], 3), round(p[1], 3)]
        return {"objects": objects, "landmarks": landmarks, "robot_xy": robot_xy, "time": time.time()}
    except Exception:
        return None


def inside(pos, lm, margin=0.0) -> bool:
    bb = lm["bbox"]
    return bb[0] - margin <= pos[0] <= bb[1] + margin and bb[2] - margin <= pos[1] <= bb[3] + margin


def where(pos, landmarks: dict) -> str:
    """物件在哪：'inside bin_balls' / 'on top of table' / 'on mat_red' / 'on the floor'。"""
    for name, lm in landmarks.items():
        if not inside(pos, lm):
            continue
        if name.startswith("bin_"):
            if pos[2] < lm["zmax"] + 0.12:
                return f"inside {name}"
        elif name.startswith("mat_"):
            if pos[2] < 0.25:
                return f"on {name}"
        elif name == "platform":
            if pos[2] < lm["top_m"] - 0.10:
                return "on the floor underneath the platform"
            return "on top of the platform"
        else:  # table / bench 這類有桌面的
            if pos[2] >= lm["top_m"] - 0.02:
                return f"on top of {name}"
    return "on the floor"


def describe(final: Optional[dict], initial: Optional[dict]) -> str:
    """給 judge LLM 看的真值摘要（權威證據）。"""
    if not final:
        return ""
    lines = []
    for name, p in sorted(final["objects"].items()):
        moved = ""
        if initial and name in initial["objects"]:
            d = math.hypot(p[0] - initial["objects"][name][0], p[1] - initial["objects"][name][1])
            moved = f", moved {d:.2f} m during the run" if d >= 0.05 else ", not moved"
        lines.append(f"- {name}: at world ({p[0]:.2f}, {p[1]:.2f}, z={p[2]:.2f}), {where(p, final['landmarks'])}{moved}")
    if final.get("robot_xy"):
        lines.append(f"- robot: at world ({final['robot_xy'][0]:.2f}, {final['robot_xy'][1]:.2f})")
    return "\n".join(lines)
