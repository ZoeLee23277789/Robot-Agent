"""
高階技能（agent 端）：go_to / approach / drive_through。

LLM 只給「目標叫什麼」，這裡負責：
    1. 把名稱解析成世界座標：'x, y' 直接用；固定地標查 /state 的 static_landmarks（伺服器給的真實座標）；
       其他（任務目標物）用俯視相機的 pointing（perception.locate_overhead）。
    2. 交給 robot_server.py 的 navigate_to / drive_through 執行：占據格地圖 + A* + 閉迴路，一次動作到位。
    3. approach 抵達後再用前鏡頭精對準，回報方位與前方距離，讓 LLM 拿到的是證據而不是「ok」。

service.py 的動作處理器直接呼叫這裡的實作。
"""

import asyncio
import time
import json
import os
import math
import re
from typing import Optional, Tuple

from robot_agent import perception

NAV_TIMEOUT_S = 320.0   # navigate_to 伺服器端預算 240 s，再留餘裕給 HTTP

_COORD_RE = re.compile(r"^\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*$")
# 描述裡夾帶的座標（「blue block at (0.90, -1.10)」）：scan_overhead 給的是兩位小數，LLM 會照抄；兩個數都要有小數點才算
_COORD_ANY_RE = re.compile(r"\(?\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)?")
_ALIASES = {
    "ballpit": "ball_pit", "pit": "ball_pit", "ball_pool": "ball_pit",
    "ball_bin": "bin_balls", "balls_bin": "bin_balls", "blue_bin": "bin_balls", "blue_storage_box": "bin_balls",
    "block_bin": "bin_blocks", "blocks_bin": "bin_blocks", "orange_bin": "bin_blocks", "orange_storage_bin": "bin_blocks",
    "red_pillar": "pillar_red", "blue_pillar": "pillar_blue", "yellow_pillar": "pillar_yellow", "green_pillar": "pillar_green",
    "stair": "stairs", "steps": "stairs", "orange_tunnel": "tunnel", "elevated_platform": "platform",
    "trash_bin": "bin_trash", "black_bin": "bin_trash", "garbage_bin": "bin_trash", "rubbish_bin": "bin_trash", "trash_can": "bin_trash",
    "black_trash_bin": "bin_trash", "waste_bin": "bin_trash",
    "recycling_bin": "bin_recycle", "recycle_bin": "bin_recycle", "white_bin": "bin_recycle", "white_recycling_bin": "bin_recycle",
}
_FRONT_RE = re.compile(r"\b(in front of (me|you|the robot)|right in front|straight ahead|ahead of (me|you)|right here)\b")


def pseudo_yaw(state: dict) -> Optional[float]:
    """給 perception 的「機器人朝向」：世界朝向（逆時針為正）。2026-10-22 起 turn_left 為正＝真正的左轉，
    perception 的公式是 turn_left = bearing - heading。沒有世界朝向（實體機）時用 -yaw_deg（SDK yaw 順時針為正）。"""
    hd = state.get("world_heading_deg")
    if hd is not None:
        return hd
    yaw = state.get("yaw_deg")
    return -yaw if yaw is not None else None


def _norm(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    key = re.sub(r"^(the|a|an)_", "", key)
    return _ALIASES.get(key, key)


def _match_landmark(target: str, landmarks: dict) -> Optional[str]:
    key = _norm(target)
    if key in landmarks:
        return key
    words = set(re.findall(r"[a-z]+", target.lower()))
    for name in landmarks:
        parts = set(name.split("_"))
        if parts <= words:              # 'red pillar' -> pillar_red, 'the bench' -> bench
            return name
    return None


_OBJECT_WORDS = {"ball", "block", "cube", "brick", "cylinder", "beam", "toy", "box",
                 "bottle", "can", "cup", "carton", "litter", "trash", "garbage", "rubbish", "waste"}
# 有「桌面」的地標：東西放在上面時 pick/place 要先把手臂升降柱升到桌面高度（Env/add_arm_lift.py）。
# 值 = 桌面高度 (m)，None 代表用伺服器量到的頂面 top_m（static_landmarks 裡的欄位）。
SURFACE_LANDMARKS = {"table": None}
LIFT_MAX_MM = 400          # 升降柱行程，跟 robot_server.py 的 LIMITS["lift_mm"] 一致


_TASK_CONTEXT = {"text": ""}
_STOP_WORDS = {"the", "a", "an", "up", "it", "from", "on", "to", "and", "into", "in", "of", "off", "top", "then"}
_PLACING_VERBS = re.compile(r"\b(put|place|drop|set|move|bring|leave)\b")


def set_task_context(text: str) -> None:
    """RobotAgent 任務開始時呼叫。LLM 常把「lime green ball from the table」縮成「lime green ball」丟給 pick，
    「在桌上」這個線索就不見了，所以解析目標時要能回頭看整句任務。"""
    _TASK_CONTEXT["text"] = text or ""


def _surface_hint(target: str, landmarks: dict) -> Optional[str]:
    """任務／目標字串有沒有說這個物件在哪個桌面上。目標字串本身寫「ball on the table」直接算；
    否則看整句任務：「... lime green ball from the table ...」而且目標的字就出現在那個片語前面
    （「put it on the table」這種是放置的目的地，片語前面有放置動詞就不算）。"""
    words = set(re.findall(r"[a-z]+", target.lower()))
    content = words - _STOP_WORDS
    for name in SURFACE_LANDMARKS:
        if name not in landmarks:
            continue
        if set(name.split("_")) <= words and (words & _OBJECT_WORDS):
            return name
        task = _TASK_CONTEXT["text"].lower()
        phrase = re.escape(name.replace("_", " "))
        for m in re.finditer(r"\b(from|off|on top of|on)\s+(the\s+)?" + phrase + r"\b", task):
            window = task[max(0, m.start() - 60):m.start()]
            if _PLACING_VERBS.search(window):
                continue
            if content & set(re.findall(r"[a-z]+", window)):
                return name
    return None


def _landmark_containing(xy, landmarks: dict) -> Optional[str]:
    """xy 落在哪個有 bbox 的地標範圍內（收納箱、地墊、桌子…）；沒有就 None。"""
    for name, lm in landmarks.items():
        bb = lm.get("bbox") if isinstance(lm, dict) else None
        if bb and bb[0] - 0.02 <= xy[0] <= bb[1] + 0.02 and bb[2] - 0.02 <= xy[1] <= bb[3] + 0.02:
            return name
    return None


def _surface_of_landmark(name: str, landmarks: dict) -> Optional[dict]:
    """地標 name 是桌子類的話回傳 {"name", "top_m", "half_m"}（桌面高度、半寬），否則 None。"""
    lm = landmarks.get(name) if name in SURFACE_LANDMARKS else None
    if not lm:
        return None
    top = SURFACE_LANDMARKS[name] if SURFACE_LANDMARKS[name] is not None else lm.get("top_m")
    if top is None or top < 0.10:
        return None
    bbox = lm.get("bbox")
    half = min(bbox[1] - bbox[0], bbox[3] - bbox[2]) / 2 if bbox else 0.20
    return {"name": name, "top_m": float(top), "half_m": float(half)}


def _approach_from_deg(bbox, robot_xy, target_xy) -> Optional[float]:
    """偏好從桌子哪一側接近：挑外法線方向最接近「目標→機器人」方向的那一側桌緣，回傳世界角度（度，目標→站位）。
    伺服器會把四個邊的正對站位都評估一遍、取離桌緣最近可站的，這裡給的只是偏好（同樣近時用它）。
    斜著接近時車頭的角會先頂到桌緣，車頭中心停不到離桌緣 2 cm 的位置（夾爪中心只在車頭前約 0.15 m），
    垂直桌緣接近才送得進指間；第一版沒指定方向，導航從西南斜著站，感測器讀 213 mm 就頂住了。"""
    if not bbox or not robot_xy:
        return None
    vx, vy = robot_xy[0] - target_xy[0], robot_xy[1] - target_xy[1]
    n = math.hypot(vx, vy) or 1.0
    best = max(((vx * nx + vy * ny) / n, ang) for nx, ny, ang in ((1, 0, 0.0), (-1, 0, 180.0), (0, 1, 90.0), (0, -1, -90.0)))
    return best[1]


async def _close_in(robot, xy, want_m: float = 0.45, max_step_m: float = 0.6) -> str:
    """導航回報「到了」但離目標還太遠（擁擠處站位會被推到 1 m 外，而 ToF 只看 0.6 m）：面向目標點再往前補一段。
    turn_left 為正會讓世界朝向減少（伺服器 TURN_SIGN），所以轉向量取負。回傳說明字串。"""
    st = await robot.state()
    rxy, hd = st.get("world_xy"), st.get("world_heading_deg")
    if not rxy or hd is None:
        return ""
    d = math.hypot(xy[0] - rxy[0], xy[1] - rxy[1])
    if d <= want_m + 0.15:
        return ""
    bearing = math.degrees(math.atan2(xy[1] - rxy[1], xy[0] - rxy[0]))
    delta = (bearing - hd + 180) % 360 - 180
    if abs(delta) >= 3:
        await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(delta, 1)})
    step = round(min(max_step_m, d - want_m), 2)
    r = await robot.act("move_chassis", {"forward_m": step, "right_m": 0, "turn_left_deg": 0})   # 不 push：前方有東西會自動停
    return f" [closed in {step:.2f} m from {d:.2f} m away: {str(r.get('message', ''))[:60]}]"


async def _tof_center(robot) -> str:
    """非球物件的左右對準：用前方距離感測器掃一遍（相對目前朝向 -8°…+8°，每 2° 一格），記下看得到物件的角度，
    轉到那些角度的中點。球用相機顏色對準；瓶子/積木沒有可靠的顏色詞，改用感測器本身——光錐在 0.2 m 處半寬約 2.8 cm，
    6-7 cm 的物件會在連續幾格裡出現，中點就是物件中心。偏 2 cm 以上一邊手指會先碰到、把輕的瓶子推走（2026-10-21 實測）。"""
    ref = await _tof(robot)
    if ref is None or ref > 400:
        return ""
    # 掃 ±16°、每 4° 一格：7 cm 的物件在 0.2 m 處要轉到 ±17° 才會離開光錐，±8° 的話每一格都看得到、找不到邊、
    # 中點永遠是 0（2026-10-21 瓶子三次都沒對準、被指尖推倒）
    seen = []
    span, step = 16, 4
    await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": -span})
    for off in range(-span, span + 1, step):
        if off != -span:
            await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": step})
        await asyncio.sleep(0.25)
        t = await _tof(robot)
        if t is not None and t <= ref + 80:
            seen.append(off)
    if not seen:
        await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": -span})
        return " (tof sweep saw nothing; heading unchanged)"
    center = (min(seen) + max(seen)) / 2.0
    back = center - span   # 現在在 +span
    if abs(back) >= 2:
        await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(back, 1)})
    elif abs(back) > 0:
        pass
    await asyncio.sleep(0.25)
    return f" (tof sweep: object seen at {min(seen)}..{max(seen)}°, centred at {center:+.0f}°)"


async def _nav_params(robot, xy, stop_m: float, surface: Optional[dict], label: Optional[str] = None) -> dict:
    """navigate_to 的參數：目標在桌上／是桌子時，指定垂直桌緣的接近方向。label 只給 /live 儀表板畫框用。"""
    params = {"x": xy[0], "y": xy[1], "stop_m": stop_m, "face": True}
    if label:
        params["label"] = str(label)[:40]
    if surface:
        st = await robot.state()
        lm = (st.get("static_landmarks") or {}).get(surface["name"]) or {}
        a = _approach_from_deg(lm.get("bbox"), st.get("world_xy"), xy)
        if a is not None:
            params["approach_from_deg"] = a
    return params


def _surface_under(xy, landmarks: dict) -> Optional[dict]:
    """世界座標 xy 落在哪個桌子類地標的地面範圍內（= 物件放在它上面）；沒有就 None。"""
    for name in SURFACE_LANDMARKS:
        lm = landmarks.get(name)
        if not lm or not lm.get("bbox"):
            continue
        x0, x1, y0, y1 = lm["bbox"]
        if x0 - 0.03 <= xy[0] <= x1 + 0.03 and y0 - 0.03 <= xy[1] <= y1 + 0.03:
            return _surface_of_landmark(name, landmarks)
    return None


async def _locate_with_front_camera(robot, llm, target: str):
    """俯視圖找不到時的備援（2026-11-13，掃描場地跑實體：使用者臨時放的箱子模擬場景裡沒有，但 /frame.jpg 是實體前鏡頭）：
    視覺模型在前鏡頭畫面指出方位 → 沒看到就原地每 60° 轉一圈找 → 轉到正對 → 前向距離感測器量距離（量不到就一段段靠近，
    伺服器的防撞縮步擋著不會撞上）→ 算成世界座標。箱子每回合會換位置（使用者），所以這裡要能從任何地方找起。回 (xy 或 None, 說明)。"""
    label = _plain_label(target)

    async def fresh():
        await asyncio.sleep(0.8)
        await robot.frame()          # H.264 串流延遲以張數計，丟掉第一張
        return await robot.frame()

    async def turn(deg):
        r = await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(deg, 1)})
        return bool(r.get("ok"))

    found = await perception.locate(llm, await fresh(), label)
    turned = 0.0
    while not found.found and turned < 300:        # 相機視角約 100°，60° 一步、最多 5 步看完一圈
        if not await turn(60):
            break
        turned += 60
        found = await perception.locate(llm, await fresh(), label)
    if not found.found:
        return None, (f"'{target}' is not visible in the overhead view, is not a known landmark, and the front camera did not see it "
                      f"anywhere around this spot (scanned a full turn). Drive to another part of the field (e.g. navigate_to the middle "
                      f"of the corridor or the far end) and try again")
    for _ in range(2):
        if abs(found.bearing_right_deg) <= 4:
            break
        if not await turn(found.turn_left_deg):
            break
        found = await perception.locate(llm, await fresh(), label)
        if not found.found:
            return None, f"the front camera saw '{label}' but lost it after turning to face it; look again"
    approached = 0.0
    for _ in range(5):
        state = await robot.state()
        tof = state.get("tof_front_mm")
        if tof is not None and tof != 9999 and 50 <= tof <= 1500:
            break
        # 距離感測器還量不到（太遠）：朝它開 0.5 m 再看，伺服器的防撞縮步會在它前面 0.15 m 停住
        r = await robot.act("move_chassis", {"forward_m": 0.5, "right_m": 0, "turn_left_deg": 0})
        if not r.get("ok"):
            break
        approached += 0.5
        found = await perception.locate(llm, await fresh(), label)
        if found.found and abs(found.bearing_right_deg) > 4:
            await turn(found.turn_left_deg)
            found = await perception.locate(llm, await fresh(), label)
    state = await robot.state()
    xy0, hd, tof = state.get("world_xy"), pseudo_yaw(state), state.get("tof_front_mm")
    if xy0 is None or hd is None:
        return None, "the front camera sees it but the robot has no world pose to convert that into coordinates"
    if tof is None or tof == 9999 or tof < 50:
        return None, (f"the front camera sees '{label}' {found.bearing_right_deg:+.0f} deg from centre (robot now facing it), but the front "
                      f"distance sensor still reads nothing after driving {approached:.1f} m closer — it may be too low for the sensor. "
                      f"Drive forward with move_chassis in 0.3 m steps while watching the camera")
    d = tof / 1000.0 + 0.16    # ToF 在車頭前緣
    ang = math.radians(hd - found.bearing_right_deg)
    xy = (round(xy0[0] + d * math.cos(ang), 2), round(xy0[1] + d * math.sin(ang), 2))
    how = (f"scanned {turned:.0f} deg, " if turned else "") + (f"drove {approached:.1f} m closer, " if approached else "")
    return xy, (f" (not in the overhead view; {how}located with the front camera {found.bearing_right_deg:+.0f} deg from centre and the "
                f"front distance sensor {tof} mm -> world ({xy[0]:.2f}, {xy[1]:.2f}); the map marks it as an obstacle)")


async def resolve_target(robot, llm, target: str) -> Tuple[Optional[Tuple[float, float]], str, str, Optional[dict]]:
    """回傳 (world_xy, kind, note, surface)。kind = 'coords' | 'landmark' | 'object'；找不到時 world_xy=None、note=原因。
    surface：目標放在哪個桌面上（{"name", "top_m", "half_m"}；地標本身就是桌子時也會給），pick/place 要先把手臂升降柱
    升到 top_m 才夾得到/放得到；在地上就是 None。"""
    if _FRONT_RE.search(target.lower()):
        # 「the bottle in front of me」：物件已經在車頭前（例如平台底下俯視相機看不到、用前相機找到並轉正的東西），
        # 不導航、不問俯視相機，直接從現在的位置用距離感測器送進指間
        return None, "front", " (the object right in front of the robot; no navigation)", None
    m = _COORD_RE.match(target) or _COORD_ANY_RE.search(target)
    state = await robot.state()
    landmarks = state.get("static_landmarks") or {}
    if m:
        # 純座標，或「描述 + 座標」（scan_overhead 之後 LLM 已經選好是哪一個，座標比再問一次俯視模型可靠：
        # 2026-10-21 r12「blue block at 0.90, -1.10」被模型指到另一塊藍積木，機器人跑過去還把它撞走）
        xy = (float(m.group(1)), float(m.group(2)))
        surface = _surface_under(xy, landmarks)   # LLM 直接給桌上物件的座標時，一樣要知道它在桌上
        desc = target[:m.start()].strip(" ,;:@-") if m is not _COORD_RE.match(target) else ""
        note = (f" (described as '{desc}')" if desc else "") + (f" (on top of the {surface['name']}, top {surface['top_m']:.2f} m)" if surface else "")
        return xy, "coords", note, surface
    name = _match_landmark(target, landmarks)
    words = set(re.findall(r"[a-z]+", target.lower()))
    if name and (words & _OBJECT_WORDS) and not (set(name.split("_")) & _OBJECT_WORDS):
        # 「the lime ball on the table」字面上對到地標 table，但目標是桌上的球、不是桌子：地標只是它所在的桌面，
        # 球本身還是用俯視相機找，找到後由 _surface_under 判斷它在桌上
        name = None
    surface_hint = _surface_hint(target, landmarks) if not name else None
    if name and landmarks[name].get("world_xy"):
        xy = landmarks[name]["world_xy"]
        return (float(xy[0]), float(xy[1])), "landmark", f" -> landmark '{name}'", _surface_of_landmark(name, landmarks)
    jpg = await robot.overhead_frame()
    if jpg is None:
        xy, note = await _locate_with_front_camera(robot, llm, target)
        return xy, "object", note, None
    depth = await robot.overhead_depth()
    low = target.lower()
    lm = landmarks.get(surface_hint) if surface_hint else None
    if lm and lm.get("world_xy") and lm.get("bbox"):
        # 任務／目標說物件在桌上（2026-10-20 實測：整張俯視圖裡 6 cm 的球只有 4 px，「lime green ball」三次有兩次被指到
        # 西北角的綠色小積木，agent 夾了積木還宣稱成功）：只把桌子那一塊裁切放大來問，候選不在桌上就老實說找不到，
        # 不去夾別的東西。
        lx, ly = lm["world_xy"]
        bb = lm["bbox"]
        found = await perception.locate_overhead(llm, jpg, target, state.get("world_xy"), pseudo_yaw(state), depth,
                                                 roi_world=(float(lx), float(ly), max(bb[1] - bb[0], bb[3] - bb[2]) / 2 + 0.25),
                                                 roi_name=f"{surface_hint} (a brown wooden block seen from above)")
        on_it = _surface_under(found.world_xy, landmarks) if found.found else None
        if not found.found or not on_it or on_it["name"] != surface_hint:
            # 桌上真的沒有（2026-10-20 實測：上一題把球放進桶子後沒歸位，agent 在桶裡看到球卻一直被導回桌子找，白燒 12 步）。
            # 再用整張圖找一次，只是為了告訴 LLM 東西現在在哪、為什麼拿不到，讓它老實用 done 回報，不是叫它去夾。
            alt = await perception.locate_overhead(llm, jpg, target, state.get("world_xy"), pseudo_yaw(state), depth)
            elsewhere = ""
            if alt.found:
                inside = _landmark_containing(alt.world_xy, landmarks)
                elsewhere = (f" The full overhead view does show a '{target}'-looking object at world ({alt.world_xy[0]:.2f}, "
                             f"{alt.world_xy[1]:.2f}), " + (f"inside the landmark '{inside}'" if inside else "on the floor") +
                             (" — at floor level, which this gripper cannot reach (fingers never go below 0.136 m above the floor)."
                              if ("ball" in low) else "."))
            return None, "object", (f"'{target}' is NOT on the {surface_hint}: the overhead close-up of the {surface_hint} shows nothing "
                                    f"matching ({found.raw[:80] if not found.found else 'candidate off the table'}).{elsewhere} The task's "
                                    f"premise (object on the {surface_hint}) does not hold right now, so do not grasp anything else or keep "
                                    f"searching the {surface_hint}: report this with done(success=false) unless the task can be satisfied another way."), None
    else:
        found = await perception.locate_overhead(llm, jpg, target, state.get("world_xy"), pseudo_yaw(state), depth)
        if not found.found and (words & _OBJECT_WORDS):
            # 指向模型常找不到 6 cm 的小東西（2026-10-21 t01：「plastic bottle」整張圖找不到）：退回 scan 的候選名單，
            # 用種類詞 + 顏色詞比對，挑離機器人最近的一個
            cat = ("balls" if "ball" in words else "small blocks" if words & {"block", "cube", "brick"} else "litter")
            items, _raw = await perception.survey_overhead(llm, jpg, cat, state.get("world_xy"), pseudo_yaw(state), depth, landmarks)
            kinds = {"bottle", "can", "cup", "carton", "ball", "block", "cube", "brick", "box"} & words
            cols = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "cyan", "lime", "white", "cream", "beige", "magenta", "teal"} & words
            cands = [it for it in items if (not kinds or any(k in it["label"] for k in kinds)) and (not cols or any(c in it["label"] for c in cols))]
            if cands:
                it = cands[0]
                found = perception.LocatedOverhead(found=True, world_xy=it["world_xy"], distance_m=it["distance_m"],
                                                   turn_left_deg=it["turn_left_deg"], height_m=it["height_m"],
                                                   raw=f"from scan_overhead({cat}): {it['label']} {it['place']}")
        if not found.found:
            xy, note = await _locate_with_front_camera(robot, llm, target)
            return xy, "object", note, None
    note = ""
    surface = _surface_under(found.world_xy, landmarks)
    if (surface is None and found.height_m is not None and found.height_m > 0.20
            and ("small" in low or "ball" in low or ("block" in low and "foam" not in low))):
        # 難題 h08：要「小積木」，俯視視覺模型指到 25 cm 的大泡棉方塊（同色、比較顯眼）。先帶著提示再問一次，
        # 還是指到大的才放棄——不然「small red block」永遠解析不到真正的小紅積木。
        # 桌上的東西從上面看也「很高」（桌面 + 物件），但那是正常的，_surface_under 認得出來就不算可疑。
        hint = f"{target} (NOT the big 25 cm foam cube of the same colour; the small block is only about 7 cm wide)"
        retry = await perception.locate_overhead(llm, jpg, hint, state.get("world_xy"), pseudo_yaw(state), depth)
        if retry.found and (retry.height_m is None or retry.height_m <= 0.20 or _surface_under(retry.world_xy, landmarks)):
            found = retry
            surface = _surface_under(found.world_xy, landmarks)
        else:
            return None, "object", (f"the overhead candidate for '{target}' is about {found.height_m * 100:.0f} cm tall — that is one of "
                                    f"the big foam blocks, not a small block/ball. Describe the target differently (colour + 'small block') "
                                    f"or pick another one."), None
    if (found.height_m is not None and found.height_m < perception.FLAT_HEIGHT_THRESHOLD_M and (words & _OBJECT_WORDS)
            and surface is None):
        # 要的是立體的小物件，候選卻是扁的：2026-10-21 r12「blue block」兩次被指到藍地墊（1 cm）跟藍桶底（2 cm），
        # 以前只加一句警告照樣去夾。帶提示再問一次，還是扁的就拒絕，並教 LLM 用 scan_overhead 的座標指名。
        hint = (f"{target} — a real 3-D object standing on the floor (a small block is a 7 cm cube, a ball sits on a short post), "
                f"NOT a flat mat, a floor marking, or the floor of a storage bin")
        retry = await perception.locate_overhead(llm, jpg, hint, state.get("world_xy"), pseudo_yaw(state), depth)
        if retry.found and (retry.height_m is None or retry.height_m >= perception.FLAT_HEIGHT_THRESHOLD_M):
            found = retry
            surface = _surface_under(found.world_xy, landmarks)
        else:
            kind = next(iter(words & _OBJECT_WORDS))
            return None, "object", (f"the overhead candidate for '{target}' at world ({found.world_xy[0]:.2f}, {found.world_xy[1]:.2f}) is "
                                    f"flat (~{found.height_m * 100:.0f} cm high) — a mat, a marking or the floor of a bin, not a {kind}. "
                                    f"Use scan_overhead('{kind}s') and then name the one you mean WITH its coordinates, e.g. "
                                    f"'{target} at (0.90, -1.10)'."), None
    if found.height_m is not None and found.height_m < perception.FLAT_HEIGHT_THRESHOLD_M:
        note = f" (warning: the overhead candidate is flat, ~{found.height_m * 100:.0f}cm high — may be a mat or marking)"
    if surface:
        note += f" (on top of the {surface['name']}, whose top is {surface['top_m']:.2f} m above the floor)"
    return (float(found.world_xy[0]), float(found.world_xy[1])), "object", note, surface


async def go_to(robot, llm, target: str, stop_m: Optional[float] = None) -> Tuple[bool, str]:
    xy, kind, note, _surface = await resolve_target(robot, llm, target)
    if kind == "front":
        return False, f"go_to({target}): 'in front of me' only makes sense for pick; give a landmark, an object description or coordinates"
    if xy is None:
        return False, f"go_to({target}): {note}"
    if stop_m is None:
        stop_m = 0.0 if kind == "coords" else 0.45      # 地標/物件：停在它前面並面向它，不要撞上去
    if kind == "landmark":
        # 地標交給伺服器：隧道/平台停在入口、其他停在前緣（伺服器知道結構幾何，agent 端不知道）
        name = note.split("'")[1]
        res = await robot.act("go_to_landmark", {"name": name, "stop_m": stop_m}, timeout=NAV_TIMEOUT_S)
        return bool(res.get("ok")), f"go_to({target}) [landmark '{name}']: {res.get('message')}"
    res = await robot.act("navigate_to", {"x": xy[0], "y": xy[1], "stop_m": stop_m, "face": stop_m > 0, "label": str(target)[:40]},
                          timeout=NAV_TIMEOUT_S)
    msg = f"go_to({target}) [{kind} at world ({xy[0]:.2f}, {xy[1]:.2f}){note}]: {res.get('message')}"
    return bool(res.get("ok")), msg


_LANDMARK_DESC = {
    "ball_pit": "ball pit with a checkered border", "bench": "wooden bench", "bin_balls": "blue storage bin",
    "bin_blocks": "orange storage bin", "tunnel": "orange tunnel", "platform": "raised platform with a deck",
    "stairs": "stairs", "slide": "blue slide", "pillar_red": "red pillar", "pillar_blue": "blue pillar",
    "pillar_yellow": "yellow pillar", "pillar_green": "green pillar",
}
_STOP_NOTE = (" You are now at the requested distance: do NOT drive closer. Anything nearer than about 15 cm drops "
              "below the front distance sensor and reads 9999, and 'close to it' checks use that sensor reading.")


async def approach(robot, llm, obj: str, stop_m: float = 0.4) -> Tuple[bool, str]:
    stop_m = min(1.0, max(0.15, float(stop_m or 0.4)))
    xy, kind, note, _surface = await resolve_target(robot, llm, obj)
    if kind == "front":
        return False, f"approach({obj}): 'in front of me' only makes sense for pick; give an object description or coordinates"
    if xy is None:
        return False, f"approach({obj}): {note}"
    # 地標名（bin_blocks）VLM 看不懂，前鏡頭辨識要用人話描述（中等批次 m06：問 'bin_blocks' 永遠「沒看到」）
    lname = note.split("'")[1] if kind == "landmark" and "'" in note else None
    label = _LANDMARK_DESC.get(lname, obj) if lname else _plain_label(obj)
    res = await robot.act("navigate_to", {"x": xy[0], "y": xy[1], "stop_m": stop_m, "face": True, "label": (lname or str(obj))[:40]},
                          timeout=NAV_TIMEOUT_S)
    msg = f"approach({obj}) [{kind} at world ({xy[0]:.2f}, {xy[1]:.2f}){note}]: {res.get('message')}"
    if not res.get("ok"):
        return False, msg
    # 前鏡頭精對準（最多修正兩次），跟 face() 同一套。前鏡頭是 H.264 串流、延遲以「張數」計：剛轉完就抓，
    # 拿到的常是轉向前的舊畫面（實測 approach 停在球前 0.3 m 卻回報「沒看到」，之後抓的畫面球就在正中央）。
    # 所以先等一下、丟掉第一張再辨識。
    async def fresh_frame():
        await asyncio.sleep(0.8)
        await robot.frame()
        return await robot.frame()
    await robot.act("recenter_arm", {})   # agent 可能自己動過手臂；相機掛在手臂上，先回標準姿態再看（h01 第 18 步）
    found = await perception.locate(llm, await fresh_frame(), label)
    for _ in range(2):
        if not found.found or abs(found.bearing_right_deg) <= 4:
            break
        r = await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": found.turn_left_deg})
        if not r.get("ok"):
            break
        found = await perception.locate(llm, await fresh_frame(), label)
    arm_note = ""
    if not found.found:
        # 小物件在預設手臂姿態下、車前 0.5 m 內的地面是視野死角（相機在手臂上）。放低手臂看夾爪正前方，
        # 再抬高看遠一點，看完把手臂歸位——姿態值來自 memory/body_notes.md 實測過的紀錄。
        # NexArm + 相機在上臂（2026-11-14）：相機俯仰 ≈ 上臂仰角 − 85°，上臂往後仰（前伸小、高度高、夾爪水平）就看天花板。
        # 舊的 (180,30)/(90,120) 是 EP 原廠手臂的座標，NexArm 會夾成 (179,188)/(88,185) → 手肘往後、看天花板。
        # NexArm 改用「上臂往前倒」的兩個姿勢：俯 30° 看車前 0.3–1 m，俯 55° 看夾爪正前方的地面/箱頂。
        nex = "arm_pose" in (await robot.state())
        poses = ((("looking 30 deg down (x=250,y=150)", 250, 150), ("looking 55 deg down (x=380,y=120)", 380, 120)) if nex
                 else (("low (x=180,y=30)", 180, 30), ("high (x=90,y=120)", 90, 120)))
        for pose_name, x_mm, y_mm in poses:
            r = await robot.act("arm_to", {"x_mm": x_mm, "y_mm": y_mm, "pitch_deg": 0.0} if nex else {"x_mm": x_mm, "y_mm": y_mm})
            if not r.get("ok"):
                continue
            found = await perception.locate(llm, await fresh_frame(), label)
            if found.found:
                arm_note = f" (seen with the arm {pose_name}; arm recentered afterwards)"
                break
        await robot.act("recenter_arm", {})
    # 用前方距離感測器收尾：站位是從地圖算的（俯視定位有 0.1-0.2 m 誤差、口袋太淺時站位會被推遠），
    # 感測器讀得到目標就直接把車頭到目標的距離修到 stop_m（前進由伺服器的防撞縮步保護，不會撞上）。
    state = await robot.state()
    tof = state.get("tof_front_mm")
    if tof is not None and tof != 9999 and 0.10 < tof / 1000.0 < 1.2:
        delta = tof / 1000.0 - stop_m
        if abs(delta) > 0.06:
            r = await robot.act("move_chassis", {"forward_m": round(delta, 2), "right_m": 0, "turn_left_deg": 0})
            if r.get("ok"):
                await asyncio.sleep(0.5)
                state = await robot.state()
                tof = state.get("tof_front_mm")
                msg += f" Adjusted {delta:+.2f} m using the front distance sensor."
    if found.found:
        msg += (f" Front camera confirms '{label}': {found.bearing_right_deg:+.0f} deg from centre (right positive), "
                f"image y={found.y}/1000{arm_note}. tof_front_mm={tof}." + _STOP_NOTE)
        return True, msg
    if lname:
        # 地標的位置是場景幾何算出來的、不是猜的：到了就是到了，前鏡頭沒認出來只是備註（矮的收納箱在近處常落在畫面下緣外）
        msg += (f" Front camera could not confirm '{label}' from here, but this is a fixed landmark at a known position and "
                f"the robot is facing it (tof_front_mm={tof})." + _STOP_NOTE)
        return True, msg
    msg += (f" Front camera does NOT see '{obj}' after arriving, even with the arm lowered and raised "
            f"(tof_front_mm={tof}) — the overhead candidate may be the wrong object or hidden; use locate() to look "
            f"around before concluding.")
    return False, msg


ARM_GRASP = (180, 30)      # 抓取姿態：夾爪放到地面高度、伸到車頭前約 0.32 m（h01/manip_001/h03 實測）
ARM_CARRY = (120, 100)     # 搬運姿態：抬高收回，物件離開距離感測錐、也不擋相機
ARM_RELEASE_LOW = (180, 40)   # 放到地墊/地面
ARM_RELEASE_HIGH = (160, 170)  # 放進收納箱：實際到 (164,149)，夾爪離地 0.248 m、在車頭前 0.106 m，夾著的積木底高過 10 cm 桶壁
GRASP_TOF_MAX_MM = 150     # 物件前緣離感測器 ≤ 這麼遠就在指間（成功的抓取實測 116-145 mm）
# 夾爪腕部力感測器（模擬，/state 的 gripper_force_n）：空夾爪 1.50 N，夾著 50 g 的球/積木多 0.49 N，靜止時沒雜訊、跟手臂姿態無關。
# 多 0.12 N 以上就算夾著：球/積木 50 g = 0.49 N，垃圾最輕的紙杯 20 g = 0.20 N、寶特瓶 30 g = 0.29 N（2026-10-21 門檻 0.25 時
# 瓶子明明夾起來了卻被判沒夾到、張開重試把它弄倒）；模擬裡靜止讀值沒雜訊，門檻可以放這麼低。
# 有這個讀值時它是主要判據：相機窗口有顏色不代表夾著（球掉在車前地上一樣看得到）。
GRIPPER_EMPTY_N = 1.50
GRIPPER_HELD_DELTA_N = 0.12

# 技能執行途中的遙測快照：一個 pick 裡「張開→關上→抬起」全發生在同一個動作內，只看動作結束時的狀態
# 永遠看不到夾爪「開著」（難題 h03/h06/h08 的遙測里程碑因此全判失敗）。技能在關鍵時刻自己記一筆，
# service.py 收進 ActionRecord.state_trace，judge 的遙測判定會一起看。
_TRACE: list = []


async def _snap(robot, tag: str) -> dict:
    st = await robot.state()
    snap = dict(st or {})
    snap["trace_tag"] = tag
    _TRACE.append(snap)
    return snap


def _save_frame(jpg, tag: str) -> str:
    """技能中途看的畫面存到 runs/_skill_frames/<時間>_<tag>.jpg（2026-11-13：pick_from_top 三個斜視姿態都「沒看到」，
    卻沒有任何畫面可以檢查是相機被線拉歪還是模型沒認出來）。回傳檔名，失敗回空字串。"""
    if not jpg:
        return ""
    try:
        import os, time as _t
        d = os.path.join("runs", "_skill_frames")
        os.makedirs(d, exist_ok=True)
        name = os.path.join(d, _t.strftime("%Y%m%d_%H%M%S_") + re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")[:40] + ".jpg")
        with open(name, "wb") as f:
            f.write(jpg)
        return name
    except Exception:
        return ""


def drain_trace() -> list:
    out = list(_TRACE)
    _TRACE.clear()
    return out


def _plain_label(obj: str) -> str:
    """給相機 VLM 看的描述：把 'blue block at (0.91, -1.11)' 的座標拿掉。帶座標問 VLM 它常回「看不到」
    （h01 第 2 步站在真積木正前方 0.56 m 仍被判沒看到；下一步問 'blue block' 就找到了）。"""
    out = re.sub(r"\s*\b(at|near|around)\b\s*\(?\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)?", "", obj)
    out = re.sub(r"\(?\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*\)?", "", out)
    return re.sub(r"\s+", " ", out).strip(" ,") or obj


async def _tof_sweep(robot):
    """前方感測器讀不到東西時左右各掃幾度，找到就停在那個角度；回傳讀值（找不到回 None）。"""
    for turn in (3, -6, 9, -12, 6):
        await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": turn})
        await asyncio.sleep(0.3)
        tof = await _tof(robot)
        if tof is not None and tof <= 700:
            return tof
    return None


async def _creep_until(robot, tof, stop_mm: int, aim_mm: int, max_iters: int):
    """用前方感測器一步步把車送到 stop_mm 以內。每走一步讀值應該差不多少掉一步的距離：連續兩步沒有變近，代表
    感測器看到的不是正前方這個東西（h01：積木在光錐外，push=true 盲推了 1.03 m 把積木推到溜滑梯邊）→ 停，不再推。"""
    blind = 0
    for _ in range(max_iters):
        if tof is None or tof <= stop_mm:
            break
        step = min(0.10, max(0.02, (tof - aim_mm) / 1000.0))
        r = await robot.act("move_chassis", {"forward_m": round(step, 2), "right_m": 0, "turn_left_deg": 0, "push": True})
        if not r.get("ok"):
            break
        new = await _tof(robot)
        if new is None or new > tof - step * 1000 * 0.4:
            blind += 1
            if blind >= 2:
                tof = new
                break
        else:
            blind = 0
        tof = new
    return tof


async def _obstacle_between(robot, xy) -> str:
    """站定後比對前方感測器讀值跟「到目標的幾何距離」：讀值短很多代表中間有別的東西。回傳說明字串，沒有就空字串。
    感測器在車心前方約 0.18 m；目標物半寬約 0.04 m。"""
    st = await robot.state()
    rxy, hd = st.get("world_xy"), st.get("world_heading_deg")
    tof = st.get("tof_front_mm")
    if rxy is None or hd is None or tof is None or tof == 9999:
        return ""
    d = math.hypot(xy[0] - rxy[0], xy[1] - rxy[1])
    rel = (math.degrees(math.atan2(xy[1] - rxy[1], xy[0] - rxy[0])) - hd + 180) % 360 - 180
    if abs(rel) > 10:
        return ""   # 沒有正對著目標，讀值本來就不是它
    expected = d - 0.18 - 0.04
    if tof / 1000.0 < expected - 0.15:
        return f"front sensor reads {tof} mm but the target is {d:.2f} m from the robot centre, so something else is in between"
    return ""


async def _creep_to_target(robot, xy, stop_m: float = 0.24, max_total_m: float = 0.5) -> str:
    """手臂放低後感測器讀 9999：站位離柱子 0.4 m 以上時，車頭差幾度，±8° 的光錐就掃不到 3 cm 的柱子
    （h05/p08 連續 13 次這樣放棄）。用世界座標先把車頭轉正對目標，再小步前進，直到感測器看到東西或只剩 stop_m。"""
    moved, note = 0.0, ""
    for _ in range(8):
        st = await robot.state()
        rxy, hd = st.get("world_xy"), st.get("world_heading_deg")
        if rxy is None or hd is None:
            break
        d = math.hypot(xy[0] - rxy[0], xy[1] - rxy[1])
        bearing = math.degrees(math.atan2(xy[1] - rxy[1], xy[0] - rxy[0]))
        delta = (bearing - hd + 180) % 360 - 180
        if abs(delta) >= 2.0:
            await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(delta, 1)})
            await asyncio.sleep(0.3)
        tof = await _tof(robot)
        if tof is not None and tof <= 450:
            note = f" (faced the target by world pose and crept {moved:.2f} m; sensor now {tof} mm)"
            break
        if d <= stop_m or moved >= max_total_m:
            break
        step = round(min(0.10, max(0.03, d - stop_m)), 2)
        r = await robot.act("move_chassis", {"forward_m": step, "right_m": 0, "turn_left_deg": 0})
        if not r.get("ok"):
            break
        moved += step
    if not note and moved:
        note = f" (faced the target by world pose and crept {moved:.2f} m; sensor still sees nothing)"
    return note


def tunnel_axis_from_landmark(geo: dict, state: dict):
    """server 的 landmark_axis 結果 → perception.TunnelAxis，給 align_to_tunnel 用（已知結構不必靠 VLM 找兩端）。"""
    rxy, hd = state.get("world_xy"), state.get("world_heading_deg")
    if rxy is None or hd is None or not geo.get("ok"):
        return None
    ent = geo["entrance"]
    dist = math.hypot(ent[0] - rxy[0], ent[1] - rxy[1])
    bearing = math.degrees(math.atan2(ent[1] - rxy[1], ent[0] - rxy[0]))
    return perception.TunnelAxis(found=True, near_end_world=tuple(geo["near_end"]), far_end_world=tuple(geo["far_end"]),
                                 axis_bearing_deg=float(geo["axis_bearing_deg"]), distance_m=round(dist, 2),
                                 turn_left_deg=round((bearing - hd + 180) % 360 - 180, 1),
                                 raw="fixed landmark geometry: " + str(geo.get("message", "")))


async def _tof(robot):
    t = (await robot.state()).get("tof_front_mm")
    return None if (t is None or t == 9999) else int(t)


async def _gripper_load(robot) -> Optional[float]:
    """夾爪上掛著的重量（N，扣掉空夾爪自重）；沒有力感測器的機器人回 None。靜止後再讀，動作中的加速度會混進去。"""
    await asyncio.sleep(0.3)
    f = (await robot.state()).get("gripper_force_n")
    return None if f is None else round(float(f) - GRIPPER_EMPTY_N, 2)


# 球的顏色 -> PIL HSV 的色相（0-255）。場景裡的球：橘、紫、粉紅、青（地上）跟萊姆綠（桌上，RGB (0.62,0.95,0.12) → 色相 60）。
_BALL_HUE = {"orange": 19, "purple": 200, "violet": 200, "pink": 234, "magenta": 234, "cyan": 131, "teal": 131,
             "lime": 60, "chartreuse": 60, "red": 0, "yellow": 35, "green": 85, "blue": 160}


def _ball_offset_mm(jpg: Optional[bytes], obj: str) -> Optional[float]:
    """抓取姿態、感測器讀柱子約 200 mm 時，用目標顏色的像素算球在夾爪中心左邊(+)或右邊(-)幾 mm。
    2026-10-08 校正（紫色球，模擬器真值）：球心像素 x=322.7 時正好在兩爪中間，每偏 1 px 約 1.13 mm；
    只認目標顏色，旁邊的紅積木、球池才不會把平均位置拉走。找不到顏色就回傳 None（不修正）。"""
    hue = next((h for word, h in _BALL_HUE.items() if word in obj.lower()), None)
    if hue is None or not jpg:
        return None
    try:
        import io
        import numpy as np
        from PIL import Image
        a = np.asarray(Image.open(io.BytesIO(jpg)).convert("HSV"), dtype=np.float32)
        h, w = a.shape[:2]
        d = np.abs(a[..., 0] - hue); d = np.minimum(d, 255 - d)
        # 色相容許 ±8：±18 時橘色會吃進黃色泡棉圓柱、黃紅樓梯（2026-10-08 量到 -72 mm、實際 +33 mm，轉錯方向）
        m = ((a[..., 1] > 90) & (a[..., 2] > 60) & (d < 8)).astype(np.uint8)
        m[: int(h * 0.45)] = 0
        import cv2
        n, _, stats, cents = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1:
            return None
        # 挑「畫面最低」的那一塊（= 離車最近的），不是最大的：背景裡的橘色隧道、橘色收納箱比眼前的球大得多，
        # 用最大塊會算出球在左邊 12 cm、車子轉 18° 離開柱子（2026-10-22 橘球 h05/p08 全部這樣失敗）。
        cand = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= 150]
        if not cand:
            return None
        k = max(cand, key=lambda i: stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT])
        off = -(float(cents[k][0]) * 640.0 / w - 322.7) * 1.13
        # 到這一步感測器已經看到柱子（光錐 ±8°），球不可能偏超過約 12 cm；更大的偏差一定是認錯東西
        return off if abs(off) <= 120 else None
    except Exception:
        return None


def _held_in_view(jpg: Optional[bytes]) -> float:
    """抬起之後，夾爪框底部那一小塊畫面裡「有顏色的像素」比例。相機裝在手臂上，夾住的東西跟相機一起動，
    會一直卡在這個窗口裡；地板、牆、柱子都是灰黑色。2026-10-07 實測：夾著 6 cm 球 0.18、空夾 0.00。
    積木一般靠前方距離感測器驗證就夠；球放在柱子上，抬起後就離開感測器光束（光束在 0.10 m 高），
    感測器只看得到留在原地的柱子，所以要靠這個。"""
    if not jpg:
        return 0.0
    try:
        import io
        import numpy as np
        from PIL import Image
        im = Image.open(io.BytesIO(jpg)).convert("HSV")
        w, h = im.size
        a = np.asarray(im, dtype=np.float32)[int(h * 330 / 360):h, int(w * 260 / 640):int(w * 380 / 640)]
        return float(((a[..., 1] > 90) & (a[..., 2] > 60)).mean())
    except Exception:
        return 0.0


async def pick(robot, llm, obj: str) -> Tuple[bool, str]:
    """抓起一個物件：站到它前面 → （在桌上的話先把手臂升降柱升到桌面高度）→ 張開、放低手臂 → 用距離感測器一步步把它
    送進指間 → 關夾爪、抬起 → 倒退 0.25 m 驗證（物件在手上，感測器讀值不會變；還在地上就會變遠或消失）。整個流程 LLM 不用介入。
    兩段目標距離：積木先到 95 mm（積木前緣在指尖內約 4 cm；實測 83-95 mm 全部成功、119 mm 會把積木推走），
    沒夾到就再深到 80 mm；球（放在柱子上）底盤送到 125 mm 以內、再用手臂往前伸補到等效 110 mm。驜證用距離感測器或手臂相機的
    夾爪窗口，任一成立即可。
    升降柱（Env/add_arm_lift.py）：前方距離感測器掛在升降座上、跟夾爪一起升，所以升到桌面高度之後，送進指間的幾何跟在地上
    完全一樣，下面的流程一步都不用改；結束時不管成敗都降回 0（重心低、感測器回到底盤高度才適合開車）。"""
    _st0 = await robot.state()
    if "real_tof_front_mm" in _st0 and "arm_pose" in _st0:
        # 2026-11-14：實體場地（鏡射 + NexArm）上舊的 pick 會頂著箱子開、把東西推倒；prompt 禁止了 agent 還是會退回來用，這裡直接擋
        return False, (f"pick({obj}): not available on the real field with the NexArm (it drives into the box and knocks things over). "
                       f"For something standing on the box use pick_from_top('<object>') with no numbers; if that says the box is "
                       f"not in front, face('<the box>') first and call pick_from_top again.")

    xy, kind, note, surface = await resolve_target(robot, llm, obj)
    if kind == "front":
        return await _pick_at_height(robot, obj, f"[{note.strip()}]", 0)
    if xy is None:
        return False, f"pick({obj}): {note}"
    where = f"[{kind} at world ({xy[0]:.2f}, {xy[1]:.2f}){note}]"
    res = await robot.act("navigate_to", await _nav_params(robot, xy, 0.30, surface, label=obj), timeout=NAV_TIMEOUT_S)
    if not res.get("ok"):
        return False, f"pick({obj}) {where}: could not reach it — {res.get('message')}"
    if not surface:
        # 擁擠處（紫球、紅積木、球池牆之間的瓶子）導航的站位會被推到 1 m 外，ToF 只看 0.6 m → 放低手臂就讀 9999；
        # 先面向目標補走到約 0.45 m（桌上的東西不做：桌緣擋著，站位本來就固定）
        where += await _close_in(robot, xy)
        # 站位與目標之間夾著別的東西（紅積木北側的黃色泡棉圓柱：ToF 讀 286 mm，積木其實在 0.57 m 外），
        # 感測器進給會停在那個東西前面、永遠到不了目標：換一側重新接近一次
        blocked = await _obstacle_between(robot, xy)
        if blocked:
            st = await robot.state()
            rxy = st.get("world_xy") or xy
            side = math.degrees(math.atan2(rxy[1] - xy[1], rxy[0] - xy[0])) + 90.0
            params = await _nav_params(robot, xy, 0.30, surface, label=obj)
            params["approach_from_deg"] = round(side, 1)
            res2 = await robot.act("navigate_to", params, timeout=NAV_TIMEOUT_S)
            if res2.get("ok"):
                where += f" [{blocked}; re-approached from another side]"
                where += await _close_in(robot, xy)
            else:
                where += f" [{blocked}; could not find another side to approach from]"
    lift_mm = min(LIFT_MAX_MM, int(round(surface["top_m"] * 1000))) if surface else 0
    if lift_mm:
        r = await robot.act("lift_arm", {"height_mm": lift_mm})
        if not r.get("ok"):
            await robot.act("lift_arm", {"height_mm": 0})
            return False, (f"pick({obj}) {where}: it sits on the {surface['name']} {surface['top_m']:.2f} m up, but the arm lift "
                           f"could not get there — {r.get('message')}")
        where += f" [arm lifted {lift_mm} mm to the {surface['name']} top]"
    try:
        return await _pick_at_height(robot, obj, where, lift_mm, target_xy=xy)
    finally:
        if lift_mm:
            await robot.act("lift_arm", {"height_mm": 0})


async def _pick_at_height(robot, obj: str, where: str, lift_mm: int, target_xy=None) -> Tuple[bool, str]:
    """pick 的抓取本體：手臂（升降柱已經在正確高度）放低、送進指間、關、抬、倒退驗證。"""
    attempts = []
    # 目標距離：積木實測 83-90 mm 四次全成功、119 mm 兩次一次失敗（指尖剛好碰到前緣，關上會把物件推走）。
    # 球放在柱子上，感測器量到的是柱子；送到 95 mm 時夾爪前方的橫桿會碰到球上半部、把球推下柱子，
    # 所以停在 120 mm（球剛好在兩個爪尖之間）就關，實測 6 cm 球連續 8 次夾起。
    is_ball = "ball" in obj.lower()
    # 圓柱形的垃圾（瓶、罐、杯）橫截面跟球一樣是 6-7 cm 的圓：送到跟球一樣的深度（中心離感測器約 14 cm），夾爪掌心的
    # 掛接感測器才偵測得到、夾爪腳本才會把它掛接到夾爪上；用積木的深度（前緣 95 mm）掛不上，開車一晃就掉（2026-10-21 實測）
    is_round = is_ball   # 垃圾改成方柱後走積木的流程（圓柱掛接不上，見 Env/add_litter.py）
    lifted = f" (arm lift lowered back to 0 from {lift_mm} mm)" if lift_mm else ""
    # 球：底盤先送到 125 mm 以內，再用手臂往前伸補到「等效 110 mm」。底盤一步最少 2 cm（再小會被當成沒動而拒絕），
    # 停下來的讀值會落在 111-119 mm 之間；2026-10-08 用模擬器真值量：關夾時 111-112 mm（球在爪尖後 58 mm）夾得起來，
    # 117-119 mm（65-67 mm）爪子會往上滑過球。手臂可以一次只伸幾 mm，所以剩下的深度交給它。
    for target_mm in ((125, 120) if is_round else (95, 80)):
        await robot.act("gripper", {"state": "open"})
        await _snap(robot, "pick: gripper opened")
        r = await robot.act("arm_to", {"x_mm": ARM_GRASP[0], "y_mm": ARM_GRASP[1]})
        if not r.get("ok"):
            await robot.act("recenter_arm", {})
            return False, f"pick({obj}) {where}: could not lower the arm — {r.get('message')}"
        tof = await _tof(robot)
        if tof is None or tof > 700:
            # 放低手臂後感測器沒看到東西：導航到位時朝向誤差可達 5°，0.5 m 外的 6-7 cm 物件（瓶子、積木）會落在 16° 光錐邊緣
            # （2026-10-21 瓶子從 0.5 m 外 pick 直接讀 9999 放棄）。左右各掃幾度，找到就繼續，不要直接放棄。
            tof = await _tof_sweep(robot)
            if (tof is None or tof > 700) and target_xy is not None and not lift_mm:
                note = await _creep_to_target(robot, target_xy)
                if note and not attempts:
                    where = where + note
                tof = await _tof(robot)
        if is_ball:
            # 先用相機把車頭轉正再看感測器：柱子只有 3 cm 粗，站位處感測器光錐左右只有約 3.4 cm，車頭偏一點就讀 9999，
            # 原本會直接放棄（2026-10-08 橘球兩次這樣失敗）。像素轉角度：校正時 1.13 mm/px、球離車心約 0.38 m → 約 0.17°/px。
            for _ in range(2):
                off = _ball_offset_mm(await robot.frame(), obj)
                if off is None:
                    break
                deg = off / 1.13 * 0.17
                if abs(deg) < 1.5:
                    break
                # 感測器已經看到柱子，球一定在光錐 ±8° 內：單次修正最多 6°，再多就是色塊認錯
                await robot.act("move_chassis", {"forward_m": 0, "right_m": 0,
                                                 "turn_left_deg": min(6.0, max(2.0, round(abs(deg)))) * (1 if deg > 0 else -1)})
                await asyncio.sleep(0.3)
            tof = await _tof(robot)
            if tof is None or tof > 700:
                tof = await _tof_sweep(robot)   # 相機修正把柱子修出光錐了：掃回來
            # 左右對準：偏超過約 1.4 cm 時，先碰到球的那一側爪子會把球推下柱子（紫色球從東側接近每次偏 14 mm、每次失敗）。
            # 先送到感測器讀柱子約 200 mm，再用目標顏色量偏差；校正時每轉 2° 球左右移約 13 mm，所以 1° ≈ 6.3 mm。
            tof = await _creep_until(robot, tof, 215, 200, 6)
            for _ in range(2):
                off = _ball_offset_mm(await robot.frame(), obj)
                if off is None or abs(off) < 8:
                    break
                turn = max(2.0, round(abs(off) / 6.3)) * (1 if off > 0 else -1)
                await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": turn})
                await asyncio.sleep(0.3)
            tof = await _tof(robot)
        if not is_ball:
            # 先送到約 200 mm，再用感測器掃描把車頭對正物件中心（相機對準只有球有可靠的顏色詞）
            tof = await _creep_until(robot, tof, 215, 200, 6)
            where_note = await _tof_center(robot)
            if where_note and not attempts:
                where = where + where_note
            tof = await _tof(robot)
        tof = await _creep_until(robot, tof, target_mm, target_mm, 10)
        if tof is None or tof > target_mm + 30:
            await robot.act("gripper", {"state": "close"})
            await robot.act("recenter_arm", {})
            return False, (f"pick({obj}) {where}: could not get the object between the fingers (front sensor reads "
                           f"{tof if tof is not None else 9999} mm with the arm lowered; attempts so far: {attempts}). "
                           f"It may have been pushed aside or be too low for the sensor; locate it again and retry.{lifted}")
        tof_before = tof
        if is_round:
            reach = max(0, min(20, tof - 110))
            if reach >= 2:
                await robot.act("arm_to", {"x_mm": ARM_GRASP[0] + reach, "y_mm": ARM_GRASP[1]})
                await asyncio.sleep(0.3)
        await robot.act("gripper", {"state": "close"})
        await _snap(robot, "pick: gripper closed on the object")
        # 球是圓的，夾爪還沒合緊就抬起會滑掉：實測等 1.5 s 才抬 8/8 夾起，pick 原本的 0.5 s 三顆球兩顆掉
        await asyncio.sleep(1.5 if is_round else 0.5)
        await robot.act("move_arm", {"forward_mm": 0, "up_mm": 60})
        await _snap(robot, "pick: lifted")
        await robot.act("move_chassis", {"forward_m": -0.25, "right_m": 0, "turn_left_deg": 0})
        await asyncio.sleep(0.3)
        tof_after = await _tof(robot)
        in_view = _held_in_view(await robot.frame())
        load = await _gripper_load(robot)
        attempts.append(f"closed at {tof_before} mm -> {tof_after if tof_after is not None else 9999} mm after backing up, "
                        f"gripper window {in_view:.2f} coloured" + (f", gripper load {load:+.2f} N" if load is not None else ""))
        # 真夾著時倒退後讀值幾乎不變（實測 92→86、93→91、94→90）；2026-10-21 t03 空夾爪讀到別的東西 86→134（差 48）
        # 被舊的 60 mm 門檻判成夾到、宣稱成功，所以收緊到 25 mm
        by_tof = tof_after is not None and abs(tof_after - tof_before) <= 25
        by_force = None if load is None else load >= GRIPPER_HELD_DELTA_N
        if by_force is False:
            # 力感測器只量得到被夾爪腳本「掛接」上去的物件（球一定會被掛接；手指連桿是靜態形狀，純靠摩擦夾住的東西
            # 接觸力傳不到感測器——2026-10-21 瓶子明明夾起來跟著車走，讀值仍是空夾爪的 1.50 N）。所以：
            # 球：沒讀到重量就是沒夾到，不管相機/ToF（ToF 可能讀到留在原地的柱子、相機可能看到掉在地上的球）。
            # 其他物件：ToF 讀值跟著車走就算夾住（原本積木的判據）；只有「只靠相機」的證據才需要力感測器背書。
            if is_ball:
                by_tof, in_view = False, 0.0
            elif not by_tof:
                in_view = 0.0
        if by_tof or in_view >= 0.05 or by_force:
            await robot.act("arm_to", {"x_mm": ARM_CARRY[0], "y_mm": ARM_CARRY[1]})
            await asyncio.sleep(0.3)
            if by_force:
                # 收到搬運姿態的瞬間最容易掉（從東側夾球實測：相機還看得到掉在車前的球，力感測器已經回到空夾爪）
                load2 = await _gripper_load(robot)
                if load2 is not None and load2 < GRIPPER_HELD_DELTA_N:
                    attempts[-1] += f", but it slipped out when the arm moved to the carry pose (gripper load {load2:+.2f} N)"
                    by_tof, in_view, by_force = False, 0.0, False
            elif not by_tof:
                # 只靠相機判定的（球）在搬運姿態再看一次：2026-10-08 實測，球被夾在抓取高度拖著走、其實沒被抬起時，
                # 倒退後的畫面也會有 13-14% 的顏色，但收手臂時就掉了；搬運姿態下真夾住 0.25、掉了 0.00。
                await asyncio.sleep(0.5)
                carry_view = _held_in_view(await robot.frame())
                if carry_view < 0.05:
                    attempts[-1] += f", but it slipped out when the arm moved to the carry pose ({carry_view:.2f} coloured)"
                    in_view = 0.0
            if not (by_tof or in_view >= 0.05 or by_force):
                await robot.act("gripper", {"state": "open"})
                await robot.act("arm_to", {"x_mm": ARM_GRASP[0], "y_mm": ARM_GRASP[1]})
                await robot.act("move_chassis", {"forward_m": 0.25, "right_m": 0, "turn_left_deg": 0, "push": True})
                continue
            # 搬運姿態下積木在感測錐裡，導航要忽略那個讀值；球在光束上方，感測器看不到它，不用遮
            await robot.act("set_held", {"tof_mm": await _tof(robot) if by_tof else None})
            if by_force:
                why = (f"the gripper's wrist force sensor carries {load:+.2f} N more than an empty gripper (the object's weight), "
                       f"still there in the carry pose after backing up 0.25 m")
            elif by_tof:
                why = f"front sensor read {tof_before} mm before lifting and {tof_after} mm after backing up 0.25 m"
            else:
                why = (f"after lifting and backing up 0.25 m the object is still between the fingers in the arm camera "
                       f"({in_view:.0%} of the gripper window is the object's colour)")
            return True, (f"pick({obj}) {where}: GRASPED — {why}, so the object moved with the robot. Arm is in the carry "
                          f"position{lifted}; keep the gripper closed and use place(target) to put it down.")
        # 沒夾到：放掉、手臂放回抓取姿態、走回去再試更深一點
        await robot.act("gripper", {"state": "open"})
        await robot.act("arm_to", {"x_mm": ARM_GRASP[0], "y_mm": ARM_GRASP[1]})
        await robot.act("move_chassis", {"forward_m": 0.25, "right_m": 0, "turn_left_deg": 0, "push": True})
    await robot.act("gripper", {"state": "close"})
    await robot.act("set_held", {"tof_mm": None})
    await robot.act("recenter_arm", {})
    if is_ball:
        # h09（2026-10-08）：球掉下柱子後 agent 花了 26 步追一顆在地上的球；手指最低在離地 0.136 m，地上的球夾不起來
        return False, (f"pick({obj}) {where}: the gripper closed twice but the ball did NOT come along ({'; '.join(attempts)}). "
                       f"Balls are only graspable while they sit on their posts; if it fell off onto the floor (or off the table) "
                       f"this gripper cannot reach it (the fingers stay at least 0.136 m above the surface the arm is lowered to). "
                       f"Check with locate_overhead whether it is still at its post's position: if it moved, do not chase it — "
                       f"report the failure honestly.{lifted}")
    return False, (f"pick({obj}) {where}: the gripper closed twice but the object did NOT come along ({'; '.join(attempts)}); "
                   f"it was not carried along (it may have rolled or been pushed away). Locate it again and retry pick({obj}).{lifted}")


# ---- 從高處頂面抓取（2026-11-13，使用者：「拿箱子上面的小瓶子」）----
# 舊 pick 靠底盤高度的前向 ToF 把東西送進指間；箱頂的瓶子 ToF 只看得到箱子正面，所以改用「夾爪上的相機 + 手臂幾何」：
# 手臂抬到斜視姿態 → 視覺模型在畫面指出瓶子 → 把那個像素的視線射到「瓶頂高度」的水平面算出它在底座座標的位置 →
# 夾爪朝下懸停在它上方再看一次修正 → 下降、合爪（NexArm 夾爪伺服卡住 = 夾到）→ 抬起 → 收到搬運姿態。
# 幾何（實測/圖面）：NexArm 底座平面離地 0.162 m、在車身正中；夾爪朝下時前伸 ≤ 300 mm 才搆得到 z 180（指尖離地 0.34 m）；
# EP 相機裝在夾爪下緣，指尖後方約 7 cm、軸線下側約 4 cm（估計值，第一次實測後再修）。
NEX_BASE_H_M = 0.162
NEX_CAM_BACK_M = 0.07
NEX_CAM_UNDER_M = 0.04
NEX_CARRY = (150, 250)     # NexArm 的搬運/準備姿態（HOME）
# 夾爪水平（pitch 0 / −20 查的是同一張表）時「前伸 r mm 最低到得了的高度 z mm」，取自 nexarm_workspace.json（REACH.z_span）。
# 越低要伸越遠：前伸 150 最低 200，伸到 325 才到得了 30。2026-11-14 使用者把箱子放倒（站著 0.35 m 抓的時候相機被短排線扯掉），
# 矮箱子的夾取高度常低於 150，要用這張表挑得到的前伸距離。
_NEX_HORIZ_MIN_Z = ((150, 200), (175, 190), (200, 180), (225, 160), (250, 140), (275, 110), (300, 70), (325, 30))


# EP 相機掛在夾爪下面：相機殼最低點在夾爪中心（arm_to 的高度）下方多少。2026-11-14 實測推回來的估計：夾爪中心在
# 箱頂上方 6–7 cm 往前伸時，相機正對箱子側面的條碼、把 0.16 m 的紙箱往前推（runs/_skill_frames/20261114_132839_*），
# 0.35 m 那次「抓取時相機掉下來」也是相機殼撞到箱子前緣。夾取/放置高度一律讓相機下緣離箱頂 ≥2 cm。量到實際值用 env 改。
# 2026-11-14 下午：相機改裝到手臂上節，夾爪下面只剩夾爪本體（約 3 cm）。相機又裝回夾爪下面的話設 0.075。
NEX_CAM_BOTTOM_M = float(os.environ.get("ROBOT_NEX_CAM_BOTTOM_M", "0.03"))


async def _creep_to_surface(robot, front_tof, notes, what, target_mm=110, max_step=0.20):
    """底盤用前向 ToF 貼到箱子前 ~11 cm。只貼一次、最多 20 cm；開完 ToF 沒跟著變小 = 箱子被推走了（或感測器沒在看它），
    回 (tof, 錯誤訊息)。2026-11-14：舊版每次呼叫都 push=True 開 (tof-110)，箱子被相機推走後 ToF 變大，三次技能一路把箱子推了 40 cm。"""
    tof = front_tof(await robot.state())
    # 15:36 agent 實跑：approach 回報的 tof_front_mm 是 min(模擬, 實體)=214，實體其實 370，舊門檻 350 直接拒絕了四次。放寬到 600，分兩段貼近
    if tof is None or tof == 9999 or tof > 700:     # 700：_drive_to_visible 的「到了」門檻是 600，讀值抖一下（o_h2 讀到 614）不該整個拒絕
        return tof, (f"{what}: the REAL front distance sensor reads {tof} mm, so the box is not right in front of the robot "
                     f"(tof_front_mm in the telemetry can be the simulated value: look at real_tof_front_mm). Turn to face the box "
                     f"(face('<the box>')) and call this again; do not use pick() instead.")
    for _ in range(3):
        if tof <= target_mm + 40:      # 底盤走不出 4 cm 以下的小步（實測送 0.02 m 完全沒動）
            return tof, None
        step = min(max_step, (tof - target_mm) / 1000.0)
        # 17:30 使用者：「你直接撞到箱子了，你不是有防撞的 sensor 嗎？」——舊版這裡送 push=True（關掉伺服器的 ToF 防撞）一次衝 0.30 m
        # 到離箱子 6 cm，煞不住就撞上。現在一律不送 push：伺服器會自己停在障礙前 0.15 m，剩下的距離由手臂伸過去。
        r = await robot.act("move_chassis", {"forward_m": round(step, 2), "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
        await asyncio.sleep(0.4)
        new = front_tof(await robot.state())
        notes.append(f"crept {step:.2f} m closer ({'ok' if r.get('ok') else 'failed'}), front sensor {tof} -> {new} mm")
        if new is None or new == 9999 or new > tof - 0.4 * step * 1000:
            return new, (f"{what}: drove {step:.2f} m toward the surface but the front distance only went {tof} -> {new} mm: "
                         f"the box is probably sliding away (being pushed) or the sensor is not looking at it. Stopped; "
                         f"check the scene before trying again.")
        tof = new
    return tof, None



async def _drive_to_visible(robot, llm, label, front_tof, notes, low_view=False, carrying=False):
    """看著目標開過去（不靠地圖）：相機找到目標 → 轉到正對 → 往前一段 → 重複，直到實體 ToF ≤ 600 mm 或目標已在畫面最下緣。
    2026-11-14：箱子在 0.8 m 以外時 ToF 讀值在 0.8–1.4 m/9999 之間亂跳，地圖上的方塊放了又拆，approach 拿到錯的位置（1.74 m）；
    相機反而一直看得到棒子。回 None = 到了；回字串 = 失敗原因。"""
    if carrying:      # 夾著東西：不能回看路姿勢（夾爪會前傾 26°，東西歪掉），用夾爪水平、相機朝前下方的姿勢邊看邊開
        await robot.act("arm_to", {"x_mm": 380, "y_mm": 120, "pitch_deg": 0.0, "yaw_deg": 0.0})
    elif low_view:    # 地上的東西：看路姿勢（平視）在 1 m 外只拍得到它的頂端或完全拍不到，改用往下看約 25° 的姿勢找
        await robot.act("arm_to", {"x_mm": 250, "y_mm": 150, "pitch_deg": 0.0, "yaw_deg": 0.0})
    else:
        await robot.act("recenter_arm", {})
    backed = False
    turns_in_row = 0
    for i in range(10):
        await asyncio.sleep(0.6)
        await robot.frame()
        jpg = await robot.frame()
        _save_frame(jpg, f"drive_to_{i+1}")
        f = await perception.locate(llm, jpg, label)
        if not f.found and i == 0:
            # 起跑時看不到（使用者把箱子換了位置）：原地每次轉 45° 找一圈
            for k in range(7):
                await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": 45.0, "push": False, "near_ok": True})
                await asyncio.sleep(0.6)
                await robot.frame()
                jpg = await robot.frame()
                _save_frame(jpg, f"search_{k+1}")
                f = await perception.locate(llm, jpg, label)
                if f.found:
                    notes.append(f"not in view at first: found it after turning {45 * (k + 1)} deg")
                    break
        if not f.found and i > 0 and carrying and not backed:
            # 夾著東西對箱子時轉完就看不到：多半是離箱子太近（箱子在畫面下緣外）或轉過頭。先直直退 0.4 m 拉開距離再看（只做一次）
            backed = True
            await robot.act("move_chassis", {"forward_m": -0.4, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
            await asyncio.sleep(0.6)
            await robot.frame()
            jpg = await robot.frame()
            f = await perception.locate(llm, jpg, label)
        if not f.found and i > 0:
            # 走到一半模型偶爾漏看（11-15 13:56：橘色棒子明明在畫面中間，第三張回「沒看到」）：換個說法、重拍再問兩次
            for alt in (f"{label} standing on the floor" if low_view else f"{label} on the box", label):
                await asyncio.sleep(0.5)
                jpg = await robot.frame()
                f = await perception.locate(llm, jpg, alt)
                if f.found:
                    break
        if not f.found:
            return (f"cannot see '{label}' with the front camera from here (after {i} moves toward it"
                    f"{', and a full turn in place' if i == 0 else ''}). Drive somewhere it can be seen from, then call this again.")
        # 2026-11-16：夾著東西時原本容許 ±3° 而且每轉一次就重看，模型給的方位本身會抖幾度，11-15 最後一趟十圈全花在轉頭、一步沒前進。
        # 改成 ±5°，而且最多連轉兩次就一定往前走一步（走近之後方位誤差對應的橫向偏差也會變小）。
        bearing = f.bearing_right_deg
        if carrying:
            # 2026-11-16 o_h2（使用者：「會一直撞到箱子」「放太邊了所以掉下去」）：模型只回箱子上的「一個點」，箱子在 0.5 m 外就佔 26° 寬，
            # 點落在左半邊還是右半邊每次不同 → 車頭左轉 20°、右轉 16° 來回擺，最後歪著貼上去（存下來的畫面：箱子中心在夾著的棒子右邊 12°）。
            # 改用影像把「整個箱子」找出來（同色區域，中間被夾著的東西擋住也接得起來），對它的水平中心；
            # 而且要對到「夾著的東西」所在的畫面位置（夾爪在畫面 x≈553，不是正中央 500）。
            ext = perception.surface_extent(jpg, f.x, f.y)
            if ext is not None:
                bearing = perception.bearing_from_x((ext[0] + ext[2]) / 2.0) - perception.bearing_from_x(NEX_GAP_REF[0])
                if perception.LOCK_HOOK is not None:
                    perception.LOCK_HOOK(ext, label)
            _drive_to_visible.last_bearing = (round(bearing, 1), ext is not None)
        if abs(bearing) > (5.0 if carrying else 4.0):
            await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(-bearing, 1), "push": False, "near_ok": True})
            await asyncio.sleep(0.4)
            turns_in_row += 1
            _drive_to_visible.last_bearing = None      # 轉過了：上面量到的方位作廢
            if carrying and turns_in_row < 2:
                continue                       # 夾著東西對箱子：轉完重看一次確認置中；連轉兩次還沒中就先往前
        turns_in_row = 0
        tof = front_tof(await robot.state())
        if isinstance(tof, (int, float)) and tof != 9999 and tof <= 600:
            lb = getattr(_drive_to_visible, "last_bearing", None)
            _drive_to_visible.last = {"bearing": lb[0], "tof": tof} if (carrying and lb and lb[1]) else None
            notes.append(f"drove toward the {label} by camera in {i} moves, front sensor now {tof} mm"
                         + (f", box centre {lb[0]:+.1f} deg from the held object" if (carrying and lb and lb[1]) else ""))
            return None
        # 不再用「目標在畫面多低」判斷太近：模型常把物件指在底部（y≥960），從 1 m 外就誤判，18:29 / 18:37 兩趟各白白放棄 2–3 次。
        # 只看 ToF：每次都先把車頭對正目標，箱子進到 0.8 m 內 ToF 就讀得到；真的太近目標會掉出畫面，下一圈回報看不到。
        near = isinstance(tof, (int, float)) and tof != 9999 and tof < 1000
        step = min(0.40, (tof - 350) / 1000.0) if near else 0.20
        await robot.act("move_chassis", {"forward_m": round(max(0.08, step), 2), "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
    return f"still not close to the {label} after 10 moves toward it."


def _nex_depth_from_base_y(reach_mm, z_mm, y1000, surface_m):
    """觀察姿勢（腕 90°、夾爪水平）下，物件「底部」在畫面的 y（0–1000）→ 物件中心離手臂底座軸的水平距離 m。
    相機在上臂：位置用 FK 算，俯仰 = 上臂仰角 − 74°（2026-11-14 17:26 成功那趟校正：底部 y=940 ↔ 距離 0.365 m、箱頂 0.25 m）。
    畫面每 40/1000 約 2 cm；y ≥ 990（貼著下緣）或視線沒往下就回 None。"""
    if y1000 is None or y1000 >= 990:
        return None
    k = _NEX_FK
    wx, wz = reach_mm - k["ax"] - k["L3"], z_mm - k["az"]
    d = min(math.hypot(wx, wz), k["L1"] + k["L2"] - 1e-6)
    p1 = math.atan2(wz, wx) + math.acos(max(-1.0, min(1.0, (k["L1"] ** 2 + d * d - k["L2"] ** 2) / (2 * k["L1"] * d))))
    cx = (k["ax"] + k["L1"] * math.cos(p1)) / 1000.0 + 0.03 * math.cos(p1)
    ch = NEX_BASE_H_M + (k["az"] + k["L1"] * math.sin(p1)) / 1000.0 + 0.03 * math.sin(p1)
    ty = math.tan(math.radians(perception.CAMERA_HFOV_DEG) / 2.0) * 9.0 / 16.0
    dep = -(math.degrees(p1) + NEX_LOOK_TILT_DEG) + math.degrees(math.atan((y1000 / 1000.0 - 0.5) * 2.0 * ty))
    if dep < 8.0 or ch <= surface_m:
        return None
    return cx + (ch - surface_m) / math.tan(math.radians(dep))


NEX_LOOK_TILT_DEG = float(os.environ.get("ROBOT_NEX_LOOK_TILT_DEG", "-76.5"))
# 相機固定座會鬆動（11-15 一小時內同一姿勢下夾爪在畫面的 y 從 429 → 512 → 790，等於俯仰變了 24°），所以畫面上的對位目標
# 一律用「相對於夾爪開口」的偏移，每次現量夾爪開口在哪。夾爪 (380,120) 與 0.73 m 外的地面點從相機看過去角度只差 3°，
# 所以這個偏移幾乎不受相機俯仰影響。基準（11-15 校正）：夾爪開口 (553,512)，白點 (471,415) = 車中心前方 0.73 m、左 0.10 m。
NEX_GAP_REF = (553.0, 512.0)
NEX_FLOOR_AHEAD_M = 0.87              # 目標對到「夾爪開口正上方 97 單位」時它在車中心正前方這麼遠（11-15 實測兩次：0.73 差 6 cm、0.79 還差 7–8 cm）
NEX_FLOOR_DY = -97.0


def _gripper_gap_cv(jpg):
    """影像備援：俯視姿勢下夾爪是畫面中下方最大的幾塊黑色區域裡「最上面」那塊（手臂本體在它下面），指尖上緣往下 25 單位當開口。"""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    m = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 50).astype(np.uint8)
    m[: int(h * 0.30)] = 0
    m[:, : int(w * 0.25)] = 0
    m[:, int(w * 0.85):] = 0
    n, lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for k in range(1, n):
        bottom = st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT]
        cx = st[k, cv2.CC_STAT_LEFT] + st[k, cv2.CC_STAT_WIDTH] / 2.0
        if (st[k, cv2.CC_STAT_AREA] >= 1200 and bottom >= 0.80 * h and 0.40 * w <= cx <= 0.72 * w
                and (best is None or st[k, cv2.CC_STAT_TOP] < st[best, cv2.CC_STAT_TOP])):
            best = k
    if best is None:
        return None
    top = st[best, cv2.CC_STAT_TOP]
    ys, xs = np.nonzero(lab[top: top + max(8, h // 25)] == best)
    if len(xs) == 0:
        return None
    return float((xs.min() + xs.max()) / 2.0 / w * 1000.0) - 15.0, float(top / h * 1000.0 - 45.0)     # 偏移對齊模型指的「開口」位置（實測差 +70/+15）


async def _gripper_gap(robot, llm):
    """俯視姿勢 (380,120) 下夾爪開口在畫面的 (x, y)/1000；量不到回 None。先問模型，問不到用影像找黑色夾爪。"""
    jpg = None
    for _ in range(2):
        await asyncio.sleep(0.5)
        await robot.frame()
        jpg = await robot.frame()
        g = await perception.locate(llm, jpg, "the gap between the two black gripper fingers")
        if g.found and g.y > 150:
            return float(g.x), float(g.y)
    return _gripper_gap_cv(jpg) if jpg else None


def _hue_ref(jpg, x1000, y1000):
    """模型指到的那一點附近的顏色（HSV 中位數）。太灰（飽和度 < 90）回 None：灰色的東西沒辦法靠顏色跟。"""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    cx, cy = int(x1000 / 1000.0 * w), int(y1000 / 1000.0 * h)
    patch = cv2.cvtColor(img[max(0, cy - 6): cy + 7, max(0, cx - 6): cx + 7], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    if len(patch) == 0:
        return None
    hh, ss, vv = (float(np.median(patch[:, k])) for k in range(3))
    return (hh, ss, vv) if ss >= 90 and vv >= 60 else None


def _color_base(jpg, ref, near=None):
    """用顏色找物件：跟 ref 同色相（±9）、夠鮮豔的最大一塊，回它「最下緣中點」(x, y)/1000 = 立在地上的物件的底部。
    視覺模型指同一個位置可以差 ±100 單位（11-15 邊看邊往前因此左右擺、漏看一次就開過頭），顏色分割每張都一樣而且不用等模型。"""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    dh = np.abs(hsv[:, :, 0] - int(ref[0]))
    dh = np.minimum(dh, 180 - dh)
    # 飽和度門檻要緊：橘色棒子 (H21,S247) 跟紙箱 (H23,S185) 色相幾乎一樣，只差在鮮豔程度。舊門檻 ref×0.55 把紙箱也算進來，
    # 11-15 18:0x 整趟追到箱子上去（讀值一直 −300~−500、+600~+700 不變）。改成比基準低不超過 35。
    m = ((dh <= 9) & (hsv[:, :, 1] >= max(80, ref[1] - 35)) & (hsv[:, :, 2] >= max(70, ref[2] - 75))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    best, best_key = None, None
    for k in range(1, n):
        a = st[k, cv2.CC_STAT_AREA]
        if a < 120:
            continue
        key = -a
        if near is not None:
            key = math.hypot(cen[k][0] / w * 1000 - near[0], (st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT]) / h * 1000 - near[1]) - a * 0.02
        if best is None or key < best_key:
            best, best_key = k, key
    if best is None:
        return None
    bot = st[best, cv2.CC_STAT_TOP] + st[best, cv2.CC_STAT_HEIGHT] - 1
    ys, xs = np.nonzero(lab[max(0, bot - 5): bot + 1] == best)
    _color_base.last_top = float(st[best, cv2.CC_STAT_TOP] / h * 1000.0)       # 這一塊的最上緣（物件頂端）
    _color_base.last_wh = (int(st[best, cv2.CC_STAT_WIDTH]), int(st[best, cv2.CC_STAT_HEIGHT]))   # 像素寬高，給「是不是直立的細長物」檢查用
    return float(xs.mean() / w * 1000.0), float(bot / h * 1000.0)


async def _close_in_by_sight(robot, llm, label, arm, r0, z_mm, gp, yaw, notes):
    """夾取高度下「邊看邊往前」（使用者 11-15：「他可以對準之後再往前吧」「相機沒有能夠非常固定的位置」）。
    手臂停在夾取姿勢 (r0, z)，同一張畫面裡量夾爪指尖那條線和物件底部：底部還在指尖線上方 = 還沒到，車往前一小段再看；
    左右偏了用底盤橫移修（不轉底座：轉了畫面整個變，11-15 轉完模型讀值亂跳把棒子弄丟）。底部降到指尖線（或被夾爪擋住）就停。
    模型指的位置會亂跳 → 每一步連看兩張，兩次差太多就再看第三張取中間值。只比同一張畫面裡的兩樣東西，相機歪了也不影響。
    回 (yaw, None) = 到指尖前了；(yaw, 原因字串) = 失敗。"""
    what = f"the bottom end of the {label}, where it touches the floor"

    color = {"ref": None, "last": None}

    async def look():
        """回 (gap, (x, y) 或 None)。有顏色基準就用顏色找（穩、快）；還沒有就問模型，問到的同時取它的顏色當基準。"""
        pts, gap = [], None
        if color["ref"] is not None:
            await asyncio.sleep(0.5)
            await robot.frame()
            jpg = await robot.frame()
            _save_frame(jpg, "closein")
            gap = _gripper_gap_cv(jpg)
            cb = _color_base(jpg, color["ref"], color["last"])
            if cb is not None:
                color["last"] = cb
            return gap, cb
        for k in range(3):
            await asyncio.sleep(0.5)
            await robot.frame()
            jpg = await robot.frame()
            if k == 0:
                _save_frame(jpg, "closein")
            g = _gripper_gap_cv(jpg)
            if g is not None:
                gap = g
            if color["ref"] is None:
                fo = await perception.locate(llm, jpg, label)          # 指物件本體（中間），取那裡的顏色
                if fo.found:
                    ref = _hue_ref(jpg, fo.x, fo.y)
                    cb = _color_base(jpg, ref, (fo.x, fo.y)) if ref else None
                    if cb is not None:
                        # 模型第一眼可能指到別的同色東西（紙箱跟橘色棒子色相一樣）。立在地上的東西影像是直立的：高要明顯大於寬，
                        # 而且不能是一大片；不像就不採用這個顏色基準，下一張再問。
                        bw, bh = getattr(_color_base, "last_wh", (0, 0))
                        if not (bh >= 1.25 * bw and bw <= 140):
                            notes.append(f"colour blob at the model's point is {bw}x{bh} px, not an upright object: not tracking that")
                            continue
                        color["ref"], color["last"] = ref, cb
                        notes.append(f"tracking the {label} by colour (hue {ref[0]:.0f})")
                        return gap, cb
            f = await perception.locate(llm, jpg, what)
            if not f.found:
                f = await perception.locate(llm, jpg, f"{label} standing on the floor")
            if f.found:
                pts.append((float(f.x), float(f.y)))
            if len(pts) == 2 and abs(pts[0][0] - pts[1][0]) < 60 and abs(pts[0][1] - pts[1][1]) < 60:
                break
            if k == 1 and not pts:
                break
        if not pts:
            return gap, None
        xs, ys = sorted(p[0] for p in pts), sorted(p[1] for p in pts)
        return gap, (xs[len(xs) // 2], ys[len(ys) // 2])

    frac_stop = 0.45        # 指尖線到物件影像高度的 45% 就停（碰到時約 0.8）：差的那一小段交給後面每次 1.5 cm 的試夾

    async def chassis(fwd, right):
        """使用者 11-15：「你還是會不小心把他推倒」「你撞到箱子了」——手臂降在地面附近時底盤一動（尤其小步修正的來回晃）夾爪就掃到東西，
        而且那個姿勢 ToF 被夾爪擋住、防撞是關的。規則：手臂降下去之後底盤不准動。要動底盤就先把手臂抬到 150（ToF 看得到前面、防撞開著），
        動完再降回夾取高度。"""
        await arm(r0, 150, 0.0, yaw)
        await _move_exact(robot, fwd, right, tol=0.02)
        await arm(r0, z_mm, gp, yaw)

    seen_close, log, blind = False, [], 0
    was_near, yaw0, side_fix = False, float(yaw), 0
    last_dy, jumps = None, 0
    r_cur = float(r0)
    NEAR = 230.0        # 底部離指尖線不到這麼多（畫面單位）= 很接近了：底盤不准再動，只動手臂
    for i in range(26):
        gap, pt = await look()
        if gap is None:
            return yaw, f"cannot see the robot's own gripper at the grasp pose (camera pointing away): {' '.join(log)}", r_cur
        tip_y = gap[1] + 45.0                      # 影像法回的是指尖上緣 − 45
        if pt is None:
            # 曾經很接近過（上一個可信讀值在 NEAR 內）之後看不到 = 被夾爪擋住了，當作到了。絕對不能再讓底盤往前
            # （11-15：到了 +57 之後一次漏看就觸發「看不到往前 12 cm」，整台車開過頭把東西撞開）
            if seen_close or (last_dy is not None and last_dy <= NEAR):
                log.append("hidden behind the gripper")
                break
            if blind < 6:                          # 這個姿勢只看得到車前約半公尺：還太遠就每次往前 12 cm 再看，最多 6 次
                blind += 1
                await chassis(0.12, 0.0)
                r_cur = float(r0)
                log.append("not seen, +0.12")
                continue
            return yaw, f"cannot see the {label} from the grasp pose ({' '.join(log)})", r_cur
        ex, dy = pt[0] - gap[0], tip_y - pt[1]     # dy > 0：底部在指尖線上方（還在前面）
        if last_dy is not None and last_dy <= NEAR and dy > last_dy + 120 and jumps < 2:
            jumps += 1                             # 已經很近了卻突然「變遠」一大截 = 模型指錯地方，這張不算，重看
            log.append(f"[ignored jump {dy:+.0f}]")
            continue
        if was_near and dy > NEAR:
            # 2026-11-16 o_h2（使用者：「明明就已經碰到他了怎麼沒有抓，然後還換位置把棍子弄到了」）：棒子已經進到兩指之間（存下來的畫面看得很清楚），
            # 它的下半截在夾爪的陰影裡、顏色變暗，顏色追蹤的「底部」一下跳到上面去 → 讀值從 +31 變成 +236 → 被當成「還很遠」→ 底盤往前開，把棒子撞倒。
            # 曾經很近過之後又「變遠」= 被夾爪擋住/遮暗了，東西就在指間：停，交給合爪。接近之後底盤絕對不再動。
            log.append(f"({ex:+.0f},{dy:+.0f}) reads far again after being close: it is between the fingers (hidden/shaded by the gripper)")
            break
        last_dy = dy
        log.append(f"({ex:+.0f},{dy:+.0f})")
        near = dy <= NEAR
        was_near = was_near or near
        # 「到了沒」改用比例判斷（11-15：只看底部會提早/太晚，因為底部在地上、指尖在離地 12 cm，兩者的畫面關係隨相機俯仰變）：
        # 指尖線落在物件影像「從底往上幾成」的位置 = frac。指尖真的碰到物件時 frac ≈ 夾點高度 ÷ 物件高度（跟相機角度無關）。
        frac = None
        if color["ref"] is not None and getattr(_color_base, "last_top", None) is not None:
            top_y = _color_base.last_top
            # 物件頂端頂到畫面上緣（相機離得很近、或相機角度變了）時「頂端」是被畫面切掉的假值，比例會被壓得很小、永遠到不了門檻：
            # 這時不用比例，退回「底部到指尖線就停」。11-16：頂端 y=0，比例算出 −0.05，棒子明明在指間還繼續伸。
            if pt[1] - top_y > 40 and top_y > 8:
                frac = (pt[1] - tip_y) / (pt[1] - top_y)
                log[-1] = log[-1][:-1] + f",f={frac:+.2f})"
        if frac is not None and abs(ex) <= 40 and frac >= frac_stop:
            break
        if abs(ex) > 25:
            if near:
                # 使用者 11-15：「很接近的時候不能讓機器人往前，這樣會把東西弄到，需要讓機械手臂往前」——夾爪已經降在車頭前面，
                # 底盤一動（前進或橫移）夾爪就掃到東西，兩次都把棍子掃倒。接近之後左右用手臂底座小角度轉。
                # 接近之後底座最多再轉 ±8°：11-16 棒子已在指間、追蹤點亂跳時連轉三次共 15°，整隻手臂橫掃把棒子掃倒
                new_yaw = max(yaw0 - 8.0, min(yaw0 + 8.0, yaw - max(-5.0, min(5.0, ex * 0.04))))
                if abs(new_yaw - yaw) < 0.5 and abs(ex) > 60 and side_fix < 2:
                    # 底座已經轉到上限、還偏很多（11-16 第三次夾：(-133,-108) 重複 12 圈）：手臂抬起來，底盤橫移，再降下來重新靠近
                    side_fix += 1
                    yaw = yaw0
                    await chassis(0.0, max(-0.08, min(0.08, 0.6 * ex * 0.0006)))
                    r_cur, was_near, last_dy = float(r0), False, None
                    log.append("sideways off with the base at its limit: lifted the arm, slid the chassis, closing in again")
                    continue
                yaw = new_yaw
                await arm(r_cur, z_mm, gp, yaw)
            else:                                  # 還遠：底盤橫移（物件在夾爪右邊 → 車往右）
                # 11-15：舊增益 (0.5+dy×0.0012) 讓左右來回擺（−208→+194→…），18 次都耗在這。減半，而且偏不多時同一圈接著往前
                mm_per_unit = 0.3 + max(0.0, dy) * 0.0005
                lat = max(-0.10, min(0.10, 0.7 * ex * mm_per_unit / 1000.0))
                # 2026-11-16 o_h1：偏 33 單位算出來只要橫移 1.3 cm，比 _move_exact 的容許誤差 (2 cm) 還小 → 底盤完全沒動，
                # 卻照樣 continue 重看，18 圈讀值一模一樣（-33,+561）一步都沒前進。橫移不到 3 cm 就不橫移，同一圈直接往前；
                # 剩下的一點左右偏差等靠近之後用手臂底座轉。
                if abs(lat) >= 0.03:
                    await chassis(0.0, lat)
                    r_cur = float(r0)
                    continue
            if near:
                continue
        if frac is None and dy <= 35:
            break
        seen_close = dy <= 160
        if near:
            if r_cur >= 438.0:                     # 手臂伸到底了還沒到線：交給後面的「縮回手臂、車往前 8 cm 再伸」
                log.append("arm at full reach")
                break
            r_cur = min(440.0, r_cur + (max(10.0, min(30.0, dy * 0.25)) if frac is None else (25.0 if frac < 0.0 else 12.0)))
            await arm(r_cur, z_mm, gp, yaw)
        else:                                      # 還遠才動底盤，而且保守：每步最多 10 cm，寧可多看幾次
            await chassis(max(0.05, min(0.10, dy * 0.0003)), 0.0)
            r_cur = float(r0)
    else:
        return yaw, f"could not get the {label} to the fingertips in 18 looks ({' '.join(log)})", r_cur
    notes.append(f"closed in by sight at grasp height {' '.join(log)}; arm now at reach {r_cur:.0f} mm")
    return yaw, None, r_cur


async def _servo_to_floor_object(robot, llm, label, notes):
    """地上的細東西 ToF 掃不到 → 跟回家看白點同一招：相機轉到俯視姿勢 (380,120)，把物件底部對到「夾爪開口正上方」的固定偏移
    （= 車中心正前方 0.73 m），再用里程計往前走到物件在手臂底座前 0.42 m。回 None = 到位；字串 = 失敗原因。"""
    # 先找到它、把車頭轉正；看不到就原地每 45° 找一圈。用俯視姿勢 (380,120) 找：相機固定不住常常朝上，
    # 看路姿勢會整個看天花板（11-15 兩次「轉一圈都找不到」就是這樣），這個姿勢朝下最多、看得到前方 0.4–2 m 的地板
    await robot.act("arm_to", {"x_mm": 380, "y_mm": 120, "pitch_deg": 0.0, "yaw_deg": 0.0})
    seen = False
    for k in range(9):
        await asyncio.sleep(0.7)
        await robot.frame()
        jpg = await robot.frame()
        _save_frame(jpg, f"floor_face_{k+1}")
        f = await perception.locate(llm, jpg, f"{label} standing on the floor")
        if f.found:
            seen = True
            if abs(f.bearing_right_deg) <= 5.0:
                break
            await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(-f.bearing_right_deg, 1), "push": False, "near_ok": True})
        elif not seen and k == 0:
            # 平視第一眼就看不到：多半是太近（15 cm 高的東西在 0.5 m 內掉在畫面下緣外）→ 先直直退 0.5 m 再看，不要馬上轉走
            await robot.act("move_chassis", {"forward_m": -0.5, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
        elif not seen:
            await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": 45.0, "push": False, "near_ok": True})
        else:
            break
    if not seen:
        return f"cannot see the {label} on the floor anywhere around (turned a full circle)."
    await robot.act("arm_to", {"x_mm": 380, "y_mm": 120, "pitch_deg": 0.0, "yaw_deg": 0.0})
    gap = await _gripper_gap(robot, llm)
    if gap is None or gap[1] > 800:
        # 這個姿勢看不到（或只看到一點）夾爪：粗對位做不了也沒關係，車頭已經對著它，剩下交給夾取高度的「邊看邊往前」
        # 改成：一邊把它保持在畫面正中、一邊往前，直到它降到畫面下緣附近（= 已經在車前半公尺左右）
        steps = 0
        for _ in range(9):
            await asyncio.sleep(0.7)
            await robot.frame()
            jpg = await robot.frame()
            _save_frame(jpg, "floor_drive")
            f = await perception.locate(llm, jpg, f"the bottom end of the {label}, where it touches the floor")
            if not f.found:
                f = await perception.locate(llm, jpg, f"{label} standing on the floor")
            if not f.found or f.y >= 820:
                break
            if abs(f.bearing_right_deg) > 5.0:
                await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(-f.bearing_right_deg, 1), "push": False, "near_ok": True})
            await _move_exact(robot, 0.25 if f.y < 650 else 0.12, 0.0, tol=0.03)
            steps += 1
        notes.append(f"gripper not in view at the look pose: drove toward it by bearing in {steps} steps")
        return None
    tx, ty = gap[0], gap[1] + NEX_FLOOR_DY
    what = f"the bottom end of the {label}, where it touches the floor"
    log = []
    for i in range(12):
        await asyncio.sleep(0.8)
        await robot.frame()
        jpg = await robot.frame()
        _save_frame(jpg, f"floor_servo_{i+1}")
        f = await perception.locate(llm, jpg, what)
        if not f.found:
            f = await perception.locate(llm, jpg, f"{label} standing on the floor")
        if not f.found:
            # 俯視看不到：車頭已經對著它，不是太遠（在畫面上緣外）就是太近（被夾爪擋住）。前幾次當成太遠往前 0.3 m，之後退 0.4 m
            for step in ((0.3, 0.3, -0.4) if i == 0 else (-0.25,)):
                await robot.act("move_chassis", {"forward_m": step, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
                await asyncio.sleep(0.8)
                await robot.frame()
                jpg = await robot.frame()
                f = await perception.locate(llm, jpg, what)
                if not f.found:
                    f = await perception.locate(llm, jpg, f"{label} standing on the floor")
                if f.found:
                    break
        if not f.found:
            return f"cannot see the {label} on the floor with the camera looking down (after {i} corrections: {' '.join(log)})."
        ex, ey = f.x - tx, f.y - ty
        log.append(f"({ex:+.0f},{ey:+.0f})")
        if abs(ex) <= 30 and abs(ey) <= 30:
            notes.append(f"lined up on the {label} on the floor by camera (gripper gap at {gap[0]:.0f},{gap[1]:.0f}) in {i} corrections {' '.join(log)}")
            await robot.act("arm_to", {"x_mm": 250, "y_mm": 200, "pitch_deg": 0.0, "yaw_deg": 0.0})
            await _move_exact(robot, max(0.0, NEX_FLOOR_AHEAD_M - 0.62), 0.0)     # 只先靠到約 0.6 m，剩下的在夾取高度邊看邊走
            return None
        if abs(ex) > 250:                          # 偏很多：先轉向它（畫面 1000 單位 ≈ 100°）
            await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(-ex * 0.09, 1), "push": False, "near_ok": True})
            continue
        fwd = max(-0.20, min(0.30, -0.9 * ey * 0.0015))        # 物件在畫面上方（遠）→ 往前
        right = max(-0.20, min(0.20, 0.9 * ex * 0.0015))
        for key, v in (("forward_m", fwd), ("right_m", right)):
            if abs(v) < 0.012:
                continue
            if abs(v) < 0.09:
                back = -0.12 if v > 0 else 0.12
                await robot.act("move_chassis", {"forward_m": back if key == "forward_m" else 0, "right_m": back if key == "right_m" else 0,
                                                 "turn_left_deg": 0, "push": False, "near_ok": True})
                v -= back
            await robot.act("move_chassis", {"forward_m": round(v, 3) if key == "forward_m" else 0,
                                             "right_m": round(v, 3) if key == "right_m" else 0, "turn_left_deg": 0, "push": False, "near_ok": True})
    return f"could not line up on the {label} on the floor after 12 corrections ({' '.join(log)})."


_NEX_FK = {"ax": -0.12, "az": 110.35, "L1": 227.45, "L2": 145.31, "L3": 129.79}      # Test_Robot/nexarm_workspace.py 擬合的平面 FK（mm）


def _nex_r_wrist90(z_mm):
    """夾爪水平、前臂垂直往下（腕關節 90°，使用者 2026-11-14 貼的尺寸圖右邊那個姿勢）時的前伸距離 mm。
    肘在腕的正上方：(r − L3)² + (z + L2 − az)² = L1²。"""
    k = _NEX_FK
    h = z_mm + k["L2"] - k["az"]
    return k["ax"] + k["L3"] + math.sqrt(max(0.0, k["L1"] ** 2 - h * h))


_TOP_GRIP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "_last_top_grip.json")
# pick_from_top 夾到東西時記下「夾爪中心在物件底部上方多高」，place_on_top 用它算放下時瓶底離箱頂多少（2026-11-14）
_LAST_TOP_GRIP = {}


def _nex_min_z(reach_mm):
    """水平夾爪在這個前伸距離最低到得了多高（mm，多留 10 mm 餘裕）。"""
    t = _NEX_HORIZ_MIN_Z
    if reach_mm <= t[0][0]:
        return t[0][1] + 10.0
    for (r0, z0), (r1, z1) in zip(t, t[1:]):
        if reach_mm <= r1:
            return z0 + (z1 - z0) * (reach_mm - r0) / float(r1 - r0) + 10.0
    return t[-1][1] + 10.0


def _nex_min_r(z_mm):
    """水平夾爪要降到 z_mm 至少得伸多遠（mm）。"""
    t = _NEX_HORIZ_MIN_Z
    z_mm = z_mm - 10.0
    if z_mm >= t[0][1]:
        return float(t[0][0])
    for (r0, z0), (r1, z1) in zip(t, t[1:]):
        if z_mm >= z1:
            return r0 + (r1 - r0) * (z0 - z_mm) / float(z0 - z1)
    return float(t[-1][0])


def _cam_ray_to_plane(reach_m, z_m, yaw_deg, pitch_deg, u, v, plane_h_m):
    """夾爪相機的一個像素（u 右為正、v 上為正，−1..1）射到離地 plane_h 的水平面，回傳 (前, 左) 相對底座的水平座標（m）；
    射線不往下或相機已低於那個面就回 None。座標：前 = 底座朝向、上 = 向上；夾爪軸 g = (cos θ, sin θ)，θ = pitch（朝下 −90）。"""
    th = math.radians(pitch_deg)
    g = (math.cos(th), math.sin(th))
    under = (math.sin(th), -math.cos(th))        # 夾爪「下側」：pitch 0 → (0, −1)；−90 → (−1, 0)（相機跑到指尖後方）
    upv = (-math.sin(th), math.cos(th))          # 影像的「上」= 夾爪的上側：朝下看時影像上方 = 車頭方向
    tcp = (reach_m, NEX_BASE_H_M + z_m)
    cam = (tcp[0] - NEX_CAM_BACK_M * g[0] + NEX_CAM_UNDER_M * under[0],
           tcp[1] - NEX_CAM_BACK_M * g[1] + NEX_CAM_UNDER_M * under[1])
    hf = math.tan(math.radians(perception.CAMERA_HFOV_DEG / 2.0))
    vf = hf * 9.0 / 16.0                         # 16:9 畫面的垂直半視角
    d_f = g[0] + v * vf * upv[0]
    d_u = g[1] + v * vf * upv[1]
    d_left = -u * hf                             # 影像右 = 機器人右
    if d_u >= -1e-6 or cam[1] <= plane_h_m:
        return None
    t = (plane_h_m - cam[1]) / d_u
    fwd, left = cam[0] + t * d_f, t * d_left
    y = math.radians(yaw_deg)                    # 底座轉了 yaw：手臂座標 → 底盤座標
    return (fwd * math.cos(y) - left * math.sin(y), fwd * math.sin(y) + left * math.cos(y))


# 量測姿勢 (320,130,水平) 的相機俯仰。2026-11-14 用實拍校正：箱高 0.16、箱面 0.42 m、箱子近/遠上緣在畫面 y=0.60/0.25 →
# 相機高 0.47 m（跟模型一致）、俯 47°。「上臂仰角 − 69°」的通用式在這個姿勢差 20° 多，量出箱高 0.29，不能用在量測。
NEX_MEASURE_CAM_PITCH_DEG = float(os.environ.get("ROBOT_NEX_MEASURE_CAM_PITCH_DEG", "-45"))
NEX_BOX_TOP_M = float(os.environ.get("ROBOT_BOX_TOP_M", "0.25"))     # 放倒的 Hiwonder 紙箱箱頂高度（使用者的固定道具）
# 2026-11-16 使用者：「箱子的尺寸又不是固定的，他不能夠自己適應嗎？」——箱頂高度不再當成固定值：place_on_top 放下時用手臂「碰到面」
# 量出實際高度，記在這個檔案；之後 30 分鐘內 pick_from_top / place_on_top 沒指定高度時就用量到的值，沒量過才退回上面的預設。
_BOX_TOP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "_box_top.json")
NEX_CARRY_Z_MM = 295.0            # 夾著東西靠近箱子時手臂舉到的高度（相機排線的上限 300 mm）：夾點離物件底部 12 cm 時底部離地 0.34 m


def _box_top_default():
    try:
        with open(_BOX_TOP_FILE, encoding="utf-8") as fh:
            d = json.load(fh)
        if time.time() - float(d.get("t", 0)) < 1800 and 0.08 <= float(d["top_m"]) <= 0.40:
            return float(d["top_m"])
    except Exception:
        pass
    return NEX_BOX_TOP_M


def _save_box_top(top_m):
    try:
        with open(_BOX_TOP_FILE, "w", encoding="utf-8") as fh:
            json.dump({"top_m": round(float(top_m), 3), "t": time.time()}, fh)
    except Exception:
        pass
NEX_GRASP_PITCH_DEG = float(os.environ.get("ROBOT_NEX_GRASP_PITCH_DEG", "0"))
NEX_SEAT_MM = float(os.environ.get("ROBOT_NEX_SEAT_MM", "30"))        # 第一次碰到後再往前伸多少重夾
NEX_AIM_LEFT_DEG = float(os.environ.get("ROBOT_NEX_AIM_LEFT_DEG", "3"))
NEX_DEPTH_EXTRA_MM = float(os.environ.get("ROBOT_NEX_DEPTH_EXTRA_MM", "30"))
NEX_GRIP_X = 547.0          # 觀察姿勢下夾爪開口中心在畫面的 x（0–1000）；模型指不到夾爪時的備援
NEX_MAX_DEPTH_M = 0.17      # 貼近後箱面離底座 0.27 m，手臂最遠 0.45 → 物件離箱緣超過這個就搆不到，要換一邊


async def pick_from_top(robot, llm, obj: str, surface_m: Optional[float] = None, object_h_m: Optional[float] = None,
                        depth_m: Optional[float] = None, _side_try: bool = False) -> Tuple[bool, str]:
    """抓放在箱頂/矮櫃上的小物件，水平夾法。2026-11-14 下午改版（相機在上臂；使用者：「你不能自己量嗎？東西是隨機刷新的」）：
    底盤用 ToF 貼到箱子前 → 手臂到 (250,150,水平)，相機俯 30° 看箱頂 → 轉底座把物件對到正中 → 自己量：讓模型指
    「箱子前上緣」（配 ToF 的箱面距離 → 箱頂高度）、「物件底部」（視線射到箱頂平面 → 離前緣多遠）、「物件頂端」（→ 物件高度）→
    夾爪在物件 7 成高度水平伸過去 → 合爪（伺服卡住 = 夾到）→ 抬、收回。surface_m/object_h_m/depth_m 有給就用給的。"""
    label = _plain_label(obj)
    given_s = None if surface_m is None else max(0.0, min(0.45, float(surface_m)))      # 0 = 站在地板上的東西
    given_h = None if not object_h_m else max(0.03, min(0.25, float(object_h_m)))
    r_look, z_look = 320.0, 0.13      # 相機俯 ~24°；箱面在 0.45 m 時箱緣、物件底、物件頂都在畫面裡（試算過 0.16 與 0.35 m 的箱子）
    notes = []

    async def fresh(tag="look"):
        await asyncio.sleep(0.8)
        await robot.frame()
        jpg = await robot.frame()
        _save_frame(jpg, "pick_from_top_" + tag)
        return jpg

    async def arm(reach_mm, z_mm, pitch, yaw=0.0):
        r = await robot.act("arm_to", {"x_mm": int(round(reach_mm)), "y_mm": int(round(z_mm)), "pitch_deg": float(pitch),
                                       "yaw_deg": round(float(yaw), 1)})
        real = r.get("real") or {}
        if real:                       # 鏡射時以實體手臂為準（模擬手臂到不了桌面以下的姿勢，實體可以）
            if not real.get("ok"):
                r["message"] = str(r.get("message", "")) + " | real arm: " + str(real.get("message", ""))[:120]
            r["ok"] = bool(real.get("ok"))
        return r

    async def arm_pose():
        ap = (await robot.state()).get("arm_pose") or {}
        return (float(ap.get("reach_mm", 0)) / 1000.0, float(ap.get("z_mm", 0)) / 1000.0,
                float(ap.get("yaw_deg", 0.0)), float(ap.get("pitch_deg", 0.0)))

    async def home():
        await robot.act("recenter_arm", {})

    # 0. 貼近：前向 ToF 讀箱子正面（感測器在底座前 0.16）。夾爪水平最遠伸 450，瓶子在箱面後 10–20 cm 的話車得幾乎貼著箱子。
    def front_tof(s):
        # 鏡射模式下 tof_front_mm 是 min(模擬, 實體)；箱子只有實體看得準（模擬那邊是 ToF 打到點放的方塊），有實體讀值就用實體的
        rt = s.get("real_tof_front_mm")
        return rt if isinstance(rt, (int, float)) else s.get("tof_front_mm")

    st = await robot.state()
    # ===== 2026-11-14 15:20 改版：不靠相機幾何量距離（近距離誤差太大，兩次實跑量出箱高 0.29 / 0.08，夾爪壓到箱頂）=====
    # 使用者：「為什麼不拿棍子中間，這樣就不會碰到箱子」。做法：夾爪固定在「箱頂上方 7.5 cm」（棍子中段）的高度，
    # 相機從上臂俯視，用畫面裡的夾爪當基準把底座轉到對正，然後水平一小步一小步往前伸，每步合一次爪，
    # 伺服被擋住 = 夾到了。高度只跟箱頂高度有關（放倒的紙箱 0.16，ROBOT_BOX_TOP_M 可改，任務有給就用給的）。
    surface_m = given_s if given_s is not None else _box_top_default()
    object_h_m = given_h if given_h is not None else 0.15
    # 15:20 實跑：夾爪俯 25° 時指尖比夾點低、實體手臂也到不了 (309,73,-25)，使用者：「爪子不是平的所以一直撞到箱子」→ 夾爪保持水平，
    # 改用「夾高一點」換空隙：箱頂上方 9 cm（15 cm 的棍子還有 6 cm 在夾點上面）
    # 15:21 直接試：實體在前伸 300 只降得到 100 mm（表上說 70），(300,88) 到不了 → 夾點再高：箱頂上 11 cm（棍子頂下方 4 cm），從前伸 320 起
    # 15:30 用合起的夾爪往下探：z≈135 mm 被箱頂頂住 → 箱頂 ≈ 0.25 m（之前一直當 0.16，所以夾爪每次都放在箱頂的高度、把箱子推走）。
    # 夾棍子正中間：箱頂上方 7.5 cm；觀察姿勢也用同一個高度、前伸 280（夾爪還沒過箱緣）
    grip_above = max(NEX_CAM_BOTTOM_M + 0.04, min(0.075, 0.5 * object_h_m))
    on_floor = surface_m < 0.03
    if on_floor:
        # 地板模式（2026-11-15 使用者：「他應該可以拿地上的東西」；實測手臂在車頭前方、前伸 355 可降到 z=−70 = 夾爪離地 9 cm，−90 頂到地）：
        # 夾點離地 10 cm，所以東西至少要 12 cm 高；車要停遠一點（腕 90° 姿勢前伸 355，東西要在更前面）
        if object_h_m < 0.12:
            return False, (f"pick_from_top({obj}): it stands on the floor and is only {object_h_m*100:.0f} cm tall; the gripper cannot "
                           f"get lower than 9 cm above the floor, so floor objects must be at least 12 cm tall. Nothing was moved.")
        # 使用者 11-15：「不用在最底下抓，可以抓棍子的中間或上半段，這樣比較好看到棍子」→ 夾在離頂端 3 cm 處（15 cm 的棍子 = 離地 12 cm）
        grip_above = max(0.10, min(0.20, object_h_m - 0.03))
    z_grasp = max(-0.070 if on_floor else 0.03, surface_m + grip_above - NEX_BASE_H_M)
    # 使用者 15:25：「機械手臂是一體的，手肘會撞到箱子」。夾爪水平時腕關節跟夾爪同高（箱頂上 7.5 cm），腕部伺服殼更低，
    # 伸到箱子上方就磨到箱頂。夾爪改成前低後高（俯 25°）：腕關節比夾點高 5.5 cm，整支前臂從上方斜斜伸進去。
    gp = NEX_GRASP_PITCH_DEG
    notes.append(f"gripping {grip_above*100:.1f} cm above the box top (surface {surface_m:.2f} m), gripper pitch {gp:.0f} deg")

    # 使用者 15:40：「從爪子往後數的那節要 90 度，不然會撞到箱子」→ 起始姿勢 = 前臂垂直、夾爪水平（前伸只有 ~24 cm），
    # 靠底盤貼到箱子前 6 cm 把夾爪送到箱緣上方；之後只在需要時才一步 2 cm 往前伸（前臂才慢慢變斜）
    r_look, z_look = _nex_r_wrist90(z_grasp * 1000), z_grasp
    notes.append(f"wrist-90 pose: reach {r_look:.0f} mm")
    if on_floor:
        why = await _servo_to_floor_object(robot, llm, label, notes)
        if why:
            await robot.act("recenter_arm", {})
            return False, f"pick_from_top({obj}) [{'; '.join(notes)}]: {why}"
    t_start = front_tof(await robot.state())
    if on_floor:
        pass
    elif not isinstance(t_start, (int, float)) or t_start == 9999 or t_start > 600:
        why = await _drive_to_visible(robot, llm, label, front_tof, notes, low_view=on_floor)       # 還遠：先看著目標開過去
        if why:
            return False, f"pick_from_top({obj}) [{'; '.join(notes)}]: {why}"
    tof, err = await _creep_to_surface(robot, front_tof, notes, f"pick_from_top({obj})", target_mm=150, max_step=0.25) if not on_floor else (9999, None)
    if err:
        return False, err
    tof0 = tof
    await robot.act("gripper", {"state": "open"})

    async def claw_now():
        return (await robot.state()).get("real_gripper_claw_deg")

    async def aim(r_mm, z_mm, yaw, tag, gain=1.0, tries=2, pitch=0.0):
        """用畫面裡的夾爪當基準，把底座轉到物件跟夾爪開口同一條直線上。回 (yaw, 有沒有看到物件)。"""
        seen = False
        for i in range(tries):
            jpg = await fresh(f"{tag}_{i+1}")
            # 指物件的「底部」：棍子在俯視畫面裡是斜的，指整根會指到上半截（偏很遠），15:18 實跑底座因此來回轉 36°/6°/30°
            f = await perception.locate(llm, jpg, f"the bottom end of the {label}, where it stands on the box")
            if not f.found:
                f = await perception.locate(llm, jpg, label)
            if not f.found:
                break
            seen = True
            g = await perception.locate(llm, jpg, "the gap between the two black gripper fingers at the bottom of the image")
            ref = g.bearing_right_deg if g.found else perception.bearing_from_x(NEX_GRIP_X)
            e = f.bearing_right_deg - ref
            if abs(e) <= 2.0:
                break
            # 相機在底座軸前方 20 cm、物件離相機 ~25 cm：畫面上的角度差大約是底座該轉角度的兩倍 → 乘 0.5，單步最多 12°
            step = max(-12.0, min(12.0, 0.5 * gain * e))
            yaw = max(-60.0, min(60.0, yaw - step))            # 物件在夾爪右邊 → 底座往右轉（yaw 負）
            r = await arm(r_mm, z_mm, pitch, yaw)
            if not r.get("ok"):
                break
        return yaw, seen

    # 1. 俯視對正（夾爪在箱緣前、比箱頂高）。地板模式先在高處伸出去再垂直降，手臂不會掃過車頭
    if on_floor:
        await arm(r_look, 150, 0.0, 0.0)
    r = await arm(r_look, z_look * 1000, 0.0, 0.0)
    if not r.get("ok"):
        await home()
        return False, f"pick_from_top({obj}): could not set the look pose (reach {r_look:.0f}, height {z_look*1000:.0f} mm): {str(r.get('message',''))[:120]}"
    if on_floor:
        yaw, seen = 0.0, True          # 地板模式：左右對正交給後面夾取高度的「邊看邊往前」，這裡的箱頂對正用詞與距離都不適用
    else:
        yaw, seen = await aim(r_look, z_look * 1000, 0.0, "aim", tries=4)
        for y0 in (20.0, -20.0, 40.0, -40.0):          # 正前方看不到（17:12 實跑：棍子在畫面左下角只露一點）→ 底座左右掃
            if seen:
                break
            await arm(r_look, z_look * 1000, 0.0, y0)
            yaw, seen = await aim(r_look, z_look * 1000, y0, f"aim_scan{y0:+.0f}", tries=4)
        if not seen:
            await home()
            return False, (f"pick_from_top({obj}) [{'; '.join(notes)}]: the camera does not see '{label}' on the surface in front "
                           f"of the gripper. Arm back home.")
    notes.append(f"aimed with the gripper in view: base yaw {yaw:+.1f} deg")

    # 2. 到夾取高度（水平夾爪降到這個高度至少要伸 r0），再對一次
    r0 = r_look
    r = await arm(r0, z_grasp * 1000, gp, yaw)
    if not r.get("ok"):
        await home()
        return False, (f"pick_from_top({obj}) [{'; '.join(notes)}]: the arm cannot get to reach {r0:.0f} mm at {z_grasp*1000:.0f} mm "
                       f"height: {str(r.get('message',''))[:140]}")
    if not on_floor:
        yaw, _ = await aim(r0, z_grasp * 1000, yaw, "level", gain=0.7, tries=2, pitch=gp)
        # 使用者 17:50 在旁邊看：「需要再往左邊一點點，不然拿不好」→ 對正完再固定往左補 NEX_AIM_LEFT_DEG（正 = 左）
        yaw = max(-60.0, min(60.0, yaw + NEX_AIM_LEFT_DEG))
        notes.append(f"plus {NEX_AIM_LEFT_DEG:+.1f} deg left bias -> base yaw {yaw:+.1f}")

    floor_ready = False
    if on_floor:
        yaw, why, r_seen = await _close_in_by_sight(robot, llm, label, arm, r0, z_grasp * 1000, gp, yaw, notes)
        if why:
            await robot.act("gripper", {"state": "open"})
            await arm(r0, 150, 0.0, 0.0)
            await home()
            return False, f"pick_from_top({obj}) [{'; '.join(notes)}]: {why}"
        floor_ready = True
    # 3. 使用者 18:00：「不用一直夾，不然快碰到瓶子的時候會移動到瓶子」→ 先用相機估距離（物件底部在畫面的高低），
    # 直接伸到定位只合一次爪；沒夾到才再往前 1.5 cm 補夾，最多補 3 次。估不出來才退回「從頭一步步試夾」。
    jpg = await fresh("depth")
    pb = await perception.locate(llm, jpg, f"the bottom end of the {label}, where it stands on the box")
    d_est = _nex_depth_from_base_y(r0, z_grasp * 1000, pb.y if pb.found else None, surface_m)
    if floor_ready:
        # 底部到指尖線時，指尖離物件還有約「夾爪離地高度 ÷ tan(視線俯角)」≈ 6–10 cm：直接再伸 7 cm 合爪，沒夾到最多補 2 次各 2 cm
        # 11-15 使用者：「你幹嘛一直推倒他」——一口氣伸 7 cm 時，只要左右差一點指頭就頂到棍子把它推倒
        # （夾爪開口 51 mm、棍子 50 mm，幾乎進不去兩指之間）。改成先伸 3 cm 合爪，沒夾到每次只多伸 1.5 cm。
        rr = min(440.0, r_seen)
        rr_max = min(450.0, rr + 75.0)
        d_est = None
        notes.append(f"trying the grip from {rr:.0f} mm in 1.5 cm steps")
    elif d_est is not None and r0 / 1000.0 <= d_est <= 0.45:
        # 18:15 使用者：「你還是移動到他了，你夾的太淺了」——伸到估計位置時瓶子只到指尖，合爪把它往外撥（兩趟都是第 2 次才夾到）。
        # 多伸 NEX_DEPTH_EXTRA_MM 讓瓶子進到夾爪裡面再合爪
        rr = max(r0, min(450.0, d_est * 1000 + NEX_DEPTH_EXTRA_MM))
        rr_max = min(450.0, rr + 45.0)
        notes.append(f"camera depth estimate {d_est:.3f} m (base at image y={pb.y}) -> reaching straight to {rr:.0f} mm")
    else:
        rr, rr_max = r0, 450.0
        notes.append(f"no usable depth estimate (base y={pb.y if pb.found else None}, est {d_est}), probing step by step")
    holding, real_msg, g, reach_mm = False, "", {}, r0
    n_close = 0
    while rr <= rr_max:
        r = await arm(rr, z_grasp * 1000, gp, yaw)
        if not r.get("ok"):
            notes.append(f"arm stopped at reach {rr:.0f} mm: {str(r.get('message',''))[:80]}")
            break
        t_now = front_tof(await robot.state())
        if isinstance(t_now, (int, float)) and t_now != 9999 and t_now > tof0 + 30:
            await robot.act("gripper", {"state": "open"})
            await arm(rr, min(z_grasp * 1000 + 80, 250), 0.0, yaw)
            await home()
            return False, (f"pick_from_top({obj}) [{'; '.join(notes)}]: the box moved away (front sensor {tof0} -> {t_now} mm) while "
                           f"reaching at {rr:.0f} mm: the gripper is pushing something. Stopped, arm back home.")
        g = await robot.act("gripper", {"state": "close"})
        n_close += 1
        real_msg = str((g.get("real") or {}).get("message", ""))
        holding = ("closed on something" in real_msg) if real_msg else bool(g.get("holding"))
        reach_mm = rr
        if holding:
            # 不做「鬆開再伸進去重夾」：NexArm 爪口全開 51 mm（原廠尺寸圖），棍子直徑 50 mm，進不到兩指之間，
            # 17:28 實測再伸 3 cm 只會把棍子推倒。只能用指尖夾住它的近側。
            # 細的物件（合爪後夾爪角度 > −5°，約 3.5 cm 以下）進得去兩指之間：鬆開、再伸 2.5 cm、重夾，讓它坐進夾爪裡；
            # 重夾沒夾到就退回原位再夾。粗的（5 cm 棍子合到 −22°）不做。
            first = await claw_now()
            if isinstance(first, (int, float)) and first > -5 and rr + 25 <= 450:
                await robot.act("gripper", {"state": "open"})
                await asyncio.sleep(0.8)
                r2 = await arm(rr + 25, z_grasp * 1000, gp, yaw)
                g2 = await robot.act("gripper", {"state": "close"})
                m2 = str((g2.get("real") or {}).get("message", ""))
                if r2.get("ok") and "closed on something" in m2:
                    g, real_msg, reach_mm = g2, m2, rr + 25
                    notes.append(f"thin object: re-seated 25 mm deeper (claw {first} -> {await claw_now()} deg)")
                else:
                    await robot.act("gripper", {"state": "open"})
                    await asyncio.sleep(0.6)
                    await arm(rr, z_grasp * 1000, gp, yaw)
                    g = await robot.act("gripper", {"state": "close"})
                    real_msg = str((g.get("real") or {}).get("message", ""))
                    holding = ("closed on something" in real_msg) if real_msg else bool(g.get("holding"))
                    notes.append("deeper re-seat missed, gripped at the first contact again")
            else:
                notes.append(f"claw stopped at {first} deg")
            break
        await robot.act("gripper", {"state": "open"})
        await asyncio.sleep(0.6)
        rr += 15.0
    # 手臂伸到底還沒夾到（使用者 11-15：「你伸長不了的時候可以往前開一點」）：夾爪張開、手臂縮回起始前伸，車往前 8 cm，再從頭伸一次；最多 3 次。
    # 只在前面沒有箱子擋著時做（地板模式）；箱子模式車已經貼在防撞距離上，不能再往前。
    advances = 0
    while not holding and on_floor and advances < 3:
        advances += 1
        await robot.act("gripper", {"state": "open"})
        await arm(r0, z_grasp * 1000, gp, yaw)
        await arm(r0, 150, 0.0, yaw)               # 手臂先抬起來才動底盤（降著時底盤不准動）
        await _move_exact(robot, 0.08, 0.0)
        await arm(r0, z_grasp * 1000, gp, yaw)
        rr = r0
        while rr <= 450.0:
            r = await arm(rr, z_grasp * 1000, gp, yaw)
            if not r.get("ok"):
                break
            g = await robot.act("gripper", {"state": "close"})
            n_close += 1
            real_msg = str((g.get("real") or {}).get("message", ""))
            holding = ("closed on something" in real_msg) if real_msg else bool(g.get("holding"))
            reach_mm = rr
            if holding:
                break
            await robot.act("gripper", {"state": "open"})
            await asyncio.sleep(0.6)
            rr += 15.0
    if advances:
        notes.append(f"arm at full reach without a grip: drove 8 cm closer {advances} time(s) and reached again")
    notes.append(f"{'closed on it at' if holding else 'nothing up to'} {reach_mm:.0f} mm after {n_close} close(s)")
    _save_frame(await robot.frame(), "pick_from_top_closed")

    if not holding:
        await robot.act("gripper", {"state": "open"})
        await arm(min(reach_mm, 400), min(z_grasp * 1000 + 80, 250), 0.0, yaw)
        await home()
        if _side_try:
            return False, (f"pick_from_top({obj}) [{'; '.join(notes)}]: nothing between the fingers from this side either. "
                           f"The object must stand within about 0.17 m of an edge.")
        # 從這一邊搆不到（長方形箱子的短邊、東西在中間）→ 用地圖繞到側面再試一次
        st = await robot.state()
        if "real_tof_front_mm" in st:        # 實體：貼著箱子繞行撞過箱子（17:30），不自動繞，回報給上層決定
            return False, (f"pick_from_top({obj}) [{'; '.join(notes)}]: nothing between the fingers within reach from this side. "
                           f"Not driving around the box automatically (collision risk); back up and approach from another side.")
        xy, hd = st.get("world_xy"), st.get("world_heading_deg")
        if xy and hd is not None:
            a2 = math.radians(hd + yaw)
            ox, oy = xy[0] + 0.50 * math.cos(a2), xy[1] + 0.50 * math.sin(a2)
            for side in (+1, -1):
                b2 = math.radians(hd + 90 * side)
                gx, gy = ox + 0.60 * math.cos(b2), oy + 0.60 * math.sin(b2)
                r = await robot.act("navigate_to", {"x": round(gx, 2), "y": round(gy, 2), "tolerance_m": 0.08, "face": False})
                if not r.get("ok"):
                    continue
                st = await robot.state()
                xy2, hd2 = st.get("world_xy") or (gx, gy), st.get("world_heading_deg") or hd
                turn = (math.degrees(math.atan2(oy - xy2[1], ox - xy2[0])) - hd2 + 180) % 360 - 180
                await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(turn, 1), "push": False})
                ok2, msg2 = await pick_from_top(robot, llm, obj, surface_m if given_s else None, object_h_m if given_h else None,
                                                depth_m, _side_try=True)
                return ok2, f"[nothing within reach from the first side; went around to the {'left' if side > 0 else 'right'} side] " + msg2
        return False, f"pick_from_top({obj}) [{'; '.join(notes)}]: nothing between the fingers within reach, and could not drive around to another side."

    # 4. 抬起確認、收回
    await asyncio.sleep(0.5)
    z_hold = min(z_grasp * 1000 + (60 if reach_mm <= 420 else 40), 295.0)     # 俯 25° 伸到 450 時最高只到 120
    await arm(reach_mm, z_hold, gp, yaw)
    await asyncio.sleep(0.5)
    st = await robot.state()
    still = bool(st.get("real_gripper_holding")) if "real_gripper_holding" in st else holding
    _save_frame(await robot.frame(), "pick_from_top_lifted")
    if still:
        # 晃動測試：底座左右各轉 12° 再回來，還夾著才算穩
        for dy in (12.0, -12.0, 0.0):
            await arm(reach_mm, z_hold, gp, yaw + dy)
        await asyncio.sleep(0.4)
        st = await robot.state()
        still = bool(st.get("real_gripper_holding")) if "real_gripper_holding" in st else still
        notes.append("shake test (base +-12 deg): " + ("still holding" if still else "DROPPED"))
    # 搬運姿勢：夾爪保持水平（看路姿勢會讓夾爪前傾 26°，棍子跟著歪、容易滑）。沒夾到才回看路姿勢
    if still:
        if on_floor:
            await arm(300, 150, 0.0, 0.0)
        await arm(_nex_r_wrist90(150.0 if on_floor else z_hold), 150.0 if on_floor else z_hold, 0.0, 0.0)
    else:
        await home()
    where = f"[{'; '.join(notes)}]"
    if still:
        _LAST_TOP_GRIP.update({"object": label, "grip_above_bottom_m": round(grip_above, 3), "object_h_m": object_h_m,
                               "surface_m": surface_m, "reach_mm": reach_mm, "yaw_deg": yaw, "z_mm": z_grasp * 1000,
                               "tof_mm": tof0, "t": time.time()})
        try:                                                    # 也寫到檔案：agent 重開或分兩次執行時 place_on_top 還找得到
            with open(_TOP_GRIP_FILE, "w", encoding="utf-8") as fh:
                json.dump(_LAST_TOP_GRIP, fh)
        except Exception:
            pass
        return True, (f"pick_from_top({obj}) {where}: GRASPED — the gripper servo stopped short of fully closed (something between the "
                      f"fingers) and it is still there after lifting and a shake test. Keep the gripper closed; the arm holds it level (do not recenter_arm while carrying, that tilts it).")
    await robot.act("gripper", {"state": "open"})
    return False, (f"pick_from_top({obj}) {where}: closed on it but it slipped out when lifting (real arm said: {real_msg[:100]}). "
                   f"Frames are in runs/_skill_frames/.")


async def place_on_top(robot, llm, surface_m: Optional[float] = None, object_h_m: Optional[float] = None,
                       depth_m: Optional[float] = None) -> Tuple[bool, str]:
    """把夾著的東西立著放回箱頂。2026-11-14 15:35 實機驗證過的路徑（pick_from_top 反過來，腕關節 90° 姿勢）：
    底盤貼到箱子前 → 手臂抬高 6 cm 移到放置點上方 → 垂直放下到比夾取高 5 mm → 爪子全開 → 水平退 2.5 cm 讓指尖離開 →
    抬高、歸位 → 用退出後的相機畫面確認它立著。剛用 pick_from_top 夾的而且車沒動過，就放回原本夾起來的位置。"""
    last = dict(_LAST_TOP_GRIP)
    aim = None                 # 夾著東西對箱子時最後量到的「箱子中心偏幾度」，放的時候用手臂底座轉過去補
    if not last:
        try:
            with open(_TOP_GRIP_FILE, encoding="utf-8") as fh:
                d = json.load(fh)
            if time.time() - float(d.get("t", 0)) < 900:
                last = d
        except Exception:
            pass
    surface = float(surface_m) if surface_m is not None else _box_top_default()      # surface_m=0 → 放到地上
    on_floor = surface < 0.03
    grip = float(last.get("grip_above_bottom_m") or 0.075)
    # 鬆手高度：比夾起來時再低一點，讓底部確實壓到面上才放。使用者 11-15：「你現在沒有放到底，所以棍子倒了」——
    # 之前用「夾取高度 + 4~5 mm」，東西在夾爪裡只要滑一點、或手臂伸長時下垂量不同，底部就懸空，一鬆手就倒。
    z_mm = (surface + grip - (0.010 if on_floor else 0.004) - NEX_BASE_H_M) * 1000
    if on_floor:
        z_mm = max(-70.0, z_mm)
    notes = [f"{'floor' if on_floor else f'surface {surface:.2f} m'}, lowering until its bottom presses on it before letting go"]

    async def arm(reach_mm, zz, yaw=0.0):
        r = await robot.act("arm_to", {"x_mm": int(round(reach_mm)), "y_mm": int(round(zz)), "pitch_deg": 0.0, "yaw_deg": round(float(yaw), 1)})
        real = r.get("real") or {}
        if real:                       # 鏡射時以實體手臂為準（模擬手臂到不了桌面以下的姿勢，實體可以）
            if not real.get("ok"):
                r["message"] = str(r.get("message", "")) + " | real arm: " + str(real.get("message", ""))[:120]
            r["ok"] = bool(real.get("ok"))
        return r

    def front_tof(s):
        rt = s.get("real_tof_front_mm")
        return rt if isinstance(rt, (int, float)) else s.get("tof_front_mm")

    def held(s):
        return bool(s.get("real_gripper_holding")) if "real_gripper_holding" in s else bool(s.get("gripper_holding"))

    if not held(await robot.state()):
        return False, "place_on_top: the gripper is not holding anything, nothing to put down."
    same_spot_hint = bool(last.get("reach_mm")) and float(last.get("surface_m") or 0.0) > 0.03      # 上一次是從箱頂夾的
    if on_floor:
        tof, err = 9999, None                      # 地上沒有箱面可以貼，就放在車前
    else:
        t0 = front_tof(await robot.state())
        if not (same_spot_hint and isinstance(t0, (int, float)) and t0 <= 250):
            # 使用者 11-15：「你沒有對準箱子」——以前只在 ToF > 0.6 m 時才用相機對箱子；ToF 剛好掃到箱角或別的東西就直接往前貼，結果歪著。
            # 現在除了「剛從這個箱子上夾起來、車沒動」以外，一律先用相機把車頭轉正對著箱子（夾著東西、看著箱子開過去）
            # 11-15 17:54 放歪的原因：離箱子不到 0.6 m 時畫面只看得到箱角（還被夾著的東西擋住），沒轉正就直直貼上去，ToF 打到箱角，
            # 東西放到箱緣外、手臂把箱子頂歪。太近就先直直退 0.5 m，讓整個箱子進畫面再對正、靠近。
            if isinstance(t0, (int, float)) and t0 <= 750:
                await robot.act("arm_to", {"x_mm": 250, "y_mm": 250, "pitch_deg": 0.0, "yaw_deg": 0.0})
                await robot.act("move_chassis", {"forward_m": -0.5, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
                notes.append("too close to see the whole box: backed up 0.5 m first")
            _drive_to_visible.last = None
            why = await _drive_to_visible(robot, llm, "cardboard box", front_tof, notes, carrying=True)
            if why:
                return False, f"place_on_top [{'; '.join(notes)}]: {why} (still holding it)"
            aim = getattr(_drive_to_visible, "last", None)
        # 使用者 11-15：「你撞到箱子了，你要舉高一點點」——貼近箱子前先把手臂舉到放置高度上方，夾著的東西下端要高過箱頂
        # 2026-11-16 使用者：「他如果舉的不夠高會撞到箱子」——不再用「假設的箱頂 + 6 cm」，一律舉到最高（NEX_CARRY_Z_MM）才靠近：
        # 箱子多高都一樣，夾著的東西底部離地 0.34 m。真正的箱頂高度等一下放下時用手臂碰出來。
        await robot.act("arm_to", {"x_mm": 250, "y_mm": int(NEX_CARRY_Z_MM), "pitch_deg": 0.0, "yaw_deg": 0.0})
        # 2026-11-16：一步最多 10 cm（原 25 cm）、停在 19 cm（原 15 cm）。使用者：「會一直撞到箱子」——歪著貼近時 ToF 沒打在箱面上，
        # 一步 20–25 cm 走完才發現讀值沒變，箱子已經被推走了。小步走，第一步讀值沒跟著變就停。
        tof, err = await _creep_to_surface(robot, front_tof, notes, "place_on_top", target_mm=190, max_step=0.10)
    if err:
        return False, err + " (still holding it)"
    r90 = _nex_r_wrist90(z_mm)
    same_spot = last.get("reach_mm") and isinstance(last.get("tof_mm"), (int, float)) and abs(tof - last["tof_mm"]) <= 25
    if on_floor:
        reach, yaw = min(440.0, max(r90 + 30.0, float(last.get("reach_mm") or 0.0))), float(last.get("yaw_deg") or 0.0)
    elif depth_m is not None:
        reach, yaw = (0.16 + tof / 1000.0 + float(depth_m)) * 1000, 0.0
    elif same_spot:
        reach, yaw = float(last["reach_mm"]), float(last.get("yaw_deg") or 0.0)
        notes.append("same standing spot as the pick: putting it back where it was")
    else:
        reach, yaw = (0.16 + tof / 1000.0 + 0.07) * 1000, 0.0      # 不知道原位：放在箱緣後 7 cm（16:03 實測放在箱緣後 1 cm，太險）
        if aim and isinstance(aim.get("tof"), (int, float)):
            # 左右：最後一眼箱子中心偏 b 度（在離手臂底座 0.16+tof+0.07 m 處）→ 橫向偏 e 公尺；直直貼近後 e 不變，
            # 用手臂底座轉 atan(e/reach) 把東西送到箱子中線上（底座 + = 左，方位 + = 右）。最多 ±20°。
            e = (0.16 + aim["tof"] / 1000.0 + 0.07) * math.tan(math.radians(aim["bearing"]))
            yaw = max(-20.0, min(20.0, -math.degrees(math.atan2(e, reach / 1000.0))))
            notes.append(f"box centre was {aim['bearing']:+.1f} deg off ({e * 100:+.0f} cm sideways): arm base yaw {yaw:+.1f} deg to put it on the centre line")
    reach = max(r90, min(450.0, reach))
    z_hi = 150.0 if on_floor else NEX_CARRY_Z_MM
    if not on_floor:
        await arm(160, z_hi)
    r = await arm(reach, z_hi, yaw)
    if not r.get("ok") and not on_floor:
        # 最高處伸不到這麼遠（手臂工作範圍）：每次降 3 cm 再試，降到「假設箱頂 + 6 cm」為止
        zz = z_hi
        while not r.get("ok") and zz - 30.0 >= z_mm + 60.0:
            zz -= 30.0
            r = await arm(reach, zz, yaw)
        if r.get("ok"):
            notes.append(f"cannot reach {reach:.0f} mm at full height: went over the box at z {zz:.0f} mm")
            z_hi = zz
    if r.get("ok") and on_floor:
        r = await arm(reach, z_mm, yaw)
    elif r.get("ok"):
        # 2026-11-16 o_h2：對準、貼近都對了，鬆手後棒子還是兩次都掉下去。箱頂高度寫死 0.25 m，但箱子換一面放就不是這個高度
        # （用旁邊 15 cm 的棒子當尺量畫面，今天的箱子大約 19 cm 高）→ 在箱頂上方 6 cm 就鬆手，棒子摔下去彈走。
        # 改成真的「放到底」：從上方每次降 8 mm，看實體手臂回報的高度；降不下去（實際高度比指令高 7 mm 以上、或回報沒到位）
        # = 底部頂到箱面了，就在那裡鬆手。最低降到假設箱頂只有 12 cm 的高度為止。
        def real_z(res):
            m = re.search(r"z=(-?\d+)", str((res.get("real") or {}).get("message", "")) or str(res.get("message", "")))
            return float(m.group(1)) if m else None
        z_low = max(_nex_min_z(reach) if callable(globals().get("_nex_min_z")) else -70.0, (0.12 + grip - NEX_BASE_H_M) * 1000)
        z_cmd, touched = z_hi, None
        while z_cmd - 10.0 >= z_low:
            z_cmd -= 10.0
            r = await arm(reach, z_cmd, yaw)
            zr = real_z(r)
            if not r.get("ok") or (zr is not None and zr - z_cmd >= 7.0):
                touched = zr if zr is not None else z_cmd + 10.0
                break
            await asyncio.sleep(0.15)
        if touched is not None:
            top = touched / 1000.0 + NEX_BASE_H_M - grip
            notes.append(f"lowered until the object pressed on the surface: arm stopped at z {touched:.0f} mm (commanded {z_cmd:.0f}) -> the top is about {top:.2f} m high")
            if 0.08 <= top <= 0.40:
                _save_box_top(top)             # 記下量到的箱頂高度，之後從這個箱子上夾東西用得到
            z_mm = touched
            r = {"ok": True}
        else:
            notes.append(f"lowered to z {z_cmd:.0f} mm without feeling a surface (box top lower than 0.12 m, or not over the box)")
            z_mm = z_cmd
            r = {"ok": True}
    if not r.get("ok"):
        await arm(160, z_hi)
        await robot.act("recenter_arm", {})
        return False, f"place_on_top [{'; '.join(notes)}]: could not get over the surface: {str(r.get('message',''))[:160]} (still holding it)"
    await asyncio.sleep(0.5)
    await robot.act("gripper", {"state": "open"})
    await asyncio.sleep(1.0)
    await arm(max(r90 - 5, reach - 30), z_mm, yaw)                 # 水平退出，指尖離開物件
    await asyncio.sleep(0.8)
    await robot.frame()
    jpg = await robot.frame()
    _save_frame(jpg, "place_on_top_check")
    name = str(last.get("object") or "object that was just put down")
    f = await perception.locate(llm, jpg, f"the {name} standing right in front of the camera")
    if not f.found:                                             # 退出後的畫面太近只看到頂端：再從看路姿勢看一次
        await arm(200, z_hi)
        await robot.act("recenter_arm", {})
        await asyncio.sleep(0.8)
        await robot.frame()
        jpg = await robot.frame()
        _save_frame(jpg, "place_on_top_check2")
        f = await perception.locate(llm, jpg, f"the {name} standing upright")
    await arm(200, z_hi)
    await robot.act("recenter_arm", {})
    still = held(await robot.state())
    _LAST_TOP_GRIP.clear()
    notes.append(f"released at reach {reach:.0f} mm, base yaw {yaw:+.1f} deg")
    if still:
        return False, f"place_on_top [{'; '.join(notes)}]: opened the gripper but it still reports holding something."
    if f.found:
        return True, (f"place_on_top [{'; '.join(notes)}]: released, the gripper is empty and the camera sees the {name} standing "
                      f"upright {'on the floor' if on_floor else 'on the box'}. The arm is back in the driving pose.")
    return False, (f"place_on_top [{'; '.join(notes)}]: released and the gripper is empty, but the camera could NOT confirm the {name} "
                   f"is standing upright (tipped over or fell off?). Not a success; frame in runs/_skill_frames/.")


async def survey_scene(robot, llm) -> Tuple[bool, str]:
    """實體場地的「先看清楚再動手」（2026-11-14 陷阱題失敗後加的）：相機轉成俯 20–30° 拍箱子一帶，列出每樣小物件
    和它站在哪（箱子上 / 地上），手臂收回。不移動底盤。模擬裡 scan_overhead 做的事，這裡用前鏡頭做。"""
    st = await robot.state()
    tof = st.get("real_tof_front_mm", st.get("tof_front_mm"))
    await robot.act("arm_to", {"x_mm": 250, "y_mm": 150, "pitch_deg": 0.0, "yaw_deg": 0.0})
    await asyncio.sleep(1.0)
    await robot.frame()
    jpg = await robot.frame()
    _save_frame(jpg, "survey_scene")
    items, raw = await perception.survey_front(llm, jpg)
    await robot.act("recenter_arm", {})
    if items is None:
        return False, f"survey_scene: could not read the scene from the camera ({raw[:120]}). Try again from 0.8-1.2 m in front of the box."
    if not items:
        return True, ("survey_scene: no small objects seen on or around the box from here. If you are closer than 0.6 m the "
                      "box may be below the view: back up to about 1 m and survey again.")
    on_box = [i["label"] for i in items if "box" in i["where"]]
    floor = [i["label"] for i in items if "floor" in i["where"]]
    other = [i["label"] for i in items if "box" not in i["where"] and "floor" not in i["where"]]
    msg = (f"survey_scene (front camera looking down): ON THE BOX: {', '.join(on_box) if on_box else 'nothing'}. "
           f"ON THE FLOOR (not on the box): {', '.join(floor) if floor else 'nothing'}."
           + (f" Elsewhere: {', '.join(other)}." if other else "")
           + " Things ON THE BOX are grasped with pick_from_top('<label>'). Things standing ON THE FLOOR can be grasped too if they are at least 12 cm tall: pick_from_top('<label>', surface_m=0)."
           + (" NOTE: you are closer than 0.6 m, part of the box may be out of view; survey from about 1 m for a full list."
              if isinstance(tof, (int, float)) and tof < 600 else ""))
    return True, msg


_HOME_DOT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "_home_dot.json")
_HOME_DOT_POSE = {"x_mm": 380, "y_mm": 120, "pitch_deg": 0.0, "yaw_deg": 0.0}


def _find_park_row(jpg):
    """停車格（2026-11-16 使用者：「我貼了多個白色的點讓他能夠知道停車的位置，要停在點點的裡面」）：俯視姿勢下地上的白點排成一個框，
    取「最遠那一排」（車頭那一邊，畫面上 y 最小的一排）最左與最右兩個點的中點當定位基準。比單一白點穩：不會認錯是哪一顆，
    踢腳板、地毯的白色碎塊也湊不成一排。回 (mid_x, mid_y, 斜率, 點數)（0–1000）或 None。"""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 185) & (hsv[:, :, 1] < 60)).astype(np.uint8)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    dots = []
    for i in range(1, n):
        a, bw, bh = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        x, y = cent[i][0] / w * 1000.0, cent[i][1] / h * 1000.0
        # 地板上的貼紙：小、接近圓（俯視會壓扁成橫的橢圓）、在畫面下半部的中間地帶（上面是遠處的牆與踢腳板，兩側是地毯）
        if 8 <= a <= 320 and 0.8 <= bw / max(1, bh) <= 3.0 and a / float(bw * bh) > 0.5 and y > 430 and 150 < x < 900:
            dots.append((x, y))
    if len(dots) < 2:
        return None
    dots.sort(key=lambda d: d[1])
    for k in range(len(dots) - 1):                      # 從最遠的點開始找「一排」：y 相近（60 內）的 2 顆以上、左右跨 100–450
        row = [d for d in dots if abs(d[1] - dots[k][1]) <= 60]
        xs = [d[0] for d in row]
        if len(row) >= 2 and 100 <= max(xs) - min(xs) <= 450:
            L, R = min(row, key=lambda d: d[0]), max(row, key=lambda d: d[0])
            return ((L[0] + R[0]) / 2.0, (L[1] + R[1]) / 2.0, (R[1] - L[1]) / max(1.0, R[0] - L[0]), len(row))
    return None


def _find_white_dot(jpg, near):
    """地上的白色原點貼紙（亮、低飽和的小圓塊）在畫面的 (x, y)/1000；取離基準最近的候選。找不到回 None。"""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 185) & (hsv[:, :, 1] < 60)).astype(np.uint8)
    mask[: int(h * 0.05)] = 0        # 相機朝下多的時候白點會在畫面很上面（11-15：y=97/1000 被舊的 30% 遮罩蓋掉）；誤判靠圓度/大小/伺服保險擋
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    cands = []
    for i in range(1, n):
        a, bw, bh = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # 11-15 14:26：踢腳板的白色碎塊被當成白點，車一路追到牆邊。白點是小圓點：面積小、接近正圓、不貼畫面邊
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        if (5 <= a <= 260 and 0.6 <= bw / max(1, bh) <= 2.2 and a / float(bw * bh) > 0.55
                and x0 > 3 and y0 > 3 and x0 + bw < w - 3 and y0 + bh < h - 3):
            cands.append((cent[i][0] / w * 1000.0, cent[i][1] / h * 1000.0))
    if not cands:
        return None
    return min(cands, key=lambda c: (c[0] - near[0]) ** 2 + (c[1] - near[1]) ** 2)


async def _move_exact(robot, fwd, right, tol=0.02, arm_low=False):
    """用實體里程計閉迴路走一段（車頭方向不變）：底盤指令的實際距離會差 ±10 cm（2026-11-15：送 −0.55 走了 0.595、送 0.642 只走 0.54），
    走完看里程計差多少再補，最多補 3 次；小於 9 cm 的補正用「反向退 12 cm 再走」。"""
    # arm_low=True：手臂降在車頭前方時夾爪擋在 ToF 前面，ToF 讀到的是自己的夾爪（0.14 m），防撞會把每一步砍成 0
    # （11-15：邊看邊往前 12 步畫面完全沒變）。這時 ToF 本來就是瞎的，只能關掉它；最前面的是夾爪，每步很短（≤15 cm）而且每步都看畫面。
    st = await robot.state()
    p0, hd = st.get("real_world_xy") or st.get("world_xy"), st.get("real_world_heading_deg", st.get("world_heading_deg", 0.0))
    h = math.radians(hd or 0.0)
    for _ in range(4):
        st = await robot.state()
        p = st.get("real_world_xy") or st.get("world_xy")
        dx, dy = p[0] - p0[0], p[1] - p0[1]
        done_f, done_r = dx * math.cos(h) + dy * math.sin(h), dx * math.sin(h) - dy * math.cos(h)
        rf, rr = fwd - done_f, right - done_r
        if abs(rf) <= tol and abs(rr) <= tol:
            break
        for key, v in (("forward_m", rf), ("right_m", rr)):
            if abs(v) <= tol:
                continue
            if abs(v) < 0.09:
                back = -0.12 if v > 0 else 0.12
                await robot.act("move_chassis", {"forward_m": back if key == "forward_m" else 0, "right_m": back if key == "right_m" else 0,
                                                 "turn_left_deg": 0, "push": bool(arm_low), "near_ok": True})
                v -= back
            await robot.act("move_chassis", {"forward_m": round(v, 3) if key == "forward_m" else 0,
                                             "right_m": round(v, 3) if key == "right_m" else 0, "turn_left_deg": 0, "push": bool(arm_low), "near_ok": True})
        await asyncio.sleep(0.5)
    return rf, rr


async def return_home(robot, llm=None) -> Tuple[bool, str]:
    """回家（2026-11-14 實機驗證，tools/home_dot.py 的 async 版）：里程計回原點 → 退到白點後方 → 相機俯視把地上的白色貼紙
    （左前輪的定位點）對到基準畫面位置 → 往前開到家。基準 runs/_home_dot.json 是使用者確認過位置的（「就是那邊不用改」）。
    沒有基準檔、不是實體場地、或看不到白點時，就只做里程計回原點並照實說。"""
    r = await robot.act("return_to_origin", {})
    base = str(r.get("message", ""))[:200]
    st = await robot.state()
    if "real_world_xy" not in st or not os.path.exists(_HOME_DOT_FILE):
        return bool(r.get("ok")), base
    with open(_HOME_DOT_FILE, encoding="utf-8") as fh:
        ref = json.load(fh)
    await _move_exact(robot, -ref["forward_m"], -ref["right_m"], tol=0.04)
    await robot.act("arm_to", _HOME_DOT_POSE)
    ok, log, last_e = False, [], None
    travelled, moved_last, fwd_seek = 0.0, False, 0
    if ref.get("row") and ref.get("gap_x") is not None:
        # 2026-11-16：停車格基準的 gap 是影像法（_gripper_gap_cv）量的；這裡原本用 LLM 指的開口（定義不同：同一張畫面 y 差約 100、x 差 13），
        # 目標位置被平移錯，使用者：「沒有回到原位，他偏右邊」。用同一個方法量，差太多（> 60）就當作沒量到、直接用基準。
        await asyncio.sleep(0.8)
        await robot.frame()
        g = _gripper_gap_cv(await robot.frame())
        if g is not None and abs(g[0] - ref["gap_x"]) <= 60 and abs(g[1] - ref["gap_y"]) <= 60:
            ref = dict(ref, x=ref["x"] + g[0] - ref["gap_x"], y=ref["y"] + g[1] - ref["gap_y"])
            log.append(f"[gripper {g[0]:.0f},{g[1]:.0f} -> target {ref['x']:.0f},{ref['y']:.0f}]")
    elif llm is not None and ref.get("gap_x") is not None:
        gap = await _gripper_gap(robot, llm)           # 相機角度會跑：白點的目標位置用「相對夾爪開口」的偏移重算
        if gap is not None:
            ref = dict(ref, x=ref["x"] + gap[0] - ref["gap_x"], y=ref["y"] + gap[1] - ref["gap_y"])
            log.append(f"[gripper gap {gap[0]:.0f},{gap[1]:.0f} -> target {ref['x']:.0f},{ref['y']:.0f}]")
        else:
            # 相機歪到看不到夾爪（使用者 11-15：「相機真的沒有能夠非常固定的位置」）：沒有基準就不要去追白點（追錯過踢腳板），只做里程計回原點並明講
            await robot.act("recenter_arm", {})
            await _move_exact(robot, ref["forward_m"], ref["right_m"])
            return bool(r.get("ok")), ("return_to_origin: back at the start by odometry only. The camera cannot see the robot's own gripper, "
                                       "so it has drifted too far to line up on the floor marker: it needs re-aiming "
                                       "(tools/cam_check.py watch). " + base)
    for _ in range(12):
        await asyncio.sleep(1.0)
        await robot.frame()
        jpg = await robot.frame()
        _save_frame(jpg, "home_dot")
        if ref.get("row"):                             # 新基準：停車格最遠那一排白點的中點
            pr = _find_park_row(jpg) if jpg else None
            dot = (pr[0], pr[1]) if pr else None
        else:
            dot = _find_white_dot(jpg, (ref["x"], ref["y"])) if jpg else None
        if dot is None:
            # 里程計漂掉時退到的位置會離白點太遠，白點在畫面上緣外（11-15：連兩次「dot not seen」，其實它就在上緣）→ 往前 0.3 m 再看，最多 3 次
            if last_e is None and fwd_seek < 3:
                fwd_seek += 1
                await robot.act("move_chassis", {"forward_m": 0.3, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
                log.append("not seen, moved 0.3 m forward to look")
                continue
            log.append("dot not seen")
            break
        ex, ey = dot[0] - ref["x"], dot[1] - ref["y"]
        log.append(f"({ex:+.0f},{ey:+.0f})")
        if (last_e is None and (abs(ex) > 330 or abs(ey) > 330)) or travelled > 0.9 or \
                (last_e is not None and moved_last and abs(ex - last_e[0]) < 8 and abs(ey - last_e[1]) < 8):
            if last_e is not None and abs(ex) <= 80 and abs(ey) <= 80:
                # 2026-11-16：誤差已經在 80 單位（約 12 cm）內卻「走了沒變」，是底盤走不出那幾公分的小步，不是追錯東西。
                # 以前這裡一律當成「不是白點」放棄，連最後補殘差那一步都跳過（兩題都停在離家 4–6 cm 還回報沒對到）。
                # 留著這個誤差，交給下面用里程計閉迴路的最後一段一起走掉。
                log.append("chassis cannot make this small correction: carrying the remainder into the final move")
                last_e = (ex, ey)
                break
            log.append("NOT the home marker (too far off / not responding to moves): stopped following it")
            last_e = None
            break
        last_e = (ex, ey)
        moved_last = False
        if abs(ex) <= 18 and abs(ey) <= 18:
            ok = True
            break
        # 畫面 1 單位 ≈ 1.5 mm；白點偏右 → 車往右，白點偏下（太近）→ 車後退。底盤走不出 5 cm 以下的小步：先反向退 8 cm 再走
        for key, v in (("forward_m", max(-0.25, min(0.25, -0.8 * ey * 0.0015))), ("right_m", max(-0.25, min(0.25, 0.8 * ex * 0.0015)))):
            if abs(v) < 0.008:
                continue
            if abs(v) < 0.09:        # 5–8 cm 的步子底盤也常常不動（19:00 回家前後方向卡在 −45 單位收斂不了）
                back = -0.12 if v > 0 else 0.12
                await robot.act("move_chassis", {"forward_m": back if key == "forward_m" else 0, "right_m": back if key == "right_m" else 0,
                                                 "turn_left_deg": 0, "push": False, "near_ok": True})
                v -= back
            await robot.act("move_chassis", {"forward_m": round(v, 3) if key == "forward_m" else 0,
                                             "right_m": round(v, 3) if key == "right_m" else 0, "turn_left_deg": 0, "push": False, "near_ok": True})
            travelled += abs(v)
            moved_last = True
    await robot.act("arm_to", {"x_mm": 250, "y_mm": 200, "pitch_deg": 0.0, "yaw_deg": 0.0})
    # 剩下沒對掉的幾公分（底盤小步走不準，前後方向常卡在 30–40 單位）直接加進最後這段用里程計閉迴路走的距離裡
    add_f = add_r = 0.0
    if last_e is not None and abs(last_e[0]) <= 80 and abs(last_e[1]) <= 80:
        add_f, add_r = -last_e[1] * 0.0015, last_e[0] * 0.0015
        ok = True
    await _move_exact(robot, ref["forward_m"] + add_f, ref["right_m"] + add_r)
    await robot.act("recenter_arm", {})
    if ok:
        return True, f"return_to_origin: back home, aligned on the white floor marker (pixel error steps {' '.join(log)}). {base}"
    return bool(r.get("ok")), (f"return_to_origin: back at the start by odometry only; the white floor marker could not be aligned "
                               f"({' '.join(log)}), so the position may be several cm off. {base}")


async def place(robot, llm, target: str) -> Tuple[bool, str]:
    """把手上的物件放到目標：收納箱 → 手臂抬高伸過桶緣再張開；桌子 → 升降柱升到桌面高度、前進到桌緣、低位張開；
    地墊/地面/其他 → 手臂放低再張開。"""
    xy, kind, note, surface = await resolve_target(robot, llm, target)
    if kind == "front":
        return False, f"place({target}): 'in front of me' only makes sense for pick; give a landmark, an object description or coordinates"
    if xy is None:
        return False, f"place({target}): {note}"
    lname = note.split("'")[1] if kind == "landmark" and "'" in note else None
    is_bin = bool(lname and lname.startswith("bin_"))
    on_surface = surface if (kind == "landmark" and surface) else None   # 目標就是桌子這個地標：放到桌面上
    where = f"[{kind} at world ({xy[0]:.2f}, {xy[1]:.2f}){note}]"
    # 收納箱也要正對桶緣接近：0.35 m 的垃圾桶從斜角站位釋放，東西會掉在桶角外面（2026-10-21 瓶子掉在 bin_recycle 西北角外）
    nav_target = on_surface or ({"name": lname} if is_bin else None)
    res = await robot.act("navigate_to", await _nav_params(robot, xy, 0.15 if is_bin else 0.20, nav_target, label=lname or target),
                          timeout=NAV_TIMEOUT_S)
    if not res.get("ok"):
        return False, f"place({target}) {where}: could not reach it — {res.get('message')}"
    lift_mm = 0
    if on_surface:
        lift_mm = min(LIFT_MAX_MM, int(round(on_surface["top_m"] * 1000)))
        r = await robot.act("lift_arm", {"height_mm": lift_mm})
        if not r.get("ok"):
            await robot.act("lift_arm", {"height_mm": 0})
            return False, (f"place({target}) {where}: could not raise the arm to the {on_surface['name']} top "
                           f"({r.get('message')}); still holding the object")
    if is_bin:
        # 釋放姿態實測：arm_to(160,170) 實際到 (164,149)，夾爪離地 0.248 m，積木（夾在上緣）底部約 0.116 m，
        # 高過 0.10 m 的桶壁；夾爪中心在 root 前 0.272 m（車頭前 0.106 m）。(180,100) 時積木底只有 0.067 m，
        # 前進會頂到桶壁、積木掉在桶緣外。
        await robot.act("arm_to", {"x_mm": ARM_RELEASE_HIGH[0], "y_mm": ARM_RELEASE_HIGH[1]})
        await asyncio.sleep(0.3)
        # 前進到車頭離桶壁 3 cm：距離用導航回報的「到桶中心距離」算（ToF 這時看到的是手上的積木，不能用）；
        # 桶外半寬從地標 bbox 取（玩具桶 0.245、垃圾桶 0.195）、車頭前緣 0.166 → 積木中心落在桶壁內側約 5-6 cm
        d = float(res.get("distance_to_target_m") or 0.6)
        lm_bin = ((await robot.state()).get("static_landmarks") or {}).get(lname) or {}
        bb = lm_bin.get("bbox")
        half = min(bb[1] - bb[0], bb[3] - bb[2]) / 2 if bb else 0.245
        creep = max(0.0, min(0.30, round(d - half - 0.166 - 0.03, 2)))
        if creep >= 0.02:
            await robot.act("move_chassis", {"forward_m": creep, "right_m": 0, "turn_left_deg": 0, "push": True})
        how = f"over the bin (arm raised above the rim, moved {creep:.2f} m up to the wall)"
    elif on_surface:
        # 升降柱已經在桌面高度：低位釋放姿態的手指在桌面上方約 0.14 m，跟放在地墊上同一套幾何。
        # 夾爪中心在車頭前約 0.15 m，所以前進到車頭離桌緣 3 cm，物件才會落在桌面內側（不是掉在桌緣外）。
        await robot.act("arm_to", {"x_mm": ARM_RELEASE_LOW[0], "y_mm": ARM_RELEASE_LOW[1]})
        await asyncio.sleep(0.3)
        d = float(res.get("distance_to_target_m") or 0.6)
        creep = max(0.0, min(0.30, round(d - on_surface["half_m"] - 0.166 - 0.03, 2)))
        if creep >= 0.02:
            await robot.act("move_chassis", {"forward_m": creep, "right_m": 0, "turn_left_deg": 0, "push": True})
        how = (f"on top of the {on_surface['name']} (arm lifted {lift_mm} mm to its {on_surface['top_m']:.2f} m top, "
               f"moved {creep:.2f} m up to its edge)")
    else:
        await robot.act("arm_to", {"x_mm": ARM_RELEASE_LOW[0], "y_mm": ARM_RELEASE_LOW[1]})
        how = "on the surface (arm lowered)"
    await _snap(robot, "place: positioned over the target")
    await robot.act("gripper", {"state": "open"})
    await robot.act("set_held", {"tof_mm": None})
    await _snap(robot, "place: gripper opened to release")
    await asyncio.sleep(1.0)
    await robot.act("move_chassis", {"forward_m": -0.25, "right_m": 0, "turn_left_deg": 0})
    if lift_mm:
        await robot.act("lift_arm", {"height_mm": 0})
        how += ", arm lift lowered back to 0"
    await robot.act("recenter_arm", {})
    tof = await _tof(robot)
    load = await _gripper_load(robot)
    if load is not None and load >= GRIPPER_HELD_DELTA_N:
        return False, (f"place({target}) {where}: opened the gripper {how}, but the wrist force sensor still carries "
                       f"{load:+.2f} N — the object is STILL in the gripper (stuck between the fingers). Try gripper(open) "
                       f"and a small arm move, then check again.")
    return True, (f"place({target}) {where}: released {how}; gripper opened, backed up 0.25 m, arm recentred "
                  f"(front sensor now {tof if tof is not None else 9999} mm"
                  + (f", gripper load back to {load:+.2f} N = empty" if load is not None else "") +
                  "). Check the front camera if the task needs visual confirmation.")


async def drive_through(robot, llm, structure: str) -> Tuple[bool, str]:
    state = await robot.state()
    landmarks = state.get("static_landmarks") or {}
    name = _match_landmark(structure, landmarks)
    if name in ("tunnel", "platform"):
        res = await robot.act("drive_through", {"name": name}, timeout=NAV_TIMEOUT_S)
        return bool(res.get("ok")), f"drive_through({structure}) -> {name}: {res.get('message')}"
    # 不是已知結構：用俯視相機找兩端，再分兩段導航
    jpg = await robot.overhead_frame()
    if jpg is None:
        return False, f"drive_through({structure}): no overhead camera available"
    axis = await perception.locate_tunnel_axis(llm, jpg, structure, state.get("world_xy"), pseudo_yaw(state))
    if not axis.found:
        return False, f"drive_through({structure}): {axis.describe(structure)}"
    (nx, ny), (fx, fy) = axis.near_end_world, axis.far_end_world
    L = math.hypot(fx - nx, fy - ny) or 1.0
    ux, uy = (fx - nx) / L, (fy - ny) / L
    entrance = (nx - ux * 0.6, ny - uy * 0.6)
    exit_pt = (fx + ux * 0.6, fy + uy * 0.6)
    r1 = await robot.act("navigate_to", {"x": entrance[0], "y": entrance[1], "tolerance_m": 0.12, "face": False, "label": str(structure)[:40]},
                         timeout=NAV_TIMEOUT_S)
    if not r1.get("ok"):
        return False, f"drive_through({structure}): could not reach the entrance — {r1.get('message')}"
    r2 = await robot.act("navigate_to", {"x": exit_pt[0], "y": exit_pt[1], "tolerance_m": 0.15, "face": False, "label": str(structure)[:40]},
                         timeout=NAV_TIMEOUT_S)
    return bool(r2.get("ok")), (f"drive_through({structure}) via overhead ends {axis.near_end_world} -> "
                                f"{axis.far_end_world}: {r2.get('message')}")
