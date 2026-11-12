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
        return None, "object", "no overhead camera available to find it", None
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
            return None, "object", f"'{target}' is not visible in the overhead view (and it is not a known landmark)", None
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
        for pose_name, x_mm, y_mm in (("low (x=180,y=30)", 180, 30), ("high (x=90,y=120)", 90, 120)):
            r = await robot.act("arm_to", {"x_mm": x_mm, "y_mm": y_mm})
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
