"""
感知工具：用 Gemini Robotics ER 的 pointing 能力，把「物體在畫面哪裡」變成「要轉幾度」。

為什麼需要它：只靠 LLM 看圖估角度，它會說「在右邊，轉 40 度」，結果轉過頭、目標跑出畫面、
再轉回來又過頭，來回震盪到步數用完。pointing 會回傳物體中心的正規化座標 [y, x] (0 到 1000)，
配上相機的水平視角，用針孔相機模型就能直接算出方位角，轉向從猜測變成計算。
"""

import base64
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from robot_agent.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL, UserMessage

# 相機水平視角 (度)。實體 EP 的鏡頭是 120 度廣角；模擬器的值不一定相同，用 tools/calibrate_fov.py 量出來再填。
CAMERA_HFOV_DEG = float(os.getenv("ROBOT_CAMERA_HFOV", "100"))

# 俯視攝影機：跟 add_overhead_camera.py 建立時用的參數要一致。攝影機架在世界座標原點正上方
# （build_playground_new.py 蓋場景時就是以房間中心為世界原點），所以只要知道正交視野寬度，
# 加上機器人在 /state 裡的 world_xy（robot_server.py 直接查 CoppeliaSim 給的真實世界座標，
# 跟 position_m 那個原點隨機的相對座標是兩回事），就能把俯視圖裡的像素座標換算成「機器人要
# 轉幾度、走幾公尺才會到那個東西前面」，不用再靠自轉一步步搜索、用猜的角度轉來轉去。
OVERHEAD_ORTHO_SIZE = float(os.getenv("ROBOT_OVERHEAD_ORTHO_SIZE", "7.0"))
OVERHEAD_CAMERA_WORLD_XY = (0.0, 0.0)
# 跟 add_overhead_camera.py 的 HEIGHT 一致：攝影機離地高度 (m)，用來把 /overhead_depth.png
# 的「攝影機到該點距離」換算成「該點離地面多高」——這是真正的高度感測，不是猜的。
OVERHEAD_CAMERA_HEIGHT_M = float(os.getenv("ROBOT_OVERHEAD_CAMERA_HEIGHT", "4.5"))
# 頂面比地板高過這個門檻 (m) 才算「立體物件」，低於這個當作地墊/地貼。3cm 留了一點誤差空間
# 給深度圖的雜訊，同時比薄地墊 (通常 <1cm) 厚很多、比最矮的箱子 (通常 >5cm) 淺很多。
FLAT_HEIGHT_THRESHOLD_M = 0.03

POINT_PROMPT = (
    'Point to the {obj} in the image. Answer ONLY with JSON in the format '
    '[{{"point": [y, x], "label": "<label>"}}], where the point is in [y, x] format normalized to 0-1000. '
    'Point to the centre of the object. If several match, return the largest or closest one first. '
    'If the {obj} is not visible in the image, answer with an empty list [].'
)

# 俯視圖用的獨立 prompt：這是由上往下看，把高度整個壓扁了，地上的一塊色墊跟一個同色箱子的頂面
# 從正上方看會長得一樣。實測發現不特別提醒的話，pointing 常常把面積比較大的地墊指成「箱子」
# （地墊通常鋪得比箱子頂面大），所以這裡明講「優先找立體物件、不要選地墊/地貼」。
OVERHEAD_POINT_PROMPT = (
    'This is a top-down view of a room, looking straight down. It flattens all height away, so a flat floor '
    'mat/rug/floor marking and the flat top of a real 3-D object (a box, bin, or piece of furniture) of the '
    'same colour can look identical from directly above. Point to the {obj} in the image. If {obj} refers to '
    'a physical object with height (a box, bin, block, ball, piece of furniture, etc.), prefer a candidate '
    'that looks like a distinct standalone item over a flat mat/rug/marking painted on the floor, even if the '
    'mat is larger or more prominent — only pick a flat floor marking if {obj} explicitly asks for a mat, rug, '
    'zone, or marking. Answer ONLY with JSON in the format '
    '[{{"point": [y, x], "label": "<label>"}}], where the point is in [y, x] format normalized to 0-1000. '
    'If several candidates match, return the one most likely to be a real 3-D object first. '
    'If {obj} is not visible in the image, answer with an empty list [].'
)


@dataclass
class Located:
    found: bool
    x: int = 0            # 0 (最左) 到 1000 (最右)
    y: int = 0            # 0 (最上) 到 1000 (最下)；在地板上的物體，y 越大代表越近
    bearing_right_deg: float = 0.0
    label: str = ""
    raw: str = ""

    @property
    def turn_left_deg(self) -> float:
        """要讓物體置中，move_chassis 的 turn_left_deg 該填多少。物體在右邊就要右轉，也就是負值。"""
        return round(-self.bearing_right_deg, 1)

    def describe(self, obj: str) -> str:
        if not self.found:
            return f"'{obj}' is NOT visible in the current camera image."
        side = "right of" if self.bearing_right_deg > 2 else "left of" if self.bearing_right_deg < -2 else "at"
        where = "lower part (near)" if self.y > 650 else "middle (mid range)" if self.y > 450 else "upper part (far)"
        return (f"'{obj}' found at image x={self.x}/1000, y={self.y}/1000: {abs(self.bearing_right_deg):.0f} deg {side} "
                f"the image centre, in the {where} of the image. "
                f"To centre it use move_chassis(turn_left_deg={self.turn_left_deg}).")


def bearing_from_x(x_norm: float, hfov_deg: float = CAMERA_HFOV_DEG) -> float:
    """針孔相機模型：畫面水平位置 → 方位角 (右為正)。"""
    u = (x_norm - 500.0) / 500.0
    return math.degrees(math.atan(u * math.tan(math.radians(hfov_deg / 2.0))))


def parse_points(text: str) -> list:
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except Exception:
        return []
    return [d for d in data if isinstance(d, dict) and isinstance(d.get("point"), list) and len(d["point"]) == 2]


def _height_above_floor_m(depth_png: bytes, x_norm: int, y_norm: int) -> Optional[float]:
    """從 /overhead_depth.png 查詢正規化座標 (0-1000，跟 pointing 回傳的座標系一致) 那個點
    離地面多高。取點周圍一小塊區域，回報其中「離攝影機最近的那 15% 像素」的高度：
    以前取中位數，實測對 8 cm 的球會誤判成平面——深度圖 512 px 涵蓋 7 m、一個像素約
    1.4 cm，7x7 區塊約 9.6 cm 寬，球的投影只佔區塊一半左右，pointing 偏 1-2 個像素就有
    過半是地板，中位數必然落在地板上，agent 因此把找對的球當成地墊丟掉（連續兩次跑分
    都發生）。用低百分位的距離 (= 高百分位的高度) 只要有幾個像素落在物體上就判得出來，
    又不像 min 那樣被單一雜訊像素騙。"""
    arr = cv2.imdecode(np.frombuffer(depth_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    h, w = arr.shape[:2]
    px, py = int(x_norm / 1000.0 * w), int(y_norm / 1000.0 * h)
    half = max(3, int(round(3 * w / 512.0)))   # 512 px 時 7x7（約 9.6 cm）；解析度提高後維持同樣的實際大小
    patch = arr[max(0, py - half):min(h, py + half + 1), max(0, px - half):min(w, px + half + 1)]
    patch = patch[patch > 0]  # 0 代表深度讀取失敗/超出範圍，排除掉
    if patch.size == 0:
        return None
    distance_m = float(np.percentile(patch, 15)) / 1000.0
    return OVERHEAD_CAMERA_HEIGHT_M - distance_m


@dataclass
class LocatedOverhead:
    found: bool
    world_xy: tuple = (0.0, 0.0)
    distance_m: float = 0.0
    turn_left_deg: float = 0.0
    height_m: Optional[float] = None  # None 代表沒有深度資料可查，不代表高度是 0
    raw: str = ""

    def describe(self, obj: str) -> str:
        if not self.found:
            return f"'{obj}' is NOT visible in the overhead view."
        msg = (f"'{obj}' seen from above at world ({self.world_xy[0]:.2f}, {self.world_xy[1]:.2f}), "
               f"about {self.distance_m:.2f}m from the robot. To face it, "
               f"move_chassis(turn_left_deg={self.turn_left_deg:.1f}); then it should be roughly straight "
               f"ahead at {self.distance_m:.2f}m, but re-check with locate() once you get close since this "
               f"is a rough overhead estimate, not a precise one.")
        if self.height_m is not None:
            # 這是深度感測器量出來的真實高度，不是猜的，直接蓋過 OVERHEAD_POINT_PROMPT 那種
            # 只能靠提示詞去引導模型別選地墊的做法——現在有實測數據可以直接判斷。
            if self.height_m < FLAT_HEIGHT_THRESHOLD_M:
                msg += (f" Depth sensor: this candidate is essentially FLAT (~{self.height_m * 100:.0f}cm above "
                        f"the floor) — almost certainly a floor mat/marking, not a real 3-D object. If you need "
                        f"an actual object, treat this as the wrong candidate and look elsewhere.")
            else:
                msg += f" Depth sensor: this candidate stands about {self.height_m * 100:.0f}cm above the floor."
        return msg


async def _point(llm, jpg: Optional[bytes], prompt: str):
    """實際呼叫 LLM 的 pointing 能力，回傳 (x_norm, y_norm, label, raw) 或 None（沒找到/沒圖）。"""
    if not jpg:
        return None, "no camera image"
    b64 = base64.b64encode(jpg).decode("ascii")
    msg = UserMessage(content=[
        ContentPartImageParam(image_url=ImageURL(url=f"data:image/jpeg;base64,{b64}", media_type="image/jpeg")),
        ContentPartTextParam(text=prompt),
    ])
    resp = await llm.ainvoke([msg])
    raw = resp.completion if isinstance(resp.completion, str) else str(resp.completion)
    pts = parse_points(raw)
    if not pts:
        return None, raw[:200]
    y, x = float(pts[0]["point"][0]), float(pts[0]["point"][1])
    return (int(x), int(y), str(pts[0].get("label", ""))), raw[:200]


OVERHEAD_ROI_PROMPT = (
    'This is a CLOSE-UP crop of a top-down view of a room, centred on the {roi}. It is magnified, so small objects '
    'look large here. Point to the {obj} in this crop. Answer ONLY with JSON in the format '
    '[{{"point": [y, x], "label": "<label>"}}], where the point is in [y, x] format normalized to 0-1000 within THIS '
    'image. If {obj} is not visible in this crop, answer with an empty list [].'
)


def world_to_overhead_norm(wx: float, wy: float):
    """世界座標 → 俯視圖正規化座標 (x_norm, y_norm)，是 locate_overhead 裡那個換算的反函數。"""
    half = OVERHEAD_ORTHO_SIZE / 2.0
    return (500.0 - (wx - OVERHEAD_CAMERA_WORLD_XY[0]) / half * 500.0,
            500.0 + (wy - OVERHEAD_CAMERA_WORLD_XY[1]) / half * 500.0)


async def _point_in_roi(llm, jpg: Optional[bytes], obj: str, roi_world, roi_name: str):
    """把俯視圖以 roi_world=(cx, cy, half_m) 為中心裁切、放大到約 512 px 再問，回傳的座標換算回整張圖的正規化座標。
    俯視圖 512 px 涵蓋 7 m，一顆 6 cm 的球只有 4 px：2026-10-20 實測「lime green ball」三次有兩次被指到
    別處的綠色積木。放大 8 倍後球有 30 幾 px，模型認得出來，而且裁切範圍本身就限制了候選只能在那張桌子附近。"""
    if not jpg:
        return None, "no camera image"
    arr = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None, "cannot decode overhead image"
    H, W = arr.shape[:2]
    cx, cy, half_m = roi_world
    xs = [world_to_overhead_norm(cx + d, cy)[0] for d in (-half_m, half_m)]
    ys = [world_to_overhead_norm(cx, cy + d)[1] for d in (-half_m, half_m)]
    x0, x1 = sorted(max(0, min(W - 1, int(v / 1000.0 * W))) for v in xs)
    y0, y1 = sorted(max(0, min(H - 1, int(v / 1000.0 * H))) for v in ys)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None, "roi too small"
    crop = arr[y0:y1 + 1, x0:x1 + 1]
    scale = max(1, int(round(512.0 / max(crop.shape[:2]))))
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", big, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None, "cannot encode crop"
    prompt = OVERHEAD_ROI_PROMPT.format(obj=obj, roi=roi_name or "target area")
    point, raw = await _point(llm, buf.tobytes(), prompt)
    if point is None:
        point, raw = await _point(llm, buf.tobytes(), prompt)   # 同一張圖偶爾回空陣列（實測 3 次 1 次），再問一次
    raw = "[close-up x%d of %.1f m around (%.2f, %.2f)] %s" % (scale, 2 * half_m, cx, cy, raw)
    if point is None:
        return None, raw
    xc, yc, label = point
    x_full = (x0 + xc / 1000.0 * (x1 - x0)) / W * 1000.0
    y_full = (y0 + yc / 1000.0 * (y1 - y0)) / H * 1000.0
    return (int(round(x_full)), int(round(y_full)), label), raw


OVERHEAD_SCAN_PROMPT = (
    'This is a top-down view of a room, looking straight down. Point to EVERY {cat} you can see, each one as a separate '
    'point, including tiny ones only a few pixels wide and ones standing on top of furniture. Do NOT list the heap of '
    'balls inside the ball pit (the large rectangle with a black-and-white checkered border) as individual items, and do '
    'not list flat floor mats. Answer ONLY with JSON in the format '
    '[{{"point": [y, x], "label": "<colour and kind, e.g. orange ball>"}}, ...], points normalized to 0-1000. '
    'If none is visible, answer with an empty list [].'
)


OVERHEAD_SCAN_PROMPT_ALT = (
    'Top-down view of a playroom. List every {cat} lying around the room — on the floor tiles, on mats, or on top of '
    'furniture — as separate points. Skip the heap inside the checkered ball pit. Reply ONLY with JSON '
    '[{{"point": [y, x], "label": "<colour> ..."}}, ...] (0-1000 normalized), or [] if there is none.'
)


async def _points_all(llm, jpg: Optional[bytes], prompt: str):
    """跟 _point 一樣，但把模型回的每一個點都留下來：[(x_norm, y_norm, label), ...]。"""
    if not jpg:
        return [], "no camera image"
    b64 = base64.b64encode(jpg).decode("ascii")
    msg = UserMessage(content=[
        ContentPartImageParam(image_url=ImageURL(url=f"data:image/jpeg;base64,{b64}", media_type="image/jpeg")),
        ContentPartTextParam(text=prompt),
    ])
    resp = await llm.ainvoke([msg])
    raw = resp.completion if isinstance(resp.completion, str) else str(resp.completion)
    pts = parse_points(raw)
    out = []
    for p in pts:
        try:
            out.append((int(float(p["point"][1])), int(float(p["point"][0])), str(p.get("label", ""))))
        except Exception:
            continue
    return out, raw[:300]


def overhead_norm_to_world(x_norm: float, y_norm: float):
    half = OVERHEAD_ORTHO_SIZE / 2.0
    return (OVERHEAD_CAMERA_WORLD_XY[0] + (500.0 - x_norm) / 500.0 * half,
            OVERHEAD_CAMERA_WORLD_XY[1] + (y_norm - 500.0) / 500.0 * half)


# 場景裡會出現的小物件顏色（Env/build_playground_new.py 與 add_*.py 的 RGB），轉成 OpenCV 的色相 (0-179)
_COLOUR_RGB = {"orange": (0.95, 0.50, 0.12), "purple": (0.55, 0.25, 0.75), "pink": (0.95, 0.45, 0.70), "cyan": (0.20, 0.80, 0.85),
               "lime": (0.62, 0.95, 0.12), "red": (0.85, 0.16, 0.16), "yellow": (0.97, 0.78, 0.12), "green": (0.22, 0.60, 0.30),
               "blue": (0.16, 0.42, 0.85)}


def _cv_hue(rgb):
    px = np.uint8([[[int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)]]])
    return int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0, 0])


_COLOUR_HUE = {name: _cv_hue(rgb) for name, rgb in _COLOUR_RGB.items()}


def _colour_blob_candidates(arr, landmarks: dict):
    """確定性的候選來源：按色相分割、找小的飽和色塊。球（6 cm）跟小積木（7 cm）在 1024 px 的俯視圖上約 40-120 px 面積；
    地墊、泡棉方塊、柱子、滑梯這些同色的大東西用面積上限排掉，球池裡那一堆球用球池的 bbox 排掉。
    為什麼要有這一道（2026-10-21）：只靠指向模型列候選，橘球四次有兩次沒被列出來（r08 數成 3 顆）；
    色塊檢測不會漏，種類再交給拼貼放大 + 深度高度判。回傳 [(x_norm, y_norm, colour), ...]。"""
    H, W = arr.shape[:2]
    pit = (landmarks or {}).get("ball_pit", {}).get("bbox")
    if not pit:
        return []   # 沒有球池範圍就不做：池裡 100 多顆球會把候選名單塞爆
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    scale = (W / 1024.0) ** 2
    a_min, a_max = 18 * scale, 360 * scale
    out = []
    for colour, hue in _COLOUR_HUE.items():
        d = np.abs(hsv[..., 0].astype(np.int16) - hue)
        d = np.minimum(d, 180 - d)
        # 飽和度門檻：木色家具（桌子、長椅，S≈125）受光的邊緣會落進橘/黃的色相，所以橘/黃/紅要 >150；
        # 紫球渲染後只有 145-165、粉紅 134（跟木色的色相差很遠，不會混），這兩個放寬到 110
        s_min = 110 if colour in ("pink", "purple") else 150
        mask = ((d <= 7) & (hsv[..., 1] > s_min) & (hsv[..., 2] > 60)).astype(np.uint8)
        n, _lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        for k in range(1, n):
            area = stats[k, cv2.CC_STAT_AREA]
            w_, h_ = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            fill = area / float(max(1, w_ * h_))
            if not (a_min <= area <= a_max) or max(w_, h_) > 1.8 * max(1, min(w_, h_)) or fill < 0.5:
                continue   # 太大/太小、細長（護欄、陰影）或不緊實（大面積物件的邊緣碎片）；球 fill≈0.78、方塊≈1.0
            x_norm, y_norm = cents[k][0] / W * 1000.0, cents[k][1] / H * 1000.0
            wx, wy = overhead_norm_to_world(x_norm, y_norm)
            if pit[0] - 0.05 <= wx <= pit[1] + 0.05 and pit[2] - 0.05 <= wy <= pit[3] + 0.05:
                continue
            name = colour
            if colour == "orange":
                # 深棕色的紙盒色相跟橘球一樣（約 14），用明度分：橘球 V≈240、深棕紙盒 V≈90
                v_mean = float(hsv[..., 2][_lab == k].mean())
                if v_mean < 150:
                    name = "brown"
            out.append((x_norm, y_norm, name, round(float(fill), 2)))   # fill：圓形色塊約 0.78、方形約 0.95-1.0
    return out


def _kind_from_colour_height(colour: str, height_rel: Optional[float], kind: str) -> str:
    """色塊候選的種類由「顏色 + 高度」決定，不靠拼貼圖猜（圓/方的 fill 在 1024 px 下太吵，0.7-0.9 混在一起；
    2026-10-21 拼貼圖把桃紅方罐標成「pink ball」，scan balls 列出 6 顆球）。這個場景的顏色分工：
    積木＝紅/藍/綠/黃；球＝橘/紫/萊姆；桃紅＝球（柱子上頂點 0.164 m）或罐（22 cm 方罐 → 量到 ≥ 0.19）；
    青＝球或瓶（24 cm 方瓶 → ≥ 0.19）。杯（白）、盒（米白）飽和度低不會成為色塊候選，由拼貼圖判。"""
    c = (colour or "").lower()
    if c in ("red", "blue", "green", "yellow"):
        # 25 cm 的泡棉方塊/圓柱顏色跟小積木一樣，量到 ≥ 20 cm 就不是積木（夾不起來，只能推）
        return "foam (big, not graspable)" if (height_rel is not None and height_rel >= 0.20) else "block"
    if c in ("orange", "purple", "lime"):
        return "ball"
    if c in ("pink", "magenta"):
        return "can" if (height_rel is not None and height_rel >= 0.19) else "ball"
    if c in ("cyan", "teal"):
        return "bottle" if (height_rel is not None and height_rel >= 0.185) else "ball"   # 20 cm 方瓶量到約 0.19-0.20、青球 0.165
    if c in ("brown", "dark brown"):
        return "carton"
    if c in ("white", "grey", "gray", "light grey", "light gray"):
        return "cup"
    return kind


def _place_of(world_xy, height_m: Optional[float], landmarks: dict) -> str:
    """物件在哪：用遙測 static_landmarks 的 bbox/top_m 判斷在桌上、桶裡、墊子上、球池裡、平台下還是地上。"""
    for name, lm in (landmarks or {}).items():
        bb = lm.get("bbox") if isinstance(lm, dict) else None
        if not bb or not (bb[0] - 0.02 <= world_xy[0] <= bb[1] + 0.02 and bb[2] - 0.02 <= world_xy[1] <= bb[3] + 0.02):
            continue
        if name.startswith("bin_"):
            return f"inside {name}"
        if name.startswith("mat_"):
            return f"on {name}"
        if name == "ball_pit":
            return "inside the ball pit"
        if name == "tunnel":
            return "inside the tunnel"
        top = lm.get("top_m")
        if top and top >= 0.10:
            if height_m is not None and height_m < top - 0.10:
                return f"on the floor underneath the {name}"
            return f"on top of the {name} (a raised surface {top:.2f} m high — needs the arm lift)"
    return "on the floor"


MOSAIC_PROMPT = (
    'Each numbered tile in this image is a magnified top-down close-up (about 0.4 m across) of one small object on a '
    'playroom floor or on furniture. For EVERY tile, give its number, the object\'s colour (one word, e.g. orange, purple, '
    'pink, cyan, lime, yellow, red, blue, green, brown, white) and its kind: "ball" (round), "block" (a small cube), "bottle" (tall thin cylinder), "can" (short cylinder), "cup" (short wider cylinder), "carton" (small tall box), "cylinder", '
    '"beam", "bin", or "other". Answer ONLY with JSON: [{{"tile": 1, "colour": "...", "kind": "..."}}, ...].'
)


def _build_mosaic(arr, items, half_m: float = 0.16, tile_px: int = 192, cols: int = 4):
    """把每個候選點周圍裁一小塊、放大，拼成一張編號的拼貼圖。回傳 (jpg bytes, 有效的 item 索引列表)。"""
    H, W = arr.shape[:2]
    tiles, idx = [], []
    for i, it in enumerate(items):
        wx, wy = it["world_xy"]
        xs = [world_to_overhead_norm(wx + d, wy)[0] for d in (-half_m, half_m)]
        ys = [world_to_overhead_norm(wx, wy + d)[1] for d in (-half_m, half_m)]
        x0, x1 = sorted(max(0, min(W - 1, int(v / 1000.0 * W))) for v in xs)
        y0, y1 = sorted(max(0, min(H - 1, int(v / 1000.0 * H))) for v in ys)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        tile = cv2.resize(arr[y0:y1 + 1, x0:x1 + 1], (tile_px, tile_px), interpolation=cv2.INTER_CUBIC)
        n = len(tiles) + 1
        cv2.putText(tile, str(n), (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, str(n), (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(tile, (0, 0), (tile_px - 1, tile_px - 1), (0, 0, 0), 2)
        tiles.append(tile)
        idx.append(i)
    if not tiles:
        return None, []
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * tile_px, cols * tile_px, 3), 255, dtype=np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas[r * tile_px:(r + 1) * tile_px, c * tile_px:(c + 1) * tile_px] = tile
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return (buf.tobytes() if ok else None), idx


async def _verify_mosaic(llm, jpg: bytes, items: list, category: str):
    """第二道：把每個候選放大成拼貼圖再問一次「這是什麼顏色、什麼東西」。整張俯視圖裡 7 cm 的積木跟 6 cm 的球
    都只有 4-5 px，指向模型把黃積木、紅積木、藍積木全標成「ball」（2026-10-21 實測），放大後圓的方的一眼就看得出來。
    回傳過濾後的 items（label 換成確認過的「顏色 種類」）；拼貼或解析失敗就原樣回傳，不要讓掃描整個失效。"""
    try:
        arr = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        mosaic, idx = _build_mosaic(arr, items)
        if mosaic is None:
            return items, "no tiles"
        b64 = base64.b64encode(mosaic).decode("ascii")
        msg = UserMessage(content=[
            ContentPartImageParam(image_url=ImageURL(url=f"data:image/jpeg;base64,{b64}", media_type="image/jpeg")),
            ContentPartTextParam(text=MOSAIC_PROMPT),
        ])
        resp = await llm.ainvoke([msg])
        raw = resp.completion if isinstance(resp.completion, str) else str(resp.completion)
        m = re.search(r"\[.*\]", raw, re.S)
        data = json.loads(m.group(0)) if m else []
        by_tile = {int(d["tile"]): d for d in data if isinstance(d, dict) and "tile" in d}
    except Exception as e:  # noqa: BLE001
        return items, f"mosaic check failed: {type(e).__name__}"
    cat = category.lower()
    want = ("ball",) if "ball" in cat else (("block", "cube", "brick") if any(w in cat for w in ("block", "cube", "brick")) else ())
    if any(w in cat for w in ("litter", "trash", "garbage", "rubbish", "waste")):
        want = ("bottle", "can", "cup", "carton")
    elif "bottle" in cat:
        want = ("bottle",)
    elif "can" in cat.split() or cat.endswith("cans"):
        want = ("can",)
    out = []
    for n, i in enumerate(idx, 1):
        d = by_tile.get(n)
        it = dict(items[i])
        if d:
            kind = str(d.get("kind", "")).lower().strip()
            colour = str(d.get("colour", "")).lower().strip()
            if it.get("fill") is not None:   # 色塊候選：顏色是算出來的
                colour = it.get("blob_colour") or colour
            rel = (it["height_m"] - (it.get("surface_top_m") or 0.0)) if it.get("height_m") is not None else None
            # 種類以「顏色 + 高度」的場景規則為準（拼貼圖把灰杯、棕盒都說成 ball）；規則沒涵蓋的顏色才用拼貼圖 + 高度仲裁
            new_kind = _kind_from_colour_height(colour, rel, "")
            kind = new_kind if new_kind else _kind_from_height(kind, it.get("height_m"), it.get("surface_top_m") or 0.0)
            it["label"] = (colour + " " + kind).strip() or it["label"]
            it["verified"] = True
            if want and not any(w in kind for w in want):
                continue   # 放大看不是題目要的種類（例如黃「球」其實是黃積木）
        out.append(it)
    return out, raw[:200]


def _kind_from_height(kind: str, height_m: Optional[float], surface_top_m: float) -> str:
    """球跟小積木的最後仲裁用深度高度：這個場景的球都放在 0.114 m 的柱子上，深度感測器量到的頂點高度 0.164-0.165 m，
    小積木 0.151 m（1024 px 實測，每個樣本都一樣，沒有重疊）；拼貼圖裡橘球在黑柱子上會被看成「block」，高度不會錯。
    放在桌上的用相對桌面的高度。只仲裁 ball/block 這兩類，泡棉方塊、圓柱之類維持拼貼圖的判斷。"""
    if height_m is None or not any(k in kind for k in ("ball", "block", "cube", "brick")):
        return kind   # 瓶/罐/杯/盒這些由拼貼圖判，高度帶跟球/積木重疊（罐子 16 cm 跟柱子上的球一樣高）
    rel = height_m - surface_top_m
    if 0.158 <= rel <= 0.195:
        return "ball"
    if 0.135 <= rel < 0.158:
        return "block"
    return kind


async def survey_overhead(llm, jpg: Optional[bytes], category: str, robot_world_xy, robot_yaw_deg: Optional[float],
                          depth_png: Optional[bytes] = None, landmarks: Optional[dict] = None):
    """列出俯視圖裡所有某一類的物件：每個都換成世界座標、距離、該轉幾度、深度高度、在哪個地標上/裡。
    給「題目沒直接講是哪一個」的任務用（最遠的球、桌上的那顆、離球池最近的積木…），LLM 拿到整張表才有辦法推理。"""
    if robot_world_xy is None or robot_yaw_deg is None:
        return [], "no world_xy/yaw telemetry"
    # 問兩次、兩種問法取聯集：單次指向常漏掉一兩個（2026-10-21 實測橘球四次有一次沒被列出來），漏掉比多列嚴重，
    # 多列的由拼貼放大 + 深度高度那一道過濾掉
    pts, raw = await _points_all(llm, jpg, OVERHEAD_SCAN_PROMPT.format(cat=category))
    pts2, raw2 = await _points_all(llm, jpg, OVERHEAD_SCAN_PROMPT_ALT.format(cat=category))
    raw = raw[:150] + " | alt: " + raw2[:150]
    blobs = []
    try:
        arr = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            blobs = _colour_blob_candidates(arr, landmarks or {})
    except Exception:
        blobs = []
    # 色塊候選排前面：它的顏色標籤是算出來的、不會漏；指向模型的點只補它沒抓到的
    pts = [(x, y, c + " ?", f) for x, y, c, f in blobs] + [(x, y, l, None) for x, y, l in list(pts) + list(pts2)]
    raw += " | blobs: %d" % len(blobs)
    items = []
    for x, y, label, fill in pts:
        wx, wy = overhead_norm_to_world(x, y)
        if any(math.hypot(wx - it["world_xy"][0], wy - it["world_xy"][1]) < 0.10 for it in items):
            continue  # 同一個東西被點兩次（球/積木本身 6-7 cm，兩個不同的物件不會近到 10 cm 內）
        height = None
        if depth_png:
            try:
                height = _height_above_floor_m(depth_png, x, y)
            except Exception:
                height = None
        vx, vy = wx - robot_world_xy[0], wy - robot_world_xy[1]   # robot_yaw_deg 現在是世界朝向（逆時針為正），直接用世界幾何
        bearing = math.degrees(math.atan2(vy, vx))
        if height is not None and height < 0.03 and not re.search(r"mat|flat|marking|sticker|tape", category or "", re.I):
            continue   # 1 cm 高的同色碎片是地墊/地貼，不是球/積木/垃圾（h01 把綠墊子列成兩個 green block）
        place = _place_of((wx, wy), height, landmarks or {})
        top = 0.0
        if place.startswith("on top of the "):
            lm = (landmarks or {}).get(place.split("on top of the ", 1)[1].split(" ")[0]) or {}
            top = float(lm.get("top_m") or 0.0)
        items.append({"label": label or category, "world_xy": (round(wx, 2), round(wy, 2)),
                      "distance_m": round(math.hypot(vx, vy), 2),
                      "turn_left_deg": round((bearing - robot_yaw_deg + 180) % 360 - 180, 1),
                      "height_m": None if height is None else round(height, 3),
                      "surface_top_m": top, "place": place, "fill": fill, "blob_colour": (label[:-2] if fill is not None else None)})
    if items:
        items, raw2 = await _verify_mosaic(llm, jpg, items, category)
        raw = raw + " | mosaic: " + raw2
    items.sort(key=lambda it: it["distance_m"])
    return items, raw


async def _point_pair(llm, jpg: Optional[bytes], prompt: str):
    """跟 _point 一樣，但預期回傳兩個點 (給通道/隧道的兩端用)。回傳 (points, raw)，
    points 是 [(x,y), (x,y)] 或 None (沒找到/沒圖/模型只給了一個點)。"""
    if not jpg:
        return None, "no camera image"
    b64 = base64.b64encode(jpg).decode("ascii")
    msg = UserMessage(content=[
        ContentPartImageParam(image_url=ImageURL(url=f"data:image/jpeg;base64,{b64}", media_type="image/jpeg")),
        ContentPartTextParam(text=prompt),
    ])
    resp = await llm.ainvoke([msg])
    raw = resp.completion if isinstance(resp.completion, str) else str(resp.completion)
    pts = parse_points(raw)
    if len(pts) < 2:
        return None, raw[:200]
    pair = [(int(float(p["point"][1])), int(float(p["point"][0]))) for p in pts[:2]]
    return pair, raw[:200]


# 跟 OVERHEAD_POINT_PROMPT 不同：這裡要的不是「東西在哪」，是「這個東西的兩端在哪」，
# 用來算出通道朝哪個方向、該站在哪一端外面才是真的「看得進去」而不是從側面斜看。
TUNNEL_AXIS_PROMPT = (
    'This is a top-down view of a room. Find the {obj} — an elongated structure you could walk or look through '
    'lengthwise (a tunnel, corridor, gate, archway, or similar). Point to its TWO short ends — the two openings '
    'you could look straight through, one at each end of its long axis — NOT its long sides. Answer ONLY with '
    'JSON in the format [{{"point": [y, x], "label": "end_a"}}, {{"point": [y, x], "label": "end_b"}}], each '
    'point normalized to 0-1000. If {obj} is not visible, or is not an elongated pass-through structure with '
    'two ends, answer with an empty list [].'
)


@dataclass
class TunnelAxis:
    """對應 locate_overhead 只給「中心點在哪」的不足：這裡給的是「這個通道朝哪個方向」，
    因為光是轉向面對一個細長物體的中心，常常還是從側面斜看，不是真的順著它的長軸看進去。"""
    found: bool
    near_end_world: tuple = (0.0, 0.0)
    far_end_world: tuple = (0.0, 0.0)
    axis_bearing_deg: float = 0.0  # 通道從近端指向遠端的世界方位角（跟 world_heading_deg 同一個座標系，逆時針為正）
    distance_m: float = 0.0        # 到建議站位 (近端外側一小段) 的距離
    turn_left_deg: float = 0.0     # 從目前朝向轉到面向建議站位需要的角度
    raw: str = ""

    def describe(self, obj: str) -> str:
        if not self.found:
            return (f"'{obj}' does not look like an elongated pass-through structure with two ends from "
                    f"directly above, or is not visible in the overhead view.")
        return (
            f"'{obj}' runs between world {self.near_end_world} (the end nearer to you) and "
            f"{self.far_end_world} (the far end). Its through-axis points at world bearing "
            f"{self.axis_bearing_deg:.0f} degrees — that is the heading you must face to look straight down its "
            f"length, not just toward its centre. This call only located it; you have not moved and have not "
            f"looked through it, so you have no evidence yet either way — do not call done() yet. "
            f"Step 1: move_chassis(turn_left_deg={self.turn_left_deg:.1f}) "
            f"then drive most of the {self.distance_m:.2f}m to get near its entrance. Step 2, once you are "
            f"there: turn until your world_heading_deg is actually close to {self.axis_bearing_deg:.0f} degrees "
            f"(turn_left_deg = that bearing minus your current world_heading_deg, normalized to -180..180) — only once "
            f"your OWN world_heading_deg telemetry confirms this, not before, may you judge what the front camera shows. "
            f"Facing the object's centre from an angle, or simply knowing the axis bearing, is not the same as "
            f"being aligned with it and is not evidence you can or cannot see through it."
        )


async def locate_tunnel_axis(llm, jpg: Optional[bytes], obj: str,
                             robot_world_xy, robot_yaw_deg: Optional[float],
                             standoff_m: float = 0.8) -> TunnelAxis:
    """跟 locate_overhead 同一種俯視圖換算方式 (像素 -> 世界座標，Y 軸方向的校正說明見
    locate_overhead 的 docstring)，差別是這裡要兩個點 (近端/遠端) 才能算出「軸線方向」，
    不是只算「中心點方位」。"""
    if robot_world_xy is None or robot_yaw_deg is None:
        return TunnelAxis(found=False, raw="no world_xy/yaw telemetry to convert pixels into a direction")
    pts, raw = await _point_pair(llm, jpg, TUNNEL_AXIS_PROMPT.format(obj=obj))
    if pts is None:
        return TunnelAxis(found=False, raw=raw)
    half = OVERHEAD_ORTHO_SIZE / 2.0

    def to_world(x, y):
        ox = OVERHEAD_CAMERA_WORLD_XY[0] + (500.0 - x) / 500.0 * half
        oy = OVERHEAD_CAMERA_WORLD_XY[1] + (y - 500.0) / 500.0 * half
        return ox, oy

    def dist_from_robot(w):
        vx = w[0] - robot_world_xy[0]
        vy = -(w[1] - robot_world_xy[1])
        return math.hypot(vx, vy)

    (x1, y1), (x2, y2) = pts
    w1, w2 = to_world(x1, y1), to_world(x2, y2)
    near_end, far_end = (w1, w2) if dist_from_robot(w1) <= dist_from_robot(w2) else (w2, w1)

    # 通道方向 (近端 -> 遠端)，換算到跟 yaw_deg 同一個座標系 (world_xy 的 y 軸是反的，
    # 理由跟 locate_overhead 裡量到的那組真實移動資料核對過的說明一樣)。
    axis_vx = far_end[0] - near_end[0]
    axis_vy = far_end[1] - near_end[1]
    axis_bearing = math.degrees(math.atan2(axis_vy, axis_vx))

    # 建議站位：從近端往「遠離遠端」的方向延伸 standoff_m，站在通道外面正對著看進去，
    # 而不是站在通道正中央 (locate_overhead 給的那個點) 從旁邊斜切過去看。
    raw_dx, raw_dy = near_end[0] - far_end[0], near_end[1] - far_end[1]
    raw_len = math.hypot(raw_dx, raw_dy) or 1.0
    approach_world = (near_end[0] + raw_dx / raw_len * standoff_m,
                      near_end[1] + raw_dy / raw_len * standoff_m)

    vx = approach_world[0] - robot_world_xy[0]
    vy = approach_world[1] - robot_world_xy[1]
    distance = math.hypot(vx, vy)
    target_world_bearing = math.degrees(math.atan2(vy, vx))
    turn_left = (target_world_bearing - robot_yaw_deg + 180) % 360 - 180

    return TunnelAxis(found=True, near_end_world=(round(near_end[0], 2), round(near_end[1], 2)),
                      far_end_world=(round(far_end[0], 2), round(far_end[1], 2)),
                      axis_bearing_deg=round(axis_bearing, 1),
                      distance_m=round(distance, 2), turn_left_deg=round(turn_left, 1), raw=raw)


async def locate_overhead(llm, jpg: Optional[bytes], obj: str,
                          robot_world_xy, robot_yaw_deg: Optional[float],
                          depth_png: Optional[bytes] = None,
                          roi_world=None, roi_name: str = "") -> LocatedOverhead:
    """跟 locate() 分開用不同 prompt（見 OVERHEAD_POINT_PROMPT 的理由），換算方式也不一樣：
    俯視圖是正交投影、由上往下看，pointing 回傳的正規化座標直接對應世界座標的位移量
    （不是像 locate() 那樣算「畫面上的方位角」），所以不能沿用 Located.bearing_right_deg
    那套針孔相機公式。這裡的世界座標平面假設跟 add_overhead_camera.py 裡的一致：
    x_norm=500 是攝影機正下方，y_norm 越大代表世界 y 越大（往上看的圖，not 螢幕座標）。

    2026-10-22 起：robot_yaw_deg 傳進來的是世界朝向 world_heading_deg（逆時針為正），turn_left = bearing − heading，
    不再對 y 分量取負號。（以前這裡把 y 反過來去配 SDK 的順時針 yaw，等於整套 turn_left 名稱反了，見 README 2026-10-22。）
    """
    if robot_world_xy is None or robot_yaw_deg is None:
        return LocatedOverhead(found=False, raw="no world_xy/yaw telemetry to convert pixels into a direction")
    if roi_world is not None:
        # 已知物件在哪個區域（例如任務說它在桌上）：只看那一塊的放大圖，小物件才認得出來，也不會指到別處同色的東西
        point, raw = await _point_in_roi(llm, jpg, obj, roi_world, roi_name)
    else:
        point, raw = await _point(llm, jpg, OVERHEAD_POINT_PROMPT.format(obj=obj))
    if point is None:
        return LocatedOverhead(found=False, raw=raw)
    x, y, _label = point
    height_m = None
    if depth_png:
        try:
            height_m = _height_above_floor_m(depth_png, x, y)
        except Exception:
            height_m = None
    half = OVERHEAD_ORTHO_SIZE / 2.0
    ox = OVERHEAD_CAMERA_WORLD_XY[0] + (500.0 - x) / 500.0 * half
    oy = OVERHEAD_CAMERA_WORLD_XY[1] + (y - 500.0) / 500.0 * half
    vx = ox - robot_world_xy[0]
    vy = oy - robot_world_xy[1]  # 2026-10-22 起 robot_yaw_deg 是世界朝向（逆時針為正），turn_left = bearing - heading
    distance = math.hypot(vx, vy)
    target_world_bearing = math.degrees(math.atan2(vy, vx))
    turn_left = (target_world_bearing - robot_yaw_deg + 180) % 360 - 180
    return LocatedOverhead(found=True, world_xy=(round(ox, 2), round(oy, 2)),
                           distance_m=round(distance, 2), turn_left_deg=round(turn_left, 1),
                           height_m=height_m, raw=raw)


async def locate(llm, jpg: Optional[bytes], obj: str) -> Located:
    point, raw = await _point(llm, jpg, POINT_PROMPT.format(obj=obj))
    if point is None:
        return Located(found=False, raw=raw)
    x, y, label = point
    return Located(found=True, x=x, y=y, bearing_right_deg=round(bearing_from_x(x), 1), label=label, raw=raw)
