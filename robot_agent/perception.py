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

from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL, UserMessage

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
    half = 3
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
    axis_bearing_deg: float = 0.0  # 通道從近端指向遠端的世界方位角 (跟 yaw_deg 同一個座標系)
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
            f"there: turn until your yaw_deg is actually close to {self.axis_bearing_deg:.0f} degrees "
            f"(turn_left_deg = that bearing minus your current yaw_deg, normalized to -180..180) — only once "
            f"your OWN yaw_deg telemetry confirms this, not before, may you judge what the front camera shows. "
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
    axis_vy = -(far_end[1] - near_end[1])
    axis_bearing = math.degrees(math.atan2(axis_vy, axis_vx))

    # 建議站位：從近端往「遠離遠端」的方向延伸 standoff_m，站在通道外面正對著看進去，
    # 而不是站在通道正中央 (locate_overhead 給的那個點) 從旁邊斜切過去看。
    raw_dx, raw_dy = near_end[0] - far_end[0], near_end[1] - far_end[1]
    raw_len = math.hypot(raw_dx, raw_dy) or 1.0
    approach_world = (near_end[0] + raw_dx / raw_len * standoff_m,
                      near_end[1] + raw_dy / raw_len * standoff_m)

    vx = approach_world[0] - robot_world_xy[0]
    vy = -(approach_world[1] - robot_world_xy[1])
    distance = math.hypot(vx, vy)
    target_world_bearing = math.degrees(math.atan2(vy, vx))
    turn_left = (target_world_bearing - robot_yaw_deg + 180) % 360 - 180

    return TunnelAxis(found=True, near_end_world=(round(near_end[0], 2), round(near_end[1], 2)),
                      far_end_world=(round(far_end[0], 2), round(far_end[1], 2)),
                      axis_bearing_deg=round(axis_bearing, 1),
                      distance_m=round(distance, 2), turn_left_deg=round(turn_left, 1), raw=raw)


async def locate_overhead(llm, jpg: Optional[bytes], obj: str,
                          robot_world_xy, robot_yaw_deg: Optional[float],
                          depth_png: Optional[bytes] = None) -> LocatedOverhead:
    """跟 locate() 分開用不同 prompt（見 OVERHEAD_POINT_PROMPT 的理由），換算方式也不一樣：
    俯視圖是正交投影、由上往下看，pointing 回傳的正規化座標直接對應世界座標的位移量
    （不是像 locate() 那樣算「畫面上的方位角」），所以不能沿用 Located.bearing_right_deg
    那套針孔相機公式。這裡的世界座標平面假設跟 add_overhead_camera.py 裡的一致：
    x_norm=500 是攝影機正下方，y_norm 越大代表世界 y 越大（往上看的圖，not 螢幕座標）。

    world_xy（CoppeliaSim 的原始世界座標，robot_server.py 直接查來的）跟 yaw_deg（RoboMaster
    SDK 自己的姿態）不是同一個座標系：拿實際跑過的任務紀錄核對過，同一段純前進的位移，
    用 position_m 算出來的方向角跟 yaw 對得上，但用 world_xy 算出來的方向角剛好差一個負號
    （y 軸相反，x 軸沒事）。這裡的俯視攝影機座標換算全部是基於 world_xy，所以算完向量之後，
    要先把 y 分量反過來，再拿去跟 yaw_deg 比較，不然算出來的建議轉向角完全是反的。
    """
    if robot_world_xy is None or robot_yaw_deg is None:
        return LocatedOverhead(found=False, raw="no world_xy/yaw telemetry to convert pixels into a direction")
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
    vy = -(oy - robot_world_xy[1])  # world_xy 的 y 軸跟 yaw 的座標系相反，見上面的說明
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
