"""
RobotSession：假裝成 BrowserSession 的機器人連線 (沿用 robot_bu.zip 原本的 duck-typing 設計，
browser_use.Agent 只碰 BrowserSession 的這幾個屬性/方法，同名提供就行，不用真的繼承)。

跟最早的版本比，多三件事：
    1. 每一步同時抓前方相機+俯視相機，合成成一張左右並排的圖當 screenshot——
       BrowserStateSummary.screenshot 只有單一欄位，不支援放兩張獨立圖片，用拼接繞過去。
    2. recent_events 除了遙測 JSON，還塞進 world_state (失敗計數/已知物體/剛恢復警示)、
       plan、long_term_summary，這幾個是 AdapterState 算好的，這裡只負責組字串塞進去。
    3. self.logger：browser_use/utils.py 的 time_execution_async 裝飾器，任何一個動作只要
       跑超過 0.25 秒就會去讀 browser_session.logger (getattr 沒給預設值，缺了就直接
       AttributeError)。實測發現 locate/face/locate_overhead 這種要等 LLM 回應的動作幾乎
       都會超過這個門檻，move_chassis/gripper 這種快動作則不會，所以只在意小工具、簡單
       smoke test 沒踩到——一定要補這個屬性，不然感知類動作在真的 Agent 底下每次都會炸。
"""

import base64
import io
import json
import logging
import uuid
from typing import Any, Optional

from browser_use.browser.profile import BrowserProfile
from browser_use.browser.views import BrowserStateSummary, TabInfo
from browser_use.dom.views import SerializedDOMState

from robot_agent.bu_common import AdapterState
from robot_agent.client import RobotClient

ROBOT_URL = "robot://robomaster-ep"


def _composite_side_by_side(front_jpg: Optional[bytes], overhead_jpg: Optional[bytes]) -> Optional[bytes]:
    """把前方相機+俯視相機拼成一張左右並排的圖，各自標上文字避免 LLM 搞混哪張是哪張。
    只有一邊有圖就直接回傳那一邊，不硬湊；合成失敗 (缺 PIL、圖片壞掉) 就退回前方相機。"""
    if front_jpg and not overhead_jpg:
        return front_jpg
    if overhead_jpg and not front_jpg:
        return overhead_jpg
    if not front_jpg and not overhead_jpg:
        return None
    try:
        from PIL import Image, ImageDraw
        front = Image.open(io.BytesIO(front_jpg)).convert("RGB")
        overhead = Image.open(io.BytesIO(overhead_jpg)).convert("RGB")
        h = max(front.height, overhead.height)
        front = front.resize((max(1, int(front.width * h / front.height)), h))
        overhead = overhead.resize((max(1, int(overhead.width * h / overhead.height)), h))
        gap, label_h = 8, 24
        canvas = Image.new("RGB", (front.width + gap + overhead.width, h + label_h), (30, 30, 30))
        canvas.paste(front, (0, label_h))
        canvas.paste(overhead, (front.width + gap, label_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), "FRONT camera (your own eye)", fill=(255, 255, 0))
        draw.text((front.width + gap + 4, 4), "OVERHEAD camera (fixed, top-down)", fill=(255, 255, 0))
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return front_jpg


class RobotSession:
    def __init__(self, robot: RobotClient, adapter_state: AdapterState):
        self.robot = robot
        self.adapter_state = adapter_state
        self.id = uuid.uuid4().hex
        self.cdp_url = None
        self.browser_profile = BrowserProfile()
        self.agent_focus = None
        self.cdp_client = None           # registry.execute_action 會讀，機器人沒有 CDP
        self.current_target_id = None
        self.downloaded_files: list = []
        self._cached_browser_state_summary: Optional[BrowserStateSummary] = None
        self.last_state: dict[str, Any] = {}
        self.mode = "?"
        self.logger = logging.getLogger("robot_session")

    # ---- Agent.run() 開頭與結尾會呼叫 ----
    async def start(self) -> None:
        health = await self.robot.health()
        if not health.get("ok"):
            raise RuntimeError(f"robot server not reachable: {health.get('message')}")
        self.mode = health.get("mode", "?")

    async def get_current_page_url(self) -> str:
        return ROBOT_URL

    async def kill(self) -> None:
        await self.robot.stop()

    async def stop(self) -> None:
        await self.robot.stop()

    # ---- 每一步 _prepare_context() 會呼叫 ----
    async def get_browser_state_summary(self, include_screenshot: bool = True,
                                        include_recent_events: bool = False, **_: Any) -> BrowserStateSummary:
        state = await self.robot.state()
        front_jpg = await self.robot.frame() if include_screenshot else None
        overhead_jpg = await self.robot.overhead_frame() if include_screenshot else None
        self.last_state = state

        composite = _composite_side_by_side(front_jpg, overhead_jpg) if include_screenshot else None

        events: dict[str, Any] = dict(state)
        events["world_state"] = json.loads(self.adapter_state.to_world_state_text())
        self.adapter_state.just_recovered = False  # 一次性警示，這次讓 LLM 看過就清掉
        events_text = json.dumps(events, ensure_ascii=False)
        if self.adapter_state.long_term_summary:
            events_text += f"\n<long_term_summary>\n{self.adapter_state.long_term_summary}\n</long_term_summary>"
        if self.adapter_state.latest_plan:
            events_text += f"\n<plan>\n{self.adapter_state.latest_plan}\n</plan>"

        summary = BrowserStateSummary(
            dom_state=SerializedDOMState(_root=None, selector_map={}),
            url=ROBOT_URL,
            title="RoboMaster EP",
            tabs=[TabInfo(url=ROBOT_URL, title="RoboMaster EP", target_id="robot0000")],
            screenshot=base64.b64encode(composite).decode("ascii") if composite else None,
            recent_events=events_text,
        )
        self._cached_browser_state_summary = summary
        return summary
