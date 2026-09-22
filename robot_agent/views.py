"""
Agent 的輸出格式。
結構刻意跟 browser_use 的 AgentOutput 對齊 (thinking / evaluation / memory / next_goal / action[])，
這樣原本評估 GUI agent 用的 step 紀錄與 checkpoint 判定邏輯可以直接沿用。
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---- 各個動作的參數 ----
class MoveChassisParams(BaseModel):
    forward_m: float = Field(0.0, description="meters; positive = forward, negative = backward. Max 1.0 per action")
    right_m: float = Field(0.0, description="meters; positive = strafe right, negative = strafe left. Max 1.0")
    turn_left_deg: float = Field(0.0, description="degrees; positive = turn left (CCW), negative = turn right. Max 180")


class MoveArmParams(BaseModel):
    forward_mm: float = Field(0.0, description="relative mm; positive extends the arm forward. Max 100")
    up_mm: float = Field(0.0, description="relative mm; positive raises the arm. Max 100")


class ArmToParams(BaseModel):
    x_mm: float = Field(description="absolute forward reach of the arm, about 80 to 200")
    y_mm: float = Field(description="absolute height of the arm, about 10 to 130")


class RememberParams(BaseModel):
    note: str = Field(description="one short, general fact about your own body or sensors, useful in FUTURE tasks")


class GripperParams(BaseModel):
    state: Literal["open", "close"]


class WaitParams(BaseModel):
    seconds: float = Field(1.0, description="0 to 10 seconds")


class EmptyParams(BaseModel):
    pass


class LocateParams(BaseModel):
    object: str = Field(description="short description of what to find, e.g. 'red box'")


class AskHumanParams(BaseModel):
    question: str


class DoneParams(BaseModel):
    success: bool
    text: str = Field(description="final report to the user: what was achieved and what was observed")


# ---- browser_use 風格的 action union：每個 action 物件只填一個欄位 ----
class RobotAction(BaseModel):
    move_chassis: Optional[MoveChassisParams] = None
    move_arm: Optional[MoveArmParams] = None
    arm_to: Optional[ArmToParams] = None
    recenter_arm: Optional[EmptyParams] = None
    gripper: Optional[GripperParams] = None
    wait: Optional[WaitParams] = None
    stop: Optional[EmptyParams] = None
    locate: Optional[LocateParams] = None
    face: Optional[LocateParams] = None
    locate_overhead: Optional[LocateParams] = None
    remember: Optional[RememberParams] = None
    ask_human: Optional[AskHumanParams] = None
    done: Optional[DoneParams] = None

    def unpack(self) -> tuple[str, dict[str, Any]]:
        """回傳 (動作名稱, 參數 dict)。如果 LLM 一次填了好幾個欄位，只取第一個。"""
        for name in type(self).model_fields:
            value = getattr(self, name)
            if value is not None:
                return name, value.model_dump()
        return "wait", {"seconds": 0.5}


class RobotAgentOutput(BaseModel):
    thinking: str = Field(description="step-by-step reasoning about the camera image, telemetry and history")
    evaluation_previous_goal: str = Field(description="one sentence: did the last action achieve its goal? Success / Failure / Uncertain, and why")
    memory: str = Field(description="1 to 3 sentences of progress tracking that should persist to later steps")
    next_goal: str = Field(description="the immediate goal for this step, one sentence")
    action: list[RobotAction] = Field(min_length=1, description="1 to 3 actions, executed in order")


# ---- 紀錄用 ----
class ActionRecord(BaseModel):
    name: str
    params: dict[str, Any]
    ok: bool
    message: str
    duration_s: float = 0.0


class StepRecord(BaseModel):
    step: int
    timestamp: float
    state_before: dict[str, Any]
    frame_path: Optional[str] = None
    thinking: str = ""
    evaluation_previous_goal: str = ""
    memory: str = ""
    next_goal: str = ""
    actions: list[ActionRecord] = []
    error: Optional[str] = None
    llm_seconds: float = 0.0


class RunResult(BaseModel):
    task: str
    is_done: bool = False
    success: Optional[bool] = None
    final_text: str = ""
    steps: list[StepRecord] = []
    final_state: dict[str, Any] = {}
    final_frame_path: Optional[str] = None
    run_dir: str = ""
    duration_s: float = 0.0
