"""
動作註冊表：對應 browser_use 的 Tools()。

以前新增一個動作要同時改三個地方才會保持一致：views.py 的 RobotAction union 手動加一個
Optional 欄位、service.py 的 ACTION_HANDLERS 手動加一筆對應、system_prompt.md 手動補文件。
三份清單各自維護，很容易新增動作時漏掉一處、悄悄不同步。

這裡把「動作叫什麼名字、參數長什麼樣子、由 RobotAgent 的哪個方法處理」集中登記成一份
ACTION_SPECS，give LLM 看的 schema (RobotAction) 是從這份登記表動態長出來的 pydantic
model，service.py 的 dispatch 表也直接從這裡衍生，兩邊永遠對得上、不會漏改。

跟 browser_use 的 Tools() 不同的一點：這裡不是把處理邏輯本身放進登記表 (browser_use
用的是不綁狀態的 free function，把 browser_session 當參數注入)，而是仍然放在
RobotAgent 的方法裡，因為每個 handler 都要用到 self.robot/self.llm/self.world_state
這些執行期才有的東西——用方法名稱字串對應，執行時用 getattr(self, handler_name) 取出來，
比硬要把 RobotAgent 拆成不綁狀態的 free function 簡單、風險也低很多。
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model


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


@dataclass
class ActionSpec:
    param_model: type
    description: str
    handler_name: str  # RobotAgent 上實際處理這個動作的方法名稱


# 動作名稱 -> 規格。新增一個動作只要在這裡加一筆，RobotAction 的 schema 跟 service.py 的
# dispatch 表都會自動跟著長出對應的項目，不用再手動同步好幾個地方。
ACTION_SPECS: dict[str, ActionSpec] = {
    "move_chassis": ActionSpec(
        MoveChassisParams,
        "Relative chassis motion in the robot's own frame. Translation happens first, then rotation.",
        "_h_physical"),
    "move_arm": ActionSpec(
        MoveArmParams,
        "Relative motion of the gripper. The camera rides on the arm, so this also tilts the view.",
        "_h_physical"),
    "arm_to": ActionSpec(
        ArmToParams,
        "Move the arm to an absolute pose. This also changes what the camera sees.",
        "_h_physical"),
    "recenter_arm": ActionSpec(EmptyParams, "Return the arm to its default pose.", "_h_physical"),
    "gripper": ActionSpec(GripperParams, "Open or close the gripper.", "_h_physical"),
    "wait": ActionSpec(WaitParams, "Do nothing, then observe again.", "_h_physical"),
    "stop": ActionSpec(EmptyParams, "Stop the chassis immediately.", "_h_physical"),
    "locate": ActionSpec(
        LocateParams,
        "Precise perception: where the object is in the current front-camera image. Does not move the robot. "
        "Ends the step.",
        "_h_locate"),
    "face": ActionSpec(
        LocateParams,
        "Locate the object and rotate the chassis to face it, correcting up to twice. Ends the step.",
        "_h_locate"),
    "locate_overhead": ActionSpec(
        LocateParams,
        "Look at the fixed overhead camera (not the front camera) to find the object from world coordinates. "
        "Ends the step.",
        "_h_locate_overhead"),
    "align_to_tunnel": ActionSpec(
        LocateParams,
        "For elongated pass-through structures only (a tunnel, corridor, gate, archway, or similar you might "
        "need to look or drive straight through): finds its two ends from the overhead camera and works out "
        "the heading you must face to look straight down its length, plus where to stand to do that — not just "
        "the bearing to its centre. Use this instead of locate_overhead when the task asks whether you can see "
        "or drive through something, since facing an elongated object's centre from an angle is not the same "
        "as being aligned with it. Ends the step.",
        "_h_align_to_tunnel"),
    "remember": ActionSpec(RememberParams, "Store one short general fact for FUTURE tasks.", "_h_remember"),
    "ask_human": ActionSpec(AskHumanParams, "Ask the human operator a question.", "_h_ask_human"),
    "done": ActionSpec(DoneParams, "Finish the task. Must be the only action in its step.", "_h_done"),
}


def _build_action_union() -> type:
    """對應 browser_use 的 Tools() 自動產生 schema：從 ACTION_SPECS 動態長出一個
    pydantic model，每個註冊過的動作各一個 Optional 欄位，不用手寫逐欄位的 union。"""
    fields = {name: (Optional[spec.param_model], None) for name, spec in ACTION_SPECS.items()}
    model = create_model("RobotAction", **fields)

    def unpack(self) -> tuple:
        """回傳 (動作名稱, 參數 dict)。如果 LLM 一次填了好幾個欄位，只取第一個。"""
        for name in type(self).model_fields:
            value = getattr(self, name)
            if value is not None:
                return name, value.model_dump()
        return "wait", {"seconds": 0.5}

    model.unpack = unpack
    return model


RobotAction = _build_action_union()
