"""
動作註冊表：對應 通用 agent 框架的 Tools()。

以前新增一個動作要同時改三個地方才會保持一致：views.py 的 RobotAction union 手動加一個
Optional 欄位、service.py 的 ACTION_HANDLERS 手動加一筆對應、system_prompt.md 手動補文件。
三份清單各自維護，很容易新增動作時漏掉一處、悄悄不同步。

這裡把「動作叫什麼名字、參數長什麼樣子、由 RobotAgent 的哪個方法處理」集中登記成一份
ACTION_SPECS，give LLM 看的 schema (RobotAction) 是從這份登記表動態長出來的 pydantic
model，service.py 的 dispatch 表也直接從這裡衍生，兩邊永遠對得上、不會漏改。

跟 通用 agent 框架的 Tools() 不同的一點：這裡不是把處理邏輯本身放進登記表 (agent 框架
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
    push: bool = Field(False, description="true ONLY when you intend to push the object in front; otherwise a forward move "
                                          "stops 0.15 m short of anything the front distance sensor sees")


class MoveArmParams(BaseModel):
    forward_mm: float = Field(0.0, description="relative mm; positive extends the arm forward. Max 100")
    up_mm: float = Field(0.0, description="relative mm; positive raises the arm. Max 100")
    turn_left_deg: Optional[float] = Field(None, description="NexArm only: rotate the ARM'S BASE (not the chassis) by this many "
                                                              "degrees, positive = left/counter-clockwise. The chassis does not move.")
    roll_deg: Optional[float] = Field(None, description="NexArm only: rotate the WRIST around the gripper's own axis by this many "
                                                         "degrees (twists the gripper, e.g. to grasp a sideways object).")


class ArmToParams(BaseModel):
    x_mm: float = Field(description="absolute forward reach of the arm, about 80 to 200 (if the telemetry has arm_limits, use those ranges instead)")
    y_mm: float = Field(description="absolute height of the arm, about 10 to 130 (if the telemetry has arm_limits, use those ranges instead)")
    pitch_deg: Optional[float] = Field(None, description="only on arms that report arm_limits: gripper pitch, 0 = horizontal "
                                                          "(reaches highest), -90 = pointing straight down (reaches lowest, "
                                                          "needed to grasp things on the table). Omit to keep it horizontal.")
    yaw_deg: Optional[float] = Field(None, description="NexArm only: absolute rotation of the arm's base, 0 = straight ahead, "
                                                        "positive = left. Omit for straight ahead.")
    roll_deg: Optional[float] = Field(None, description="NexArm only: absolute wrist roll, 0 = fingers level. Omit to keep the current roll.")


class LiftArmParams(BaseModel):
    height_mm: float = Field(description="absolute height of the arm lift column, 0 to 400 mm. 0 = arm down at chassis "
                                         "level (the normal driving pose); 330 puts the gripper at the height of the table top")


class RememberParams(BaseModel):
    note: str = Field(description="one short, general fact about your own body or sensors, useful in FUTURE tasks")


class GripperParams(BaseModel):
    state: Literal["open", "close"]


class WaitParams(BaseModel):
    seconds: float = Field(1.0, description="0 to 10 seconds")


class EmptyParams(BaseModel):
    pass


class RecenterArmParams(BaseModel):
    pose: Optional[Literal["home", "stow"]] = Field(None, description="omit or 'home' = default ready pose (gripper horizontal in "
                                                                       "front). 'stow' (NexArm only) = fold the arm over its own base "
                                                                       "with the gripper pointing down: the most compact pose, use it "
                                                                       "before driving through low or narrow places or when asked to "
                                                                       "fold/retract the arm fully.")


class LocateParams(BaseModel):
    object: str = Field(description="short description of what to find, e.g. 'red box'")


class ScanParams(BaseModel):
    category: str = Field(description="what kind of things to list, e.g. 'balls', 'small blocks', 'storage bins'")


class AskHumanParams(BaseModel):
    question: str


class GoToParams(BaseModel):
    target: str = Field(description="a fixed landmark name (tunnel, ball_pit, bench, bin_balls, bin_blocks, platform, "
                                    "stairs, slide, pillar_red/blue/yellow/green), an object description seen from "
                                    "above ('purple ball'), or world coordinates 'x, y'")


class ApproachParams(BaseModel):
    object: str = Field(description="the task target to reach, e.g. 'red box', 'small green block'")
    stop_m: float = Field(0.4, description="how far in front of the object to stop, metres (0.15 to 1.0)")


class PickParams(BaseModel):
    object: str = Field(description="the object to grasp, e.g. 'purple ball', 'small green block' (or 'x, y')")


class PlaceParams(BaseModel):
    target: str = Field(description="where to put the held object: a landmark (bin_blocks, bin_balls, mat_red, mat_green, "
                                    "mat_blue, ...), an object description, or world coordinates 'x, y'")


class DriveThroughParams(BaseModel):
    structure: str = Field(description="'tunnel' or 'platform' (drive underneath), or another pass-through structure")


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
    "recenter_arm": ActionSpec(RecenterArmParams, "Return the arm to its default pose, or fold it (pose='stow').", "_h_physical"),
    "lift_arm": ActionSpec(
        LiftArmParams,
        "Raise or lower the vertical lift column that carries the whole arm (gripper, arm camera and the front distance "
        "sensor all move with it). For objects on raised surfaces the arm alone cannot reach; lower it back to 0 before "
        "driving far. pick/place do this by themselves for objects on the table.",
        "_h_physical"),
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
    "scan_overhead": ActionSpec(
        ScanParams,
        "List EVERY object of one kind that the fixed overhead camera can see, each with its world position, distance "
        "and bearing from you, height above the floor, and what it stands on or in (floor, the table, a bin, a mat). "
        "Use it when the task describes its target indirectly (by place, by comparison, by exclusion, by a goal) so you "
        "can work out which concrete object is meant BEFORE approach/pick. Ends the step.",
        "_h_scan_overhead"),
    "align_to_tunnel": ActionSpec(
        LocateParams,
        "For elongated pass-through structures only (a tunnel, corridor, gate, archway, or similar you might "
        "need to look or drive straight through): finds its two ends from the overhead camera and works out "
        "the heading you must face to look straight down its length, plus where to stand to do that — not just "
        "the bearing to its centre. Use this instead of locate_overhead when the task asks whether you can see "
        "or drive through something, since facing an elongated object's centre from an angle is not the same "
        "as being aligned with it. Ends the step.",
        "_h_align_to_tunnel"),
    "go_to": ActionSpec(
        GoToParams,
        "Autonomous navigation: plans a collision-free path on the room map and drives there in ONE call "
        "(landmark name, object description, or 'x, y'). Stops ~0.45 m in front of landmarks/objects, facing them. "
        "Ends the step.",
        "_h_go_to"),
    "approach": ActionSpec(
        ApproachParams,
        "Reach a task target and face it at close range: finds it from above, drives there on a planned path, "
        "stops stop_m in front, fine-aligns with the front camera and reports bearing + tof_front_mm. Use for every "
        "'go to / find and stop near X' task. Ends the step.",
        "_h_approach"),
    "pick": ActionSpec(
        PickParams,
        "Grasp an object in ONE call: drives to it, opens the gripper, lowers the arm, feeds the object between the "
        "fingers using the front distance sensor, closes, lifts, and VERIFIES the object came along. Reports GRASPED "
        "or exactly why not. Ends the step.",
        "_h_pick"),
    "place": ActionSpec(
        PlaceParams,
        "Put down the held object at a target in ONE call: drives there, raises the arm over a bin's rim or lowers it "
        "onto a mat/floor, opens the gripper, backs up and recentres the arm. Ends the step.",
        "_h_place"),
    "drive_through": ActionSpec(
        DriveThroughParams,
        "Drive through the tunnel or underneath the platform along its long axis, entrance to exit, and report "
        "whether the robot really came out the other side. Ends the step.",
        "_h_drive_through"),
    "return_to_origin": ActionSpec(EmptyParams, "Drive back to where this task started and face the start heading.", "_h_physical"),
    "remember": ActionSpec(RememberParams, "Store one short general fact for FUTURE tasks.", "_h_remember"),
    "ask_human": ActionSpec(AskHumanParams, "Ask the human operator a question.", "_h_ask_human"),
    "done": ActionSpec(DoneParams, "Finish the task. Must be the only action in its step.", "_h_done"),
}


def _build_action_union() -> type:
    """對應 通用 agent 框架的 Tools() 自動產生 schema：從 ACTION_SPECS 動態長出一個
    pydantic model，每個註冊過的動作各一個 Optional 欄位，不用手寫逐欄位的 union。"""
    # name：這一格到底是哪個動作。Gemini 的結構化輸出（response_schema）沒辦法表達「欄位不存在」，實測它會把每一個
    # Optional 的動作欄位全部填上值（2026-10-21 r12：一格裡 move_chassis、move_arm、…、pick、place 全部非空），
    # 「取第一個非空欄位」就常常執行到 move_chassis 而不是它真正要的 approach/pick。所以逼模型用一個列舉欄位宣告動作，
    # 解析只信這個欄位，其他填充的欄位一律忽略。
    names = tuple(ACTION_SPECS.keys())
    fields = {"name": (Literal[names], Field(description="the ONE action this object performs; fill only that action's "
                                                         "parameter object and leave every other action null"))}
    fields.update({name: (Optional[spec.param_model], None) for name, spec in ACTION_SPECS.items()})
    model = create_model("RobotAction", **fields)

    def filled(self) -> list:
        """LLM 在這一格填了哪些動作的參數物件（正常只該有一個）。"""
        return [name for name in type(self).model_fields if name != "name" and getattr(self, name) is not None]

    def unpack(self) -> tuple:
        """回傳 (動作名稱, 參數 dict)：以 name 宣告的動作為準，它的參數物件沒填就用預設值（沒參數的動作本來就是空的）。
        name 無效或缺少時才退回下面的啟發式。
        LLM 一次填了好幾個欄位時，以前「取第一個」——而 move_chassis 剛好是第一個欄位。gemini-robotics-er 的結構化輸出
        常把 move_chassis 填成全 0 再填真正要的 pick/approach（2026-10-21 r05：記憶寫「用 pick」，送出的卻是 move_chassis(0,0,0)，
        連續五次被拒絕），真正的動作就這樣被丟掉。所以先把「全 0 的 move_chassis/move_arm」這種沒意義的填充排除，再取第一個。"""
        declared = getattr(self, "name", None)
        if declared in ACTION_SPECS:
            value = getattr(self, declared, None)
            if value is None:
                try:
                    value = ACTION_SPECS[declared].param_model()   # 全部有預設值的參數模型（EmptyParams、WaitParams…）
                except Exception:
                    value = None
            if value is not None:
                return declared, value.model_dump()
        names = self.filled()
        if not names:
            return "wait", {"seconds": 0.5}
        if len(names) > 1:
            zero_keys = {"move_chassis": ("forward_m", "right_m", "turn_left_deg"), "move_arm": ("forward_mm", "up_mm")}

            def trivial(name):
                keys = zero_keys.get(name)
                if not keys:
                    return False
                d = getattr(self, name).model_dump()
                return all(abs(float(d.get(k, 0) or 0)) < 1e-6 for k in keys)

            real = [n for n in names if not trivial(n)]
            if real:
                names = real
        name = names[0]
        return name, getattr(self, name).model_dump()

    model.filled = filled
    model.unpack = unpack
    return model


RobotAction = _build_action_union()
