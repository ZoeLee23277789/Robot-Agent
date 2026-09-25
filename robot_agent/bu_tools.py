"""
機器人版的 Tools()：這次是真的 browser_use.Tools() registry，不是自己重寫的版本
(對照 robot_agent/service.py 裡自己重寫的 ACTION_HANDLERS)。

跟最早的 robot_bu.zip 版本比，這裡多了 locate/face/locate_overhead/remember 四個動作——
這幾天在自製版本裡做出來、對任務有實際幫助的能力，這裡直接重用 perception.py 的判斷邏輯
(pointing、深度換算) port 過來，不是重寫一份新的。

動作參數的 pydantic model 用 robot_agent/actions.py 那份登記表 (ACTION_SPECS)，
同一份參數定義兩邊共用，不會有兩套 MoveChassisParams 各自維護、慢慢長歪。
"""

import asyncio
import time
from pathlib import Path
from typing import Optional

from browser_use.agent.views import ActionResult
from browser_use.tools.service import Tools

from robot_agent import perception
from robot_agent.actions import (
    ArmToParams,
    AskHumanParams,
    EmptyParams,
    GripperParams,
    LocateParams,
    MoveArmParams,
    MoveChassisParams,
    RememberParams,
)
from robot_agent.bu_common import RECOVERY_MARKER, AdapterState
from robot_agent.client import RobotClient

BROWSER_ACTIONS = [
    "search", "navigate", "go_back", "click", "input", "upload_file", "switch", "close",
    "extract", "scroll", "send_keys", "find_text", "screenshot", "dropdown_options",
    "select_dropdown", "write_file", "replace_file", "read_file", "evaluate",
]

tools = Tools(exclude_actions=BROWSER_ACTIONS)

# 由進入點設定：連到哪台機器人、感知動作用哪個 LLM、跟 RobotSession 共用的狀態、
# 實體動作前要不要先問人、筆記存哪裡。
_robot: Optional[RobotClient] = None
_llm = None
_adapter_state: Optional[AdapterState] = None
_confirm_each_action = False
_notes_path: Optional[Path] = None
MAX_NOTES = 20


def configure(robot: RobotClient, llm, adapter_state: AdapterState, *,
             confirm_each_action: bool = False, notes_path: Optional[str] = "memory/body_notes.md") -> None:
    global _robot, _llm, _adapter_state, _confirm_each_action, _notes_path
    _robot, _llm, _adapter_state, _confirm_each_action = robot, llm, adapter_state, confirm_each_action
    _notes_path = Path(notes_path) if notes_path else None


async def _run(name: str, params: dict) -> ActionResult:
    assert _robot is not None and _adapter_state is not None, "bu_tools.configure() 還沒被呼叫"
    if _confirm_each_action:
        ans = (await asyncio.to_thread(input, f"⚠️  要執行 {name}({params}) 嗎？ [Enter=執行 / n=拒絕]：")).strip().lower()
        if ans in ("n", "no"):
            return ActionResult(error=f"{name} rejected by the human operator. Propose something safer or ask_human.")
    res = await _robot.act(name, params)
    msg = str(res.get("message", ""))
    ok = bool(res.get("ok"))
    if RECOVERY_MARKER in msg:
        # 伺服器剛自動重啟模擬救回卡死：根因已經處理掉了，不該算進 Agent 自己的
        # consecutive_failures 額度，所以回報成功而不是 error (error 才會被框架計入失敗次數)。
        # 同時標記 just_recovered，讓下一步的 prompt 提醒 LLM 姿態可能跳動。
        _adapter_state.just_recovered = True
        return ActionResult(extracted_content=msg, long_term_memory=f"{name}: {msg}")
    if ok:
        return ActionResult(extracted_content=msg, long_term_memory=f"{name}: {msg}")
    # error 會讓 multi_act 中斷這一步剩下的動作，並計入 Agent 自己的 consecutive_failures
    return ActionResult(error=f"{name} failed: {msg}")


def _remember_object(obj: str, step: int, source: str, summary: str) -> None:
    _adapter_state.known_objects[obj] = f"(step {step}, via {source}) {summary}"


@tools.action(
    "Relative chassis motion in the robot's own frame: forward_m (+forward), right_m (+strafe right), "
    "turn_left_deg (+CCW). Translation first, then rotation. Max 1 m / 180 deg per call.",
    param_model=MoveChassisParams,
)
async def move_chassis(params: MoveChassisParams) -> ActionResult:
    thr = {"forward_m": 0.02, "right_m": 0.02, "turn_left_deg": 2.0}
    p = params.model_dump()
    if all(abs(p.get(k, 0) or 0) < v for k, v in thr.items()):
        return ActionResult(
            error=f"move_chassis is too small to have any real effect (every value is near zero). "
                  f"If you just want to look again without actually moving, use wait(seconds) instead.")
    result = await _run("move_chassis", p)
    if result.error is None:
        # 底盤真的動了，之前記下的方位角/轉向建議全部失效，留著只會誤導 LLM。
        _adapter_state.known_objects.clear()
    return result


@tools.action("Relative gripper motion: forward_mm and up_mm, max 100 mm each. The camera rides on the arm.",
              param_model=MoveArmParams)
async def move_arm(params: MoveArmParams) -> ActionResult:
    return await _run("move_arm", params.model_dump())


@tools.action(
    "Move the arm to an absolute pose. x is forward reach (about 80 to 200), y is height (about 10 to 130). "
    "This also changes what the camera sees.",
    param_model=ArmToParams,
)
async def arm_to(params: ArmToParams) -> ActionResult:
    return await _run("arm_to", params.model_dump())


@tools.action("Return the arm to its default pose.", param_model=EmptyParams)
async def recenter_arm(params: EmptyParams) -> ActionResult:
    return await _run("recenter_arm", {})


@tools.action('Open or close the gripper: state = "open" | "close".', param_model=GripperParams)
async def gripper(params: GripperParams) -> ActionResult:
    return await _run("gripper", params.model_dump())


@tools.action("Stop the chassis immediately.", param_model=EmptyParams)
async def stop(params: EmptyParams) -> ActionResult:
    return await _run("stop", {})


async def _do_locate(name: str, obj: str) -> ActionResult:
    """locate/face 共用：對應 robot_agent/service.py 的 _perceive，
    face 額外做最多兩次修正轉向，邏輯原封不動搬過來。"""
    t0 = time.time()
    try:
        jpg = await _robot.frame()
        found = await perception.locate(_llm, jpg, obj)
        if not found.found:
            await asyncio.sleep(0.3)
            retry = await perception.locate(_llm, await _robot.frame(), obj)
            if retry.found:
                found = retry
        msg = found.describe(obj)
        if name == "face" and found.found:
            for _ in range(2):
                if abs(found.bearing_right_deg) <= 4:
                    break
                res = await _robot.act("move_chassis", {"forward_m": 0, "right_m": 0,
                                                        "turn_left_deg": found.turn_left_deg})
                if not res.get("ok"):
                    msg += f" Turn failed: {res.get('message')}"
                    break
                again = await perception.locate(_llm, await _robot.frame(), obj)
                msg += f" Turned {found.turn_left_deg:+.0f} deg (left positive)."
                if not again.found:
                    msg += " After turning, the object is no longer visible (the turn may have overshot)."
                    found = again
                    break
                found = again
                msg += f" Now it is {found.bearing_right_deg:+.0f} deg from centre (right positive), image y={found.y}/1000."
            if found.found and abs(found.bearing_right_deg) <= 4:
                msg += " The robot is now FACING the object; driving forward will approach it."
        ok = found.found
    except Exception as e:
        ok, msg = False, f"perception failed: {type(e).__name__}: {' '.join(str(e).split())[:200]}"
    if ok:
        _remember_object(obj, 0, name, msg)
    # 感知動作沒有實際移動機器人，找不到東西不代表卡住，用 extracted_content (不是 error)
    # 回報，這樣不會被框架計入 Agent 的 consecutive_failures——那是給「真的執行失敗」用的。
    duration = round(time.time() - t0, 2)
    return ActionResult(extracted_content=f"{msg} (took {duration}s)",
                        long_term_memory=f"{name}({obj}): {msg}")


@tools.action(
    "Precise perception: where the object is in the current front-camera image, as a bearing in degrees and "
    "the exact turn_left_deg that would centre it, or tells you it is not visible. Does not move the robot. "
    "Ends the step.",
    param_model=LocateParams,
)
async def locate(params: LocateParams) -> ActionResult:
    return await _do_locate("locate", params.object)


@tools.action(
    "Locate the object and rotate the chassis to face it, correcting up to twice. "
    "Use this instead of guessing a turn angle. Ends the step.",
    param_model=LocateParams,
)
async def face(params: LocateParams) -> ActionResult:
    return await _do_locate("face", params.object)


@tools.action(
    "Look at a fixed camera mounted above the whole room (not your own camera) and, when it can see the "
    "object, tell you roughly how far away it is and the turn_left_deg to face it directly, computed from "
    "real world coordinates. When a depth sensor is available it also reports the candidate's real height "
    "above the floor, so a flat floor mat will not be confused with a real 3-D object. Not every robot has "
    "this; if it says unavailable, search normally with locate/face instead. Ends the step.",
    param_model=LocateParams,
)
async def locate_overhead(params: LocateParams) -> ActionResult:
    obj = params.object
    t0 = time.time()
    try:
        jpg = await _robot.overhead_frame()
        if jpg is None:
            ok, msg = False, ("no overhead camera available (run add_overhead_camera.py on the "
                              "robot side first, or this robot simply does not have one)")
        else:
            depth_png = await _robot.overhead_depth()
            state = await _robot.state()
            found = await perception.locate_overhead(
                _llm, jpg, obj, state.get("world_xy"), state.get("yaw_deg"), depth_png)
            ok, msg = found.found, found.describe(obj)
    except Exception as e:
        ok, msg = False, f"perception failed: {type(e).__name__}: {' '.join(str(e).split())[:200]}"
    if ok:
        _remember_object(obj, 0, "locate_overhead", msg)
    duration = round(time.time() - t0, 2)
    return ActionResult(extracted_content=f"{msg} (took {duration}s)",
                        long_term_memory=f"locate_overhead({obj}): {msg}")


@tools.action(
    "For elongated pass-through structures only (a tunnel, corridor, gate, archway, or similar you might need "
    "to look or drive straight through): finds its two ends from the overhead camera and works out the heading "
    "you must face to look straight down its length, plus where to stand to do that — not just the bearing to "
    "its centre. Use this instead of locate_overhead when the task asks whether you can see or drive through "
    "something, since facing an elongated object's centre from an angle is not the same as being aligned with "
    "it. Ends the step.",
    param_model=LocateParams,
)
async def align_to_tunnel(params: LocateParams) -> ActionResult:
    obj = params.object
    t0 = time.time()
    try:
        jpg = await _robot.overhead_frame()
        if jpg is None:
            ok, msg = False, ("no overhead camera available (run add_overhead_camera.py on the "
                              "robot side first, or this robot simply does not have one)")
        else:
            state = await _robot.state()
            found = await perception.locate_tunnel_axis(
                _llm, jpg, obj, state.get("world_xy"), state.get("yaw_deg"))
            ok, msg = found.found, found.describe(obj)
    except Exception as e:
        ok, msg = False, f"perception failed: {type(e).__name__}: {' '.join(str(e).split())[:200]}"
    duration = round(time.time() - t0, 2)
    return ActionResult(extracted_content=f"{msg} (took {duration}s)",
                        long_term_memory=f"align_to_tunnel({obj}): {msg}")


def _load_notes() -> list:
    if not _notes_path or not _notes_path.exists():
        return []
    lines = [ln[2:].strip() for ln in _notes_path.read_text(encoding="utf-8").splitlines() if ln.startswith("- ")]
    return lines[-MAX_NOTES:]


@tools.action(
    "Store one short general fact about your own body or sensors for FUTURE tasks. Not for task progress; "
    "use memory for that.",
    param_model=RememberParams,
)
async def remember(params: RememberParams) -> ActionResult:
    note = " ".join(params.note.split())[:240]
    if not _notes_path or not note:
        return ActionResult(extracted_content="nothing stored")
    notes = _load_notes()
    if note in notes:
        return ActionResult(extracted_content="already known")
    notes = (notes + [note])[-MAX_NOTES:]
    _notes_path.parent.mkdir(parents=True, exist_ok=True)
    _notes_path.write_text("# Notes the robot agent wrote about its own body\n\n"
                           + "\n".join(f"- {n}" for n in notes) + "\n", encoding="utf-8")
    return ActionResult(extracted_content=f"stored ({len(notes)}/{MAX_NOTES} notes). Available in future tasks.")


@tools.action(
    "Ask the human operator a question when the task is ambiguous, when you are stuck, "
    "or before anything that could damage the robot or its surroundings.",
    param_model=AskHumanParams,
)
async def ask_human(params: AskHumanParams) -> ActionResult:
    print(f"\n🟡 Agent 想問你：{params.question}")
    answer = (await asyncio.to_thread(input, "👉 你的回答：")).strip() or "(no answer)"
    return ActionResult(extracted_content=f"Human answered: {answer}", long_term_memory=f"Human answered: {answer}")


def load_learned_notes() -> list:
    """讓進入點在組 system prompt 時可以讀到跨任務筆記，跟 configure() 分開是因為
    這個要在 Agent 建立之前 (system prompt 準備階段) 就能用，不用等 configure() 先跑。"""
    if not _notes_path:
        return []
    return _load_notes()


__all__ = ["tools", "configure", "load_learned_notes"]
