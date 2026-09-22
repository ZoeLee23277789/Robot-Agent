"""
RobotAgent：觀察 → 思考 → 行動 的主迴圈。

跟 browser_use.Agent 的對應關係：
    browser state (DOM + screenshot)   →  robot state (遙測 JSON + 相機畫面)
    Tools / action registry            →  robot_server 的 /action 加上本地的 ask_human、done
    AgentOutput                        →  RobotAgentOutput (欄位名稱相同)
    AgentHistoryList                   →  RunResult，逐步存成 history.json 與 step_XXX.jpg
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

from browser_use.llm.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)

from robot_agent import perception
from robot_agent.client import RobotClient
from robot_agent.views import ActionRecord, RobotAgentOutput, RunResult, StepRecord

LOCAL_ACTIONS = {"ask_human", "done"}
# 底盤/手臂動作只要每個分量都小於這個門檻，就視為「實質上沒動」而退回，不是只擋精確的 0。
# 模型第一版只擋精確 0 時，被發現有人填 0.0001、1.0 這種技術上非零但毫無效果的數字來繞過去。
NOOP_THRESHOLDS = {
    "move_chassis": {"forward_m": 0.02, "right_m": 0.02, "turn_left_deg": 2.0},
    "move_arm": {"forward_mm": 2.0, "up_mm": 2.0},
}
PHYSICAL_ACTIONS = {"move_chassis", "move_arm", "arm_to", "recenter_arm", "gripper"}
MAX_NOTES = 20

AskHuman = Callable[[str], Awaitable[str]]
Confirm = Callable[[RobotAgentOutput], Awaitable[bool]]


async def _console_ask(question: str) -> str:
    print(f"\n🟡 Agent 想問你：{question}")
    return (await asyncio.to_thread(input, "👉 你的回答：")).strip() or "(no answer)"


async def _console_confirm(output: RobotAgentOutput) -> bool:
    ans = (await asyncio.to_thread(input, "⚠️  要執行以上動作嗎？ [Enter=執行 / n=拒絕]：")).strip().lower()
    return ans not in ("n", "no")


class RobotAgent:
    def __init__(
        self,
        task: str,
        llm,
        robot: RobotClient,
        *,
        max_steps: int = 30,
        max_actions_per_step: int = 3,
        max_failures: int = 3,
        run_dir: Optional[str] = None,
        confirm_each_step: bool = False,
        ask_human: AskHuman = _console_ask,
        confirm: Confirm = _console_confirm,
        extend_system_message: Optional[str] = None,
        history_items: int = 12,
        notes_path: Optional[str] = "memory/body_notes.md",
    ):
        self.task = task
        self.llm = llm
        self.robot = robot
        self.max_steps = max_steps
        self.max_actions_per_step = max_actions_per_step
        self.max_failures = max_failures
        self.confirm_each_step = confirm_each_step
        self.ask_human = ask_human
        self.confirm = confirm
        self.history_items = history_items

        self.run_dir = Path(run_dir or f"runs/{time.strftime('%Y%m%d_%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 跨任務的長期記憶：LLM 自己探索身體之後寫下的筆記，下一個任務開始時會讀回來。
        self.notes_path = Path(notes_path) if notes_path else None
        prompt = (Path(__file__).parent / "system_prompt.md").read_text(encoding="utf-8")
        notes = self._load_notes()
        if notes:
            prompt += ("\n<learned_notes>\nFacts you discovered about your own body in EARLIER tasks and chose to remember. "
                       "Trust them, and correct them with remember() if you find one is wrong.\n"
                       + "\n".join(f"- {n}" for n in notes) + "\n</learned_notes>\n")
        if extend_system_message:
            prompt += "\n" + extend_system_message
        self.system_message = SystemMessage(content=prompt)

        self.result = RunResult(task=task, run_dir=str(self.run_dir))

    # ------------------------------------------------------------------ 觀察
    async def _observe(self, step: int):
        state, jpg = await asyncio.gather(self.robot.state(), self.robot.frame())
        frame_path = None
        if jpg:
            frame_path = str(self.run_dir / f"step_{step:03d}.jpg")
            Path(frame_path).write_bytes(jpg)
        return state, jpg, frame_path

    # ------------------------------------------------------------------ prompt
    def _history_text(self) -> str:
        steps = self.result.steps
        if not steps:
            return "(no steps yet)"
        lines = []
        omitted = len(steps) - self.history_items
        if omitted > 0:
            lines.append(f"[{omitted} earlier steps omitted; rely on your memory field]")
        for s in steps[-self.history_items:]:
            lines.append(f"<step_{s.step}>")
            if s.error:
                lines.append(f"Error: {s.error}")
            else:
                lines.append(f"Evaluation of previous step: {s.evaluation_previous_goal}")
                lines.append(f"Memory: {s.memory}")
                lines.append(f"Goal: {s.next_goal}")
                for a in s.actions:
                    flag = "ok" if a.ok else "FAILED"
                    lines.append(f"Action {a.name}({json.dumps(a.params)}) -> {flag}: {a.message}")
            lines.append(f"</step_{s.step}>")
        return "\n".join(lines)

    def _build_user_message(self, step: int, state: dict, jpg: Optional[bytes]) -> UserMessage:
        text = (
            f"<user_request>\n{self.task}\n</user_request>\n\n"
            f"<agent_history>\n{self._history_text()}\n</agent_history>\n\n"
            f"<robot_state>\n{json.dumps(state, ensure_ascii=False)}\n</robot_state>\n\n"
            f"<step_info>Step {step} of {self.max_steps}.</step_info>\n"
        )
        if step >= self.max_steps:
            text += "This is your LAST step. You must call done now and report what was achieved.\n"
        parts: list = [ContentPartTextParam(text=text)]
        if jpg:
            parts.append(ContentPartTextParam(text="Current camera image:"))
            b64 = base64.b64encode(jpg).decode("ascii")
            parts.append(ContentPartImageParam(
                image_url=ImageURL(url=f"data:image/jpeg;base64,{b64}", media_type="image/jpeg", detail="auto")))
        else:
            parts.append(ContentPartTextParam(text="(No camera image is available for this step. Be extra careful.)"))
        return UserMessage(content=parts)

    # ------------------------------------------------------------------ 行動
    async def _execute(self, output: RobotAgentOutput, record: StepRecord) -> None:
        actions = output.action[: self.max_actions_per_step]
        for i, action in enumerate(actions):
            name, params = action.unpack()

            if name == "done":
                if i > 0:  # done 必須單獨一步，前面的動作結果還沒被看過
                    record.actions.append(ActionRecord(
                        name=name, params=params, ok=False,
                        message="done ignored: it must be the only action in its step, so you can verify first"))
                    return
                self.result.is_done = True
                self.result.success = bool(params.get("success"))
                self.result.final_text = str(params.get("text", ""))
                record.actions.append(ActionRecord(name=name, params=params, ok=True, message="task finished"))
                return

            if name in ("locate", "face"):
                await self._perceive(name, params, record)
                return  # 感知結果要先讓 LLM 看過，所以這一步到此為止

            if name == "locate_overhead":
                await self._perceive_overhead(params, record)
                return

            if name == "remember":
                msg = self._remember(str(params.get("note", "")))
                record.actions.append(ActionRecord(name=name, params=params, ok=True, message=msg))
                print(f"   📝 remember → {params.get('note')}")
                continue

            if name == "ask_human":
                answer = await self.ask_human(str(params.get("question", "")))
                record.actions.append(ActionRecord(name=name, params=params, ok=True, message=f"Human answered: {answer}"))
                continue

            # 防呆：擋下實質上不會有效果的移動，不只擋精確的 0 (見上面 NOOP_THRESHOLDS 的註解)。
            thr = NOOP_THRESHOLDS.get(name)
            if thr and all(abs(float(params.get(k, 0) or 0)) < v for k, v in thr.items()):
                record.actions.append(ActionRecord(
                    name=name, params=params, ok=False,
                    message=f"rejected: this {name} is too small to have any real effect (every value is near zero). "
                            f"If you just want to look again without actually moving, use wait(seconds) instead. "
                            f"If you meant to move a different part of the robot, check the action name."))
                print(f"   ❌ {name} 幅度太小、等同沒動，已退回給 LLM")
                return

            t0 = time.time()
            res = await self.robot.act(name, params)
            rec = ActionRecord(
                name=name, params=params, ok=bool(res.get("ok")),
                message=str(res.get("message", "")), duration_s=float(res.get("duration_s", time.time() - t0)))
            record.actions.append(rec)
            print(f"   {'✅' if rec.ok else '❌'} {name}({json.dumps(params)}) → {rec.message}")
            if not rec.ok:
                return  # 後面的動作是建立在這個動作成功的前提上，直接跳過

    # ------------------------------------------------------------------ 長期記憶
    def _load_notes(self) -> list[str]:
        if not self.notes_path or not self.notes_path.exists():
            return []
        lines = [ln[2:].strip() for ln in self.notes_path.read_text(encoding="utf-8").splitlines() if ln.startswith("- ")]
        return lines[-MAX_NOTES:]

    def _remember(self, note: str) -> str:
        note = " ".join(note.split())[:240]
        if not self.notes_path or not note:
            return "nothing stored"
        notes = self._load_notes()
        if note in notes:
            return "already known"
        notes = (notes + [note])[-MAX_NOTES:]
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        self.notes_path.write_text("# Notes the robot agent wrote about its own body\n\n"
                                   + "\n".join(f"- {n}" for n in notes) + "\n", encoding="utf-8")
        return f"stored ({len(notes)}/{MAX_NOTES} notes). It will be available in future tasks."

    # ------------------------------------------------------------------ 感知
    async def _perceive(self, name: str, params: dict, record: StepRecord) -> None:
        """locate：只回報目標在畫面哪裡。face：回報之後直接轉過去對準，最多修正兩次。"""
        obj = str(params.get("object", "")).strip() or "target"
        t0 = time.time()
        try:
            found = await perception.locate(self.llm, await self.robot.frame(), obj)
            if not found.found:
                # pointing 對小物體偶爾會漏判，同一個畫面重問一次的成本很低，先排除是不是看漏了
                await asyncio.sleep(0.3)
                retry = await perception.locate(self.llm, await self.robot.frame(), obj)
                if retry.found:
                    found = retry
            msg = found.describe(obj)
            if name == "face" and found.found:
                for _ in range(2):
                    if abs(found.bearing_right_deg) <= 4:
                        break
                    res = await self.robot.act("move_chassis", {"forward_m": 0, "right_m": 0,
                                                                "turn_left_deg": found.turn_left_deg})
                    if not res.get("ok"):
                        msg += f" Turn failed: {res.get('message')}"
                        break
                    again = await perception.locate(self.llm, await self.robot.frame(), obj)
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
        record.actions.append(ActionRecord(name=name, params=params, ok=ok, message=msg,
                                           duration_s=round(time.time() - t0, 2)))
        print(f"   {'🎯' if ok else '🔍'} {name}({obj}) → {msg}")

    async def _perceive_overhead(self, params: dict, record: StepRecord) -> None:
        """跟 _perceive 是同一種「感知動作，這一步就此結束」的寫法，圖跟換算方式不同。"""
        obj = str(params.get("object", "")).strip() or "target"
        t0 = time.time()
        try:
            jpg = await self.robot.overhead_frame()
            if jpg is None:
                ok, msg = False, ("no overhead camera available (run add_overhead_camera.py on the "
                                  "robot side first, or this robot simply does not have one)")
            else:
                state = record.state_before or {}
                found = await perception.locate_overhead(
                    self.llm, jpg, obj, state.get("world_xy"), state.get("yaw_deg"))
                ok, msg = found.found, found.describe(obj)
        except Exception as e:
            ok, msg = False, f"perception failed: {type(e).__name__}: {' '.join(str(e).split())[:200]}"
        record.actions.append(ActionRecord(name="locate_overhead", params=params, ok=ok, message=msg,
                                           duration_s=round(time.time() - t0, 2)))
        print(f"   {'🛰️' if ok else '🔍'} locate_overhead({obj}) → {msg}")

    # ------------------------------------------------------------------ 主迴圈
    async def run(self) -> RunResult:
        t_start = time.time()
        failures = 0
        try:
            health = await self.robot.health()
            if not health.get("ok"):
                raise RuntimeError(f"robot server 沒有回應正常：{health}")
            print(f"🤖 已連上 robot server (mode={health.get('mode')})，任務：{self.task}")

            for step in range(1, self.max_steps + 1):
                state, jpg, frame_path = await self._observe(step)
                record = StepRecord(step=step, timestamp=time.time(), state_before=state, frame_path=frame_path)

                try:
                    t0 = time.time()
                    completion = await self.llm.ainvoke(
                        [self.system_message, self._build_user_message(step, state, jpg)],
                        output_format=RobotAgentOutput,
                    )
                    record.llm_seconds = round(time.time() - t0, 2)
                    output: RobotAgentOutput = completion.completion
                except Exception as e:
                    failures += 1
                    detail = " ".join(str(e).split())[:300]
                    record.error = f"LLM call failed ({type(e).__name__}): {detail}. Your previous output was invalid or too long; no action was executed. Keep numbers short (at most 2 decimals)."
                    print(f"📍 Step {step}: ❌ {record.error}")
                    self.result.steps.append(record)
                    self._save()
                    if failures >= self.max_failures:
                        break
                    continue

                record.thinking = output.thinking
                record.evaluation_previous_goal = output.evaluation_previous_goal
                record.memory = output.memory
                record.next_goal = output.next_goal
                print(f"\n📍 Step {step}  ({record.llm_seconds}s)")
                print(f"   👍 Eval: {output.evaluation_previous_goal}")
                print(f"   🧠 Memory: {output.memory}")
                print(f"   🎯 Goal: {output.next_goal}")
                for a in output.action[: self.max_actions_per_step]:
                    n, p = a.unpack()
                    print(f"   ▶ {n}({json.dumps(p, ensure_ascii=False)})")

                is_physical = any(a.unpack()[0] in PHYSICAL_ACTIONS for a in output.action)
                if self.confirm_each_step and is_physical and not await self.confirm(output):
                    record.actions.append(ActionRecord(
                        name="(human veto)", params={}, ok=False,
                        message="The human operator rejected these actions. Propose something safer or ask_human."))
                else:
                    await self._execute(output, record)

                self.result.steps.append(record)
                self._save()

                if self.result.is_done:
                    break
                if record.actions and not record.actions[-1].ok and record.actions[-1].name not in ("locate", "face", "locate_overhead"):
                    failures += 1
                    if failures >= self.max_failures:
                        print(f"⛔ 連續失敗 {failures} 次，中止任務")
                        break
                else:
                    failures = 0
        finally:
            # 跟 test_movement.py 的 finally 同一個精神：不管怎麼結束，先讓底盤停下來
            await self.robot.stop()
            try:
                state, jpg, _ = await self._observe(0)
                self.result.final_state = state
                if jpg:
                    final = self.run_dir / "final.jpg"
                    final.write_bytes(jpg)
                    (self.run_dir / "step_000.jpg").unlink(missing_ok=True)
                    self.result.final_frame_path = str(final)
            except Exception:
                pass
            self.result.duration_s = round(time.time() - t_start, 1)
            self._save()

        status = "✅ 成功" if self.result.success else ("❌ 失敗" if self.result.is_done else "⏹ 未完成")
        print(f"\n{status}｜{len(self.result.steps)} 步｜{self.result.duration_s}s｜紀錄在 {self.run_dir}")
        if self.result.final_text:
            print(f"📝 {self.result.final_text}")
        return self.result

    def _save(self) -> None:
        (self.run_dir / "history.json").write_text(
            self.result.model_dump_json(indent=2), encoding="utf-8")
