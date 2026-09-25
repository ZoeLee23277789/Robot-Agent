"""
RobotAgent：觀察 → 思考 → 行動 的主迴圈。

跟 browser_use.Agent 的對應關係 (自己重寫，不依賴那個套件)：
    browser state (DOM + screenshot)   →  robot state (遙測 JSON + 相機畫面)
    Tools() registry                   →  actions.ACTION_SPECS + _h_* 方法，動作名稱/參數/
                                           handler 對應集中登記在 actions.py，這裡只是照著跑
    ActionResult 的 error/多動作中斷    →  _classify_result() + ActionRecord.counts_as_failure/
                                           resets_streak，統一的連續失敗判定
    AgentOutput                        →  RobotAgentOutput (欄位名稱相同)
    AgentHistoryList                   →  RunResult，逐步存成 history.json 與 step_XXX.jpg
    planner                            →  _maybe_plan()，每 planner_interval 步用完整歷史
                                           檢查一次「是不是在原地打轉」，結果塞進 <plan>
    長期記憶 (procedural memory)        →  _update_long_term_summary()，步數一多、舊步驟快被
                                           歷史視窗擠出去之前，先摘要進 <long_term_summary>
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
from robot_agent.actions import ACTION_SPECS
from robot_agent.client import RobotClient
from robot_agent.views import (
    ActionRecord,
    ActionResult,
    KnownObject,
    RobotAgentOutput,
    RunResult,
    StepRecord,
    WorldState,
)

# 底盤/手臂動作只要每個分量都小於這個門檻，就視為「實質上沒動」而退回，不是只擋精確的 0。
# 模型第一版只擋精確 0 時，被發現有人填 0.0001、1.0 這種技術上非零但毫無效果的數字來繞過去。
NOOP_THRESHOLDS = {
    "move_chassis": {"forward_m": 0.02, "right_m": 0.02, "turn_left_deg": 2.0},
    "move_arm": {"forward_mm": 2.0, "up_mm": 2.0},
}
PHYSICAL_ACTIONS = {"move_chassis", "move_arm", "arm_to", "recenter_arm", "gripper"}
MAX_NOTES = 20
RECOVERY_MARKER = "auto-recovered:"  # 跟 robot_server.py 的 _note_stall 約定好的標記字串


def _classify_result(ok: bool, message: str) -> tuple[bool, bool]:
    """把 robot_server 回應分類成 (counts_as_failure, recovered)。

    robot_server 在自己偵測到控制腳本卡死、自動重啟模擬之後，會把 RECOVERY_MARKER
    這個標記附加在 message 裡 (見 robot_server.py 的 _note_stall)。那次動作本身通常
    還是 ok=False (要求的效果確實沒發生)，但卡住的根因已經被伺服器端處理掉了——如果
    這次也算進 agent 這邊的連續失敗次數，等於伺服器和 agent 各自用同一種「連續3次」
    門檻在互搶額度，伺服器才剛恢復、agent 這邊的額度卻可能已經被之前的失敗用完，
    直接中止任務。所以 recovered 時兩邊都不算：agent 拿到的是全新的開始。"""
    if RECOVERY_MARKER in message:
        return False, True
    return (not ok), False


def _estimate_tokens(text: str) -> int:
    """不接外部 tokenizer，用字元數/4 粗估——夠用來決定「歷史該從哪裡開始截斷」，
    不需要精準到跟 provider 收費算法一致。"""
    return max(1, len(text) // 4)


# 對應 browser_use 的 planner：逐步反應式的主迴圈，每一步自己看起來都合理，只有拉長時間看
# 完整歷史才看得出「一直在原地打轉、重複同一個錯誤判斷」。這裡特別強調要點名這種迴圈，
# 因為那正是今天實測任務失敗的原因——locate_overhead 連續 8 次指向同一個錯誤物體，
# 每一步的主 LLM 都各自覺得「這次應該不一樣」，要退一步看歷史才會發現是同一個迴圈。
PLANNER_PROMPT = """You are the strategic planner for a robot agent. You do NOT control the robot directly \
and cannot see the current camera image — you only review its history and state, then advise.

Task: {task}

Review the history below and answer in 2 to 4 short sentences:
1. Is progress being made, or has the same action/outcome pattern repeated several times without progress \
(e.g. the same candidate object being re-identified as correct and then failing to appear on the front \
camera, repeated turns that end up back near a heading already tried, the same rejected/failed action type \
recurring)? If so, name the loop explicitly.
2. If it is looping or stuck, suggest one concrete different approach to break out of it (a different search \
strategy, a different object to trust, when to give up on a candidate and ask_human).
3. Otherwise, give one sentence of high-level guidance for the next few steps.

Keep it short: this is strategic guidance for the next several steps, not a specific action list. Reply with \
plain text only, no JSON."""


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
        max_history_tokens: int = 6000,
        notes_path: Optional[str] = "memory/body_notes.md",
        planner_llm=None,
        planner_interval: int = 5,
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
        self.max_history_tokens = max_history_tokens
        # 系統自己維護的結構化狀態，跟 LLM 自己口頭寫的 memory 欄位分開，見 views.WorldState。
        self.world_state = WorldState(max_failures=max_failures)
        # 對應 browser_use 的 planner：預設沿用同一個 llm，不強迫使用者多設定一個 provider。
        self.planner_llm = planner_llm or llm
        self.planner_interval = planner_interval
        self.latest_plan = ""
        # 對應 browser_use 的長期記憶：步數一多，_history_text() 的 token 預算會開始丟掉最舊的
        # 步驟，這裡用一次小型 LLM 呼叫把整批要被丟掉的步驟濃縮成摘要，不會像純截斷那樣直接消失。
        self.long_term_summary = ""

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
        # state()+frame() 併發沒問題 (從以前就這樣跑，穩定)，但 overhead_frame() 一起併發送出
        # 實測會拖垮前方相機：CoppeliaSim 本來就跑得很吃緊 (真實時間的 3.8%~5%)，俯視圖那次
        # 渲染會跟 RoboMaster SDK 的相機串流搶模擬器資源，讓 _fresh_image() 的等待逾時、整段
        # 變成 None——一整場任務的前方相機幾乎每一步都拿不到畫面，agent 等於半瞎在跑。
        # 改成前方相機先抓完、俯視圖再抓，兩邊不再搶同一時間點的模擬器資源。
        state, jpg = await asyncio.gather(self.robot.state(), self.robot.frame())
        overhead_jpg = await self.robot.overhead_frame()
        frame_path = None
        if jpg:
            frame_path = str(self.run_dir / f"step_{step:03d}.jpg")
            Path(frame_path).write_bytes(jpg)
        overhead_path = None
        if overhead_jpg:
            overhead_path = str(self.run_dir / f"step_{step:03d}_overhead.jpg")
            Path(overhead_path).write_bytes(overhead_jpg)
        return state, jpg, frame_path, overhead_jpg, overhead_path

    # ------------------------------------------------------------------ prompt
    @staticmethod
    def _step_block(s: StepRecord) -> str:
        lines = [f"<step_{s.step}>"]
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

    def _history_text(self) -> str:
        """對應 browser_use 的 message manager 裁剪：不是單純「留最近 N 步」，是留最近 N 步裡面，
        由最新往回塞、直到湊滿 token 預算為止 (最新一步不管多大都一定留著，避免歷史空白)。
        history_items 仍然是硬上限，防止大量很短的步驟把預算全部塞滿、context 卻暴增。"""
        capped = self.result.steps[-self.history_items:]
        hard_omitted = len(self.result.steps) - len(capped)
        if not capped:
            return "(no steps yet)"
        chosen: list[str] = []
        used = 0
        soft_omitted = 0
        for s in reversed(capped):
            block = self._step_block(s)
            cost = _estimate_tokens(block)
            if chosen and used + cost > self.max_history_tokens:
                soft_omitted += 1
                continue
            chosen.append(block)
            used += cost
        chosen.reverse()
        omitted = hard_omitted + soft_omitted
        lines = []
        if omitted > 0:
            lines.append(f"[{omitted} earlier steps omitted to stay within the context budget; rely on your memory field]")
        lines.extend(chosen)
        return "\n".join(lines)

    def _full_history_text(self, steps: list) -> str:
        """跟 _history_text 不同：不截斷，給 planner/摘要用——這兩個都是低頻率呼叫
        (每 N 步一次)，需要看到完整範圍才有辦法判斷「是不是在繞圈」或濃縮進摘要。"""
        if not steps:
            return "(no steps yet)"
        return "\n".join(self._step_block(s) for s in steps)

    async def _maybe_plan(self, step: int) -> None:
        """每 planner_interval 步呼叫一次規劃者，把結果存進 self.latest_plan，下一次
        _build_user_message 會塞進 <plan> 區塊。第一步還沒有任何歷史可看，跳過不呼叫。
        規劃者呼叫失敗不該拖垮任務本身，安靜略過、沿用上一版計畫即可。"""
        if step % self.planner_interval != 0:
            return
        prompt = (
            PLANNER_PROMPT.format(task=self.task) +
            f"\n\n<history>\n{self._full_history_text(self.result.steps)}\n</history>\n\n"
            f"<world_state>\n{json.dumps(self.world_state.to_prompt_dict(), ensure_ascii=False)}\n</world_state>"
        )
        try:
            resp = await self.planner_llm.ainvoke([UserMessage(content=prompt)])
            text = resp.completion if isinstance(resp.completion, str) else str(resp.completion)
            self.latest_plan = " ".join(text.split())[:600]
            print(f"   🗺️  plan: {self.latest_plan}")
        except Exception as e:
            print(f"   ⚠️ planner 呼叫失敗（{type(e).__name__}），沿用上一版計畫")

    async def _update_long_term_summary(self) -> None:
        """對應 browser_use 的長期記憶：每累積滿一個 history_items 大小的視窗，就把「即將被
        _history_text 的 token 預算擠出視窗」的那一整批舊步驟濃縮成一段摘要，累加進
        self.long_term_summary。跟逐步截斷不同，這裡的內容不會再消失，只會越摘越精簡。
        用批次 (每 history_items 步一次) 而不是每步都摘要，避免長任務多花太多額外 LLM 呼叫。"""
        n = len(self.result.steps)
        if n == 0 or n % self.history_items != 0:
            return
        chunk = self.result.steps[n - self.history_items:n]
        prompt = (
            "You maintain a running summary of a robot's completed steps for a long-horizon task, "
            "so older details are not lost even after they scroll out of the visible history window.\n\n"
            f"Existing summary so far:\n{self.long_term_summary or '(none yet)'}\n\n"
            f"New steps to fold in:\n{self._full_history_text(chunk)}\n\n"
            "Write the UPDATED summary: 2 to 5 short sentences covering everything still relevant "
            "(what has been tried, what worked or failed, where the robot ended up). Be concise. "
            "Reply with ONLY the updated summary text, nothing else."
        )
        try:
            resp = await self.llm.ainvoke([UserMessage(content=prompt)])
            text = resp.completion if isinstance(resp.completion, str) else str(resp.completion)
            self.long_term_summary = " ".join(text.split())[:800]
        except Exception:
            pass  # 摘要失敗不影響任務，維持舊摘要

    def _build_user_message(self, step: int, state: dict, jpg: Optional[bytes],
                            overhead_jpg: Optional[bytes] = None) -> UserMessage:
        world_state = self.world_state.to_prompt_dict()
        self.world_state.just_recovered = False  # 一次性警示：這次讓 LLM 看過就清掉，不然會一直顯示
        text = f"<user_request>\n{self.task}\n</user_request>\n\n"
        if self.long_term_summary:
            text += f"<long_term_summary>\n{self.long_term_summary}\n</long_term_summary>\n\n"
        if self.latest_plan:
            text += f"<plan>\n{self.latest_plan}\n</plan>\n\n"
        text += (
            f"<agent_history>\n{self._history_text()}\n</agent_history>\n\n"
            f"<world_state>\n{json.dumps(world_state, ensure_ascii=False)}\n</world_state>\n\n"
            f"<robot_state>\n{json.dumps(state, ensure_ascii=False)}\n</robot_state>\n\n"
            f"<step_info>Step {step} of {self.max_steps}.</step_info>\n"
        )
        if step >= self.max_steps:
            text += "This is your LAST step. You must call done now and report what was achieved.\n"
        parts: list = [ContentPartTextParam(text=text)]
        if jpg:
            parts.append(ContentPartTextParam(text="Current FRONT camera image (your own eye, mounted on the arm):"))
            b64 = base64.b64encode(jpg).decode("ascii")
            parts.append(ContentPartImageParam(
                image_url=ImageURL(url=f"data:image/jpeg;base64,{b64}", media_type="image/jpeg", detail="auto")))
        else:
            parts.append(ContentPartTextParam(text="(No front camera image is available for this step. Be extra careful.)"))
        if overhead_jpg:
            parts.append(ContentPartTextParam(
                text="Current OVERHEAD camera image (fixed camera above the whole room, looking straight down; "
                     "not your own eye, does not move with you):"))
            b64o = base64.b64encode(overhead_jpg).decode("ascii")
            parts.append(ContentPartImageParam(
                image_url=ImageURL(url=f"data:image/jpeg;base64,{b64o}", media_type="image/jpeg", detail="auto")))
        return UserMessage(content=parts)

    # ------------------------------------------------------------------ 行動
    # 對應 browser_use 的 Tools() registry：每個動作叫什麼名字、參數長什麼樣子、由哪個方法
    # 處理，集中登記在 actions.ACTION_SPECS，這裡的 dispatch 表直接從那份登記表衍生，不用
    # 再手動維護一份重複的名稱清單——新增一個動作只要在 actions.py 加一筆就好。
    # 每個 handler 統一簽章 (name, params, record, index) -> ActionResult；handler 只負責
    # 回報「發生了什麼」，不直接碰 record.actions/world_state——那是 _execute 單一的職責，
    # 這樣「一個結果該怎麼影響 agent 狀態」只有一個地方在決定 (對應 ActionResult)。
    ACTION_HANDLERS: dict[str, str] = {name: spec.handler_name for name, spec in ACTION_SPECS.items()}

    _ICONS = {"remember": "📝", "ask_human": "🟡"}

    @classmethod
    def _icon(cls, name: str, ok: bool) -> str:
        if name in ("locate", "face"):
            return "🎯" if ok else "🔍"
        if name == "locate_overhead":
            return "🛰️" if ok else "🔍"
        return cls._ICONS.get(name, "✅" if ok else "❌")

    async def _execute(self, output: RobotAgentOutput, record: StepRecord) -> None:
        actions = output.action[: self.max_actions_per_step]
        for i, action in enumerate(actions):
            name, params = action.unpack()
            handler = getattr(self, self.ACTION_HANDLERS[name])
            result: ActionResult = await handler(name, params, record, i)

            record.actions.append(ActionRecord(
                name=name, params=params, ok=result.ok, message=result.message,
                duration_s=result.duration_s,
                counts_as_failure=result.counts_as_failure, resets_streak=result.resets_streak))
            print(f"   {self._icon(name, result.ok)} {name}({json.dumps(params, ensure_ascii=False)}) → {result.message}")

            if result.recovered:
                self.world_state.just_recovered = True
            if name == "move_chassis" and result.ok:
                # 底盤真的動了，之前記下的方位角/轉向建議全部失效，留著只會誤導 LLM。
                self.world_state.known_objects.clear()

            if result.stop_step:
                return

    async def _h_done(self, name: str, params: dict, record: StepRecord, index: int) -> ActionResult:
        if index > 0:  # done 必須單獨一步，前面的動作結果還沒被看過
            return ActionResult(
                ok=False, message="done ignored: it must be the only action in its step, so you can verify first")
        self.result.is_done = True
        self.result.success = bool(params.get("success"))
        self.result.final_text = str(params.get("text", ""))
        return ActionResult(ok=True, message="task finished")

    async def _h_locate(self, name: str, params: dict, record: StepRecord, index: int) -> ActionResult:
        return await self._perceive(name, params, record)

    async def _h_locate_overhead(self, name: str, params: dict, record: StepRecord, index: int) -> ActionResult:
        return await self._perceive_overhead(params, record)

    async def _h_remember(self, name: str, params: dict, record: StepRecord, index: int) -> ActionResult:
        msg = self._remember(str(params.get("note", "")))
        return ActionResult(ok=True, message=msg, stop_step=False)

    async def _h_ask_human(self, name: str, params: dict, record: StepRecord, index: int) -> ActionResult:
        answer = await self.ask_human(str(params.get("question", "")))
        return ActionResult(ok=True, message=f"Human answered: {answer}", stop_step=False)

    async def _h_physical(self, name: str, params: dict, record: StepRecord, index: int) -> ActionResult:
        # 防呆：擋下實質上不會有效果的移動，不只擋精確的 0 (見上面 NOOP_THRESHOLDS 的註解)。
        # 這是 LLM 選錯動作格式，根本沒送到 server，所以既不算失敗、也不打斷失敗連勝紀錄
        # (resets_streak=False)：它跟一次真的執行失敗性質不同，不該讓已經累積的真失敗歸零，
        # 也不該自己再往上加。
        thr = NOOP_THRESHOLDS.get(name)
        if thr and all(abs(float(params.get(k, 0) or 0)) < v for k, v in thr.items()):
            return ActionResult(
                ok=False,
                message=f"rejected: this {name} is too small to have any real effect (every value is near zero). "
                        f"If you just want to look again without actually moving, use wait(seconds) instead. "
                        f"If you meant to move a different part of the robot, check the action name.",
                counts_as_failure=False, resets_streak=False)

        t0 = time.time()
        res = await self.robot.act(name, params)
        ok = bool(res.get("ok"))
        message = str(res.get("message", ""))
        counts_as_failure, recovered = _classify_result(ok, message)
        return ActionResult(
            ok=ok, message=message, duration_s=float(res.get("duration_s", time.time() - t0)),
            counts_as_failure=counts_as_failure, resets_streak=recovered, recovered=recovered,
            stop_step=not ok)

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
    def _remember_object(self, obj: str, step: int, source: str, summary: str) -> None:
        """把感知結果存進 world_state，不用等 LLM 自己把座標/方位角抄進 memory 欄位——
        這樣即使這一步之後被歷史截斷 (見 _history_text)，事實還在，不會失憶。"""
        self.world_state.known_objects[obj] = KnownObject(step=step, source=source, summary=summary)

    async def _perceive(self, name: str, params: dict, record: StepRecord) -> ActionResult:
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
        if ok:
            self._remember_object(obj, record.step, name, msg)
        # 感知動作沒有實際移動機器人，找不到東西不代表卡住，所以不計入失敗連勝，
        # 而且找或沒找到都直接把連勝歸零：agent 明顯還在積極嘗試，不是卡死。
        # stop_step=True：感知結果要先讓 LLM 看過，所以這一步到此為止。
        return ActionResult(ok=ok, message=msg, duration_s=round(time.time() - t0, 2),
                             counts_as_failure=False, resets_streak=True)

    async def _perceive_overhead(self, params: dict, record: StepRecord) -> ActionResult:
        """跟 _perceive 是同一種「感知動作，這一步就此結束」的寫法，圖跟換算方式不同。"""
        obj = str(params.get("object", "")).strip() or "target"
        t0 = time.time()
        try:
            jpg = await self.robot.overhead_frame()
            if jpg is None:
                ok, msg = False, ("no overhead camera available (run add_overhead_camera.py on the "
                                  "robot side first, or this robot simply does not have one)")
            else:
                depth_png = await self.robot.overhead_depth()  # None 就沒有高度資訊，locate_overhead 會照舊運作
                state = record.state_before or {}
                found = await perception.locate_overhead(
                    self.llm, jpg, obj, state.get("world_xy"), state.get("yaw_deg"), depth_png)
                ok, msg = found.found, found.describe(obj)
        except Exception as e:
            ok, msg = False, f"perception failed: {type(e).__name__}: {' '.join(str(e).split())[:200]}"
        if ok:
            self._remember_object(obj, record.step, "locate_overhead", msg)
        return ActionResult(ok=ok, message=msg, duration_s=round(time.time() - t0, 2),
                             counts_as_failure=False, resets_streak=True)

    async def _h_align_to_tunnel(self, name: str, params: dict, record: StepRecord, index: int) -> ActionResult:
        """跟 _perceive_overhead 同一種寫法，但算的是「通道朝哪個方向」，不是「中心點在哪」——
        見 perception.locate_tunnel_axis 的說明：光是轉向面對一個細長物體的中心，常常還是
        從側面斜看，不是真的順著它的長軸看進去。"""
        obj = str(params.get("object", "")).strip() or "target"
        t0 = time.time()
        try:
            jpg = await self.robot.overhead_frame()
            if jpg is None:
                ok, msg = False, ("no overhead camera available (run add_overhead_camera.py on the "
                                  "robot side first, or this robot simply does not have one)")
            else:
                state = record.state_before or {}
                found = await perception.locate_tunnel_axis(
                    self.llm, jpg, obj, state.get("world_xy"), state.get("yaw_deg"))
                ok, msg = found.found, found.describe(obj)
        except Exception as e:
            ok, msg = False, f"perception failed: {type(e).__name__}: {' '.join(str(e).split())[:200]}"
        return ActionResult(ok=ok, message=msg, duration_s=round(time.time() - t0, 2),
                             counts_as_failure=False, resets_streak=True)

    async def _call_llm(self, messages) -> tuple[RobotAgentOutput, float]:
        """有些 provider（實測見於 Gemini Robotics-ER 的 thinking_config：隱藏推理跟最終 JSON
        共用同一個 token 額度）偶爾會生成途中暴走，額度燒光時 JSON 還沒寫完就被截斷、整步作廢。
        這種情況是機率性的，同樣的輸入重打一次通常就正常了，立即重試一次的成本（一次 API 呼叫）
        遠低於白白浪費掉一整個 step、甚至因此把任務的失敗額度燒光。"""
        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(2):
            try:
                t0 = time.time()
                completion = await self.llm.ainvoke(messages, output_format=RobotAgentOutput)
                return completion.completion, round(time.time() - t0, 2)
            except Exception as e:
                last_exc = e
                if attempt == 0:
                    print(f"   ⚠️ LLM 呼叫失敗（{type(e).__name__}），立即重試一次")
        raise last_exc

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
                state, jpg, frame_path, overhead_jpg, overhead_path = await self._observe(step)
                record = StepRecord(step=step, timestamp=time.time(), state_before=state,
                                    frame_path=frame_path, overhead_frame_path=overhead_path)
                await self._maybe_plan(step)

                try:
                    output, record.llm_seconds = await self._call_llm(
                        [self.system_message, self._build_user_message(step, state, jpg, overhead_jpg)])
                except Exception as e:
                    failures += 1
                    self.world_state.consecutive_failures = failures  # 這條路徑跳過下面的同步，得在這裡補上
                    detail = " ".join(str(e).split())[:300]
                    record.error = f"LLM call failed ({type(e).__name__}): {detail}. Your previous output was invalid or too long; no action was executed. Keep numbers short (at most 2 decimals)."
                    print(f"📍 Step {step}: ❌ {record.error}")
                    self.result.steps.append(record)
                    self._save()
                    await self._update_long_term_summary()
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
                await self._update_long_term_summary()

                if self.result.is_done:
                    break
                # 每個 ActionRecord 在建立的當下就已經決定好 counts_as_failure / resets_streak
                # (格式錯誤、感知動作、伺服器自動恢復都各自在建立處標好)，這裡不用再用字串
                # 前綴或動作名稱清單去猜——這正是今天炸掉的地方：agent 端和 server 端各自維護
                # 一個門檻 3 次的計數器，agent 端把格式錯誤也算進去，搶在 server 端真正的卡死
                # 偵測有機會累積之前就先放棄了任務。
                last = record.actions[-1] if record.actions else None
                if last:
                    if last.resets_streak:
                        failures = 0
                    elif last.counts_as_failure and not last.ok:
                        failures += 1
                    elif last.ok:
                        failures = 0
                    # 其餘情況 (格式錯誤但不計失敗)：連勝次數不變
                    self.world_state.consecutive_failures = failures
                    if failures >= self.max_failures:
                        print(f"⛔ 連續失敗 {failures} 次，中止任務")
                        break
        finally:
            # 跟 test_movement.py 的 finally 同一個精神：不管怎麼結束，先讓底盤停下來
            await self.robot.stop()
            try:
                state, jpg, _, overhead_jpg, _ = await self._observe(0)
                self.result.final_state = state
                if jpg:
                    final = self.run_dir / "final.jpg"
                    final.write_bytes(jpg)
                    (self.run_dir / "step_000.jpg").unlink(missing_ok=True)
                    self.result.final_frame_path = str(final)
                if overhead_jpg:
                    final_overhead = self.run_dir / "final_overhead.jpg"
                    final_overhead.write_bytes(overhead_jpg)
                    (self.run_dir / "step_000_overhead.jpg").unlink(missing_ok=True)
            except Exception:
                pass
            self.result.duration_s = round(time.time() - t_start, 1)
            self._save()

        status = "✅ 成功" if self.result.is_successful() else ("❌ 失敗" if self.result.is_done else "⏹ 未完成")
        print(f"\n{status}｜{self.result.number_of_steps()} 步｜{self.result.duration_s}s｜紀錄在 {self.run_dir}")
        if self.result.final_result():
            print(f"📝 {self.result.final_result()}")
        return self.result

    def _save(self) -> None:
        self.result.latest_plan = self.latest_plan
        self.result.long_term_summary = self.long_term_summary
        (self.run_dir / "history.json").write_text(
            self.result.model_dump_json(indent=2), encoding="utf-8")
