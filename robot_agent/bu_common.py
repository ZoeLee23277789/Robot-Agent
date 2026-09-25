"""
用「真的」browser_use.Agent 控制 RoboMaster EP 時，agent/tools/session 三邊共用的狀態跟輔助函式。

browser_use.Agent 本身已經有 Tools()/Registry (動態動作註冊表) 跟 self.state.consecutive_failures
(連續失敗計數)，這兩個不用重做。但它沒有 planner、沒有長期記憶摘要、也沒有 known_objects 這種
「感知結果暫存」的概念——這些是我們自己這幾天做出來、對這個機器人任務有實際幫助的東西，這裡用
on_step_start hook + 一個共用的 AdapterState 掛上去，不用改動 browser_use 本身的原始碼。
"""

import json
from dataclasses import dataclass, field
from typing import Optional

RECOVERY_MARKER = "auto-recovered:"  # 跟 robot_server.py 的 _note_stall 約定好的標記字串
FLAT_NOTE_MARKER = "essentially FLAT"  # 跟 perception.py 的 LocatedOverhead.describe() 用字一致


@dataclass
class AdapterState:
    """RobotSession (組 prompt) 跟 on_step_start hook (算 plan/摘要) 共用的狀態，
    不需要互相持有對方的參照——這樣 RobotSession 建立時不用等 Agent 先存在。"""
    known_objects: dict = field(default_factory=dict)  # obj 名稱 -> 描述文字 (含 step/來源)
    latest_plan: str = ""
    long_term_summary: str = ""
    just_recovered: bool = False
    consecutive_failures: int = 0
    max_failures: int = 3

    def to_world_state_text(self) -> str:
        data = {
            "consecutive_failures": f"{self.consecutive_failures}/{self.max_failures} "
                                     "(task auto-aborts once this reaches the max)",
            "just_recovered": (
                "YES - the simulation was just auto-restarted because the control channel was stuck. "
                "Your chassis/arm pose may have jumped; re-check telemetry and the camera before continuing."
                if self.just_recovered else False
            ),
            "known_objects": self.known_objects or "(nothing located yet)",
        }
        return json.dumps(data, ensure_ascii=False)


def step_block(step_num: int, model_output, results: list) -> str:
    """把 browser_use 的 AgentHistory 一步轉成人看得懂的文字，給 planner/摘要用。
    對應 robot_agent/service.py 的 _step_block，資料來源換成 AgentOutput + ActionResult。"""
    lines = [f"<step_{step_num}>"]
    if model_output is not None:
        lines.append(f"Evaluation of previous step: {model_output.evaluation_previous_goal}")
        lines.append(f"Memory: {model_output.memory}")
        lines.append(f"Goal: {model_output.next_goal}")
    for r in results or []:
        ok = r.error is None
        msg = r.error or r.extracted_content or r.long_term_memory or ""
        lines.append(f"Action -> {'ok' if ok else 'FAILED'}: {' '.join(str(msg).split())[:300]}")
    lines.append(f"</step_{step_num}>")
    return "\n".join(lines)


def full_history_text(history_items: list) -> str:
    if not history_items:
        return "(no steps yet)"
    return "\n".join(
        step_block(i + 1, h.model_output, h.result) for i, h in enumerate(history_items)
    )


PLANNER_PROMPT = """You are the strategic planner for a robot agent. You do NOT control the robot directly \
and cannot see the current camera image — you only review its history, then advise.

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


async def call_planner(llm, task: str, history_items: list) -> Optional[str]:
    """回傳新的 plan 文字，失敗回傳 None (呼叫端沿用舊的，不當成任務致命錯誤)。"""
    prompt = PLANNER_PROMPT.format(task=task) + f"\n\n<history>\n{full_history_text(history_items)}\n</history>"
    try:
        resp = await llm.ainvoke([_plain_user_message(prompt)])
        text = resp.completion if isinstance(resp.completion, str) else str(resp.completion)
        return " ".join(text.split())[:600]
    except Exception:
        return None


async def call_summary(llm, existing_summary: str, chunk_history_items: list) -> Optional[str]:
    prompt = (
        "You maintain a running summary of a robot's completed steps for a long-horizon task, "
        "so older details are not lost even after they scroll out of the visible history window.\n\n"
        f"Existing summary so far:\n{existing_summary or '(none yet)'}\n\n"
        f"New steps to fold in:\n{full_history_text(chunk_history_items)}\n\n"
        "Write the UPDATED summary: 2 to 5 short sentences covering everything still relevant "
        "(what has been tried, what worked or failed, where the robot ended up). Be concise. "
        "Reply with ONLY the updated summary text, nothing else."
    )
    try:
        resp = await llm.ainvoke([_plain_user_message(prompt)])
        text = resp.completion if isinstance(resp.completion, str) else str(resp.completion)
        return " ".join(text.split())[:800]
    except Exception:
        return None


def _plain_user_message(text: str):
    from browser_use.llm.messages import UserMessage
    return UserMessage(content=text)


def make_on_step_start(adapter_state: AdapterState, planner_llm, summary_llm,
                       planner_interval: int = 5, history_items: int = 12):
    """對應 browser_use.Agent 的 on_step_start hook：每一步都先把框架自己算好的
    consecutive_failures 同步過來 (不重複計數，直接讀)，再視情況觸發 planner / 長期摘要。"""
    async def on_step_start(agent) -> None:
        adapter_state.consecutive_failures = agent.state.consecutive_failures
        adapter_state.max_failures = agent.settings.max_failures
        step = agent.state.n_steps
        history = agent.history.history

        if step > 1 and step % planner_interval == 0:
            plan = await call_planner(planner_llm, agent.task, history)
            if plan is not None:
                adapter_state.latest_plan = plan

        if len(history) > 0 and len(history) % history_items == 0:
            chunk = history[-history_items:]
            summary = await call_summary(summary_llm, adapter_state.long_term_summary, chunk)
            if summary is not None:
                adapter_state.long_term_summary = summary

    return on_step_start
