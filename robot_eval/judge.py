"""
Milestone 判定與任務裁決。

跟原本 evaluation/milestone_checker.py 的差別：
    1. 讀的是 runs/<時間>/history.json 和 step_XXX.jpg，不需要瀏覽器的 AgentHistoryList
    2. milestone 可以帶 telemetry 條件，用遙測數值直接判定，不花 LLM 也不會誤判
       (機器人比 GUI 多了這個優勢：yaw、距離、夾爪狀態都是量得到的事實)
    3. 其餘 milestone 和整體成敗交給 judge LLM，一次呼叫判完，用 structured output 取代手動解析 JSON
"""

import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL, SystemMessage, UserMessage


@dataclass
class MilestoneResult:
    milestone_id: str
    description: str
    passed: bool
    method: str  # "telemetry" 或 "llm"
    reasoning: str = ""
    step_completed_at: Optional[int] = None


class _MilestoneVerdict(BaseModel):
    id: str
    verdict: bool
    reasoning: str
    step_completed_at: Optional[int] = Field(None, description="first step number whose image or log shows it was achieved")


class _JudgeOutput(BaseModel):
    milestones: list[_MilestoneVerdict]
    task_success: bool
    task_reasoning: str
    failure_reason: Optional[str] = None


JUDGE_SYSTEM = (
    "You are a strict evaluator of a mobile robot (DJI RoboMaster EP) that was controlled by an LLM agent. "
    "You receive the task, the agent's step log with telemetry, a sample of camera images labelled with their step number, "
    "and the final camera image. Decide from EVIDENCE only. The agent's own claims (its memory, its final report, "
    "'ok' action results) are not evidence; camera images and telemetry are. "
    "A milestone passes if it was true at any point of the run, unless it describes the end state. "
    "task_success is true only if the user's request is fully satisfied at the end of the run."
)


# ---------------------------------------------------------------- telemetry
def _get(state: dict, field: str) -> Any:
    cur: Any = state
    for key in field.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _match(value: Any, cond: dict) -> bool:
    if value is None:
        return False
    if "equals" in cond:
        return value == cond["equals"]
    if "in" in cond:
        return value in cond["in"]
    try:
        v = float(value)
    except Exception:
        return False
    if "abs_min" in cond or "abs_max" in cond:
        v = abs(v)
        return cond.get("abs_min", float("-inf")) <= v <= cond.get("abs_max", float("inf"))
    return cond.get("min", float("-inf")) <= v <= cond.get("max", float("inf"))


def check_telemetry(milestone: dict, history: dict) -> Optional[MilestoneResult]:
    """回傳 None 代表這個 milestone 沒辦法用遙測判定 (沒寫條件，或機器人沒回報那個欄位)，交給 LLM。"""
    cond = milestone.get("telemetry")
    if not cond:
        return None
    field, when = cond["field"], cond.get("when", "any")
    final = history.get("final_state") or {}
    steps = history.get("steps", [])
    states = [(s["step"], s.get("state_before") or {}) for s in steps]
    states.append((None, final))
    if all(_get(st, field) is None for _, st in states):
        return None

    # relative_to_start：像「轉 90 度」這種任務，量的是「跟一開始比變化了多少」，不是最終角度
    # 的絕對值本身——機器人上一個任務結束時卡在哪個角度是任意的，直接看 final yaw_deg 的絕對值
    # 會把「剛好卡在符合區間的角度、其實根本沒轉」誤判成通過。這裡改成先減掉任務開始時的初值，
    # 再套用同一組 abs_min/abs_max/min/max/equals/in 條件。
    baseline = 0.0
    if cond.get("relative_to_start"):
        if not steps:
            return None
        initial = steps[0].get("state_before") or {}
        base_val = _get(initial, field)
        if base_val is None:
            return None
        try:
            baseline = float(base_val)
        except Exception:
            return None

    def relval(st):
        v = _get(st, field)
        if v is None or not cond.get("relative_to_start"):
            return v
        try:
            delta = float(v) - baseline
        except Exception:
            return None
        if cond.get("angular"):
            # yaw_deg 這類角度欄位會被正規化到 -180~180，直接相減在邊界附近會算錯
            # (例如 170 度轉 90 度變成 -100 度，直接相減得到 -270 而不是 90)，
            # 這裡改成取最短路徑的角度差。
            delta = (delta + 180) % 360 - 180
        return delta

    candidates = [(None, final)] if when == "final" else states
    for step, st in candidates:
        if _match(relval(st), cond):
            where = "final state" if step is None else f"before step {step}"
            shown = relval(st) if cond.get("relative_to_start") else _get(st, field)
            return MilestoneResult(milestone["id"], milestone["description"], True, "telemetry",
                                   f"{field}={shown} at {where}" +
                                   (f" (baseline {baseline})" if cond.get("relative_to_start") else ""), step)
    seen = relval(final) if when == "final" else [relval(st) for _, st in states]
    return MilestoneResult(milestone["id"], milestone["description"], False, "telemetry",
                           f"{field} never satisfied {json.dumps({k: v for k, v in cond.items() if k not in ('field', 'when')})}; observed {seen}")


# ---------------------------------------------------------------- LLM judge
def _step_log(history: dict) -> str:
    lines = []
    for s in history.get("steps", []):
        lines.append(f"[step {s['step']}] telemetry={json.dumps(s.get('state_before', {}))}")
        if s.get("error"):
            lines.append(f"  error: {s['error']}")
            continue
        lines.append(f"  goal: {s.get('next_goal', '')}")
        for a in s.get("actions", []):
            lines.append(f"  action {a['name']}({json.dumps(a['params'])}) -> {'ok' if a['ok'] else 'FAILED'}: {a['message']}")
    lines.append(f"[final] telemetry={json.dumps(history.get('final_state', {}))}")
    lines.append(f"[agent's final report, NOT evidence] success={history.get('success')} text={history.get('final_text', '')!r}")
    return "\n".join(lines)


def _sample_frames(history: dict, max_frames: int) -> list[tuple[str, str]]:
    frames = [(f"step {s['step']}", s["frame_path"]) for s in history.get("steps", [])
              if s.get("frame_path") and Path(s["frame_path"]).exists()]
    if len(frames) > max_frames - 1:  # 平均取樣，頭尾一定保留
        idx = sorted({round(i * (len(frames) - 1) / (max_frames - 2)) for i in range(max_frames - 1)})
        frames = [frames[i] for i in idx]
    final = history.get("final_frame_path")
    if final and Path(final).exists():
        frames.append(("FINAL", final))
    return frames


async def judge_run(task: dict, history: dict, judge_llm, max_frames: int = 8) -> tuple[list[MilestoneResult], dict]:
    """回傳 (每個 milestone 的結果, {"verdict", "reasoning", "failure_reason"})。"""
    milestones = task.get("milestones", [])
    results: dict[str, MilestoneResult] = {}
    for m in milestones:
        r = check_telemetry(m, history)
        if r is not None:
            results[m["id"]] = r
    pending = [m for m in milestones if m["id"] not in results]

    settled = "\n".join(f"- [{r.milestone_id}] {r.description}: {'PASSED' if r.passed else 'FAILED'} by telemetry ({r.reasoning})"
                        for r in results.values()) or "(none)"
    todo = "\n".join(f"- [{m['id']}] {m['description']}" for m in pending) or "(none, return an empty milestones list)"
    text = (
        f"<task>\n{task['description']}\n</task>\n\n"
        f"<milestones_already_settled_by_telemetry>\n{settled}\n</milestones_already_settled_by_telemetry>\n\n"
        f"<milestones_to_judge>\n{todo}\n</milestones_to_judge>\n\n"
        f"<run_log>\n{_step_log(history)}\n</run_log>\n\n"
        "Camera images follow, each preceded by its label."
    )
    parts: list = [ContentPartTextParam(text=text)]
    for label, path in _sample_frames(history, max_frames):
        b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        parts.append(ContentPartTextParam(text=f"Image at {label}:"))
        parts.append(ContentPartImageParam(image_url=ImageURL(url=f"data:image/jpeg;base64,{b64}", media_type="image/jpeg")))

    judgement = {"verdict": None, "reasoning": None, "failure_reason": None}
    try:
        resp = await judge_llm.ainvoke([SystemMessage(content=JUDGE_SYSTEM), UserMessage(content=parts)],
                                       output_format=_JudgeOutput)
        out: _JudgeOutput = resp.completion
        by_id = {v.id: v for v in out.milestones}
        for m in pending:
            v = by_id.get(m["id"])
            results[m["id"]] = MilestoneResult(
                m["id"], m["description"], bool(v and v.verdict), "llm",
                v.reasoning if v else "judge did not return this milestone", v.step_completed_at if v else None)
        judgement = {"verdict": out.task_success, "reasoning": out.task_reasoning, "failure_reason": out.failure_reason}
    except Exception as e:
        for m in pending:
            results[m["id"]] = MilestoneResult(m["id"], m["description"], False, "llm", f"judge failed: {e}")
        judgement["failure_reason"] = f"judge failed: {type(e).__name__}: {e}"

    return [results[m["id"]] for m in milestones], judgement


def milestone_dicts(results: list[MilestoneResult]) -> list[dict]:
    return [asdict(r) for r in results]
