"""
Milestone 判定與任務裁決。

跟原本 evaluation/milestone_checker.py 的差別：
    1. 讀的是 runs/<時間>/history.json 和 step_XXX.jpg，不需要瀏覽器的 AgentHistoryList
    2. milestone 可以帶 telemetry 條件，用遙測數值直接判定，不花 LLM 也不會誤判
       (機器人比 GUI 多了這個優勢：yaw、距離、夾爪狀態都是量得到的事實)
    3. 其餘 milestone 和整體成敗交給 judge LLM，一次呼叫判完，用 structured output 取代手動解析 JSON
"""

import asyncio
import base64
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from robot_agent.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL, SystemMessage, UserMessage


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
    "task_success is true only if the user's request is fully satisfied at the end of the run. "
    "Odometry noise: position_m drifts by a few millimetres and yaw_deg by up to about 3 degrees even when the chassis "
    "never moves; changes below 0.03 m or 3 degrees are sensor noise, never evidence of movement. To decide whether the "
    "chassis moved, use world_xy (real world coordinates) and whether any move_chassis action was actually executed. "
    "Milestones already settled by telemetry are facts; do not contradict them. "
    "Images labelled 'overhead camera' come from a fixed camera above the room looking straight down; the robot "
    "itself appears in them from above. An object that is visible there, or that the log shows was located from "
    "the overhead camera with world coordinates, counts as seen/found for milestones about seeing or locating it. "
    "The step log lists, from telemetry, which fixed landmarks (pillars, tunnel, ball pit, ...) were within the front "
    "camera's field-of-view ANGLE at each step. That is angle only: a landmark in that list can still be hidden behind "
    "other objects (the slide, the platform, a wall corner). Treat a landmark that is NOT in the list as not visible, but "
    "for one that IS in the list, decide from the image whether it was actually visible before blaming the agent for "
    "not reporting it. An agent that truthfully reports only what its images showed has not failed by omitting a "
    "landmark that was occluded. "
    "The image sample may not include every step: if the step log shows an action that would plausibly have produced "
    "the evidence at a step whose image is not shown, say so in your reasoning instead of treating the missing image as "
    "proof that it never happened."
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
    states = []
    for s in steps:
        states.append((s["step"], s.get("state_before") or {}))
        # 每個動作之後的遙測也算：一步裡「開夾爪、等、關夾爪」三個動作，只看步驟開頭永遠看不到「開著」的狀態
        for i, a in enumerate(s.get("actions", [])):
            # 技能（pick/place）途中的快照也算：夾爪「開著」只在技能執行到一半時看得到
            for k, tr in enumerate(a.get("state_trace") or []):
                states.append((f"{s['step']}.{i + 1} ({tr.get('trace_tag', 'trace %d' % k)})", tr))
            if a.get("state_after"):
                states.append((f"{s['step']}.{i + 1}", a["state_after"]))
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
            where = "final state" if step is None else (f"after action {step}" if isinstance(step, str) else f"before step {step}")
            shown = relval(st) if cond.get("relative_to_start") else _get(st, field)
            return MilestoneResult(milestone["id"], milestone["description"], True, "telemetry",
                                   f"{field}={shown} at {where}" +
                                   (f" (baseline {baseline})" if cond.get("relative_to_start") else ""), step)
    seen = relval(final) if when == "final" else [relval(st) for _, st in states]
    return MilestoneResult(milestone["id"], milestone["description"], False, "telemetry",
                           f"{field} never satisfied {json.dumps({k: v for k, v in cond.items() if k not in ('field', 'when')})}; observed {seen}")


# ---------------------------------------------------------------- LLM judge
def _landmarks_in_view(state: dict, half_fov_deg: float = 45.0) -> str:
    """從遙測算出這一步相機視野內有哪些固定地標（|turn_left| 在半視角內），給 judge 當客觀證據——
    judge 只抽樣看部分畫面，光靠影像常把「掃描時依序看到什麼」判錯（實測 e07）。"""
    lm = (state or {}).get("static_landmarks") or {}
    seen = [f"{n} ({v['turn_left_deg']:+.0f} deg, {v['distance_m']:.1f} m)" for n, v in sorted(lm.items())
            if isinstance(v, dict) and abs(v.get("turn_left_deg", 999)) <= half_fov_deg]
    return ", ".join(seen) if seen else "(none of the fixed landmarks)"


def _all_states(history: dict) -> list:
    out = []
    for s in history.get("steps", []):
        out.append(s.get("state_before") or {})
        for a in s.get("actions", []):
            out.extend(a.get("state_trace") or [])
            if a.get("state_after"):
                out.append(a["state_after"])
    out.append(history.get("final_state") or {})
    return out


def check_report(milestone: dict, history: dict) -> Optional[MilestoneResult]:
    """report 型 milestone（確定性的，不用 LLM）：
    regex           最終報告要符合的正規表示式（例如「四種顏色」這種有標準答案的題）。
    landmark_order  掃描類任務：用遙測（每一步/每個動作後的 static_landmarks 方位角）推算固定地標「依序進入
                    相機視野」的順序，對照報告裡對應顏色詞第一次出現的順序；judge 只抽樣看部分畫面，
                    光靠影像判這種順序實測會判錯（e07：agent 回報的順序其實是對的）。"""
    cond = milestone.get("report")
    if not cond:
        return None
    text = history.get("final_text") or ""
    if cond.get("regex"):
        ok = re.search(cond["regex"], text, re.I) is not None
        return MilestoneResult(milestone["id"], milestone["description"], ok, "report",
                               f"final report {'matches' if ok else 'does not match'} /{cond['regex']}/: {text[:120]!r}")
    if cond.get("landmark_order"):
        names = cond["landmark_order"]
        half = float(cond.get("half_fov_deg", 45.0))
        seen = []
        for st in _all_states(history):
            lm = st.get("static_landmarks") or {}
            for n in names:
                v = lm.get(n)
                if isinstance(v, dict) and abs(v.get("turn_left_deg", 999)) <= half and n not in seen:
                    seen.append(n)
        low = text.lower()
        reported = sorted(((low.find(n.split("_")[-1]), n) for n in names if n.split("_")[-1] in low))
        reported = [n for _, n in reported]
        # 「進入視野」只是用方位角算的，不知道有沒有被擋住：2026-10-07 的 e07，yaw -97.5 時黃柱子在
        # +43.5°（視野邊緣），但被滑梯擋住，畫面裡根本沒有；agent 照實回報藍、紅、綠，舊的「整串
        # 相等」判定卻判它錯。所以遙測只能確認兩件事：報出來的每一根都真的轉進過視野（沒有憑空捏造），
        # 而且相對順序對（10/6 那次在過期畫面下報成藍、綠、紅、黃，綠在紅前面，這裡照樣抓得到）。
        # 少報被擋住的那根不算錯；至少要報 min_seen 根。
        pos = {n: i for i, n in enumerate(seen)}
        in_view = all(n in pos for n in reported)
        in_order = in_view and [pos[n] for n in reported] == sorted(pos[n] for n in reported)
        ok = len(reported) >= int(cond.get("min_seen", 1)) and in_order
        why = ("ok" if ok else
               "reports a pillar that never entered the view" if not in_view else
               "relative order differs from telemetry" if not in_order else
               "fewer pillars reported than min_seen")
        return MilestoneResult(milestone["id"], milestone["description"], ok, "report",
                               f"telemetry says the landmarks entered the camera view in this order: {seen} "
                               f"(angle only, occlusion unknown); the report lists them as: {reported} -> {why}")
    return None


def _step_log(history: dict) -> str:
    lines = []
    for s in history.get("steps", []):
        lines.append(f"[step {s['step']}] telemetry={json.dumps(s.get('state_before', {}))}")
        lines.append(f"  fixed landmarks within the front camera's field-of-view angle at this step (telemetry, angle only, may be occluded): "
                     f"{_landmarks_in_view(s.get('state_before'))}")
        if s.get("error"):
            lines.append(f"  error: {s['error']}")
            continue
        lines.append(f"  goal: {s.get('next_goal', '')}")
        for a in s.get("actions", []):
            lines.append(f"  action {a['name']}({json.dumps(a['params'])}) -> {'ok' if a['ok'] else 'FAILED'}: {a['message']}")
    lines.append(f"[final] telemetry={json.dumps(history.get('final_state', {}))}")
    lines.append(f"  fixed landmarks within the front camera's field-of-view angle at the end (telemetry, angle only, may be occluded): "
                 f"{_landmarks_in_view(history.get('final_state'))}")
    lines.append(f"[agent's final report, NOT evidence] success={history.get('success')} text={history.get('final_text', '')!r}")
    return "\n".join(lines)


def _sample_frames(history: dict, max_frames: int) -> list[tuple[str, str]]:
    steps = [s for s in history.get("steps", []) if s.get("frame_path") and Path(s["frame_path"]).exists()]
    if max_frames > 0 and len(steps) > max_frames - 1:  # 平均取樣，頭尾一定保留；max_frames<=0 代表全部都給
        idx = sorted({round(i * (len(steps) - 1) / (max_frames - 2)) for i in range(max_frames - 1)})
        steps = [steps[i] for i in idx]
    frames = []
    # 俯視相機的畫面也給 judge（低解析度、每隔一步一張）：agent 用 locate_overhead 找到的東西只出現在這張圖，
    # 只給前鏡頭會把「從天上看到球、正確回報距離」判成「球從來沒出現在畫面裡」（實測 e06）。
    # 每張都給、高解析度的話一次請求會超過 gpt-4o 的 TPM 上限（實測 429）。
    for i, s in enumerate(steps):
        frames.append((f"step {s['step']} (front camera)", s["frame_path"], "auto"))
        ov = s.get("overhead_frame_path")
        if ov and Path(ov).exists() and (i % 2 == 0 or i == len(steps) - 1):
            frames.append((f"step {s['step']} (overhead camera, fixed top-down view of the whole room)", ov, "low"))
    # pick/place 之後手臂放低拍的證據影格：夾爪裡的物件 / 放下的物件，每張都給（數量很少）
    for s in history.get("steps", []):
        for a in s.get("actions", []):
            ev = a.get("evidence_frame_path")
            if ev and Path(ev).exists():
                frames.append((f"step {s['step']} right after {a.get('name')} (front camera, arm lowered to look at the gripper/floor)", ev, "auto"))
    final = history.get("final_frame_path")
    if final and Path(final).exists():
        frames.append(("FINAL (front camera)", final, "auto"))
    return frames


async def judge_run(task: dict, history: dict, judge_llm, max_frames: int = 12) -> tuple[list[MilestoneResult], dict]:
    """回傳 (每個 milestone 的結果, {"verdict", "reasoning", "failure_reason"})。"""
    milestones = task.get("milestones", [])
    results: dict[str, MilestoneResult] = {}
    for m in milestones:
        r = check_telemetry(m, history) or check_report(m, history)
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
    def build_parts(frames):
        parts: list = [ContentPartTextParam(text=text)]
        for label, path, detail in frames:
            b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            parts.append(ContentPartTextParam(text=f"Image at {label}:"))
            parts.append(ContentPartImageParam(image_url=ImageURL(url=f"data:image/jpeg;base64,{b64}",
                                                                  media_type="image/jpeg", detail=detail)))
        return parts

    judgement = {"verdict": None, "reasoning": None, "failure_reason": None}
    frames = _sample_frames(history, max_frames)
    try:
        resp = None
        last_err = None
        # 429（TPM 超限）就退避重試；還是不行就把圖減半再試一次，不要讓整題的 LLM milestone 全部變成「judge failed」
        for attempt, wait_s in enumerate((0, 20, 45, 70)):
            if wait_s:
                await asyncio.sleep(wait_s)
            try:
                resp = await judge_llm.ainvoke([SystemMessage(content=JUDGE_SYSTEM), UserMessage(content=build_parts(frames))],
                                               output_format=_JudgeOutput)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if "429" not in str(e) and "RateLimit" not in type(e).__name__:
                    raise
                if attempt == 1 and len(frames) > 4:
                    frames = frames[::2]  # 請求本身太大時，抽樣減半
        if resp is None:
            raise last_err
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
