"""任務指標與彙總報告。欄位與成功判準沿用原本的 evaluation/metrics.py。"""

from dataclasses import asdict, dataclass
from typing import Optional

from robot_eval.judge import MilestoneResult


@dataclass
class TaskMetrics:
    task_id: str
    difficulty: str
    category: Optional[str]
    total_milestones: int
    completed_milestones: int
    milestone_completion_rate: float
    agent_marked_done: bool
    agent_success: Optional[bool]
    judge_verdict: Optional[bool]
    judge_reasoning: Optional[str]
    judge_failure_reason: Optional[str]
    total_steps: int
    total_duration_seconds: float
    error_count: int          # LLM 呼叫失敗或動作失敗的步數
    total_distance_m: float   # 機器人專屬：下令的總平移距離，可以拿來比較路徑效率
    is_successful: bool
    run_dir: str = ""
    judge_overruled: bool = False  # judge 判敗、但所有 milestone 都已由遙測證實通過，最終仍算成功


def create_task_metrics(task: dict, history: dict, milestones: list[MilestoneResult], judgement: dict) -> TaskMetrics:
    total = len(milestones)
    done = sum(1 for m in milestones if m.passed)
    verdict = judgement.get("verdict")
    # judge 的裁決優先；沒有裁決時，退回用 milestone 全過當作成功。
    # 例外：所有 milestone 都是「遙測」直接證實通過（例如夾爪狀態、yaw 變化這種量得到的事實），
    # judge 就不能只憑影像或里程計雜訊否決——實測 gpt-4o judge 曾因 3 mm 的里程計漂移把一個
    # 動作完全正確的 e08 判成「底盤動了」。這種情況記成成功並標記 judge_overruled，方便事後複核。
    telemetry_all_passed = total > 0 and all(m.passed and m.method in ("telemetry", "report") for m in milestones)
    overruled = verdict is False and telemetry_all_passed
    success = True if overruled else (verdict if verdict is not None else (total > 0 and done == total))

    steps = history.get("steps", [])
    errors = sum(1 for s in steps if s.get("error") or any(not a["ok"] for a in s.get("actions", [])))
    dist = sum((a["params"].get("forward_m", 0) ** 2 + a["params"].get("right_m", 0) ** 2) ** 0.5
               for s in steps for a in s.get("actions", []) if a["name"] == "move_chassis" and a["ok"])

    return TaskMetrics(
        task_id=task["id"], difficulty=task.get("difficulty", "unknown"), category=task.get("category"),
        total_milestones=total, completed_milestones=done,
        milestone_completion_rate=done / total if total else 0.0,
        agent_marked_done=bool(history.get("is_done")), agent_success=history.get("success"),
        judge_verdict=verdict, judge_reasoning=judgement.get("reasoning"),
        judge_failure_reason=judgement.get("failure_reason"),
        total_steps=len(steps), total_duration_seconds=float(history.get("duration_s", 0.0)),
        error_count=errors, total_distance_m=round(dist, 2), is_successful=bool(success),
        run_dir=history.get("run_dir", ""), judge_overruled=overruled,
    )


def _rate(items: list[TaskMetrics]) -> dict:
    n = len(items)
    return {
        "tasks": n,
        "success_rate": sum(m.is_successful for m in items) / n if n else 0.0,
        "avg_milestone_completion": sum(m.milestone_completion_rate for m in items) / n if n else 0.0,
        "avg_steps": sum(m.total_steps for m in items) / n if n else 0.0,
        "avg_duration_s": sum(m.total_duration_seconds for m in items) / n if n else 0.0,
    }


def build_report(all_metrics: list[TaskMetrics]) -> dict:
    def group(key):
        out: dict[str, list[TaskMetrics]] = {}
        for m in all_metrics:
            out.setdefault(str(getattr(m, key)), []).append(m)
        return {k: _rate(v) for k, v in sorted(out.items())}

    # agent 自評與 judge 不一致的比例：自評成功但其實失敗，是實體機器人上最危險的一種錯
    overclaim = [m.task_id for m in all_metrics if m.agent_success and not m.is_successful]
    return {
        "overall": _rate(all_metrics),
        "by_difficulty": group("difficulty"),
        "by_category": group("category"),
        "agent_overclaimed_success": overclaim,
        "tasks": [asdict(m) for m in all_metrics],
    }
