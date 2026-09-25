"""
Agent 的輸出格式。
結構刻意跟 browser_use 的 AgentOutput 對齊 (thinking / evaluation / memory / next_goal / action[])，
這樣原本評估 GUI agent 用的 step 紀錄與 checkpoint 判定邏輯可以直接沿用。

動作的參數 model 跟 RobotAction (給 LLM 看的 action union) 都搬到 actions.py 了，
那邊是動態從 ACTION_SPECS registry 長出來的，對應 browser_use 的 Tools()——這裡繼續
re-export RobotAction，是因為 RobotAgentOutput.action 需要用到它，其他地方要用action
相關的東西應該直接 import robot_agent.actions。
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from robot_agent.actions import RobotAction

__all__ = [
    "RobotAction", "RobotAgentOutput", "ActionResult", "ActionRecord", "StepRecord",
    "RunResult", "KnownObject", "WorldState",
]


class RobotAgentOutput(BaseModel):
    thinking: str = Field(description="step-by-step reasoning about the camera image, telemetry and history")
    evaluation_previous_goal: str = Field(description="one sentence: did the last action achieve its goal? Success / Failure / Uncertain, and why")
    memory: str = Field(description="1 to 3 sentences of progress tracking that should persist to later steps")
    next_goal: str = Field(description="the immediate goal for this step, one sentence")
    action: list[RobotAction] = Field(min_length=1, description="1 to 3 actions, executed in order")


# ---- 動作執行結果 ----
# 對應 browser_use 的 ActionResult：每個 handler 執行完不直接寫歷史紀錄，而是回報
# 「發生了什麼」，由 _execute 統一決定要不要記錄失敗、要不要中斷這一步、要不要更新
# world_state。這樣「一個動作結果該如何影響 agent 的狀態」只有一個地方在決定，
# 不會散落在每個 handler 裡各自判斷。
class ActionResult(BaseModel):
    ok: bool
    message: str
    duration_s: float = 0.0
    # 這次 ok=False 要不要計入連續失敗次數 (格式錯誤、感知動作找不到東西都不算——
    # 那不是機器人卡住的證據)。ok=True 時這個欄位沒有意義。
    counts_as_failure: bool = True
    # 不管 ok 是什麼，都把連續失敗次數歸零 (感知動作、伺服器端剛自動恢復)。
    resets_streak: bool = False
    # 專指「伺服器剛自動重啟模擬救回卡死」——跟 resets_streak 分開是因為感知動作也會
    # resets_streak=True，但那不是 recovery，不該觸發 world_state 的「姿態可能跳動」警示。
    recovered: bool = False
    # 這個結果出來後，這一步還要不要繼續跑下一個動作 (True = 這一步到此為止)。
    stop_step: bool = True


# ---- 紀錄用 ----
class ActionRecord(BaseModel):
    name: str
    params: dict[str, Any]
    ok: bool
    message: str
    duration_s: float = 0.0
    counts_as_failure: bool = True
    resets_streak: bool = False


class StepRecord(BaseModel):
    step: int
    timestamp: float
    state_before: dict[str, Any]
    frame_path: Optional[str] = None
    overhead_frame_path: Optional[str] = None
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
    latest_plan: str = ""
    long_term_summary: str = ""

    # ---- 對應 browser_use 的 AgentHistoryList 便利方法 ----
    def is_successful(self) -> Optional[bool]:
        """任務還沒 done 就沒有答案，回傳 None (不是 False)，避免跟「明確失敗」混淆。"""
        return self.success if self.is_done else None

    def number_of_steps(self) -> int:
        return len(self.steps)

    def action_names(self) -> list[str]:
        """依執行順序列出送出過的每個動作名稱，跨所有步驟攤平。"""
        return [a.name for s in self.steps for a in s.actions]

    def final_result(self) -> str:
        return self.final_text

    def screenshot_paths(self) -> list[Optional[str]]:
        return [s.frame_path for s in self.steps]

    def errors(self) -> list[Optional[str]]:
        """LLM 呼叫本身失敗 (格式錯誤、逾時) 的訊息，依步驟列出；沒出錯的步驟是 None。"""
        return [s.error for s in self.steps]


class KnownObject(BaseModel):
    """一次成功的 locate/face/locate_overhead 結果，留給之後的步驟查，
    不用等 LLM 自己把座標/方位角抄進 memory 欄位、也不會隨著歷史截斷而消失。"""
    step: int
    source: str
    summary: str


class WorldState(BaseModel):
    """系統自己維護的結構化狀態，跟 RobotAgentOutput.memory (LLM 自己口頭寫的進度) 分開：
    這裡的內容是程式碼算出來的事實，不會被 LLM 的幻覺污染，也不受歷史截斷影響。"""
    consecutive_failures: int = 0
    max_failures: int = 3
    just_recovered: bool = False
    known_objects: dict[str, KnownObject] = {}

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "consecutive_failures": f"{self.consecutive_failures}/{self.max_failures} "
                                     "(task auto-aborts once this reaches the max)",
            "just_recovered": (
                "YES - the simulation was just auto-restarted because the control channel was stuck. "
                "Your chassis/arm pose may have jumped; re-check telemetry and the camera before continuing."
                if self.just_recovered else False
            ),
            "known_objects": {
                name: f"(step {o.step}, via {o.source}) {o.summary}" for name, o in self.known_objects.items()
            } or "(nothing located yet)",
        }
