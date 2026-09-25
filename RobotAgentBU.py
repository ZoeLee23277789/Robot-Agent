#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用「真的」browser_use.Agent 控制 RoboMaster EP (對照 RobotAgent.py 那個自己重寫的版本)。

跟 RobotAgent.py 用的是同一個 robot_server.py、同一份 robot_agent/perception.py，
差別只在主迴圈換成 browser_use.Agent 本身：
    RobotAgent (robot_agent/service.py，自己重寫)     browser_use.Agent (這個檔案)
    ACTION_HANDLERS + _h_* 方法                       bu_tools.tools (真的 Tools()/Registry)
    _classify_result + counts_as_failure              bu_tools._run() 回報 ActionResult(error=...)，
                                                        框架自己的 self.state.consecutive_failures 算
    _maybe_plan / _update_long_term_summary            bu_common.make_on_step_start() 這個 hook
    world_state (自己組 JSON)                          bu_common.AdapterState + RobotSession 塞進 recent_events
    _build_user_message 兩張圖分開送                    RobotSession 合成一張並排圖 (screenshot 只有單一欄位)

    python RobotAgentBU.py --provider robotics-er --task "Find the red box and stop 30 cm in front of it"
    python RobotAgentBU.py --loop
"""

import argparse
import asyncio
import os
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv

# 這支程式只在本機控制機器人，不使用 browser_use 的任何雲端/對外服務：關掉它預設會開的
# posthog 匿名遙測 (ANONYMIZED_TELEMETRY)、cloud sync (BROWSER_USE_CLOUD_SYNC，預設跟著
# 遙測一起開)，以及每次啟動都去 PyPI 查新版的檢查 (BROWSER_USE_VERSION_CHECK，見
# browser_use/utils.py)。必須在 import browser_use 之前設定，CONFIG 是讀取當下才看環境變數。
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("BROWSER_USE_CLOUD_SYNC", "false")
os.environ.setdefault("BROWSER_USE_VERSION_CHECK", "false")

from browser_use.agent.service import Agent  # noqa: E402

from robot_agent import bu_tools
from robot_agent.bu_common import AdapterState, make_on_step_start
from robot_agent.bu_session import RobotSession
from robot_agent.client import RobotClient, check_link
from robot_agent.llm_factory import build_llm

SYSTEM_PROMPT_PATH = Path(__file__).parent / "robot_agent" / "system_prompt.md"


def build_system_prompt(extend: str = None) -> str:
    prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    notes = bu_tools.load_learned_notes()
    if notes:
        prompt += ("\n<learned_notes>\nFacts you discovered about your own body in EARLIER tasks and chose to "
                   "remember. Trust them, and correct them with remember() if you find one is wrong.\n"
                   + "\n".join(f"- {n}" for n in notes) + "\n</learned_notes>\n")
    if extend:
        prompt += "\n" + extend
    return prompt


def make_agent(task: str, llm, robot: RobotClient, adapter_state: AdapterState, *,
              judge_llm=None, **kw) -> Agent:
    """所有進入點都用這個函式造 Agent，換機器人只換這裡。"""
    return Agent(
        task=task,
        llm=llm,
        browser=RobotSession(robot, adapter_state),
        tools=bu_tools.tools,
        override_system_message=build_system_prompt(),
        include_recent_events=True,      # 遙測 JSON + world_state/plan/摘要都是透過 recent_events 塞進去的
        use_vision=True,                 # 相機影像 (合成後的雙圖) 走原本的 screenshot 通道
        max_actions_per_step=3,
        use_judge=judge_llm is not None,
        judge_llm=judge_llm,
        calculate_cost=False,
        **kw,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="真的 browser_use.Agent 控制 RoboMaster EP")
    ap.add_argument("--task")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--provider", default=os.getenv("ROBOT_LLM_PROVIDER", "openai"),
                    choices=["openai", "anthropic", "google", "gemini", "robotics-er", "ollama", "mlx"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--robot-url", default=os.getenv("ROBOT_SERVER_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--robot-token", default=os.getenv("ROBOT_SERVER_TOKEN", ""))
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--planner-interval", type=int, default=5)
    ap.add_argument("--history-items", type=int, default=12, help="長期摘要每累積這麼多步觸發一次")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--confirm", dest="confirm", action="store_true", default=None)
    g.add_argument("--no-confirm", dest="confirm", action="store_false")
    return ap.parse_args()


async def amain() -> None:
    load_dotenv()
    args = parse_args()
    llm = build_llm(args.provider, args.model)
    robot = RobotClient(args.robot_url, args.robot_token)

    health = await robot.health()
    if not health.get("ok"):
        raise SystemExit(f"❌ {health.get('message')}\n   請先在機器人那一端執行：./start_all.sh（或 ./start_server.sh sim）")
    check_link(health)
    mode = health.get("mode", "?")
    confirm = args.confirm if args.confirm is not None else (mode == "real")
    adapter_state = AdapterState(max_failures=3)
    bu_tools.configure(robot, llm, adapter_state, confirm_each_action=confirm)
    print(f"LLM: {args.provider}/{llm.model}｜Robot: {args.robot_url} (mode={mode})｜逐步確認: {'開' if confirm else '關'}")

    async def run(task: str):
        agent = make_agent(task, llm, robot, adapter_state, max_failures=3)
        on_step_start = make_on_step_start(adapter_state, llm, llm,
                                           planner_interval=args.planner_interval,
                                           history_items=args.history_items)
        try:
            history = await agent.run(max_steps=args.max_steps, on_step_start=on_step_start)
        finally:
            await robot.stop()
            # browser_use.Agent 預設把截圖存到系統暫存目錄 (agent.agent_directory)，也從不會
            # 自己呼叫 save_history()——不接上這段的話，任務跑完 runs/ 底下什麼都不會有。
            # 這裡另外複製一份進 runs/<timestamp>/，習慣跟 RobotAgent.py 的舊版一致。
            try:
                run_dir = Path(f"runs/{time.strftime('%Y%m%d_%H%M%S')}")
                run_dir.mkdir(parents=True, exist_ok=True)
                agent.history.save_to_file(run_dir / "history.json")
                src_screenshots = agent.agent_directory / "screenshots"
                if src_screenshots.exists():
                    shutil.copytree(src_screenshots, run_dir / "screenshots", dirs_exist_ok=True)
                print(f"📁 紀錄存在 {run_dir}")
            except Exception as e:
                print(f"⚠️  存檔失敗（不影響任務結果）：{type(e).__name__}: {e}")
        print(f"\n{'✅ 成功' if history.is_successful() else '❌ 未成功'}｜{history.number_of_steps()} 步｜{history.final_result() or ''}")
        return history

    if args.task and not args.loop:
        await run(args.task)
        return

    print("互動模式。輸入任務後按 Enter；輸入 stop 讓機器人停車；輸入 exit 離開。")
    while True:
        task = (await asyncio.to_thread(input, "\n🗣  Task> ")).strip()
        if task.lower() in ("exit", "quit"):
            break
        if task.lower() == "stop":
            print(await robot.stop())
            continue
        if task:
            await run(task)


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n已中止")
