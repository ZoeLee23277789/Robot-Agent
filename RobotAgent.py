#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RoboMaster LLM Agent 的進入點 (對應原本的 Agent.py)。

先在機器人那一端啟動 robot_server.py，然後在這個 repo 的環境 (Python 3.11+) 執行：

    python RobotAgent.py --task "Find the red box and stop 30 cm in front of it"
    python RobotAgent.py --loop                          # 互動模式，連續下任務
    python RobotAgent.py --provider anthropic --task "..."
    python RobotAgent.py --robot-url http://100.x.y.z:8765 --task "..."   # 遠端機器人
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

from robot_agent import RobotAgent, RobotClient
from robot_agent.client import check_link
from robot_agent.llm_factory import build_llm


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM high-level controller for DJI RoboMaster EP")
    ap.add_argument("--task", help="自然語言任務；不給就進入互動模式")
    ap.add_argument("--loop", action="store_true", help="互動模式：同一個連線連續下多個任務")
    ap.add_argument("--provider", default=os.getenv("ROBOT_LLM_PROVIDER", "openai"),
                    choices=["openai", "anthropic", "google", "gemini", "robotics-er", "ollama", "mlx"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--robot-url", default=os.getenv("ROBOT_SERVER_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--robot-token", default=os.getenv("ROBOT_SERVER_TOKEN", ""))
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--run-dir", default=None, help="這次執行的紀錄資料夾，預設 runs/<timestamp>")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--confirm", dest="confirm", action="store_true", default=None,
                   help="每一步實體動作前都先問人")
    g.add_argument("--no-confirm", dest="confirm", action="store_false",
                   help="不問直接執行 (real 模式預設會問)")
    return ap.parse_args()


async def run_task(task: str, args, llm, robot: RobotClient, confirm: bool):
    agent = RobotAgent(task=task, llm=llm, robot=robot, max_steps=args.max_steps,
                       run_dir=args.run_dir, confirm_each_step=confirm)
    return await agent.run()


async def amain() -> None:
    load_dotenv()
    args = parse_args()
    llm = build_llm(args.provider, args.model)
    robot = RobotClient(args.robot_url, args.robot_token)

    health = await robot.health()
    if not health.get("ok"):
        raise SystemExit(f"❌ {health.get('message')}\n   請先在機器人那一端執行：python robot_server.py --mode sim")
    check_link(health)
    mode = health.get("mode", "?")
    # 實體機器人預設每一步都要人按 Enter；模擬器和 mock 預設直接跑
    confirm = args.confirm if args.confirm is not None else (mode == "real")
    print(f"LLM: {args.provider}/{llm.model}｜Robot: {args.robot_url} (mode={mode})｜逐步確認: {'開' if confirm else '關'}")

    if args.task and not args.loop:
        await run_task(args.task, args, llm, robot, confirm)
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
            try:
                await run_task(task, args, llm, robot, confirm)
            except KeyboardInterrupt:
                print("\n中斷，已送出停止指令")
                await robot.stop()


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n已中止")


if __name__ == "__main__":
    main()
