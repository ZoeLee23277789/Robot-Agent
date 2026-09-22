#!/usr/bin/env python
"""
RoboMaster Agent 評估 CLI。

    python -m robot_eval.run_evaluation                         # 跑 dataset 裡全部任務
    python -m robot_eval.run_evaluation --task nav_001          # 只跑一個
    python -m robot_eval.run_evaluation --task nav_001 --repeat 5   # 同一個任務重複 5 次，看成功率的變異
    python -m robot_eval.run_evaluation --rejudge runs/20260916_101500 --task nav_002   # 不動機器人，重新評一次舊紀錄

流程：CLI → (reset) → RobotAgent 執行 → telemetry 判定 + judge LLM → metrics → report.json
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from robot_agent import RobotAgent, RobotClient
from robot_agent.llm_factory import build_llm
from robot_eval.judge import judge_run, milestone_dicts
from robot_eval.metrics import build_report, create_task_metrics


async def _auto_answer(question: str) -> str:
    # 評估時不讓人介入，否則量到的是人的能力
    return "No human is available during evaluation. Decide by yourself and continue."


async def evaluate_history(task: dict, history: dict, judge_llm, out_dir: Path):
    milestones, judgement = await judge_run(task, history, judge_llm)
    metrics = create_task_metrics(task, history, milestones, judgement)
    for m in milestones:
        print(f"    {'✓' if m.passed else '✗'} {m.milestone_id} [{m.method}] {m.reasoning}")
    print(f"    ⚖️  judge: {judgement['verdict']}  {judgement.get('reasoning') or judgement.get('failure_reason') or ''}")
    (out_dir / "evaluation.json").write_text(json.dumps(
        {"task": task, "milestones": milestone_dicts(milestones), "judgement": judgement, "metrics": asdict(metrics)},
        indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


async def amain() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Checkpoint-based evaluation for the RoboMaster LLM agent")
    ap.add_argument("--dataset", default="robot_eval/dataset.json")
    ap.add_argument("--task", action="append", help="task id，可以重複給；不給就跑全部")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--provider", default=os.getenv("ROBOT_LLM_PROVIDER", "openai"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge-provider", default=os.getenv("JUDGE_PROVIDER", "openai"))
    ap.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-4o"))
    ap.add_argument("--robot-url", default=os.getenv("ROBOT_SERVER_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--robot-token", default=os.getenv("ROBOT_SERVER_TOKEN", ""))
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--output-dir", default="robot_eval/results")
    ap.add_argument("--no-pause", action="store_true", help="任務之間不停下來等人重置場景 (mock 模式會自動略過)")
    ap.add_argument("--rejudge", metavar="RUN_DIR", help="重新評一個既有的 run 資料夾，要搭配一個 --task")
    args = ap.parse_args()

    tasks = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["tasks"]
    if args.task:
        tasks = [t for t in tasks if t["id"] in args.task]
        missing = set(args.task) - {t["id"] for t in tasks}
        if missing:
            raise SystemExit(f"dataset 裡找不到：{sorted(missing)}")
    judge_llm = build_llm(args.judge_provider, args.judge_model)

    if args.rejudge:
        if len(tasks) != 1:
            raise SystemExit("--rejudge 需要剛好一個 --task")
        run_dir = Path(args.rejudge)
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        await evaluate_history(tasks[0], history, judge_llm, run_dir)
        return

    llm = build_llm(args.provider, args.model)
    robot = RobotClient(args.robot_url, args.robot_token)
    health = await robot.health()
    if not health.get("ok"):
        raise SystemExit(f"❌ {health.get('message')}")
    mode = health.get("mode")

    session = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for task in tasks:
        for rep in range(1, args.repeat + 1):
            name = task["id"] + (f"_r{rep}" if args.repeat > 1 else "")
            print(f"\n{'=' * 70}\n▶ {name} [{task.get('difficulty')}/{task.get('category')}] {task['description']}")
            print("   reset:", (await robot.reset()).get("message"))
            if mode != "mock" and not args.no_pause:
                await asyncio.to_thread(input, "   請把機器人和場景擺回起始狀態 (模擬器就按重置)，好了按 Enter ...")

            run_dir = session / name
            agent = RobotAgent(task=task["description"], llm=llm, robot=robot, max_steps=args.max_steps,
                               run_dir=str(run_dir), confirm_each_step=False, ask_human=_auto_answer)
            try:
                result = await agent.run()
                history = json.loads(result.model_dump_json())
            except Exception as e:
                print(f"   ✗ 執行失敗：{e}")
                await robot.stop()
                history = {"steps": [], "run_dir": str(run_dir), "final_text": f"execution failed: {e}"}
            all_metrics.append(await evaluate_history(task, history, judge_llm, run_dir))

    report = build_report(all_metrics)
    report["config"] = {"agent": f"{args.provider}/{llm.model}", "judge": f"{args.judge_provider}/{judge_llm.model}",
                        "robot_mode": mode, "max_steps": args.max_steps, "repeat": args.repeat}
    (session / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    o = report["overall"]
    print(f"\n{'=' * 70}\n📊 {o['tasks']} runs｜成功率 {o['success_rate']:.0%}｜milestone 平均完成 {o['avg_milestone_completion']:.0%}"
          f"｜平均 {o['avg_steps']:.1f} 步 / {o['avg_duration_s']:.0f}s")
    for k, v in report["by_difficulty"].items():
        print(f"   {k:<8} {v['success_rate']:.0%} ({v['tasks']} runs)")
    if report["agent_overclaimed_success"]:
        print(f"   ⚠️  Agent 自評成功但實際失敗：{report['agent_overclaimed_success']}")
    print(f"   報告：{session / 'report.json'}")


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n已中止")
