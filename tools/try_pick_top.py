#!/usr/bin/env python
"""直接呼叫技能（不經 agent 迴圈），調抓取/放置用。
用法：.venv/bin/python tools/try_pick_top.py [approach] [pick] [place] --obj "orange stick" """
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from robot_agent import RobotClient, skills, skills_nexarm
from robot_agent.llm_factory import build_llm


async def main():
    args = sys.argv[1:]
    obj = args[args.index("--obj") + 1] if "--obj" in args else "orange stick"
    steps = [a for a in args if a in ("approach", "pick", "place", "home")] or ["pick"]
    llm = build_llm(os.environ.get("ROBOT_LLM_PROVIDER", "robotics-er"), None)
    robot = RobotClient(os.environ.get("ROBOT_URL", "http://127.0.0.1:8765"), os.environ.get("ROBOT_TOKEN"))
    for st in steps:
        if st == "approach":
            ok, msg = await skills.approach(robot, llm, "box", 0.2)
        elif st == "pick":
            ok, msg = await skills_nexarm.pick_from_top(robot, llm, obj, float(os.environ['SURF']) if os.environ.get('SURF') else None, float(os.environ['OBJ_H']) if os.environ.get('OBJ_H') else None)
        elif st == "place":
            ok, msg = await skills_nexarm.place_on_top(robot, llm, float(os.environ['PLACE_SURF']) if os.environ.get('PLACE_SURF') else None)
        else:
            ok, msg = await skills_nexarm.return_home(robot, llm)
        print("STEP %s ok=%s\n%s\n" % (st, ok, msg))
        if not ok and st != "place":
            if "home" in steps and st != "home":           # 失敗也要回原位（使用者 2026-11-14：「每次記得自己走回原位」）
                print("STEP home ok=%s\n%s\n" % (await skills_nexarm.return_home(robot, llm)))
            break

asyncio.run(main())
