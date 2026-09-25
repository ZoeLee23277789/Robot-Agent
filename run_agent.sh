#!/bin/bash
# Agent 端：固定使用 .venv，預設用 Gemini Robotics ER。
# 已切換成真的 browser_use.Agent (RobotAgentBU.py)；舊的自己重寫版本 (RobotAgent.py) 還在，
# 想跑舊版對照就直接 .venv/bin/python RobotAgent.py ...。
cd "$(dirname "$0")"
exec .venv/bin/python RobotAgentBU.py --provider robotics-er "$@"
