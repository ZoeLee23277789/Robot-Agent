#!/bin/bash
# Agent 端：固定使用 .venv，預設用 Gemini Robotics ER
cd "$(dirname "$0")"
exec .venv/bin/python RobotAgent.py --provider robotics-er "$@"
