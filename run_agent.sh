#!/bin/bash
# 啟動 LLM agent（robot_agent/service.py 的迴圈）。用法跟 RobotAgent.py 一樣：
#   ./run_agent.sh --task "..."     或    ./run_agent.sh --loop
cd "$(dirname "$0")"
exec .venv/bin/python RobotAgent.py --provider "${ROBOT_LLM_PROVIDER:-robotics-er}" "$@"
