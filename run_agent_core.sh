#!/bin/bash
# 啟動「框架迴圈」的 agent（robot_agent/core 的 Agent，RobotAgentCore.py）。
# 另一個入口 ./run_agent.sh 是自己寫的迴圈（robot_agent/service.py），跑分用的是那一條。
#   ./run_agent_core.sh --task "..."     或    ./run_agent_core.sh --loop
cd "$(dirname "$0")"
exec .venv/bin/python RobotAgentCore.py --provider "${ROBOT_LLM_PROVIDER:-robotics-er}" "$@"
