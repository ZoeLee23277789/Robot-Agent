#!/bin/bash
# 一個指令搞定：確認 CoppeliaSim/robot_server 準備好，然後直接執行任務，不用開兩個終端機。
#
# 用法：
#   ./run_task.sh "Rotate to find any colored box, then drive up to it and stop about 30 cm away"
set -e
cd "$(dirname "$0")"

if [ -z "$1" ]; then
    echo "用法：./run_task.sh \"任務描述\""
    echo "例如：./run_task.sh \"Rotate to find any colored box, then drive up to it and stop about 30 cm away\""
    exit 1
fi

./start_all.sh
echo
echo "== 開始執行任務 =="
./run_agent.sh --task "$1"
