#!/usr/bin/env bash
# 辦公室實體測試（2026-11-16）：依序跑 robot_eval/office_tasks.json 的題目（簡單 5、中等 3、困難 2），走你自己的 agent 迴圈（run_agent.sh）。
# 實體場地沒有模擬器真值可以自動判分，所以每題跑完由你看現場按 y / n 判定，結果寫進 robot_eval/results/office_<時間>/results.tsv。
# 先決條件：鏡射環境已經開好（8765 鏡射 + 8767 實體）、車在原點。
#
# 用法：
#   ./run_office_tasks.sh                       跑全部 10 題
#   ./run_office_tasks.sh --difficulty easy     只跑簡單（easy / medium / hard）
#   ./run_office_tasks.sh --task o_m1           只跑一題（可以重複給）
#   ./run_office_tasks.sh --list                只列出題目
cd "$(dirname "$0")"
exec .venv/bin/python tools/run_office_tasks.py "$@"
