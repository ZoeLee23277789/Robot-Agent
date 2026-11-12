#!/usr/bin/env bash
# 一個指令跑完 robot_eval/dataset.json 的全部任務：無視窗開 CoppeliaSim → 背景跑 run_batch.py（--restart-sim）。
# 關掉終端機也會繼續跑；模擬器當掉會自己重開、當掉那題重跑一次。
#
# 用法：
#   ./run_all.sh                                   跑全部
#   ./run_all.sh --difficulty hard                 其他參數原樣交給 run_batch.py（--task、--model、--resume ...）
#   ./run_all.sh --stop                            中止批次（會先叫機器人停下、彙整已跑完的 report.json）
#
# 看進度：瀏覽器開 http://127.0.0.1:8765/live（地圖上的綠線是規劃路徑，下方 batch results 是每題的成績）
#         或 tail -f /tmp/run_batch.log
cd "$(dirname "$0")"
PID_FILE=/tmp/run_batch.pid
LOG=/tmp/run_batch.log

# 不能用程序群組判斷批次還在不在：start_all.sh 開的 robot_server.py 會留在批次的群組裡，
# 批次結束後群組還活著（2026-11-06 實測：重跑被誤擋成「已經有一個批次在跑」）。直接確認 pid 還是 run_batch。
batch_pid() {
    [ -f "$PID_FILE" ] || return 1
    local p; p=$(cat "$PID_FILE")
    tr '\0' ' ' 2>/dev/null < "/proc/$p/cmdline" | grep -q "robot_eval.run_batch" && echo "$p"
}

if [ "$1" = "--stop" ]; then
    if P=$(batch_pid); then
        # SIGINT 而不是 SIGTERM：run_batch.py 收到 KeyboardInterrupt 才會送 /stop、彙整 report.json；
        # 只送給批次本身和它正在跑的那題，不要波及 robot_server.py
        pkill -INT -P "$P"; kill -INT "$P" && echo "已送出中止，等它彙整完（tail $LOG）"
    else
        echo "沒有在跑的批次"
    fi
    exit 0
fi

if P=$(batch_pid); then
    echo "已經有一個批次在跑（pid $P，./run_all.sh --stop 可以中止），進度：tail -f $LOG"
    exit 1
fi

if ! pgrep -x coppeliaSim > /dev/null; then
    ./start_sim_headless.sh || exit 1
else
    echo "CoppeliaSim 已經在跑，直接沿用（--restart-sim 當掉時會用無視窗模式重開）"
fi

ARGS=("$@")
if [[ " $* " != *" --resume "* && " $* " != *" --output-dir "* ]]; then
    OUT="robot_eval/results/all_$(date +%Y%m%d_%H%M%S)"
    ARGS+=(--output-dir "$OUT")
fi

setsid nohup .venv/bin/python -u -m robot_eval.run_batch --restart-sim "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
sleep 2
if ! batch_pid > /dev/null; then
    echo "批次沒有啟動成功："; tail -n 20 "$LOG"; exit 1
fi

echo
echo "批次已在背景執行，關掉這個終端機也會繼續跑。"
echo "  即時畫面：http://127.0.0.1:8765/live   （每題開始前會重開 server，斷線幾十秒是正常的）"
echo "  進度    ：tail -f $LOG"
[ -n "$OUT" ] && echo "  結果    ：$OUT/report.json（跑完才會有；中途想看就 .venv/bin/python -m robot_eval.run_batch --report-only $OUT）"
echo "  中止    ：./run_all.sh --stop"
