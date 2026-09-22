#!/bin/bash
# 一次把「機器人端」準備好，取代手動打好幾個指令：
#   1. 確認 CoppeliaSim 有反應，模擬是停止狀態的話自動幫你按播放
#   2. 啟動 robot_server.py（用 hybrid backend，背景執行），已經在跑的話就重用
#   3. 加上三個真的距離感測器（第一次加完之後 server 要重開一次才讀得到，這裡會自動處理）
#
# 用法：
#   ./start_all.sh              一般啟動
#   ./start_all.sh --restart    先強制關掉舊的 robot_server.py 再重開（例如你剛改過程式碼）
#
# 做完這一步之後，另開一個終端機執行：
#   ./run_agent.sh --task "..."   或   ./run_agent.sh --loop
set -e
cd "$(dirname "$0")"

SERVER_URL="http://127.0.0.1:8765"
LOG="/tmp/robot_server.log"

start_server() {
    echo "啟動 robot_server.py（log 在 $LOG）..."
    nohup ~/rmenv/bin/python robot_server.py --mode sim --chassis-backend hybrid > "$LOG" 2>&1 &
    disown
    for i in $(seq 1 40); do
        sleep 1
        if curl -s -m 2 "$SERVER_URL/state" > /dev/null 2>&1; then
            sleep 1.5  # 等第一筆真的姿態遙測進來，不要拿預設值去校正感測器方向
            return 0
        fi
        # 連線失敗 robot_server.py 會直接結束程序（內建重試過了才會結束），
        # 程序都不在了就不用繼續等滿 40 秒。
        if ! pgrep -f "robot_server.py --mode sim" > /dev/null 2>&1; then
            break
        fi
    done
    echo "沒連上，看一下 log："
    tail -n 30 "$LOG"
    return 1
}

echo "== 1. 確認 CoppeliaSim 有反應、模擬有在跑 =="
if ! timeout 8 .venv/bin/python -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient().require('sim')
if sim.getSimulationState() == sim.simulation_stopped:
    print('模擬是停止狀態，幫你按下播放...')
    sim.startSimulation()
" 2>&1; then
    echo "連不上 CoppeliaSim（8 秒沒回應）。請先手動：開啟 CoppeliaSim → 載入你的場景，再重跑這支腳本。"
    exit 1
fi
sleep 1
echo "CoppeliaSim 正常，模擬進行中。"

echo "== 2. robot_server.py =="
if [ "$1" = "--restart" ]; then
    pkill -f "robot_server.py" 2>/dev/null || true
    sleep 1
fi
if curl -s -m 2 "$SERVER_URL/state" > /dev/null 2>&1; then
    echo "已經有 robot_server.py 在跑，直接沿用（想重開就加 --restart）。"
else
    start_server || exit 1
fi

echo "== 3. 距離感測器 =="
if curl -s -m 2 "$SERVER_URL/state" | grep -q tof_front_mm; then
    echo "感測器已經在線，跳過。"
else
    .venv/bin/python add_proximity_sensors.py
    echo "感測器剛建立，robot_server.py 只在啟動當下掃描一次，重開一次讓它讀到..."
    pkill -f "robot_server.py" 2>/dev/null || true
    sleep 1
    start_server || exit 1
    if ! curl -s -m 2 "$SERVER_URL/state" | grep -q tof_front_mm; then
        echo "重開後還是沒讀到感測器，需要再查一下，其他功能不受影響（會退回用死掉的 tof_distance_mm）。"
    fi
fi

echo
echo "== 完成，目前狀態 =="
curl -s "$SERVER_URL/state"
echo
echo
echo "接下來另開一個終端機執行： ./run_agent.sh --task \"...\"   或   ./run_agent.sh --loop"
