#!/bin/bash
# 一次把「機器人端」準備好，取代手動打好幾個指令：
#   1. 確認 CoppeliaSim 有反應，模擬是停止狀態的話自動幫你按播放
#   2. 啟動 robot_server.py（用 hybrid backend，背景執行），已經在跑的話就重用
#   3. 確認三個真的距離感測器有上線（robot_server.py 自己會建立/重建，這裡只是檢查一下）
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
# 只比對真正的啟動命令「python -u [Test_Robot/]robot_server.py」，不要用裸的 "robot_server.py"：
# pkill -f 比對的是整條命令列，裸字串會連 vim robot_server.py、tail -f /tmp/robot_server.log、
# 甚至呼叫這支腳本的 shell（命令列裡剛好提到這個檔名）都一起砍掉。實測踩過。
# (Test_Robot/)? 是為了也能砍掉搬家前從舊路徑啟動、還在跑的舊 server。
SERVER_PATTERN="python -u (Test_Robot/)?robot_server\.py"

start_server() {
    echo "啟動 robot_server.py（log 在 $LOG）..."
    # -u：關掉 stdout/stderr 的區塊緩衝。輸出導向檔案（不是終端機）時 Python 預設不會
    # 每行都立刻寫檔，要等緩衝區滿了或程序結束才會真的落地——之前 world_xy 的 log
    # 就是因為這樣，明明有在印，但 robot_server.py 還在跑的時候 grep /tmp/robot_server.log
    # 永遠是空的，看起來像 log 沒生效，其實只是還沒被沖進磁碟。
    # robot_server.py 跟它 import 的 rm_connection.py / patch_ftp.py 都在 Test_Robot/，
    # Python 會把腳本所在目錄放進 sys.path，所以從 repo 根目錄執行也找得到那兩個模組。
    nohup ~/rmenv/bin/python -u Test_Robot/robot_server.py --mode sim --chassis-backend hybrid > "$LOG" 2>&1 &
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
SIM_RESTARTED=0
if ! SIM_OUT=$(timeout 8 .venv/bin/python -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient().require('sim')
if sim.getSimulationState() == sim.simulation_stopped:
    print('模擬是停止狀態，幫你按下播放... [SIM_WAS_STOPPED]')
    sim.startSimulation()
" 2>&1); then
    if [ -n "$SIM_OUT" ]; then echo "$SIM_OUT"; fi
    echo "連不上 CoppeliaSim（8 秒沒回應）。請先手動：開啟 CoppeliaSim → 載入你的場景，再重跑這支腳本。"
    exit 1
fi
if [ -n "$SIM_OUT" ]; then echo "$SIM_OUT"; fi
if echo "$SIM_OUT" | grep -q "SIM_WAS_STOPPED"; then
    SIM_RESTARTED=1
fi
sleep 1
echo "CoppeliaSim 正常，模擬進行中。"

echo "== 2. robot_server.py =="
if [ "$SIM_RESTARTED" = "1" ] && [ "$1" != "--restart" ]; then
    # 模擬停止再播放之後，舊的 robot_server.py 那條 SDK 連線會變成「遙測照常推送、底盤/手臂
    # 指令全部沒反應」：/health 的 robot_link 看起來正常，agent 卻每個 rotation 都回報轉了 0 度
    # （實測連續三次跑分都是這樣，直到 _auto_recover 自己再停止/重播一次模擬並重連 SDK 才恢復）。
    # SDK 連線必須在模擬重播之後重新建立，所以只要是這支腳本自己按的播放，就等同 --restart。
    echo "模擬剛被重新播放，舊的 robot_server.py 對它的控制通道已經失效，強制重開..."
fi
if [ "$1" = "--restart" ] || [ "$SIM_RESTARTED" = "1" ]; then
    pkill -f "$SERVER_PATTERN" 2>/dev/null || true
    sleep 1
fi
if curl -s -m 2 "$SERVER_URL/state" > /dev/null 2>&1; then
    # HTTP 有回應不代表機器人還連著：模擬只要停止/重播過一次，server 這個 process 會活著，
    # 但它跟 RoboMaster SDK 的連線已經斷了、遙測不再更新，/state 吐的全是快取的舊值。
    # 以前這裡只看 HTTP 就沿用，結果 run_agent.sh 一啟動就被 check_link() 擋下來，
    # 連續好幾次都在沿用一個殭屍 server。改成再看 /health 的 robot_link，斷了就自動重開。
    if curl -s -m 2 "$SERVER_URL/health" | grep -Eq '"robot_link": *"lost"'; then
        echo "robot_server.py 還在，但它跟機器人的 SDK 連線已經斷了（模擬曾被停止/重播），自動重開..."
        pkill -f "$SERVER_PATTERN" 2>/dev/null || true
        sleep 1
        start_server || exit 1
    else
        echo "已經有 robot_server.py 在跑，直接沿用（想重開就加 --restart）。"
    fi
else
    start_server || exit 1
fi

echo "== 3. 距離感測器 =="
# robot_server.py 現在會自己在啟動時建立三個距離感測器（見 _create_prox_sensors()），
# 不再需要外部的 add_proximity_sensors.py + 重開一次的舊流程；這裡只是確認一下有沒有成功。
if curl -s -m 2 "$SERVER_URL/state" | grep -q tof_front_mm; then
    echo "感測器已經在線。"
else
    echo "沒讀到 tof_front_mm，看一下 log 裡『距離感測器』相關訊息："
    grep "距離感測器" "$LOG" | tail -5
fi

echo
echo "== 完成，目前狀態 =="
curl -s "$SERVER_URL/state"
echo
echo
echo "接下來另開一個終端機執行： ./run_agent.sh --task \"...\"   或   ./run_agent.sh --loop"
