#!/usr/bin/env bash
# 一個指令把「鏡射批次」要的東西全部開起來，然後跑批次：
#   無視窗 CoppeliaSim（23000，agent 的眼睛）→ 視窗版 CoppeliaSim（23001，數位分身，只看不跑）→ sim_twin（底盤位姿複製到視窗版）
#   → 電腦 Wi-Fi 連手臂熱點 → 實體 server 8767（EP 路由器模式 + NexArm 熱點，背景跑、自動重開）→ run_batch（前景，Ctrl+C 可中止）
# 已經開著的東西會直接沿用，不會重開。
#
# 用法：
#   ./run_mirror.sh --task r01                  跑一題（後面的參數原樣交給 run_batch.py：--difficulty easy、--resume DIR、--model ...）
#   ./run_mirror.sh --difficulty easy
#   ./run_mirror.sh --check                     只把環境開好、檢查一遍，不跑批次
#   ./run_mirror.sh --no-gui --task r01         不開視窗版
#   ./run_mirror.sh --ep-ap --arm-usb --task r01 EP 直連熱點 + 手臂 USB（預設是 EP 路由器模式 + 手臂熱點）
#   ./run_mirror.sh --stop                      關掉 8767、sim_twin、視窗版（無視窗模擬器留著；批次本身用 Ctrl+C）
#
# 預設：EP 走辦公室 Wi-Fi（RM_CONN_TYPE=sta）、手臂走熱點（NEXARM_PORT=tcp://192.168.4.1:8080）、場景 Capstone_Playground_nexarm.ttt。
cd "$(dirname "$0")"
SIM_DIR="${COPPELIASIM_DIR:-$HOME/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04}"
export COPPELIASIM_SCENE="${COPPELIASIM_SCENE:-scenes/Capstone_Playground_nexarm.ttt}"
# 連線方式不看 shell 繼承的環境變數（實測使用者 shell 裡留著 RM_CONN_TYPE=ap，害 start_arm 跑去連直連熱點）：
# 預設 EP 路由器模式 + 手臂熱點；要改用 --ep-ap / --ep-usb / --arm-usb
export RM_CONN_TYPE=sta
export NEXARM_PORT=tcp://192.168.4.1:8080
export ROBOT_MIRROR_REAL="${ROBOT_MIRROR_REAL:-http://127.0.0.1:8767}"
REAL_URL="$ROBOT_MIRROR_REAL"
GUI=1; CHECK=0; ARGS=()
for a in "$@"; do
    case "$a" in
        --no-gui) GUI=0 ;;
        --check) CHECK=1 ;;
        --ep-ap) export RM_CONN_TYPE=ap ;;        # EP 直連熱點（電腦 Wi-Fi 要連 RMEP，手臂就只能 USB）
        --ep-usb) export RM_CONN_TYPE=rndis ;;    # EP 走 USB 線（192.168.42.2）
        --arm-usb) export NEXARM_PORT=/dev/nexarm ;;
        --stop)
            pkill -f "robot_eval.run_batch" 2>/dev/null && echo "批次已送出中止"
            pkill -f "start_arm.sh" 2>/dev/null; pkill -f "robot_server.py --mode real" 2>/dev/null && echo "8767 已關"
            pkill -f "tools/sim_twin.py" 2>/dev/null && echo "sim_twin 已關"
            pkill -f "coppeliaSim.sh scenes/" 2>/dev/null; pgrep -f "coppeliaSim $SIM_DIR" | xargs -r kill 2>/dev/null && echo "視窗版 CoppeliaSim 已關"
            echo "無視窗模擬器留著（要關：./start_sim_headless.sh --stop）"; exit 0 ;;
        *) ARGS+=("$a") ;;
    esac
done
port_open() { (echo > /dev/tcp/127.0.0.1/"$1") 2>/dev/null; }
health_ok() { curl -s -m 3 "$1/health" 2>/dev/null | grep -q '"ok": *true'; }

echo "== 1. 無視窗 CoppeliaSim（23000）=="
if pgrep -f "coppeliaSim -h " >/dev/null; then
    NAME=$(timeout 10 .venv/bin/python -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient(port=23000).require('sim'); print(sim.getStringParam(sim.stringparam_scene_name))" 2>/dev/null)
    echo "已在跑，場景：${NAME:-?}"
    case "$NAME" in *nexarm*) ;; *) echo "!! 載的不是 NexArm 場景（要 Capstone_Playground_nexarm.ttt）。先 ./start_sim_headless.sh --stop 再重跑這支"; exit 1 ;; esac
else
    ./start_sim_headless.sh || exit 1
fi

if [ "$GUI" = 1 ]; then
    echo "== 2. 視窗版 CoppeliaSim（23001，數位分身，保持停止、不要存檔）=="
    if port_open 23001; then
        echo "已在跑"
    else
        (cd "$SIM_DIR" && nohup ./coppeliaSim.sh "$COPPELIASIM_SCENE" > /tmp/coppeliasim_gui.log 2>&1 < /dev/null &)
        for i in $(seq 1 30); do port_open 23001 && break; sleep 2; done
        port_open 23001 && echo "視窗版就緒" || echo "視窗版 60 秒內沒開起來（看 /tmp/coppeliasim_gui.log），先繼續"
    fi
    echo "== 3. sim_twin（底盤位姿複製到視窗版）=="
    if pgrep -f "tools/sim_twin.py" >/dev/null; then echo "已在跑"; else
        nohup ~/rmenv/bin/python tools/sim_twin.py > /tmp/sim_twin.log 2>&1 < /dev/null & disown; echo "已啟動（log /tmp/sim_twin.log）"
    fi
fi

echo "== 4. 網路（EP：$RM_CONN_TYPE，手臂：$NEXARM_PORT）=="
case "$NEXARM_PORT" in
    tcp://*)
        HOST="${NEXARM_PORT#tcp://}"; HOST="${HOST%%:*}"
        if ! ping -c 1 -W 1 "$HOST" >/dev/null 2>&1; then
            echo "手臂熱點 ping 不到，電腦 Wi-Fi 切到 NexArm..."; nmcli con up NexArm >/dev/null 2>&1; sleep 2
            ping -c 1 -W 1 "$HOST" >/dev/null 2>&1 || { echo "還是連不到手臂（$HOST）：手臂有開機嗎？nmcli dev wifi list 看得到 NexArm 嗎？"; exit 1; }
        fi
        echo "手臂熱點 OK（$HOST）" ;;
    *) [ -e "$NEXARM_PORT" ] || [ -e /dev/ttyUSB0 ] || { echo "找不到手臂的 USB（$NEXARM_PORT）"; exit 1; }; echo "手臂 USB OK" ;;
esac

echo "== 5. 實體 server 8767 =="
if health_ok "$REAL_URL"; then
    echo "已在跑而且連線正常（EP 跟手臂都接著），沿用"
else
    if port_open 8767; then echo "8767 有東西但不健康，關掉重開"; pkill -f "start_arm.sh" 2>/dev/null; pkill -f "robot_server.py --mode real" 2>/dev/null; sleep 2; fi
    if [ "$RM_CONN_TYPE" = sta ]; then
        # 只在要開 server 時才聽廣播：server 開著時 SDK 自己佔著 UDP 40927/45678，這裡會聽不到（不代表 EP 不在）
        EP=$(~/rmenv/bin/python -c 'import sys; sys.path.insert(0, "Test_Robot"); import rm_connection as r; print(r.pick_real_ep(timeout=6))' 2>/dev/null | tail -1)
        [ -n "$EP" ] || { echo "網段上聽不到 EP 的廣播：EP 開機了嗎？中控的開關在「路由器」那側嗎？（它記得 Arriba_5G，開機會自己連）"; exit 1; }
        echo "EP OK（$EP）"
    fi
    # ROBOT_MIRROR_REAL 是給模擬 server 用的，不能傳給實體 server：實測 8767 繼承後會把每個動作再鏡射給自己（409 busy、每段多等 6 秒）
    setsid nohup env -u ROBOT_MIRROR_REAL -u ROBOT_MIRROR_ACTIONS ./start_arm.sh --with-ep > /tmp/start_arm.log 2>&1 < /dev/null &
    for i in $(seq 1 60); do health_ok "$REAL_URL" && break; sleep 2; done
    health_ok "$REAL_URL" || { echo "8767 120 秒內沒就緒，log："; tail -n 15 /tmp/start_arm.log; exit 1; }
    echo "就緒（背景跑，log /tmp/start_arm.log；EP 斷線會自動重開）"
fi
echo "   實體畫面：http://127.0.0.1:8767/live   模擬畫面：http://127.0.0.1:8765/live"

if [ "$CHECK" = 1 ]; then echo; echo "== 環境都開好了（--check 不跑批次）=="; exit 0; fi
echo "== 6. 批次（模擬 8765 每題由 run_batch 自己重開並掛上鏡射；Ctrl+C 中止、--resume <dir> 續跑）=="
echo "   把實體車放回起點、周圍清空。"
exec .venv/bin/python -m robot_eval.run_batch --mirror-real "$REAL_URL" --restart-sim "${ARGS[@]}"
