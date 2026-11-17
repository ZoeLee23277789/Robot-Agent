#!/usr/bin/env bash
# 陌生環境模式（2026-11-12）：實體 EP 自己當眼睛，地圖從零開始由前向 ToF + 里程計邊走邊建，
# 視窗版 CoppeliaSim 載「空場景」當機器人自己建出來的模擬環境（灰方塊 = 掃到的障礙）。
# 跟 run_mirror.sh 互不干擾：鏡射那條路（模擬當眼睛、實體照抄）一個字都沒動，差別只在這裡用 --explore 開 8767。
#
# 用法：
#   ./run_explore.sh --check                 把環境開好：視窗版（空場景）、sim_twin --real、8767 --explore；不跑 agent
#   ./run_explore.sh --task "scan around and report what is near you"
#   ./run_explore.sh --no-gui --check
#   ./run_explore.sh --no-arm --check        手臂沒開（只建圖、不夾東西）：8767 不接 NexArm
#   ./run_explore.sh --bounds -2,2,-2,1 ...   探索範圍（左邊只到 1 m：那邊有小門）；車被搬過位置就 --stop 再重跑（里程計不知道）
#   ./run_explore.sh --stop                  關掉 8767、sim_twin（視窗版留著）
# 手動下指令（--check 之後）：
#   curl -s -X POST http://127.0.0.1:8767/action -H 'Content-Type: application/json' -d '{"name":"scan_around","params":{}}'
#   curl -s -X POST http://127.0.0.1:8767/action -H 'Content-Type: application/json' -d '{"name":"explore","params":{"steps":2}}'
#   curl -s -X POST http://127.0.0.1:8767/action -H 'Content-Type: application/json' -d '{"name":"return_to_origin","params":{}}'
#   地圖：http://127.0.0.1:8767/live（左上 map）、/map.png、/map.json
cd "$(dirname "$0")"
SIM_DIR="${COPPELIASIM_DIR:-$HOME/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04}"
SCENE="scenes/Capstone_Explore_nexarm.ttt"
export RM_CONN_TYPE=sta
export NEXARM_PORT=tcp://192.168.4.1:8080
export ROBOT_TOF_OFFSET_M="${ROBOT_TOF_OFFSET_M:-0.16}"
# 探索範圍 xmin,xmax,ymin,ymax（公尺，原點 = 開 server 時車的位置、+x = 車頭、+y = 車的左邊；外面當牆）。--bounds 可改
export ROBOT_EXPLORE_BOUNDS="${ROBOT_EXPLORE_BOUNDS:--2,2,-2,1}"
REAL_URL=http://127.0.0.1:8767
GUI=1; CHECK=0; NOARM=0; TASK=""; ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-gui) GUI=0 ;;
        --no-arm) NOARM=1 ;;
        --check) CHECK=1 ;;
        --task) TASK="$2"; shift ;;
        --bounds) export ROBOT_EXPLORE_BOUNDS="$2"; shift ;;
        --ep-ap) export RM_CONN_TYPE=ap ;;
        --ep-usb) export RM_CONN_TYPE=rndis ;;
        --arm-usb) export NEXARM_PORT=/dev/nexarm ;;
        --stop)
            pkill -f "start_arm.sh" 2>/dev/null; pkill -f "robot_server.py --mode real" 2>/dev/null && echo "8767 已關"
            pkill -f "tools/sim_twin.py" 2>/dev/null && echo "sim_twin 已關"
            echo "視窗版留著（要關就直接關視窗）"; exit 0 ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done
port_open() { (echo > /dev/tcp/127.0.0.1/"$1") 2>/dev/null; }
health_ok() { curl -s -m 3 "$1/health" 2>/dev/null | grep -q '"ok": *true'; }
[ -f "$SIM_DIR/$SCENE" ] || { echo "找不到空場景 $SIM_DIR/$SCENE"; exit 1; }

if [ "$GUI" = 1 ]; then
    echo "== 1. 視窗版 CoppeliaSim（23001，空場景 $SCENE，保持停止、不要存檔）=="
    # 鏡射用的 sim_twin（來源 23000）要先關：不然換完場景它馬上把無視窗那台的位姿（可能是掉出世界的）蓋到新場景的機器人上
    pkill -f "tools/sim_twin.py" 2>/dev/null && sleep 1
    if port_open 23001; then
        NAME=$(timeout 10 ~/rmenv/bin/python -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient(port=23001).require('sim'); print(sim.getStringParam(sim.stringparam_scene_name))" 2>/dev/null)
        case "$NAME" in
            *Explore*) echo "已在跑，場景 $NAME" ;;
            *) echo "視窗版開著別的場景（$NAME），換成空場景..."
               timeout 60 ~/rmenv/bin/python -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient(port=23001).require('sim')
if sim.getSimulationState() != 0: sim.stopSimulation()
sim.loadScene('$SIM_DIR/$SCENE'); print('已載入', sim.getStringParam(sim.stringparam_scene_name))" || { echo "換場景失敗，請手動在視窗裡開 $SCENE"; exit 1; } ;;
        esac
    else
        (cd "$SIM_DIR" && nohup ./coppeliaSim.sh "$SCENE" > /tmp/coppeliasim_gui.log 2>&1 < /dev/null &)
        for i in $(seq 1 30); do port_open 23001 && break; sleep 2; done
        port_open 23001 && echo "視窗版就緒" || echo "視窗版 60 秒內沒開起來（看 /tmp/coppeliasim_gui.log），先繼續"
    fi
fi

echo "== 2. 網路（EP：$RM_CONN_TYPE，手臂：$([ "$NOARM" = 1 ] && echo 不接 || echo "$NEXARM_PORT")）=="
[ "$NOARM" = 1 ] && NEXARM_PORT=none
case "$NEXARM_PORT" in
    none) ;;
    tcp://*)
        HOST="${NEXARM_PORT#tcp://}"; HOST="${HOST%%:*}"
        if ! ping -c 1 -W 1 "$HOST" >/dev/null 2>&1; then
            echo "手臂熱點 ping 不到，電腦 Wi-Fi 切到 NexArm..."; nmcli con up NexArm >/dev/null 2>&1; sleep 2
            ping -c 1 -W 1 "$HOST" >/dev/null 2>&1 || { echo "還是連不到手臂（$HOST）：手臂有開機嗎？"; exit 1; }
        fi
        echo "手臂熱點 OK（$HOST）" ;;
    *) [ -e "$NEXARM_PORT" ] || [ -e /dev/ttyUSB0 ] || { echo "找不到手臂的 USB（$NEXARM_PORT）"; exit 1; }; echo "手臂 USB OK" ;;
esac

echo "== 3. 實體 server 8767（--explore）=="
if health_ok "$REAL_URL" && curl -s -m 3 "$REAL_URL/map.json" | grep -q '"cell_m"'; then
    echo "已在跑而且是 explore 模式，沿用（地圖不重來；要重來請 --stop 再跑）"
    # 範圍套到正在跑的 server（不動車）；舊版 server 沒這個動作就提示重開
    B="[$(echo "$ROBOT_EXPLORE_BOUNDS" | tr -d ' ')]"
    R=$(curl -s -m 5 -X POST "$REAL_URL/action" -H 'Content-Type: application/json' -d "{\"name\":\"set_bounds\",\"params\":{\"bounds\":$B}}")
    echo "$R" | grep -q '"ok": *true' && echo "探索範圍已套用：$ROBOT_EXPLORE_BOUNDS" || echo "!! 範圍沒套上（$(echo "$R" | cut -c1-80)）：要用新範圍請 ./run_explore.sh --stop 再重跑"
else
    if port_open 8767; then echo "8767 有東西但不是 explore 模式，關掉重開"; pkill -f "start_arm.sh" 2>/dev/null; pkill -f "robot_server.py --mode real" 2>/dev/null; sleep 2; fi
    if [ "$RM_CONN_TYPE" = sta ]; then
        EP=$(~/rmenv/bin/python -c 'import sys; sys.path.insert(0, "Test_Robot"); import rm_connection as r; print(r.pick_real_ep(timeout=6))' 2>/dev/null | tail -1)
        [ -n "$EP" ] || { echo "網段上聽不到 EP 的廣播：EP 開機了嗎？中控開關在「路由器」那側嗎？"; exit 1; }
        echo "EP OK（$EP）"
    fi
    echo "   把實體車放在出發點、車頭朝你要當作 +x 的方向：連線那一刻就是地圖原點（0, 0）朝向 0°。"
    if [ "$NOARM" = 1 ]; then
        setsid nohup env -u ROBOT_MIRROR_REAL -u ROBOT_MIRROR_ACTIONS ~/rmenv/bin/python -u Test_Robot/robot_server.py --mode real --arm sdk --chassis-backend move --port 8767 --explore > /tmp/start_arm.log 2>&1 < /dev/null &
    else
        setsid nohup env -u ROBOT_MIRROR_REAL -u ROBOT_MIRROR_ACTIONS ./start_arm.sh --with-ep --explore > /tmp/start_arm.log 2>&1 < /dev/null &
    fi
    for i in $(seq 1 60); do health_ok "$REAL_URL" && break; sleep 2; done
    health_ok "$REAL_URL" || { echo "8767 120 秒內沒就緒，log："; tail -n 15 /tmp/start_arm.log; exit 1; }
    echo "就緒（背景跑，log /tmp/start_arm.log）"
fi

if [ "$GUI" = 1 ]; then
    echo "== 4. sim_twin --real（里程計位姿 + ToF 地圖畫到視窗版）=="
    if pgrep -f "tools/sim_twin.py --real" >/dev/null; then echo "已在跑"; else
        nohup ~/rmenv/bin/python tools/sim_twin.py --real "$REAL_URL" > /tmp/sim_twin.log 2>&1 < /dev/null & disown
        echo "已啟動（log /tmp/sim_twin.log）"
    fi
fi
echo "   即時畫面：$REAL_URL/live（map = 自己建的占據格：灰 = 未知、白 = 掃過沒東西、橘 = 障礙）"
if [ "$CHECK" = 1 ] || [ -z "$TASK" ]; then echo; echo "== 環境都開好了（沒給 --task 就不跑 agent）=="; exit 0; fi
echo "== 5. agent =="
exec ./run_agent.sh --robot-url "$REAL_URL" --task "$TASK" "${ARGS[@]}"
