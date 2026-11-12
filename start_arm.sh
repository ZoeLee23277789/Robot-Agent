#!/usr/bin/env bash
# 實體 NexArm 的 server，跟模擬的那套（8765）同時跑：自己聽 8767，並把實體手臂的動作即時映到視窗版 CoppeliaSim（數位分身）。
#
# 用法：
#   ./start_arm.sh                 # armonly（桌上的 NexArm，沒有底盤），8767，數位分身到 zmq 23001
#   ./start_arm.sh --no-mirror     # 不要數位分身（視窗版 CoppeliaSim 沒開時）
#   ./start_arm.sh --with-ep       # 底盤也接上：EP 走 Wi-Fi 直連（電腦要連 RMEP-xxxxxx 熱點）+ NexArm 走 USB，一個 server 管兩個
#   RM_CONN_TYPE=rndis NEXARM_PORT=tcp://192.168.4.1:8080 ./start_arm.sh --with-ep # EP 走 USB 線（中控 micro-USB）+ 手臂走熱點
#   RM_CONN_TYPE=sta NEXARM_PORT=tcp://192.168.4.1:8080 ./start_arm.sh --with-ep   # 兩台都無線：EP 路由器模式（要先用相機掃 QR code）+ 手臂熱點
#   ARM_SERVER_PORT=8768 ./start_arm.sh
#   之後下任務：./run_agent.sh --robot-url http://127.0.0.1:8767 --task "..."
#
# 數位分身要先有一個開著 Capstone_Playground_nexarm.ttt 的視窗版 CoppeliaSim：
#   cd ~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04 && ./coppeliaSim scenes/Capstone_Playground_nexarm.ttt &
# （無視窗那個已經佔了 23000，第二個實例會自動用 23001）
cd "$(dirname "$0")"
PORT="${ARM_SERVER_PORT:-8767}"
MIRROR=(--mirror-sim "${MIRROR_SIM_PORT:-23001}")
MODE=(--mode armonly)
ARGS=()
for a in "$@"; do
    case "$a" in
        --no-mirror) MIRROR=() ;;
        --with-ep)
            MODE=(--mode real --arm nexarm --chassis-backend move)
            export RM_CONN_TYPE="${RM_CONN_TYPE:-ap}"
            if [ "$RM_CONN_TYPE" = "sta" ]; then
                # 路由器模式：EP 已用 App 掃 QR code 加入辦公室 Wi-Fi（Arriba_5G，跟電腦有線同一台路由器），
                # SDK 靠 UDP 40927 廣播找它；電腦的 Wi-Fi 可以同時連手臂熱點（NEXARM_PORT=tcp://192.168.4.1:8080）
                if OUT=$(~/rmenv/bin/python -c 'import sys; sys.path.insert(0, "Test_Robot"); import rm_connection as r; print(r.pick_real_ep())' 2>&1); then
                    echo "EP 在路由器模式：$(echo "$OUT" | tail -1)"
                else
                    echo "$OUT" | grep -E 'RuntimeError|rm_connection' | tail -1; exit 1
                fi
            elif [ "$RM_CONN_TYPE" = "rndis" ]; then
                # USB 線接智能中控的 micro-USB：EP 固定 192.168.42.2，電腦的 Wi-Fi 可以留給手臂熱點
                ping -c 1 -W 1 192.168.42.2 >/dev/null 2>&1 || { echo "連不到 EP（192.168.42.2）：USB 線要接在智能中控的 micro-USB 孔、EP 要開機；ip -4 addr 看有沒有 192.168.42.x 的介面"; exit 1; }
            else
                ping -c 1 -W 1 192.168.2.1 >/dev/null 2>&1 || { echo "連不到 EP（192.168.2.1）：電腦的 Wi-Fi 要先連到 RMEP-xxxxxx 熱點、機器人要開機（USB 線請加 RM_CONN_TYPE=rndis，路由器模式 RM_CONN_TYPE=sta）"; exit 1; }
            fi
            ;;
        *) ARGS+=("$a") ;;
    esac
done
if P=$(lsof -t -i:"$PORT" 2>/dev/null); then
    echo "8767 已經有 server（pid $P），先關掉：kill $P"; exit 1
fi
ARM_PORT="${NEXARM_PORT:-/dev/nexarm}"
case "$ARM_PORT" in
    tcp://*)
        # 手臂自己的 Wi-Fi 熱點：SSID NexArm / 密碼 hiwonder，手臂是 192.168.4.1、TCP 8080（韌體 EspNow_Ctrl.cpp）。
        # 電腦的 Wi-Fi 要先連上它：nmcli con up NexArm（第一次才用 nmcli dev wifi connect NexArm password hiwonder；
        # 設定檔已存在時那句會報「key-mgmt 找不到屬性」，改用 con up）
        HOST="${ARM_PORT#tcp://}"; HOST="${HOST%%:*}"
        ping -c 1 -W 1 "$HOST" >/dev/null 2>&1 || { echo "連不到手臂的熱點（$HOST）：電腦 Wi-Fi 要連 NexArm（密碼 hiwonder），手臂要開機"; exit 1; }
        ;;
    *)
        [ -e "$ARM_PORT" ] || [ -e /dev/ttyUSB0 ] || { echo "找不到 $ARM_PORT 或 /dev/ttyUSB0：手臂的 USB 有接上嗎？（走 Wi-Fi 就 NEXARM_PORT=tcp://192.168.4.1:8080）"; exit 1; }
        ;;
esac
echo "實體 NexArm server → http://127.0.0.1:$PORT  （手臂 $ARM_PORT；/live 可看遙測；agent 用 --robot-url http://127.0.0.1:$PORT）"
# EP 相機：2026-11-11 確認沒壞，之前「開串流逾時」是相機線沒接、或機器人開著才插線（中控只在開機時找相機）。
# 現在預設會開串流（0.2 秒就起來）；相機沒接時想省掉 3 次逾時約 20 秒，用 RM_NO_CAMERA=1 ./start_arm.sh --with-ep
export RM_NO_CAMERA="${RM_NO_CAMERA:-0}"
# 看門狗：EP 的 SDK 連線死掉（遙測停 20 秒）server 會以 exit code 3 退出，這裡自動重開；Ctrl+C 或其他退出碼就結束。
while true; do
    # 同時寫一份到 /tmp/robot_server_real.log（append），事後才能對實體每一段的里程計；終端機照樣看得到
    ~/rmenv/bin/python -u Test_Robot/robot_server.py "${MODE[@]}" --port "$PORT" --arm-port "$ARM_PORT" "${MIRROR[@]}" "${ARGS[@]}" 2>&1 | tee -a /tmp/robot_server_real.log
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 3 ] || exit "$rc"
    echo "[start_arm] EP 連線死掉，3 秒後自動重開 server（EP 要還開著、電腦還在 RMEP 熱點上）..."
    sleep 3
    # 等 EP 回來再重開：直連 192.168.2.1、USB 線 192.168.42.2；路由器模式 IP 不固定，改聽廣播
    case "$RM_CONN_TYPE" in
        rndis) EP_HOST=192.168.42.2 ;;
        sta) EP_HOST="" ;;
        *) EP_HOST=192.168.2.1 ;;
    esac
    for i in 1 2 3 4 5 6 7 8 9 10; do
        if [ -n "$EP_HOST" ]; then
            ping -c 1 -W 1 "$EP_HOST" >/dev/null 2>&1 && break
        else
            ~/rmenv/bin/python -c 'import sys; sys.path.insert(0, "Test_Robot"); import rm_connection as r; sys.exit(0 if r.find_real_ep(timeout=4) else 1)' 2>/dev/null && break
        fi
        echo "[start_arm] 還找不到 EP（${EP_HOST:-路由器模式廣播}），再等 5 秒..."; sleep 5
    done
done
