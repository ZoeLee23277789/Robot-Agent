#!/usr/bin/env bash
# 辦公室實體（鏡射）環境一次開好（2026-11-16）：Wi-Fi → 無視窗模擬器載掃描場景 → 實體 server 8767 → 鏡射的模擬 server 8765。
# 執行前：車放在原點（左前輪對白點）、EP 與手臂開機。8767 啟動那一刻車的位置 = 原點。
#   ./start_office.sh           開好（已經開著的 8765/8767 會重開）
#   ./start_office.sh --stop    關掉 8765、8767（模擬器留著）
# 手臂原位 = 手臂開機時自己擺的姿勢（reach 152、高 98、俯仰 -40°）：相機在上臂，這個姿勢看得到前方地面與箱子；
# 舊的 160/136/-26 在相機重新固定後只看得到天花板（11-16 實拍比對）。
cd "$(dirname "$0")"
SIM_DIR="${COPPELIASIM_DIR:-$HOME/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04}"
SCENE="${OFFICE_SCENE:-scenes/Scan_2026_11_12.ttt}"
export NEXARM_HOME="${NEXARM_HOME:-152,0,98,-40}" NEXARM_Z_MIN=-110 NEXARM_CLAW_OPEN=-90
port_pid() { ss -ltnp 2>/dev/null | grep ":$1 " | sed 's/.*pid=\([0-9]*\).*/\1/' | head -1; }
stop_port() { local p; p=$(port_pid "$1"); [ -n "$p" ] && kill "$p" && echo "關掉 $1 (pid $p)"; for i in $(seq 1 30); do [ -z "$(port_pid "$1")" ] && break; sleep 1; done
    # 模擬器關閉偶爾拖很久：埠沒放掉就開新的，新的會改佔 23001，被誤認成視窗版（11-17）
    p=$(port_pid "$1"); [ -n "$p" ] && kill -9 "$p" && sleep 2; }
stop_port 8765; stop_port 8767
[ "$1" = "--stop" ] && exit 0

echo "== 1. 網路 =="
ping -c1 -W1 192.168.4.1 >/dev/null 2>&1 || { nmcli con up NexArm >/dev/null 2>&1; sleep 3; }
ping -c1 -W2 192.168.4.1 >/dev/null 2>&1 || { echo "連不到手臂熱點：手臂開機了嗎？"; exit 1; }
echo "手臂熱點 OK"

echo "== 2. 無視窗模擬器（$SCENE）=="
NAME=$(timeout 10 ~/rmenv/bin/python -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient(port=23000).require('sim'); print(sim.getStringParam(sim.stringparam_scene_name))" 2>/dev/null)
if [ "$NAME" != "$(basename "$SCENE")" ]; then
    stop_port 23000
    (cd "$SIM_DIR" && nohup bash -c "sleep infinity | ./coppeliaSim -h $SCENE" > /tmp/coppeliasim_headless.log 2>&1 &)
    for i in $(seq 1 40); do (echo > /dev/tcp/127.0.0.1/23000) 2>/dev/null && break; sleep 2; done
    echo "已載入 $SCENE"
else
    timeout 20 ~/rmenv/bin/python -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
sim = RemoteAPIClient(port=23000).require('sim')
if sim.getSimulationState() != 0: sim.stopSimulation(); time.sleep(3)" 2>/dev/null
    echo "沿用（模擬重置回起點）"
fi

# 2026-11-17（使用者：「我每次都要有開這個」）：視窗版沒開就自己開、載同一個場景。無視窗那台已經佔住 23000，視窗版會拿到 23001。
if ! (echo > /dev/tcp/127.0.0.1/23001) 2>/dev/null; then
    echo "== 2b. 視窗版 CoppeliaSim（23001，只看不跑：保持停止、不要存檔）=="
    (cd "$SIM_DIR" && nohup ./coppeliaSim.sh "$SCENE" > /tmp/coppeliasim_gui.log 2>&1 < /dev/null &)
    for i in $(seq 1 30); do (echo > /dev/tcp/127.0.0.1/23001) 2>/dev/null && break; sleep 2; done
    (echo > /dev/tcp/127.0.0.1/23001) 2>/dev/null && echo "視窗版就緒" || echo "視窗版 60 秒內沒開起來（看 /tmp/coppeliasim_gui.log），先繼續，開起來後會自動同步"
fi

# 視窗版不會跟著重載場景：前幾輪實體 ToF 放的橘色方塊還留著，開新一輪前先掃掉
(echo > /dev/tcp/127.0.0.1/23001) 2>/dev/null && timeout 20 ~/rmenv/bin/python -c "
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
sim = RemoteAPIClient(port=23001).require('sim')
hs = [o for o in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0) if sim.getObjectAlias(o).startswith('real_obstacle_')]
if hs: sim.removeObjects(hs); print('視窗版清掉 %d 個舊障礙方塊' % len(hs))" 2>/dev/null

echo "== 3. 實體 server 8767 =="
# 視窗版 CoppeliaSim（23001）有開的話：手臂關節同步到視窗版（--mirror-sim），底盤位姿由 sim_twin 複製（第 5 步）
# 2026-11-17：不管視窗版現在有沒有開都接上（--mirror-sim 與 sim_twin 會自己等、自己重連），視窗版晚開、重開都會跟上
GUI=0; (echo > /dev/tcp/127.0.0.1/23001) 2>/dev/null && GUI=1
MS=(--mirror-sim 23001)
nohup env RM_CONN_TYPE=sta ROBOT_TOF_OFFSET_M=0.16 ~/rmenv/bin/python -u Test_Robot/robot_server.py --mode real --arm nexarm \
    --chassis-backend move --port 8767 --arm-port tcp://192.168.4.1:8080 "${MS[@]}" > /tmp/start_arm.log 2>&1 &
for i in $(seq 1 45); do curl -s -m 2 http://127.0.0.1:8767/health 2>/dev/null | grep -q '"ok": *true' && break; sleep 2; done
curl -s -m 3 http://127.0.0.1:8767/health | grep -q '"ok": *true' || { echo "8767 沒起來："; tail -n 15 /tmp/start_arm.log; exit 1; }
echo "就緒"

echo "== 4. 模擬 server 8765（鏡射）=="
ROBOT_MIRROR_REAL=http://127.0.0.1:8767 ./start_all.sh --restart > /tmp/start_all_office.log 2>&1 || { tail -n 15 /tmp/start_all_office.log; exit 1; }
curl -s -m 30 -X POST http://127.0.0.1:8765/action -H 'Content-Type: application/json' -d '{"name":"recenter_arm","params":{}}' > /dev/null
curl -s -m 5 http://127.0.0.1:8765/state | ~/rmenv/bin/python -c "
import sys, json; s = json.load(sys.stdin)
print('模擬位置 %s｜實體位置 %s｜實體前方 %s mm｜實體電量 %s%%｜手臂 %s' % (s.get('world_xy'), s.get('real_world_xy'), s.get('real_tof_front_mm'), s.get('real_battery_percent'), {k: (s.get('arm_pose') or {}).get(k) for k in ('reach_mm', 'z_mm', 'pitch_deg')}))
sys.exit(0 if 'real_tof_front_mm' in s else 1)" || { echo "!! 鏡射沒接上"; exit 1; }
echo "== 5. 視窗版同步（sim_twin：底盤；手臂由 8767 同步）=="
for p in $(pgrep -f "tools/sim_twin.py"); do kill "$p" 2>/dev/null; done      # 舊的可能卡住或連著已關掉的視窗：一律換新的
nohup ~/rmenv/bin/python tools/sim_twin.py > /tmp/sim_twin.log 2>&1 < /dev/null & disown
if [ "$GUI" = 1 ]; then echo "已啟動（視窗版保持「停止」，不要按播放）"
else echo "視窗版 CoppeliaSim 還沒開：之後開起來、載入 $(basename "$SCENE") 就會自動同步，不用重跑這支腳本"; fi
echo "== 好了：./run_office_tasks.sh ；即時畫面 http://127.0.0.1:8765/live =="
