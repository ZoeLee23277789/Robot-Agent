#!/usr/bin/env bash
# 無視窗啟動 CoppeliaSim 並載入場景（跑分批次建議用這個：視窗版 2026-10-22 一天內崩潰四次，而且只跑 0.55 倍即時）。
# 用法：./start_sim_headless.sh            # 背景啟動，log 在 /tmp/coppeliasim_headless.log
#       ./start_sim_headless.sh --stop     # 關掉
# 注意：-h 模式在 stdin 收到 EOF 時會直接退出，所以用一條永不結束的管線把 stdin 撐著。
SIM_DIR="${COPPELIASIM_DIR:-$HOME/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04}"
SCENE="${COPPELIASIM_SCENE:-scenes/Capstone_Playground.ttt}"
LOG=/tmp/coppeliasim_headless.log
# 只管無視窗（-h）那一個實例：視窗版 CoppeliaSim（實體 NexArm 的數位分身、23001）是另一個實例，不能一起殺
# （2026-11-09：pkill -x coppeliaSim 把視窗版也殺了，之後啟動又撞到還沒退乾淨的程序）
HEADLESS_PATTERN="coppeliaSim -h "
if [ "$1" = "--stop" ]; then
    if pkill -f "$HEADLESS_PATTERN" 2>/dev/null; then
        for i in $(seq 1 30); do pgrep -f "$HEADLESS_PATTERN" >/dev/null || break; sleep 0.5; done
        echo "無視窗 coppeliaSim 已關閉"
    else
        echo "沒有在跑的無視窗 coppeliaSim"
    fi
    pkill -f "sleep infinity" 2>/dev/null || true
    exit 0
fi
if pgrep -f "$HEADLESS_PATTERN" >/dev/null; then
    echo "已經有無視窗 coppeliaSim 在跑（先 ./start_sim_headless.sh --stop）"; exit 1
fi
if (echo > /dev/tcp/127.0.0.1/23000) 2>/dev/null; then
    echo "23000 埠已被佔用（視窗版 CoppeliaSim 先開的話會佔走 23000）：先關掉視窗版，先開無視窗的、再開視窗版"; exit 1
fi
cd "$SIM_DIR" || { echo "找不到 $SIM_DIR"; exit 1; }
nohup bash -c "sleep infinity | ./coppeliaSim -h $SCENE" > "$LOG" 2>&1 &
for i in $(seq 1 40); do
    (echo > /dev/tcp/127.0.0.1/23000) 2>/dev/null && { echo "CoppeliaSim 無視窗模式就緒（zmq 23000），接著跑 ./start_all.sh"; exit 0; }
    sleep 2
done
echo "等了 80 秒 zmq 埠還沒開，看 $LOG"; exit 1
