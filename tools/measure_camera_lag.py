#!/usr/bin/env python
"""
量相機畫面的延遲：機器人轉完之後，畫面還要多久才「跟上」。

做法：命令機器人轉 30 度，動作一結束就開始連續抓原始畫面 (跳過 server 的等待機制)，
比較相鄰兩張的差異。畫面還在變，代表收到的仍是轉動過程中的舊畫面；不再變的那一刻就是延遲。
    .venv/bin/python tools/measure_camera_lag.py
"""
import asyncio
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot_agent.client import RobotClient, check_link  # noqa: E402


def decode(jpg):
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.int16)


async def main():
    robot = RobotClient(os.getenv("ROBOT_SERVER_URL", "http://127.0.0.1:8765"), os.getenv("ROBOT_SERVER_TOKEN", ""))
    health = await robot.health()
    if not health.get("ok"):
        sys.exit("❌ 連不上 robot_server：" + str(health.get("message")))
    check_link(health)
    first = await robot.frame(raw_mode=True)
    if first is None:
        sys.exit("❌ server 拿不到相機畫面。通常代表模擬沒有在播放，或 server 是模擬重啟之前開的。請重開 server。")
    before = decode(first)
    res = await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": 30})
    print(res.get("message"))
    t0, prev, last_change, rows = time.time(), None, 0.0, []
    while time.time() - t0 < 6.0:
        img = decode(await robot.frame(raw_mode=True))
        t = time.time() - t0
        d_prev = float(np.abs(img - prev).mean()) if prev is not None else 0.0
        d_before = float(np.abs(img - before).mean())
        rows.append((t, d_prev, d_before))
        if d_prev > 1.5:
            last_change = t
        prev = img
        await asyncio.sleep(0.1)
    for t, a, b in rows:
        print(f"  t={t:4.1f}s  與上一張差異={a:6.2f}  與轉動前差異={b:6.2f}  {'<- 畫面還在變' if a > 1.5 else ''}")
    await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": -30})
    stale = [t for t, _, b in rows if b < 1.5]
    print()
    if stale:
        print(f"⚠️  動作結束後 {max(stale):.1f} 秒內，抓到的仍然是轉動前的舊畫面。")
    print(f"📷 畫面在動作結束後約 {last_change:.1f} 秒才穩定下來。這就是相機延遲。")
    print("   server 預設會等 8 張新影格。如果延遲超過 1.5 秒，啟動 server 前設 ROBOT_FRESH_FRAMES=12 之類更大的值。")


if __name__ == "__main__":
    asyncio.run(main())
