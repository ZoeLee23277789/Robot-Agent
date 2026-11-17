#!/usr/bin/env python
"""實體 EP 的 ToF（紅外線深度感測器）診斷：印出 SDK 訂閱到的 4 個插槽原始值（模組編號、方向、flag、距離 mm），5 秒。
用法：RM_CONN_TYPE=sta ~/rmenv/bin/python tools/ep_tof_probe.py   （直連熱點改 ap、USB 線改 rndis）
全部 0 = 中控沒認到模組：線要接底盤的感測器接口（不是傳感器轉接模組）、接好後要重開機。
不要在 8767 server 開著時跑（同一台 EP 只能有一條 SDK 連線）。"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Test_Robot"))
import rm_connection as r  # noqa: E402
from robomaster import sensor  # noqa: E402

raw = []
_orig = sensor.TofSubject.decode


def _decode(self, buf):
    _orig(self, buf)
    raw.append((list(self._cmd_id), list(self._direct), list(self._flag), list(self._distance)))


sensor.TofSubject.decode = _decode   # 類別層級改掉 decode，SDK 內部建的 subject 也會走這裡
ep = r.connect("real")
ep.sensor.sub_distance(freq=5, callback=lambda d: None)
try:
    for _ in range(5):
        time.sleep(1.0)
        if raw:
            ids, direct, flag, dist = raw[-1]
            print("id=%s direct=%s flag=%s distance_mm=%s" % (ids, direct, flag, dist), flush=True)
        else:
            print("還沒收到 ToF 資料", flush=True)
finally:
    ep.sensor.unsub_distance()
    ep.close()
