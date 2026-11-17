#!/usr/bin/env python
"""相機角度對位工具（2026-11-15 改版）。手臂擺到 (380,120,水平)，用影像找黑色夾爪，告訴你相機該往上還是往下調。
目標：夾爪開口在畫面的 (553, 512)/1000（11-15 校正白點基準時的位置）。
  .venv/bin/python tools/cam_check.py          # 量一次
  .venv/bin/python tools/cam_check.py watch    # 每 1.5 秒量一次，邊調相機邊看，Ctrl+C 結束"""
import json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot_agent import skills_nexarm
U, REAL = "http://127.0.0.1:8765", "http://127.0.0.1:8767"
TX, TY = skills_nexarm.NEX_GAP_REF
r = urllib.request.Request(U + "/action", json.dumps({"name": "arm_to", "params": {"x_mm": 380, "y_mm": 120, "pitch_deg": 0, "yaw_deg": 0}}).encode(), {"Content-Type": "application/json"})
urllib.request.urlopen(r, timeout=40).read(); time.sleep(1.2)
try:
    for _ in range(600 if len(sys.argv) > 1 else 1):
        urllib.request.urlopen(REAL + "/frame.jpg", timeout=8).read(); time.sleep(0.2)
        jpg = urllib.request.urlopen(REAL + "/frame.jpg", timeout=8).read()
        g = skills_nexarm._gripper_gap_cv(jpg)
        if g is None:
            print("畫面裡找不到夾爪：相機偏太多了，先大致對著夾爪前方的地板", flush=True)
        else:
            dx, dy = g[0] - TX, g[1] - TY
            if abs(dy) <= 110 and abs(dx) <= 90:
                msg = "✔ 可以了（在容許範圍內，不用對到剛好）"
            else:
                msg = ("相機要往%s轉約 %.0f°" % ("下（鏡頭朝地板）" if dy > 0 else "上", abs(dy) * 0.067)) if abs(dy) > 110 else ""
                if abs(dx) > 90: msg += "；往%s轉約 %.0f°" % ("右" if dx > 0 else "左", abs(dx) * 0.1)
            print("夾爪在 (%.0f, %.0f)，目標 (%.0f, %.0f) → %s" % (g[0], g[1], TX, TY, msg), flush=True)
        if len(sys.argv) > 1: time.sleep(1.3)
except KeyboardInterrupt:
    pass
