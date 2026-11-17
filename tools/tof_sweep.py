#!/usr/bin/env python
"""原地轉一圈，每 15° 記一次實體 ToF（取 5 次中位數）與里程計朝向 → runs/_tof_sweep.json。定位用（2026-11-14）。"""
import json, statistics, sys, time, urllib.request
U = "http://127.0.0.1:8765"
def act(n, p):
    r = urllib.request.Request(U + "/action", json.dumps({"name": n, "params": p}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=60))
def st(): return json.load(urllib.request.urlopen(U + "/state", timeout=5))
step = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
out = []
act("recenter_arm", {})
for k in range(int(round(360 / step))):
    time.sleep(0.7)
    vals = []
    for _ in range(5):
        s = st(); vals.append(s.get("real_tof_front_mm")); time.sleep(0.22)
    good = [v for v in vals if isinstance(v, (int, float)) and v < 9999]
    out.append({"heading_deg": s.get("real_world_heading_deg"), "xy": s.get("real_world_xy"), "tof_mm": statistics.median(good) if len(good) >= 3 else None,
                "raw": vals})
    print(k, out[-1]["heading_deg"], out[-1]["tof_mm"], vals)
    act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": step, "push": False})
json.dump(out, open("runs/_tof_sweep.json", "w"))
s = st(); print("end pose real", s.get("real_world_xy"), s.get("real_world_heading_deg"))
