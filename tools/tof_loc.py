#!/usr/bin/env python
"""ToF 掃一圈 + 掃描地圖 → 定位，並把車修回「家」（2026-11-14，使用者：「你沒有辦法定位嗎？」）。
  .venv/bin/python tools/tof_loc.py set-home     # 車已經擺在正確的家：掃一圈，把定位結果存成基準 runs/_home_ref.json
  .venv/bin/python tools/tof_loc.py fix          # 回到家附近之後：掃一圈定位，跟基準比，補平移與轉角
  .venv/bin/python tools/tof_loc.py where        # 只定位、不動
地圖取自 8765 的 /map.png（深色 = 牆/家具），格點搜尋後取截尾平均誤差最小的位姿。"""
import json, math, statistics, sys, time, urllib.request
import numpy as np, cv2
U = "http://127.0.0.1:8765"
REF = "runs/_home_ref.json"
def act(n, p):
    r = urllib.request.Request(U + "/action", json.dumps({"name": n, "params": p}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=90))
def st(): return json.load(urllib.request.urlopen(U + "/state", timeout=5))

def load_map():
    buf = np.frombuffer(urllib.request.urlopen(U + "/map.png", timeout=15).read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR); n = img.shape[0] // 4
    small = cv2.resize(img, (n, n), interpolation=cv2.INTER_NEAREST)
    return small.astype(int).sum(axis=2) < 200, n

def sweep(step=15.0):
    out = []
    act("recenter_arm", {})
    for k in range(int(round(360 / step))):
        time.sleep(0.7); vals = []
        for _ in range(5):
            s = st(); vals.append(s.get("real_tof_front_mm")); time.sleep(0.2)
        good = [v for v in vals if isinstance(v, (int, float)) and v < 9999]
        out.append({"heading_deg": s.get("real_world_heading_deg"), "tof_mm": statistics.median(good) if len(good) >= 3 else None})
        act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": step, "push": False})
    time.sleep(0.8)
    return out, st().get("real_world_heading_deg")

def estimate(sw, occ, n, x0=0.0, y0=0.0):
    half = n * 0.05 / 2
    def cell(x, y): return int((x + half) / 0.05), int((half - y) / 0.05)
    def ray(x, y, th):
        c, s = math.cos(th), math.sin(th); t = 0.0
        while t < 7.0:
            ix, iy = cell(x + c * t, y + s * t)
            if not (0 <= ix < n and 0 <= iy < n) or occ[iy, ix]: return t
            t += 0.025
        return 7.0
    beams = [d for d in sw if d["tof_mm"] and d["tof_mm"] < 5500]
    def cost(x, y, dth):
        e = []
        for d in beams:
            th = math.radians(d["heading_deg"] + dth)
            e.append(abs(ray(x + 0.16 * math.cos(th), y + 0.16 * math.sin(th), th) - d["tof_mm"] / 1000.0))
        e.sort(); k = max(3, int(len(e) * 0.75))
        return sum(e[:k]) / k
    def search(cx, cy, cth, span, step, tspan, tstep):
        best = (9e9, cx, cy, cth)
        m = int(round(span / step)); tm = int(round(tspan / tstep))
        for it in range(-tm, tm + 1):
            for ix in range(-m, m + 1):
                for iy in range(-m, m + 1):
                    x, y, th = cx + ix * step, cy + iy * step, cth + it * tstep
                    jx, jy = cell(x, y)
                    if not (0 <= jx < n and 0 <= jy < n) or occ[jy, jx]: continue
                    c = cost(x, y, th)
                    if c < best[0]: best = (c, x, y, th)
        return best
    b = search(x0, y0, 0.0, 0.7, 0.10, 24, 3)
    b = search(b[1], b[2], b[3], 0.10, 0.025, 3, 1)
    return {"cost_m": round(b[0], 3), "x": round(b[1], 3), "y": round(b[2], 3), "dth_deg": b[3], "beams": len(beams)}

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "where"
    occ, n = load_map()
    if mode == "from-file":       # 用既有的掃描檔算（不動車）
        sw = json.load(open("runs/_tof_sweep.json")); print(estimate(sw, occ, n)); return
    sw, hd_now = sweep()
    est = estimate(sw, occ, n)
    pose = {"x": est["x"], "y": est["y"], "heading_deg": ((hd_now + est["dth_deg"] + 180) % 360) - 180}   # 掃完後車現在的地圖位姿
    print("定位：", est, "→ 現在位姿", pose)
    if mode == "set-home":
        # 家的朝向 = 掃描開始時的朝向（掃一圈後車沒有剛好轉回 360°，等一下 fix 會轉回去）
        ref = {"x": est["x"], "y": est["y"], "heading_deg": ((sw[0]["heading_deg"] + est["dth_deg"] + 180) % 360) - 180, "cost_m": est["cost_m"]}
        json.dump(ref, open(REF, "w")); print("家的基準存到", REF, ref)
    if mode in ("fix", "set-home"):
        ref = json.load(open(REF))
        dx, dy = ref["x"] - pose["x"], ref["y"] - pose["y"]
        th = math.radians(pose["heading_deg"])
        fwd, left = dx * math.cos(th) + dy * math.sin(th), -dx * math.sin(th) + dy * math.cos(th)
        turn = (ref["heading_deg"] - pose["heading_deg"] + 180) % 360 - 180
        print("修正：前進 %+.2f m、往左 %+.2f m、左轉 %+.1f°" % (fwd, left, turn))
        if abs(fwd) > 0.6 or abs(left) > 0.6:
            print("差太多（>0.6 m），不自動修，請檢查"); return
        if abs(fwd) > 0.03 or abs(left) > 0.03:
            act("move_chassis", {"forward_m": round(fwd, 3), "right_m": round(-left, 3), "turn_left_deg": 0, "push": False})
        if abs(turn) > 1.5:
            act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(turn, 1), "push": False})
        print("修正完成")

if __name__ == "__main__":
    main()
