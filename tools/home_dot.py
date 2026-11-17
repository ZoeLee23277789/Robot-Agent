#!/usr/bin/env python
"""看著地上的白色原點貼紙回家（2026-11-14，使用者：「我在地上本來就有貼一個白色的原點」）。
相機在俯視姿勢 (前伸 380, 高 120) 看得到夾爪前方地板上的白點。
  calib：車在「白點後方約 0.55 m、朝向 0°」時執行，記下白點在畫面的位置與「還要往前/往右多少才到家」→ runs/_home_dot.json
  servo：把白點對回基準畫面位置（底盤前後/左右小步修正）
  go   ：servo 之後往前開到家、手臂回看路姿勢
白點用影像找（亮、低飽和、小圓塊，取離基準最近的），不靠 LLM。"""
import json, sys, time, urllib.request
import numpy as np, cv2
U, REAL = "http://127.0.0.1:8765", "http://127.0.0.1:8767"
REF = "runs/_home_dot.json"
POSE = {"x_mm": 380, "y_mm": 120, "pitch_deg": 0, "yaw_deg": 0}
MM_PER_UNIT = 1.5          # 畫面 1/1000 約等於地面 1 mm（俯視姿勢、白點在 0.5 m 左右）
def act(n, p):
    r = urllib.request.Request(U + "/action", json.dumps({"name": n, "params": p}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=90))
def st(): return json.load(urllib.request.urlopen(U + "/state", timeout=5))
def frame():
    urllib.request.urlopen(REAL + "/frame.jpg", timeout=8).read(); time.sleep(0.3)
    return cv2.imdecode(np.frombuffer(urllib.request.urlopen(REAL + "/frame.jpg", timeout=8).read(), np.uint8), cv2.IMREAD_COLOR)
def find_dot(img, near=None):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 185) & (hsv[:, :, 1] < 60)).astype(np.uint8)
    mask[: int(h * 0.30)] = 0                                   # 上面 30% 是遠處（牆、窗、燈的反光）
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    cands = []
    for i in range(1, n):
        a, bw, bh = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if 5 <= a <= 600 and 0.4 <= bw / max(1, bh) <= 3.5 and a / float(bw * bh) > 0.5:
            cands.append((cent[i][0] / w * 1000, cent[i][1] / h * 1000, int(a)))
    if not cands: return None, []
    if near: cands.sort(key=lambda c: (c[0] - near[0]) ** 2 + (c[1] - near[1]) ** 2)
    else: cands.sort(key=lambda c: -c[2])
    return cands[0], cands
def look():
    act("arm_to", POSE); time.sleep(1.0)
def servo(ref, tries=5, tol=15):
    for i in range(tries):
        img = frame(); dot, cands = find_dot(img, (ref["x"], ref["y"]))
        cv2.imwrite("runs/_home_dot_last.jpg", img)
        if dot is None:
            print("看不到白點"); return False
        ex, ey = dot[0] - ref["x"], dot[1] - ref["y"]
        print("  白點在 (%.0f, %.0f)，基準 (%.0f, %.0f)，差 (%+.0f, %+.0f)" % (dot[0], dot[1], ref["x"], ref["y"], ex, ey))
        if abs(ex) <= tol and abs(ey) <= tol: return True
        right = max(-0.25, min(0.25, 0.8 * ex * MM_PER_UNIT / 1000.0))      # 白點偏右 → 車往右
        fwd = max(-0.25, min(0.25, -0.8 * ey * MM_PER_UNIT / 1000.0))       # 白點偏下（太近）→ 車後退
        # 底盤走不出 5 cm 以下的小步（實測送 2 cm 不動）：小修正用「先反方向退 8 cm，再走 8 cm + 修正量」
        for key, v in (("forward_m", fwd), ("right_m", right)):
            if abs(v) < 0.008: continue
            if abs(v) < 0.09:        # 5–8 cm 的步子底盤也常常不動（19:00 回家前後方向卡在 −45 單位收斂不了）
                back = -0.12 if v > 0 else 0.12
                act("move_chassis", {"forward_m": back if key == "forward_m" else 0, "right_m": back if key == "right_m" else 0, "turn_left_deg": 0, "push": False})
                v = v - back
            act("move_chassis", {"forward_m": round(v, 3) if key == "forward_m" else 0, "right_m": round(v, 3) if key == "right_m" else 0, "turn_left_deg": 0, "push": False})
        time.sleep(0.8)
    return False
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "servo"
    if mode == "home":        # 完整回家：里程計回原點 → 後退到白點後方 → 看白點對準 → 開上去
        r = act("return_to_origin", {}); print(str(r.get("message"))[:200])
        ref = json.load(open(REF))
        act("move_chassis", {"forward_m": -ref["forward_m"], "right_m": -ref["right_m"], "turn_left_deg": 0, "push": False})
        mode = "go"
    look()
    if mode == "calib":
        img = frame(); dot, cands = find_dot(img, (477, 553) if len(sys.argv) < 4 else (float(sys.argv[2]), float(sys.argv[3])))
        s = st(); xy = s["real_world_xy"]
        ref = {"x": dot[0], "y": dot[1], "forward_m": round(-xy[0], 3), "right_m": round(xy[1], 3)}      # 假設里程計原點 = 家
        ref = {k: float(v) for k, v in ref.items()}; json.dump(ref, open(REF, "w")); cv2.imwrite("runs/_home_dot_ref.jpg", img)
        print("基準：", ref, "候選", cands[:4]); return
    ref = json.load(open(REF))
    ok = servo(ref)
    print("servo", "對準了" if ok else "沒對準")
    if mode == "go" and ok:
        act("arm_to", {"x_mm": 250, "y_mm": 200, "pitch_deg": 0, "yaw_deg": 0})
        act("move_chassis", {"forward_m": ref["forward_m"], "right_m": ref["right_m"], "turn_left_deg": 0, "push": False})
        act("recenter_arm", {}); print("到家")
if __name__ == "__main__":
    main()
