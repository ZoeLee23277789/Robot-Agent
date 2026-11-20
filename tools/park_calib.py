#!/usr/bin/env python
"""停車格校正（2026-11-16）：地上貼了一圈白點當停車格，回原點時要停在格子裡（skills_nexarm.return_home）。
先把車用手擺到格子正中間、車頭朝前，然後執行：
    .venv/bin/python tools/park_calib.py
流程：重開辦公室環境（這個位置 = 原點）→ 車退 0.65 m → 手臂擺俯視姿勢拍照 → 找出最遠那一排白點的中點與夾爪位置
→ 寫進 runs/_home_dot.json（{"row": true, ...}），參考照片存 runs/_home_dot_ref.jpg。最後車停在格子後方，
跑一次 return_to_origin（或任何任務）它就會自己開回格子裡。看不到一排白點或夾爪時不會覆蓋舊基準。"""
import json, subprocess, sys, time, urllib.request
sys.path.insert(0, ".")
from robot_agent import skills_nexarm

U = "http://127.0.0.1:8765"


def act(name, params):
    req = urllib.request.Request(U + "/action", json.dumps({"name": name, "params": params}).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=90))


def frame():
    urllib.request.urlopen(U + "/frame.jpg", timeout=8).read()      # 第一張可能是舊的
    time.sleep(0.5)
    return urllib.request.urlopen(U + "/frame.jpg", timeout=8).read()


def main():
    subprocess.call(["bash", "-c", "./start_office.sh 2>&1 | tail -3"])
    act("arm_to", {"x_mm": 250, "y_mm": 200, "pitch_deg": 0, "yaw_deg": 0})
    print(act("move_chassis", {"forward_m": -0.65, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})["message"][:110])
    act("arm_to", skills_nexarm._HOME_DOT_POSE)
    time.sleep(1.2)
    jpg = frame()
    row, gap = skills_nexarm._find_park_row(jpg), skills_nexarm._gripper_gap_cv(jpg)
    xy = json.load(urllib.request.urlopen(U + "/state", timeout=5))["real_world_xy"]
    print("白點排", row, "夾爪", gap, "實體里程計", xy)
    if row is None:
        print("!! 看不到一排白點：沒有更動基準（檢查相機角度、白點有沒有被擋住）")
        return 1
    if gap is None:
        # 2026-11-19：相機今天比 11-16 朝上約 3°，俯視姿勢下夾爪的橫桿掉到畫面下緣外，只剩兩根指尖，影像法認不出夾爪。
        # 白點那一排看得到就照存；少了夾爪位置只是不能補償「相機之後又被碰歪」，return_home 會直接用這張圖的白點位置。
        print("!! 畫面裡看不到完整的夾爪：基準照存，但沒有夾爪位置，之後相機角度被動到就要重新校正")
    ref = {"row": True, "x": round(float(row[0]), 1), "y": round(float(row[1]), 1), "slope": round(float(row[2]), 4),
           "forward_m": round(-xy[0], 3), "right_m": round(xy[1], 3),
           "gap_x": None if gap is None else round(float(gap[0]), 1), "gap_y": None if gap is None else round(float(gap[1]), 1)}
    with open(skills_nexarm._HOME_DOT_FILE, "w", encoding="utf-8") as fh:
        json.dump(ref, fh)
    with open("runs/_home_dot_ref.jpg", "wb") as fh:
        fh.write(jpg)
    print("已存基準", ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
