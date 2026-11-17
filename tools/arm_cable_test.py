#!/usr/bin/env python
"""相機排線安全範圍測試（2026-11-14，相機改裝到手臂頂端之後）。
手臂每次只動一小步，人看著排線：Enter = 沒事繼續、n = 線緊了（退回上一步，這個方向到此為止）、q = 結束。
只動手臂、不動底盤。結果寫到 Test_Robot/nexarm_cable_limits.json，每一步的相機畫面存在 runs/_cable_test/。

用法：.venv/bin/python tools/arm_cable_test.py [--url http://127.0.0.1:8767]
（8767 要用 ROBOT_ARM_MAX_Z_MM=0 啟動，不然高度會被夾在 300）
"""
import argparse
import json
import os
import sys
import time
import urllib.request

HOME = dict(x_mm=150, y_mm=250, pitch_deg=0.0, yaw_deg=0.0)
# (名稱, 說明, 變動的欄位, 其他固定值, 依序要試的值)
AXES = [
    ("height_at_home", "抬高（前伸 150）", "y_mm", dict(x_mm=150), [280, 310, 340, 370, 400, 430, 460]),
    ("reach_at_250", "往前伸（高度 250）", "x_mm", dict(y_mm=250), [200, 250, 300, 350, 400, 450]),
    ("reach_at_150", "往前伸（高度 150）", "x_mm", dict(y_mm=150), [250, 300, 350, 400, 450]),
    ("low_at_340", "放低（前伸 340）", "y_mm", dict(x_mm=340), [200, 150, 110, 80, 50, 30]),
    ("high_at_350", "伸出去再抬高（前伸 350）", "y_mm", dict(x_mm=350), [250, 290, 330, 370, 400]),
    ("pitch_down", "夾爪往下轉（HOME）", "pitch_deg", dict(), [-15, -30, -45, -60, -75, -90]),
    ("pitch_down_far", "夾爪往下轉（前伸 300、高 200）", "pitch_deg", dict(x_mm=300, y_mm=200), [-15, -30, -45]),
    ("yaw_left", "底座往左轉（HOME）", "yaw_deg", dict(), [15, 30, 45, 60]),
    ("yaw_right", "底座往右轉（HOME）", "yaw_deg", dict(), [-15, -30, -45, -60]),
    ("yaw_left_far", "伸出去往左轉（前伸 350、高 150）", "yaw_deg", dict(x_mm=350, y_mm=150), [15, 30, 45]),
    ("yaw_right_far", "伸出去往右轉（前伸 350、高 150）", "yaw_deg", dict(x_mm=350, y_mm=150), [-15, -30, -45]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8767")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "Test_Robot", "nexarm_cable_limits.json")
    fdir = os.path.join(root, "runs", "_cable_test")
    os.makedirs(fdir, exist_ok=True)

    def act(name, params):
        req = urllib.request.Request(a.url + "/action", json.dumps({"name": name, "params": params}).encode(),
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=40))

    def frame(tag):
        try:
            open(os.path.join(fdir, tag + ".jpg"), "wb").write(urllib.request.urlopen(a.url + "/frame.jpg", timeout=8).read())
        except Exception:
            pass

    def go(pose):
        r = act("arm_to", pose)
        return bool(r.get("ok")), str(r.get("message", ""))[:150]

    print("排線測試：只動手臂。每一步看線：Enter=沒事 / n=線緊了 / s=跳過這個方向 / q=結束")
    if input("手臂要先回 HOME（前伸 150、高 250）。可以動了嗎？[Enter=開始 / q=取消] ").strip().lower() == "q":
        return 0
    go(HOME)
    results = {"date": time.strftime("%Y-%m-%d %H:%M"), "axes": {}}
    quit_all = False
    for key, desc, field, fixed, values in AXES:
        if quit_all:
            break
        base = dict(HOME); base.update(fixed)
        print("\n=== %s ===" % desc)
        ans = input("  先到起點 %s [Enter=走 / s=跳過 / q=結束] " % json.dumps(base)).strip().lower()
        if ans == "q":
            break
        if ans == "s":
            continue
        ok, msg = go(base)
        last_safe, tight_at, why = base[field], None, "all steps fine"
        ans = input("  起點到了（%s）。線還好嗎？[Enter=好 / n=緊了] " % ("ok" if ok else "手臂沒到: " + msg)).strip().lower()
        if ans == "n":
            results["axes"][key] = {"desc": desc, "fixed": fixed, "field": field, "last_safe": None, "tight_at": base[field],
                                    "note": "start pose already tight"}
            go(HOME)
            continue
        for v in values:
            pose = dict(base); pose[field] = v
            ok, msg = go(pose)
            time.sleep(0.5)
            frame("%s_%s" % (key, str(v).replace("-", "m")))
            if not ok:
                print("  %s=%s：手臂到不了（%s），這個方向到此為止" % (field, v, msg))
                why = "arm limit at %s" % v
                break
            ans = input("  %s=%s 到了。線還好嗎？[Enter=好 / n=緊了 / q=結束] " % (field, v)).strip().lower()
            if ans == "n":
                tight_at, why = v, "cable tight"
                safe = dict(base); safe[field] = last_safe
                go(safe)
                break
            if ans == "q":
                quit_all = True
                break
            last_safe = v
        results["axes"][key] = {"desc": desc, "fixed": fixed, "field": field, "last_safe": last_safe, "tight_at": tight_at, "note": why}
        print("  → 安全到 %s=%s（%s）" % (field, last_safe, why))
        go(base)
        go(HOME)
        json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    go(HOME)
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n結果寫在 %s" % out)
    for k, r in results["axes"].items():
        print("  %-28s 安全到 %s=%s%s" % (r["desc"], r["field"], r["last_safe"], ("，%s 就緊了" % r["tight_at"]) if r["tight_at"] is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
