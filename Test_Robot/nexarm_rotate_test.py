"""
實測 NexArm 兩個旋轉軸真正能轉多少：底座旋轉（ID1，y 方向）跟手腕滾轉（ID5，roll）。

nexarm.py 的 YAW_LIMIT_DEG=135 是照模型抄的、WORKSPACE roll ±60 是照韌體 PS3 搖桿模式抄的，都不是 CMD 8 的真實極限。
做法跟 nexarm_fold_test.py 一樣：在 HOME（前伸 150、高 250、夾爪水平）每次多轉 10°，實測停在離目標 > 5° 就算到底，
正負兩邊各掃一次，最後回 HOME。同時印出方向：底座「正」是不是真的向左（站在手臂後面、面向夾爪看）請人眼確認。

server 要先關掉（串列埠不能共用）：kill $(lsof -t -i:8767)
    ~/rmenv/bin/python Test_Robot/nexarm_rotate_test.py --go --mirror 23001
結果：nexarm_rotate.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import nexarm                                   # noqa: E402
from nexarm import HOME, NexArm, rotate_xy, yaw_of   # noqa: E402

OUT = Path(__file__).with_name("nexarm_rotate.json")
TOL_DEG = 5.0


def sweep_yaw(arm, sign, step, log):
    last_ok = 0.0
    for yaw in range(step, 181, step):
        yaw *= sign
        x, y = rotate_xy(HOME[0], 0.0, yaw)
        r = arm.move_to(x=x, y=y, z=HOME[2], pitch=HOME[3], correct=False)
        st = r.get("pose") or {}
        got = yaw_of(st.get("x", 0), st.get("y", 0)) if st else None
        err = abs(((got - yaw) + 180) % 360 - 180) if got is not None else None
        ok = bool(r.get("ok")) and not r.get("clamped") and err is not None and err <= TOL_DEG
        log.append({"axis": "yaw", "target_deg": yaw, "measured_deg": round(got, 1) if got is not None else None,
                    "servo1": (st.get("servos") or [None])[0], "ok": ok, "clamped": r.get("clamped", ""), "pose": st})
        print("  底座 目標 %+4d° → 實測 %s°  伺服1=%s  %s%s" % (
            yaw, ("%+.1f" % got) if got is not None else "?", (st.get("servos") or ["?"])[0],
            "OK" if ok else "到不了", ("  [%s]" % r["clamped"]) if r.get("clamped") else ""))
        if not ok:
            break
        last_ok = yaw
        time.sleep(0.2)
    return last_ok


def sweep_roll(arm, sign, step, log):
    last_ok = 0.0
    for roll in range(step, 181, step):
        roll *= sign
        r = arm.move_to(x=HOME[0], y=0.0, z=HOME[2], pitch=HOME[3], roll=float(roll), correct=False)
        st = r.get("pose") or {}
        got = st.get("roll")
        err = abs(got - roll) if got is not None else None
        ok = bool(r.get("ok")) and not r.get("clamped") and err is not None and err <= TOL_DEG
        log.append({"axis": "roll", "target_deg": roll, "measured_deg": got, "servo5": (st.get("servos") or [None] * 5)[4],
                    "ok": ok, "clamped": r.get("clamped", ""), "pose": st})
        print("  手腕 目標 %+4d° → 實測 %s°  伺服5=%s  %s%s" % (
            roll, got, (st.get("servos") or ["?"] * 5)[4], "OK" if ok else "到不了",
            ("  [%s]" % r["clamped"]) if r.get("clamped") else ""))
        if not ok:
            break
        last_ok = roll
        time.sleep(0.2)
    return last_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/nexarm")
    ap.add_argument("--mirror", type=int, default=None)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if not a.go:
        print("會在 HOME 把底座轉 ±10° 一路到 ±180°、手腕轉 ±10° 一路到 ±180°，到不了就停；加 --go 才會動。")
        return
    nexarm.YAW_LIMIT_DEG = 180.0                                  # 放開，這支腳本就是要量極限
    nexarm.WORKSPACE["roll"] = (-180.0, 180.0)

    arm = NexArm(a.port)
    t0 = time.time()
    while not arm.link_ok() and time.time() - t0 < 5:
        time.sleep(0.1)
    if not arm.link_ok():
        print("手臂沒有狀態回報：server 還開著？USB 有接？")
        return
    mirror = None
    if a.mirror:
        from nexarm_sim import SimMirror
        mirror = SimMirror(lambda: arm.status()[0], port=a.mirror)
    log, summary = [], {}
    try:
        print("回 HOME")
        arm.home()
        print("底座向「正」方向（應該是向左）")
        summary["yaw_max_deg"] = sweep_yaw(arm, +1, a.step, log)
        arm.home()
        print("底座向「負」方向")
        summary["yaw_min_deg"] = sweep_yaw(arm, -1, a.step, log)
        arm.home()
        print("手腕正向")
        summary["roll_max_deg"] = sweep_roll(arm, +1, a.step, log)
        arm.home()
        print("手腕負向")
        summary["roll_min_deg"] = sweep_roll(arm, -1, a.step, log)
    finally:
        print("回 HOME")
        try:
            arm.home()
        except Exception as e:
            print("home 失敗：", e)
        OUT.write_text(json.dumps({"summary": summary, "steps": log, "tol_deg": TOL_DEG,
                                   "scanned": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1, ensure_ascii=False), encoding="utf-8")
        print("寫入", OUT)
        if mirror:
            mirror.close()
        arm.close()
    print("\n摘要：", summary)


if __name__ == "__main__":
    main()
