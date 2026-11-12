"""
實測 NexArm 真正能「收多緊」：從 HOME 一次往底座縮 10 mm，看伺服實際停在哪裡。

韌體的 IK 表（nexarm_workspace.json）說 x=0（夾爪在底座正上方）數學上到得了，但那張表不管手臂會不會撞到自己，
所以 nexarm.py 的 WORKSPACE 把 x 下限保守設在 80。這支腳本把下限暫時放開，真的走過去量：
  A. 夾爪水平（pitch 0），高度 250：x 150 → 0，每步 10 mm，停在離目標 > 12 mm 就算到底
  B. 在 A 找到的最緊 x，往下壓：z 250 → 120
  C. 夾爪朝下（pitch −90），高度 150：x 150 → 0，再往下壓到 80
  D. 在 A 的最緊 x 往上抬：z 250 → 450（手臂直立，佔地最小但最高）
每一步都用擬合 FK 把肘、腕的位置也算出來，才知道手臂折回來時肘有沒有伸到底座後面（裝在車上會撞升降柱）。

server 要先關掉（兩個程式不能共用同一個串列埠）：kill $(lsof -t -i:8767)
    ~/rmenv/bin/python Test_Robot/nexarm_fold_test.py            # 約 3 分鐘；有數位分身就加 --mirror 23001
結果：nexarm_fold.json，最後一段印出建議的「收攏姿勢」。
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import nexarm                                   # noqa: E402
from nexarm import HOME, NexArm                # noqa: E402
from nexarm_workspace import FK, COUNTS_PER_DEG   # noqa: E402

OUT = Path(__file__).with_name("nexarm_fold.json")
TOL = nexarm.REACH_TOL_MM


def joints_xz(servos):
    """肘、腕、末端在手臂平面上的 (x, z)，用擬合出來的三連桿模型算（跟 nexarm_workspace.fk 同一組參數）。"""
    d2, d3, d4 = ((s - 2048) / COUNTS_PER_DEG for s in servos[1:4])
    p1 = math.radians(FK["o1"] + d2)
    p2 = p1 + math.radians(FK["o2"] + d3)
    p3 = math.radians(FK["c"] + d2 + d3 - d4)
    elbow = (FK["ax"] + FK["L1"] * math.cos(p1), FK["az"] + FK["L1"] * math.sin(p1))
    wrist = (elbow[0] + FK["L2"] * math.cos(p2), elbow[1] + FK["L2"] * math.sin(p2))
    tip = (wrist[0] + FK["L3"] * math.cos(p3), wrist[1] + FK["L3"] * math.sin(p3))
    return {"elbow": [round(v) for v in elbow], "wrist": [round(v) for v in wrist], "tip": [round(v) for v in tip]}


def step_until_stuck(arm, label, points, pitch, log):
    """依序走 points [(x, z), ...]，第一個停在離目標 > TOL 的點就停，回傳最後一個真的到的 (x, z)。"""
    last_ok = None
    for x, z in points:
        r = arm.move_to(x=x, y=0.0, z=z, pitch=pitch, correct=False)
        st = r.get("pose") or {}
        # 可達表把目標夾掉（例如 z 夾到該前伸距離的下限）也算到底：手臂根本沒被送去原本的點
        reached = bool(r.get("ok")) and not r.get("clamped")
        rec = {"seq": label, "target": [x, z, pitch], "ok": reached, "error_mm": r.get("error_mm"),
               "unreachable": bool(r.get("unreachable")), "clamped": r.get("clamped", ""),
               "measured": [st.get("x"), st.get("z"), st.get("pitch")],
               "servos": st.get("servos"), "joints": joints_xz(st["servos"]) if st.get("servos") else None}
        log.append(rec)
        j = rec["joints"] or {}
        print("  %s 目標 x=%3d z=%3d → 實測 x=%s z=%s pitch=%s  err=%s mm  %s  肘=%s 腕=%s%s" % (
            label, x, z, st.get("x"), st.get("z"), st.get("pitch"), r.get("error_mm"),
            "OK " if reached else ("表外" if r.get("unreachable") else ("被夾" if r.get("clamped") else "到不了")),
            j.get("elbow"), j.get("wrist"), ("  [%s]" % r["clamped"]) if r.get("clamped") else ""))
        if not reached:
            break
        last_ok = (x, z)
        time.sleep(0.2)
    return last_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/nexarm")
    ap.add_argument("--mirror", type=int, default=None, help="數位分身的 zmq 埠（視窗版 CoppeliaSim，通常 23001）")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--go", action="store_true", help="真的動；不給只印計畫")
    a = ap.parse_args()

    nexarm.WORKSPACE["x"] = (0.0, nexarm.WORKSPACE["x"][1])     # 放開 x 下限，這支腳本就是要量 80 以下
    s = a.step
    plan = {
        "A 水平縮回 (pitch 0, z 250)": (0.0, [(x, 250) for x in range(150, -1, -s)]),
        "C 朝下縮回 (pitch -90, z 150)": (-90.0, [(x, 150) for x in range(150, -1, -s)]),
    }
    if not a.go:
        for k, (p, pts) in plan.items():
            print(k, "pitch", p, "點數", len(pts), "從", pts[0], "到", pts[-1])
        print("B / D 的 z 範圍依 A 的結果決定；C 之後再往下壓到 z 80。加 --go 才會動。")
        return

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
        pitch, pts = plan["A 水平縮回 (pitch 0, z 250)"]
        a_x = step_until_stuck(arm, "A", pts, pitch, log)
        summary["A_tightest_x_at_z250_pitch0"] = a_x
        if a_x:
            print("B 在 x=%d 往下壓" % a_x[0])
            b = step_until_stuck(arm, "B", [(a_x[0], z) for z in range(250, 119, -s)], 0.0, log)
            summary["B_lowest_at_tight_x_pitch0"] = b
            print("D 在 x=%d 往上抬" % a_x[0])
            d = step_until_stuck(arm, "D", [(a_x[0], z) for z in range(250, 451, s)], 0.0, log)
            summary["D_highest_at_tight_x_pitch0"] = d
        print("回 HOME 再做 C（夾爪朝下）")
        arm.home()
        pitch, pts = plan["C 朝下縮回 (pitch -90, z 150)"]
        c_x = step_until_stuck(arm, "C", pts, pitch, log)
        summary["C_tightest_x_at_z150_pitch-90"] = c_x
        if c_x:
            c2 = step_until_stuck(arm, "C2", [(c_x[0], z) for z in range(150, 79, -s)], -90.0, log)
            summary["C2_lowest_at_tight_x_pitch-90"] = c2
    finally:
        print("回 HOME")
        try:
            arm.home()
        except Exception as e:
            print("home 失敗：", e)
        OUT.write_text(json.dumps({"summary": summary, "steps": log, "tol_mm": TOL,
                                   "scanned": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        print("寫入", OUT)
        if mirror:
            mirror.close()
        arm.close()
    print("\n摘要：")
    for k, v in summary.items():
        print("  %s: %s" % (k, v))


if __name__ == "__main__":
    main()
