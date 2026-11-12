#!/usr/bin/env python
"""整理 /tmp/mirror.log（模擬 server 鏡射到實體的每一筆）：哪些實體失敗、只做一半、被夾、補送。
用法：python tools/mirror_report.py [--since HH:MM] [--log /tmp/mirror.log]"""
import argparse, json, re

ap = argparse.ArgumentParser(); ap.add_argument("--log", default="/tmp/mirror.log"); ap.add_argument("--since", default="00:00")
a = ap.parse_args()
rows = []
for line in open(a.log, encoding="utf-8"):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("t", "") >= a.since:
        rows.append(d)
print("筆數 %d（%s 之後）" % (len(rows), a.since))
fails = [d for d in rows if not d["real_ok"]]
flags = [d for d in rows if d["real_ok"] and re.search(r"NOT achieved|clamped|rejected|did NOT|corrected|barely", d["real_msg"])]
short = []
for d in rows:
    sm, rm = d.get("sim_measured") or {}, d.get("real_measured") or {}
    if d["name"] == "move_chassis" and sm and rm:
        if abs((sm.get("distance_m") or 0) - (rm.get("distance_m") or 0)) > 0.08 or abs(abs(sm.get("dyaw_deg") or 0) - abs(rm.get("dyaw_deg") or 0)) > 8:
            short.append(d)
def show(title, lst):
    print("\n== %s：%d 筆" % (title, len(lst)))
    for d in lst[:15]:
        print(" %s %-13s %-55s | sim %s | real %s | %s" % (d["t"], d["name"], json.dumps(d["params"], ensure_ascii=False)[:55], d.get("sim_measured"), d.get("real_measured"), d["real_msg"][:90]))
show("實體回 FAIL", fails); show("實體回 ok 但有被夾/沒達成/補送", flags); show("模擬與實體量到的位移差太多", short)
