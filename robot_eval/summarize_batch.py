"""彙整一個批次目錄下所有任務的 evaluation.json + history.json，輸出 Markdown 總表與失敗分類。"""
import collections
import glob
import json
import sys
from pathlib import Path

batch = Path(sys.argv[1])
dataset = json.load(open("robot_eval/dataset.json", encoding="utf-8"))["tasks"]
order = {t["id"]: i for i, t in enumerate(dataset)}
by_id = {t["id"]: t for t in dataset}

rows = {}
for ev_path in sorted(glob.glob(str(batch / "*/*/evaluation.json"))):
    ev = json.load(open(ev_path, encoding="utf-8"))
    hist = json.load(open(Path(ev_path).parent / "history.json", encoding="utf-8"))
    m = ev["metrics"]
    tid = m["task_id"]
    mv = collections.Counter()
    for s in hist["steps"]:
        for a in s.get("actions", []):
            if a["name"] != "move_chassis":
                continue
            p, msg = a["params"], a.get("message", "")
            kind = "T" if abs(p.get("forward_m", 0)) >= 0.02 or abs(p.get("right_m", 0)) >= 0.02 else "R"
            if "too small" in msg:
                mv["tiny"] += 1
            elif "timed out" in msg:
                mv["timeout"] += 1
            elif "rejected after the fact" in msg:
                mv["nomotion"] += 1
            elif a["ok"]:
                mv[kind + "ok"] += 1
            else:
                mv["otherfail"] += 1
    ms_all = m["total_milestones"] > 0 and m["completed_milestones"] == m["total_milestones"]
    ms_none = m["completed_milestones"] == 0
    verdict = m["judge_verdict"]
    disagree = ""
    if ms_all and verdict is False:
        disagree = "milestone 全過但 judge 判敗"
    elif ms_none and verdict is True:
        disagree = "milestone 全沒過但 judge 判成功"
    overclaim = bool(m["agent_success"]) and not m["is_successful"]
    reason = (m["judge_failure_reason"] or ("" if m["is_successful"] else m["judge_reasoning"]) or "").strip()
    rows[tid] = dict(m=m, mv=mv, disagree=disagree, overclaim=overclaim, reason=reason,
                     ms=ev["milestones"], hit_max=m["total_steps"] >= 30)

ids = sorted(rows, key=lambda t: order.get(t, 10 ** 6))
done_ids = set(ids)
pending = [t["id"] for t in dataset if t["id"] not in done_ids]

def pct(n, d):
    return f"{n / d:.0%}" if d else "-"

lines = []
lines.append(f"# 評估統整：{batch.name}")
lines.append("")
lines.append(f"已完成 {len(ids)}/{len(dataset)} 個任務" + (f"，尚未跑到：{', '.join(pending)}" if pending else "，全部完成"))
lines.append("")
succ = [t for t in ids if rows[t]["m"]["is_successful"]]
lines.append(f"**整體成功率 {pct(len(succ), len(ids))}（{len(succ)}/{len(ids)}）**，milestone 平均完成 "
             f"{sum(rows[t]['m']['milestone_completion_rate'] for t in ids) / max(len(ids), 1):.0%}")
lines.append("")
lines.append("| 難度 | 成功 | 任務數 | 成功率 |")
lines.append("| --- | --- | --- | --- |")
for diff in ("easy", "medium", "hard"):
    g = [t for t in ids if rows[t]["m"]["difficulty"] == diff]
    s = [t for t in g if rows[t]["m"]["is_successful"]]
    lines.append(f"| {diff} | {len(s)} | {len(g)} | {pct(len(s), len(g))} |")
lines.append("")
lines.append("| 類別 | 成功 | 任務數 | 成功率 |")
lines.append("| --- | --- | --- | --- |")
for cat in sorted({rows[t]["m"]["category"] or "?" for t in ids}):
    g = [t for t in ids if (rows[t]["m"]["category"] or "?") == cat]
    s = [t for t in g if rows[t]["m"]["is_successful"]]
    lines.append(f"| {cat} | {len(s)} | {len(g)} | {pct(len(s), len(g))} |")
lines.append("")
lines.append("## 逐一任務")
lines.append("")
lines.append("| 任務 | 難度 | 結果 | milestone | 步數 | 秒數 | 自評 | 平移 ok/逾時/沒動 | 旋轉 ok/太小 | 備註 |")
lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for t in ids:
    r, m, mv = rows[t], rows[t]["m"], rows[t]["mv"]
    notes = []
    if r["overclaim"]:
        notes.append("自評成功但失敗")
    if r["disagree"]:
        notes.append(r["disagree"])
    if r["hit_max"]:
        notes.append("用滿 30 步")
    if r["reason"] and not m["is_successful"]:
        notes.append(r["reason"][:90])
    lines.append(f"| {t} | {m['difficulty']} | {'✅' if m['is_successful'] else '❌'} | "
                 f"{m['completed_milestones']}/{m['total_milestones']} | {m['total_steps']} | {m['total_duration_seconds']:.0f} | "
                 f"{'成功' if m['agent_success'] else ('失敗' if m['agent_success'] is False else '-')} | "
                 f"{mv['Tok']}/{mv['timeout']}/{mv['nomotion']} | {mv['Rok']}/{mv['tiny']} | {'；'.join(notes)} |")
lines.append("")
lines.append("## 橫向觀察")
lines.append("")
over = [t for t in ids if rows[t]["overclaim"]]
dis = [f"{t}（{rows[t]['disagree']}）" for t in ids if rows[t]["disagree"]]
maxed = [t for t in ids if rows[t]["hit_max"]]
tot = collections.Counter()
for t in ids:
    tot.update(rows[t]["mv"])
lines.append(f"- Agent 自評成功但實際失敗：{len(over)} 個 → {', '.join(over) or '無'}")
lines.append(f"- judge 與 milestone 不一致（需人工複核）：{len(dis)} 個 → {'; '.join(dis) or '無'}")
lines.append(f"- 用滿 30 步仍未完成：{len(maxed)} 個 → {', '.join(maxed) or '無'}")
T = tot["Tok"] + tot["timeout"] + tot["nomotion"]
R = tot["Rok"] + tot["tiny"]
lines.append(f"- 平移指令 {T} 次：成功 {tot['Tok']}、回報逾時 {tot['timeout']}（{pct(tot['timeout'], T)}）、事後判定沒動 {tot['nomotion']}")
lines.append(f"- 旋轉指令 {R} 次：成功 {tot['Rok']}、太小被拒 {tot['tiny']}（{pct(tot['tiny'], R)}）")
lines.append("")
lines.append("## 失敗任務的 milestone 明細")
lines.append("")
for t in ids:
    if rows[t]["m"]["is_successful"]:
        continue
    lines.append(f"**{t}** — {by_id[t]['description']}")
    for ms in rows[t]["ms"]:
        lines.append(f"- {'✓' if ms['passed'] else '✗'} {ms['milestone_id']} [{ms['method']}] {ms['reasoning'][:200]}")
    if rows[t]["reason"]:
        lines.append(f"- judge：{rows[t]['reason'][:250]}")
    lines.append("")

out = "\n".join(lines)
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(out, encoding="utf-8")
print(out)
