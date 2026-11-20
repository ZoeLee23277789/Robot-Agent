#!/usr/bin/env python
"""每個任務用了多少 token、大約多少錢：讀 runs/ 與 robot_eval/results/ 底下所有 history.json 彙整。

    .venv/bin/python tools/token_report.py                      # 每一題（dataset 的 task id）的中位數
    .venv/bin/python tools/token_report.py --price-per-m 38     # 用這個平均單價換算（每一百萬 token，幣別看 .env 的 ROBOT_PRICE_CURRENCY）
    .venv/bin/python tools/token_report.py --since 2026-11-01   # 只看這天之後的紀錄
    .venv/bin/python tools/token_report.py --runs               # 不彙整，逐次列出（最新的在最後）
    .venv/bin/python tools/token_report.py --csv out.csv

單價：沒給 --price-per-m 就用 .env 的 ROBOT_PRICE_*（見 .env.example）；都沒有就只列 token。
注意：2026-11-20 之前的紀錄沒有算進感知／技能內部的 LLM 呼叫（perception_tokens 欄位是 0），token 偏低；
批次評估的 judge（預設 OpenAI gpt-4o）不在 history.json 裡，是另一張帳單。
"""

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from robot_agent.llm_factory import estimate_cost  # noqa: E402


def load_runs(since_ts: float):
    dataset = {}
    for name in ("dataset.json", "office_tasks.json"):
        try:
            data = json.load(open(ROOT / "robot_eval" / name, encoding="utf-8"))
            for t in (data if isinstance(data, list) else data.get("tasks", [])):
                dataset[t["id"]] = t
        except Exception:
            pass
    rows = []
    files = glob.glob(str(ROOT / "runs/*/history.json")) + glob.glob(str(ROOT / "robot_eval/results/**/history.json"), recursive=True)
    for f in files:
        try:
            h = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not h.get("llm_calls"):
            continue   # token 統計功能加入之前的紀錄
        steps = h.get("steps") or []
        ts = (steps[0].get("timestamp") if steps else None) or os.path.getmtime(f)
        if ts < since_ts:
            continue
        task_id = Path(f).parent.name if "robot_eval" in Path(f).parts else ""
        meta = dataset.get(task_id, {})
        rows.append({
            "time": ts, "task_id": task_id if meta else "", "difficulty": meta.get("difficulty", ""),
            "category": meta.get("category", ""), "task": " ".join(str(h.get("task", "")).split()),
            "steps": len(steps), "llm_calls": h["llm_calls"], "prompt": h.get("total_prompt_tokens", 0),
            "completion": h.get("total_completion_tokens", 0), "total": h.get("total_tokens", 0),
            "perception": h.get("perception_tokens", 0), "success": h.get("success"),
            "dir": str(Path(f).parent.relative_to(ROOT)),
        })
    rows.sort(key=lambda r: r["time"])
    return rows


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="per-task token and cost report")
    ap.add_argument("--price-per-m", type=float, default=None, help="平均單價：每一百萬 token 多少錢")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD")
    ap.add_argument("--runs", action="store_true", help="逐次列出，不依題目彙整")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    if a.price_per_m is not None:
        os.environ["ROBOT_PRICE_PER_M"] = str(a.price_per_m)
        os.environ.pop("ROBOT_PRICE_INPUT_PER_M", None)
    since = time.mktime(time.strptime(a.since, "%Y-%m-%d")) if a.since else 0.0
    rows = load_runs(since)
    if not rows:
        print("沒有帶 token 統計的紀錄")
        return
    for r in rows:
        c = estimate_cost(r["prompt"], r["completion"], r["total"])
        r["cost"] = round(c[0], 2) if c else None
    priced = rows[0]["cost"] is not None
    cur = os.getenv("ROBOT_PRICE_CURRENCY", "TWD")

    def money(v):
        return f"{v:8.2f}" if v is not None else "       -"

    if a.runs:
        out = rows
        print(f"{'時間':16} {'題目':6} {'步':>3} {'呼叫':>4} {'token':>9} {'錢(' + cur + ')':>9}  任務")
        for r in rows:
            print(f"{time.strftime('%m-%d %H:%M', time.localtime(r['time'])):16} {r['task_id'] or '-':6} {r['steps']:3d} "
                  f"{r['llm_calls']:4d} {r['total']:9d} {money(r['cost'])}  {r['task'][:60]}")
    else:
        groups = defaultdict(list)
        for r in rows:
            groups[r["task_id"] or "(互動/其他)"].append(r)
        out = []
        for tid, rs in groups.items():
            med = lambda k: statistics.median(x[k] for x in rs)   # noqa: E731
            out.append({"task_id": tid, "difficulty": rs[0]["difficulty"], "category": rs[0]["category"], "runs": len(rs),
                        "steps_median": med("steps"), "llm_calls_median": med("llm_calls"), "tokens_median": int(med("total")),
                        "tokens_max": max(x["total"] for x in rs),
                        "cost_median": round(med("cost"), 2) if priced else None,
                        "cost_max": max(x["cost"] for x in rs) if priced else None,
                        "task": rs[-1]["task"] if tid != "(互動/其他)" else ""})
        out.sort(key=lambda g: -g["tokens_median"])
        print(f"{'題目':10} {'難度':7} {'類別':13} {'次數':>4} {'步(中位)':>7} {'token(中位)':>11} {'token(最大)':>11} "
              f"{'錢中位(' + cur + ')':>11} {'錢最大':>8}")
        for g in out:
            print(f"{g['task_id']:10} {g['difficulty']:7} {g['category']:13} {g['runs']:4d} {g['steps_median']:7.0f} "
                  f"{g['tokens_median']:11d} {g['tokens_max']:11d} {money(g['cost_median']):>11} {money(g['cost_max'])}")

    print("\n== 每月合計 ==")
    months = defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        m = months[time.strftime("%Y-%m", time.localtime(r["time"]))]
        m[0] += 1
        m[1] += r["total"]
        m[2] += r["cost"] or 0.0
    for k in sorted(months):
        n, tok, c = months[k]
        print(f"{k}: {n} 次任務, {tok / 1e6:.1f} M tokens" + (f", 約 {c:.0f} {cur}" if priced else ""))
    tot = [r["total"] for r in rows]
    print(f"\n全部 {len(rows)} 次：每次任務 token 中位數 {int(statistics.median(tot))}、平均 {int(statistics.mean(tot))}"
          + (f"；錢 中位數 {statistics.median(r['cost'] for r in rows):.2f}、平均 {statistics.mean(r['cost'] for r in rows):.2f} {cur}" if priced else
             "（沒設單價：加 --price-per-m 或在 .env 設 ROBOT_PRICE_PER_M）"))

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"已寫出 {a.csv}")


if __name__ == "__main__":
    main()
