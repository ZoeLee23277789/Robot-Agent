#!/usr/bin/env python
"""
把多份 tools/nav_selftest.py 的結果並排比較（預設：robot_eval/results/ 下最近的三份）。

    .venv/bin/python tools/nav_selftest_compare.py
    .venv/bin/python tools/nav_selftest_compare.py robot_eval/results/nav_selftest_A.json robot_eval/results/nav_selftest_B.json
"""
import glob
import json
import sys
from pathlib import Path


def load(paths):
    return [(Path(p).stem.replace("nav_selftest_", ""), json.loads(Path(p).read_text(encoding="utf-8"))) for p in paths]


def main():
    paths = sys.argv[1:] or sorted(glob.glob("robot_eval/results/nav_selftest_*.json"))[-3:]
    runs = load(paths)
    if not runs:
        raise SystemExit("沒有 nav_selftest 結果")
    names = [n for n, _ in runs]
    print(f"{'指標':<34}" + "".join(f"{n:>18}" for n in names))
    print("-" * (34 + 18 * len(names)))
    rows = []
    for sec in ("rotate", "goals", "landmarks", "passages", "objects"):
        def cell(r, what):
            s = r["sections"].get(sec)
            if not s:
                return "-"
            n = len(s)
            if what == "pass":
                return f"{sum(1 for x in s if x.get('ok'))}/{n}"
            if what == "coll":
                return str(sum(len(x.get("collisions", [])) for x in s))
            if what == "maxerr" and sec == "rotate":
                return f"{max(abs(x['error_deg']) for x in s):.1f}°"
            if what == "meanerr" and sec == "goals":
                return f"{sum(x['final_err_m'] for x in s) / n:.2f}m"
            if what == "time":
                return f"{sum(x.get('s', 0) for x in s):.0f}s"
            return "-"
        rows.append((f"{sec} 通過", "pass"))
        if sec == "rotate":
            rows.append((f"{sec} 最大誤差", "maxerr"))
        if sec == "goals":
            rows.append((f"{sec} 平均終點誤差", "meanerr"))
        if sec != "rotate":
            rows.append((f"{sec} 碰撞次數", "coll"))
        rows.append((f"{sec} 耗時", "time"))
        for label, what in rows:
            print(f"{label:<34}" + "".join(f"{cell(r, what):>18}" for _, r in runs))
        rows = []
    print("-" * (34 + 18 * len(names)))
    print(f"{'總碰撞':<34}" + "".join(f"{sum(len(x.get('collisions', [])) for s in r['sections'].values() for x in s):>18}" for _, r in runs))
    print(f"{'總時間':<34}" + "".join(f"{r.get('duration_s', 0):>17}s" for _, r in runs))
    print(f"{'CoppeliaSim RSS 起→終 (MB)':<34}" + "".join(f"{str(r.get('coppelia_rss_mb_start')) + '→' + str(r.get('coppelia_rss_mb_end')):>18}" for _, r in runs))


if __name__ == "__main__":
    main()
