#!/usr/bin/env python
"""辦公室實體題組的執行器：一題一題跑 run_agent.sh，跑完問人判定（實體沒有真值）。見 run_office_tasks.sh。"""
import argparse, json, os, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def state(url):
    try:
        with urllib.request.urlopen(url + "/state", timeout=5) as r:
            return json.load(r)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="robot_eval/office_tasks.json")
    ap.add_argument("--task", action="extend", nargs="+", help="題號，可以一次給多個（空白或逗號分開）：--task o_h2 o_m1")
    ap.add_argument("--difficulty", action="extend", nargs="+")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--robot-url", default="http://127.0.0.1:8765")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--fresh", action="store_true", help="每題開始前都整套重開環境歸零（預設只在模擬與實體漂開超過 15 cm 時才重開）")
    a = ap.parse_args()
    os.chdir(ROOT)
    spec = json.load(open(a.tasks, encoding="utf-8"))
    split = lambda v: [x for part in (v or []) for x in part.replace("，", ",").split(",") if x.strip()]
    a.task, a.difficulty = [x.strip() for x in split(a.task)], [x.strip() for x in split(a.difficulty)]
    unknown = [x for x in a.task if x not in {t["id"] for t in spec["tasks"]}]
    if unknown:
        print("!! 沒有這個題號：%s（可用的：%s）" % (", ".join(unknown), ", ".join(t["id"] for t in spec["tasks"])))
        return 1
    tasks = [t for t in spec["tasks"] if (not a.task or t["id"] in a.task) and (not a.difficulty or t["difficulty"] in a.difficulty)]
    if a.task:                                         # 照你打的順序跑
        tasks.sort(key=lambda t: a.task.index(t["id"]))
    print("場景：%s" % spec["scene"])
    for t in tasks:
        print("  %-5s %-7s %s" % (t["id"], t["difficulty"], t["description"]))
    if a.list or not tasks:
        return 0
    st = state(a.robot_url)
    if st is None:
        print("!! 連不到 %s：環境還沒開好" % a.robot_url); return 1
    if "real_tof_front_mm" not in st:
        print("!! %s 沒有實體的資料（不是鏡射模式）：現在跑只會動模擬裡的車。先把鏡射環境開好。" % a.robot_url); return 1
    out = os.path.join("robot_eval", "results", "office_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    rows = []
    # /live 下方的成績表讀的是「最新的、有 driver.log 的批次資料夾」：照 run_batch 的格式寫，辦公室題組的進度與結果就會顯示在那裡
    json.dump([{"id": t["id"], "difficulty": t["difficulty"]} for t in tasks], open(os.path.join(out, "batch_tasks.json"), "w", encoding="utf-8"))

    def dlog(msg):
        with open(os.path.join(out, "driver.log"), "a", encoding="utf-8") as f:
            f.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    dlog("辦公室實體題組 %s：%d 題" % (os.path.basename(out), len(tasks)))
    for i, t in enumerate(tasks, 1):
        print("\n" + "=" * 90)
        print("[%d/%d] %s（%s）\n任務：%s\n預期：%s\n開始前：%s" % (i, len(tasks), t["id"], t["difficulty"], t["description"], t["expected"], t["reset"]))
        ans = input("場景擺好、車在原點了嗎？Enter = 開始，s = 跳過這題，q = 結束：").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            rows.append((t, "skip", 0, "")); continue
        log = os.path.join(out, t["id"] + ".log")
        run_dir = os.path.join(out, "run", t["id"])
        # 每題開始前把模擬環境整理乾淨（使用者 11-16：「模擬環境的可移動障礙物為什麼不會根據每個任務重新更新」）：
        # 1. 上一題實體 ToF 放的橘色方塊全部拆掉（箱子被撞歪、棒子被拿走之後，舊方塊還留在原地）
        # 2. 模擬裡的車跟實體里程計差超過 15 cm（跑一題下來常常漂 30–50 cm，之後放的方塊位置全錯）就整套重開：
        #    使用者剛確認車在原點，重開等於把模擬、實體、地圖一起歸零（約 2 分鐘）
        try:
            req = urllib.request.Request(a.robot_url + "/action", data=json.dumps({"name": "forget_real_obstacles", "params": {}}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            print("(清掉上一題留下的障礙方塊：%s)" % str(json.load(urllib.request.urlopen(req, timeout=40)).get("message", ""))[:80])
        except Exception as e:
            print("(清障礙方塊失敗：%s)" % e)
        s0 = state(a.robot_url) or {}
        sx, rx = s0.get("world_xy"), s0.get("real_world_xy")
        drift = ((sx[0] - rx[0]) ** 2 + (sx[1] - rx[1]) ** 2) ** 0.5 if sx and rx else 0.0
        off = (rx[0] ** 2 + rx[1] ** 2) ** 0.5 if rx else 0.0
        if a.fresh or drift > 0.15 or off > 0.15:
            print("(模擬與實體差 %.2f m、實體里程計離原點 %.2f m：整套重開歸零，約 2 分鐘…)" % (drift, off))
            dlog("重開環境歸零（模擬/實體差 %.2f m）" % drift)
            subprocess.call(["bash", "-c", "./start_office.sh 2>&1 | tail -4"])
        dlog("===== [%d/%d] %s =====" % (i, len(tasks), t["id"]))
        t0 = time.time()
        cmd = "./run_agent.sh --no-judge --max-steps %d --robot-url %s --run-dir %s --task %s 2>&1 | tee %s" % (
            a.max_steps, a.robot_url, run_dir, json.dumps(t["description"], ensure_ascii=False), log)
        try:
            subprocess.call(["bash", "-c", cmd])
        except KeyboardInterrupt:
            print("\n(中斷這一題)")
            try:
                urllib.request.urlopen(urllib.request.Request(a.robot_url + "/stop", data=b"{}", method="POST"), timeout=5)
            except Exception:
                pass
        dur = time.time() - t0
        # 不管這題怎麼結束（成功、放棄、中斷），手臂都收回原位（使用者 11-16：「你每次都需要把手臂回覆到原位吧」）；夾著東西就不動，免得摔掉
        st2 = state(a.robot_url) or {}
        if not st2.get("gripper_holding"):
            try:
                req = urllib.request.Request(a.robot_url + "/action", data=json.dumps({"name": "recenter_arm", "params": {}}).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=40).read()
            except Exception as e:
                print("(手臂收回原位失敗：%s)" % e)
        else:
            print("(夾爪還夾著東西：手臂沒有自動收回，請先處理)")
        print("\n預期：%s" % t["expected"])
        v = ""
        while v not in ("y", "n"):
            v = input("看現場：這題算通過嗎？(y/n)：").strip().lower()
        note = input("備註（可空白，例如「撞到箱子」「沒回原點」）：").strip()
        rows.append((t, "pass" if v == "y" else "fail", dur, note))
        steps = 0
        try:
            steps = len(json.load(open(os.path.join(run_dir, "history.json"), encoding="utf-8")).get("steps", []))
        except Exception:
            pass
        os.makedirs(run_dir, exist_ok=True)
        json.dump({"task": t, "metrics": {"task_id": t["id"], "difficulty": t["difficulty"], "is_successful": v == "y",
                                          "completed_milestones": 1 if v == "y" else 0, "total_milestones": 1, "total_steps": steps,
                                          "total_duration_seconds": dur, "judge_verdict": v == "y", "judge": "human",
                                          "judge_failure_reason": note or ("" if v == "y" else "判定失敗（現場人工）"),
                                          "judge_reasoning": note}},
                  open(os.path.join(run_dir, "evaluation.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        dlog("%s: %s %ds %s" % (t["id"], "通過" if v == "y" else "失敗", dur, note))
        with open(os.path.join(out, "results.tsv"), "w", encoding="utf-8") as f:
            f.write("id\tdifficulty\tresult\tseconds\tnote\n")
            for tt, r, d, n in rows:
                f.write("%s\t%s\t%s\t%d\t%s\n" % (tt["id"], tt["difficulty"], r, d, n))
    dlog("批次結束")
    print("\n" + "=" * 90)
    for tt, r, d, n in rows:
        print("%-5s %-7s %s  %4ds  %s" % (tt["id"], tt["difficulty"], {"pass": "✓", "fail": "✗", "skip": "-"}[r], d, n))
    ran = [r for _, r, _, _ in rows if r != "skip"]
    if ran:
        print("通過 %d / %d；結果在 %s/results.tsv" % (ran.count("pass"), len(ran), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
