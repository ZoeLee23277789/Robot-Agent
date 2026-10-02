#!/usr/bin/env python
"""
整批跑 robot_eval/dataset.json 的評估，每個任務之間把 CoppeliaSim 模擬停止再重播，讓機器人回到起點。

    python -m robot_eval.run_batch                               # 全部任務
    python -m robot_eval.run_batch --difficulty easy             # 只跑某個難度
    python -m robot_eval.run_batch --task e01 --task m02         # 指定任務
    python -m robot_eval.run_batch --report-only robot_eval/results/all_20260925_101500   # 只重新彙整報表

為什麼不直接用 run_evaluation.py 跑全部：sim 模式的 /reset 只會收手臂、開夾爪，底盤位置不會動，
一個任務結束後機器人停在哪裡，下一個任務就從哪裡開始。要真的回到起點只能把模擬停止再播放
（CoppeliaSim 會把場景還原成存檔狀態），而模擬重播後 robot_server.py 跟 SDK 的連線會失效，
所以每個任務前都走一次 ./start_all.sh（它會自己偵測到模擬剛被重播並重開 server），
再用 zmqRemoteApi 讀機器人的真實世界姿態確認真的回到起點，才開始跑任務。

每個任務各自呼叫一次 run_evaluation.py（--no-pause），輸出在 <batch_dir>/<session>/<task_id>/，
console 輸出存成 <batch_dir>/<task_id>.log，流程紀錄在 <batch_dir>/driver.log，
最後把所有 evaluation.json 彙整成 <batch_dir>/report.json。
"""

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER_URL = "http://127.0.0.1:8765"
START_XY = (-1.1, -2.51)   # Env/build_playground_new.py 的 ROBOT_START
START_HEADING_DEG = 0.0    # 起點面向 +x（build_playground_new.ROBOT_HEADING）
POS_TOL_M = 0.15
HEADING_TOL_DEG = 15.0


def log(batch_dir: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(batch_dir / "driver.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


class SimUnreachable(RuntimeError):
    pass


def _sim():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    return RemoteAPIClient().require("sim")


def _with_timeout(fn, seconds: float):
    """zmq 遠端 API 的呼叫在 CoppeliaSim 當掉時會無限期卡住（實測驅動程式因此停了 1 小時 46 分），
    所以所有 zmq 呼叫都丟到 worker thread 並設硬性逾時。"""
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=seconds)
    except concurrent.futures.TimeoutError:
        raise SimUnreachable(f"CoppeliaSim 在 {seconds:.0f}s 內沒有回應 zmq 呼叫")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def wait_for_sim(batch_dir: Path, max_wait_s: float = 20 * 60) -> bool:
    """CoppeliaSim 沒回應（當掉/重開中）時，每 30 秒探測一次、最多等 max_wait_s，讓人有時間重開它。"""
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        try:
            _with_timeout(lambda: _sim().getSimulationState(), 15)
            return True
        except Exception as e:  # noqa: BLE001
            log(batch_dir, f"!! CoppeliaSim 沒有回應（{type(e).__name__}），已等 {time.time() - t0:.0f}s；請確認它還開著、場景有載入")
            time.sleep(30)
    return False


def stop_simulation(batch_dir: Path, timeout: float = 15.0) -> bool:
    sim = _with_timeout(_sim, 20)
    if _with_timeout(sim.getSimulationState, 20) == sim.simulation_stopped:
        log(batch_dir, "模擬本來就是停止狀態")
        return True
    _with_timeout(sim.stopSimulation, 20)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.5)
        if _with_timeout(sim.getSimulationState, 20) == sim.simulation_stopped:
            log(batch_dir, f"模擬已停止 ({time.time() - t0:.1f}s)")
            return True
    log(batch_dir, f"!! 模擬 {timeout:.0f}s 內沒有停下來")
    return False


def find_robot(sim) -> int:
    """跟 robot_server.py 的 _find_robot_root 同一套規則：場景若有多台 RoboMaster，挑有 GyroSensor 的完整模型
    （外掛真正驅動的那台），並警告。"""
    roots = [o for o in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2) if sim.getObjectAlias(o) == "RoboMaster"]
    if not roots:
        raise RuntimeError("找不到 /RoboMaster")
    good = [h for h in roots if any(sim.getObjectAlias(o) == "GyroSensor" for o in sim.getObjectsInTree(h, sim.handle_all, 0))]
    if len(roots) > 1:
        print(f"!! 場景裡有 {len(roots)} 台 RoboMaster 模型 {roots}，完整的是 {good}；請刪掉多餘的並存檔", flush=True)
    return (good or roots)[0]


def robot_pose() -> tuple[float, float, float]:
    """機器人在世界座標的 (x, y, heading_deg)。朝向用車上相機的視線方向判斷，跟 Test_Robot/reset_robot.py 同一套。"""
    return _with_timeout(_robot_pose_inner, 30)


def _robot_pose_inner() -> tuple[float, float, float]:
    sim = _sim()
    h = find_robot(sim)
    p = sim.getObjectPosition(h, sim.handle_world)
    cams = sim.getObjectsInTree(h, sim.object_visionsensor_type, 0)
    if cams:
        m = sim.getObjectMatrix(cams[0], sim.handle_world)
        heading = math.degrees(math.atan2(m[6], m[2]))
    else:
        m = sim.getObjectMatrix(h, sim.handle_world)
        heading = math.degrees(math.atan2(m[4], m[0]))
    return p[0], p[1], heading


def http_json(path: str, method: str = "GET", timeout: float = 3.0) -> dict:
    req = urllib.request.Request(SERVER_URL + path, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def emergency_stop() -> None:
    for method in ("POST", "GET"):
        try:
            http_json("/stop", method=method)
            return
        except Exception:
            continue


def prepare(batch_dir: Path, attempts: int = 4) -> bool:
    """停止模擬 → start_all.sh（重播模擬、重開 server）→ 確認連線正常且機器人真的在起點。"""
    for i in range(1, attempts + 1):
        try:
            stop_simulation(batch_dir)
        except Exception as e:  # noqa: BLE001  CoppeliaSim 沒回應：等它被重開，等不到就放棄這一題
            log(batch_dir, f"!! 停止模擬失敗（{type(e).__name__}: {e}），等待 CoppeliaSim 恢復...")
            if not wait_for_sim(batch_dir):
                return False
            continue
        with open(batch_dir / "start_all.log", "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%H:%M:%S')} start_all.sh (attempt {i}) =====\n")
            f.flush()
            rc = subprocess.call(["./start_all.sh"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        try:
            health = http_json("/health")
        except Exception as e:
            health = {"ok": False, "message": f"{type(e).__name__}: {e}"}
        try:
            x, y, hd = robot_pose()
            pose_ok = (math.hypot(x - START_XY[0], y - START_XY[1]) < POS_TOL_M
                       and abs((hd - START_HEADING_DEG + 180) % 360 - 180) < HEADING_TOL_DEG)
            pose_txt = f"({x:.2f}, {y:.3f}, {hd:.1f})"
        except Exception as e:
            pose_ok, pose_txt = False, f"讀取失敗 {type(e).__name__}: {e}"
        log(batch_dir, f"reset #{i}: start_all rc={rc} | health={health} | 起點 pose={pose_txt}")
        if rc == 0 and health.get("ok") and health.get("robot_link") == "ok" and pose_ok:
            return True
        time.sleep(2 + 10 * i)  # 場景重新載入中的話多等一會兒
    return False


def llm_outage(log_path: Path) -> str:
    """任務 log 裡 LLM 呼叫連續失敗（額度用盡、配額、認證）而且沒有任何一步成功 → 回傳錯誤摘要，否則空字串。"""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    fails = text.count("LLM call failed")
    markers = ("RESOURCE_EXHAUSTED", "credits are depleted", "insufficient_quota", "PERMISSION_DENIED", "API key not valid", "401 ", "402 ")
    hit = next((m for m in markers if m in text), None)
    if fails >= 2 and hit and "✅" not in text and "Step 4" not in text:
        line = next((ln for ln in text.splitlines() if hit in ln), hit)
        return line.strip()[:160]
    return ""


def run_task(batch_dir: Path, task: dict, args) -> tuple:
    cmd = [sys.executable, "-m", "robot_eval.run_evaluation", "--task", task["id"], "--no-pause",
           "--output-dir", str(batch_dir), "--max-steps", str(args.max_steps)]
    for opt in ("provider", "model", "judge_provider", "judge_model", "judge_frames", "notes_mode", "notes_path"):
        if getattr(args, opt) is not None:
            cmd += [f"--{opt.replace('_', '-')}", str(getattr(args, opt))]
    t0 = time.time()
    with open(batch_dir / f"{task['id']}.log", "w", encoding="utf-8") as f:
        try:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=args.task_timeout)
        except subprocess.TimeoutExpired:
            rc = "timeout"
            emergency_stop()
    dur = time.time() - t0
    summary = ""
    for line in (batch_dir / f"{task['id']}.log").read_text(encoding="utf-8", errors="ignore").splitlines():
        if "📊" in line:
            summary = line.strip()
    log(batch_dir, f"{task['id']}: rc={rc} {dur:.0f}s | {summary or '(沒有找到 📊 結果行，看 ' + task['id'] + '.log)'}")
    return rc, dur


def aggregate(batch_dir: Path, tasks: list) -> dict:
    from robot_eval.metrics import TaskMetrics, build_report

    order = {t["id"]: i for i, t in enumerate(tasks)}
    latest: dict[str, TaskMetrics] = {}
    for ev in sorted(batch_dir.glob("*/*/evaluation.json")):       # 路徑含 session 時間戳，排序後最後一個是最新的
        m = json.loads(ev.read_text(encoding="utf-8"))["metrics"]
        latest[m["task_id"]] = TaskMetrics(**m)
    metrics = [latest[tid] for tid in sorted(latest, key=lambda t: order.get(t, 10 ** 6))]
    missing = [t["id"] for t in tasks if t["id"] not in latest]

    report = build_report(metrics)
    report["not_evaluated"] = missing
    for sub in sorted(batch_dir.glob("*/report.json")):
        cfg = json.loads(sub.read_text(encoding="utf-8")).get("config")
        if cfg:
            report["config"] = cfg
            break
    (batch_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * 100}")
    print(f"{'task':<14}{'diff':<8}{'ok':<4}{'milestones':<12}{'steps':<7}{'time':<8}{'overclaim':<11}reason")
    for m in metrics:
        over = "YES" if (m.agent_success and not m.is_successful) else ""
        reason = (m.judge_failure_reason or ("" if m.is_successful else m.judge_reasoning) or "")[:60]
        print(f"{m.task_id:<14}{m.difficulty:<8}{'✓' if m.is_successful else '✗':<4}"
              f"{m.completed_milestones}/{m.total_milestones:<10}{m.total_steps:<7}{m.total_duration_seconds:<8.0f}{over:<11}{reason}")
    o = report["overall"]
    print(f"{'=' * 100}\n📊 {o['tasks']} runs｜成功率 {o['success_rate']:.0%}｜milestone 平均完成 {o['avg_milestone_completion']:.0%}"
          f"｜平均 {o['avg_steps']:.1f} 步 / {o['avg_duration_s']:.0f}s")
    for k, v in report["by_difficulty"].items():
        print(f"   {k:<8} {v['success_rate']:.0%} ({v['tasks']} runs)")
    if report["agent_overclaimed_success"]:
        print(f"   ⚠️  Agent 自評成功但實際失敗：{report['agent_overclaimed_success']}")
    if missing:
        print(f"   ⚠️  沒有評估結果：{missing}")
    print(f"   報告：{batch_dir / 'report.json'}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the whole robot_eval dataset, resetting the simulation between tasks")
    ap.add_argument("--dataset", default="robot_eval/dataset.json")
    ap.add_argument("--task", action="append", help="task id，可以重複給；不給就跑全部")
    ap.add_argument("--difficulty", action="append", help="easy / medium / hard，可以重複給")
    ap.add_argument("--output-dir", default=None, help="預設 robot_eval/results/all_<timestamp>")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--task-timeout", type=int, default=1500, help="單一任務的秒數上限，超過就強制結束、跳下一個")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-frames", type=int, default=None, help="透傳給 run_evaluation 的 --judge-frames")
    ap.add_argument("--notes-mode", choices=["fixed", "accumulate", "off"], default=None,
                   help="透傳給 run_evaluation 的 --notes-mode；不給就用它自己的預設 fixed。"
                        "accumulate 且沒給 --notes-path 時，整個批次共用同一個檔案"
                        "（<batch_dir>/accumulated_notes.md），讓筆記真的跨題累積，不是每個子程序各自的空白檔案。")
    ap.add_argument("--notes-path", default=None, help="透傳給 run_evaluation 的 --notes-path")
    ap.add_argument("--report-only", metavar="BATCH_DIR", help="不跑任務，只重新彙整既有批次的 report.json")
    args = ap.parse_args()

    tasks = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))["tasks"]
    if args.task:
        tasks = [t for t in tasks if t["id"] in args.task]
    if args.difficulty:
        tasks = [t for t in tasks if t.get("difficulty") in args.difficulty]
    if not tasks:
        raise SystemExit("沒有符合條件的任務")

    if args.report_only:
        aggregate(Path(args.report_only), tasks)
        return

    batch_dir = Path(args.output_dir or f"robot_eval/results/all_{time.strftime('%Y%m%d_%H%M%S')}")
    batch_dir = batch_dir if batch_dir.is_absolute() else ROOT / batch_dir
    batch_dir.mkdir(parents=True, exist_ok=True)
    # accumulate 模式每個任務各自是獨立子程序，run_evaluation.py 自己算的路徑（session 底下）每次
    # 呼叫都不一樣，筆記不會真的跨題累積；在這裡把路徑固定成整個批次共用同一份，批次之間因為
    # batch_dir 本身是新的時間戳記資料夾，天然就是「每次重跑都從空白開始」。
    if args.notes_mode == "accumulate" and not args.notes_path:
        args.notes_path = str(batch_dir / "accumulated_notes.md")
    (batch_dir / "driver.pid").write_text(str(os.getpid()))
    log(batch_dir, f"批次 {batch_dir.name}：{len(tasks)} 個任務 → {[t['id'] for t in tasks]}")
    log(batch_dir, f"輸出目錄：{batch_dir}")

    skipped = []
    try:
        for i, task in enumerate(tasks, 1):
            log(batch_dir, f"===== [{i}/{len(tasks)}] {task['id']} =====")
            if not prepare(batch_dir):
                log(batch_dir, f"!! {task['id']}：機器人沒有回到起點或 server 沒連上，跳過")
                skipped.append(task["id"])
                continue
            run_task(batch_dir, task, args)
            outage = llm_outage(batch_dir / f"{task['id']}.log")
            if outage:
                # 難題批次實測：Gemini 預付額度用完後每題 2 秒內連續三次 402，8 題全部「失敗」還有一題被誤判成功。
                # 這種結果沒有意義，直接中止，剩下的任務不算分；額度恢復後用 --task 重跑。
                log(batch_dir, f"!! LLM API 不可用（{outage}），批次中止；{task['id']} 之後的任務不算分，額度/配額恢復後用 --task 重跑")
                break
    except KeyboardInterrupt:
        log(batch_dir, "!! 被中斷，送出停止指令")
        emergency_stop()
    finally:
        emergency_stop()
        if skipped:
            log(batch_dir, f"準備失敗而跳過的任務：{skipped}")
        aggregate(batch_dir, tasks)
        log(batch_dir, "批次結束")


if __name__ == "__main__":
    main()
