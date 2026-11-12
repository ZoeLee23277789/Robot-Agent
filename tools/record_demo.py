"""
錄一段 agent 執行任務的展示影片（1920x1080 mp4），不需要螢幕錄影軟體，無視窗模擬器也能錄。

畫面跟 /live 儀表板一樣：占據格地圖（路徑、鎖定目標框）、俯視鏡頭（轉 180° 跟地圖同方向）、前鏡頭（洋紅框＝鎖定目標）、
遙測，下方是任務與 agent 最新一步的 eval / thinking / goal / 動作結果，任務結束時疊上 SUCCESS / FAILED 橫幅。
資料全部來自 robot_server 的 HTTP 端點，跟 agent 用的是同一份，不會干擾執行。

用法（從 repo 根目錄，模擬器 + server 已經在跑）：
    .venv/bin/python tools/record_demo.py --task "Pick up the lime ball from the table and put it into the blue bin."
    .venv/bin/python tools/record_demo.py --task-id r11          # 直接用 dataset.json 裡的題目
    .venv/bin/python tools/record_demo.py                         # 只錄影，任務你自己在另一個終端機跑；Ctrl+C 結束

選項：
    --no-reset     不先把場景重置回起點（預設會：停止模擬 → start_all.sh 重播，跟跑分批次一樣）
    --speed 3      影片相對實際時間的倍速（模擬器只有約 0.55 倍即時，3 倍看起來接近正常速度）
    --out PATH     輸出檔（預設 videos/<時間>_<題號>.mp4；agent 的終端輸出存在同名 .agent.log）

有 ffmpeg 時錄完會自動轉成 H.264（瀏覽器、簡報、手機都能播）；沒有的話是 mp4v 編碼，只有 VLC 這類播放器能播。
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SERVER = os.getenv("ROBOT_SERVER_URL", "http://127.0.0.1:8765")
W, H = 1920, 1080
BG = (30, 30, 30)
CARD = (42, 42, 42)
ACCENT = (255, 204, 153)    # BGR 淡藍
WHITE = (240, 240, 240)
GREY = (170, 170, 170)


def get(path, timeout=3.0):
    with urllib.request.urlopen(SERVER + path, timeout=timeout) as r:
        return r.read()


def get_img(path):
    try:
        b = get(path)
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def get_json(path):
    try:
        return json.loads(get(path))
    except Exception:
        return None


def fit(img, w, h):
    """等比例縮放後置中貼進 w x h 的底圖。"""
    canvas = np.full((h, w, 3), CARD, np.uint8)
    if img is None:
        cv2.putText(canvas, "no image", (w // 2 - 50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 1, cv2.LINE_AA)
        return canvas
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def put(img, text, org, scale=0.6, color=WHITE, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def card(frame, x, y, w, h, title, content):
    cv2.rectangle(frame, (x, y), (x + w, y + h), CARD, -1)
    put(frame, title, (x + 10, y + 26), 0.62, ACCENT)
    frame[y + 36:y + h - 6, x + 6:x + w - 6] = fit(content, w - 12, h - 42)


def wrap_lines(text, width):
    out = []
    for para in str(text or "").splitlines() or [""]:
        out.extend(textwrap.wrap(para, width) or [""])
    return out


def compose(st, log, imgs, elapsed_s, speed):
    frame = np.full((H, W, 3), BG, np.uint8)
    # 上排：地圖 / 俯視 / 前鏡頭＋遙測
    card(frame, 16, 16, 600, 620, "occupancy map  (green = planned path, magenta = locked target)", imgs.get("map"))
    ov = imgs.get("overhead")
    card(frame, 632, 16, 600, 620, "overhead camera", None if ov is None else cv2.rotate(ov, cv2.ROTATE_180))
    card(frame, 1248, 16, 656, 400, "front camera on the arm  (magenta = locked target)", imgs.get("front"))
    # 遙測
    x, y = 1248, 432
    cv2.rectangle(frame, (x, y), (x + 656, y + 204), CARD, -1)
    put(frame, "telemetry", (x + 10, y + 26), 0.62, ACCENT)
    if st:
        xy = st.get("world_xy") or ["-", "-"]
        rows = [
            ("position", "(%s, %s) m" % tuple(("%.2f" % v) if isinstance(v, (int, float)) else v for v in xy)),
            ("heading", "%s deg" % (("%.0f" % st["world_heading_deg"]) if st.get("world_heading_deg") is not None else "-")),
            ("tof front/left/right", " / ".join(str(st.get(k, "-")) for k in ("tof_front_mm", "tof_left_mm", "tof_right_mm")) + " mm"),
            ("arm / lift", "%s / %s mm" % (st.get("arm_mm"), st.get("lift_mm"))),
            ("gripper load", ("%+.2f N" % (st["gripper_force_n"] - 1.5)) if st.get("gripper_force_n") is not None else "-"),
        ]
        for i, (k, v) in enumerate(rows):
            put(frame, k, (x + 14, y + 60 + i * 28), 0.55, GREY)
            put(frame, str(v), (x + 250, y + 60 + i * 28), 0.55, WHITE)
    # 下排：任務與 agent 思考
    x, y, w, h = 16, 652, 1888, 412
    cv2.rectangle(frame, (x, y), (x + w, y + h), CARD, -1)
    task = (log or {}).get("task") or "(waiting for the agent to start)"
    lines = wrap_lines("TASK: " + task, 120)[:2]
    for i, ln in enumerate(lines):
        put(frame, ln, (x + 14, y + 32 + i * 30), 0.72, WHITE, 2)
    yy = y + 32 + len(lines) * 30 + 10
    steps = (log or {}).get("steps") or []
    if steps:
        s = steps[-1]
        put(frame, "step %s   (%ss thinking)" % (s.get("step"), s.get("llm_seconds")), (x + 14, yy), 0.62, ACCENT)
        yy += 30
        for label, key in (("eval", "evaluation"), ("thinking", "thinking"), ("goal", "next_goal")):
            for j, ln in enumerate(wrap_lines(s.get(key), 150)[:2]):
                put(frame, ("%-9s " % (label + ":") if j == 0 else " " * 10) + ln, (x + 14, yy), 0.56, GREY if label != "goal" else WHITE)
                yy += 26
        for a in (s.get("actions") or [])[:2]:
            mark = "OK " if a.get("ok") else "X  "
            col = (120, 220, 120) if a.get("ok") else (110, 110, 240)
            msg = "%s%s(%s) -> %s" % (mark, a.get("name"), json.dumps(a.get("params"), ensure_ascii=False)[:70], a.get("message") or "")
            for j, ln in enumerate(wrap_lines(msg, 150)[:3]):
                put(frame, ln, (x + 14, yy), 0.56, col)
                yy += 26
    # 時間戳
    put(frame, "t = %d:%02d   (video %.0fx speed)" % (elapsed_s // 60, elapsed_s % 60, speed), (W - 380, H - 18), 0.55, GREY)
    # 結束橫幅
    done = (log or {}).get("done")
    if done:
        ok = bool(done.get("success"))
        col = (60, 170, 60) if ok else (60, 60, 200)
        by = H - 150   # 放在下方任務欄底部，不蓋住地圖和相機
        cv2.rectangle(frame, (16, by), (W - 16, H - 40), col, -1)
        put(frame, "TASK %s" % ("SUCCESS" if ok else "FAILED"), (40, by + 48), 1.3, WHITE, 3)
        for i, ln in enumerate(wrap_lines(done.get("text"), 150)[:2]):
            put(frame, ln, (40, by + 80 + i * 26), 0.6, WHITE, 1)
    return frame


def reset_scene():
    """跟跑分批次一樣：停止模擬 → start_all.sh 重播並重開 server → 確認機器人在起點。"""
    sys.path.insert(0, str(ROOT))
    from robot_eval import run_batch
    tmp = ROOT / "videos" / ".reset_log"
    tmp.mkdir(parents=True, exist_ok=True)
    ok = run_batch.prepare(tmp)
    print("場景重置%s（細節在 %s/driver.log）" % ("完成" if ok else "失敗", tmp))
    return ok


def task_from_dataset(tid):
    d = json.loads((ROOT / "robot_eval" / "dataset.json").read_text(encoding="utf-8"))
    tasks = d["tasks"] if isinstance(d, dict) and "tasks" in d else d
    for t in tasks:
        if t.get("id") == tid:
            return t.get("description") or t.get("task")
    raise SystemExit("dataset.json 裡沒有 %s" % tid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", help="要 agent 執行的任務句子")
    ap.add_argument("--task-id", help="dataset.json 的題號（例如 r11）")
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--interval", type=float, default=0.5, help="擷取間隔（秒，實際時間）")
    ap.add_argument("--tail", type=float, default=6.0, help="任務結束後多錄幾秒（實際時間）")
    ap.add_argument("--max-minutes", type=float, default=20.0)
    ap.add_argument("--out")
    args = ap.parse_args()

    task = task_from_dataset(args.task_id) if args.task_id else args.task
    if task and not args.no_reset:
        if not reset_scene():
            raise SystemExit("場景重置失敗，加 --no-reset 可以略過")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "videos" / ("%s_%s.mp4" % (stamp, args.task_id or ("task" if task else "watch")))
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1.0, args.speed / args.interval)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise SystemExit("無法建立影片檔 %s" % out)

    agent = None
    if task:
        print("啟動 agent：%s" % task)
        agent = subprocess.Popen([str(ROOT / "run_agent.sh"), "--task", task], cwd=ROOT,
                                 stdout=open(out.with_suffix(".agent.log"), "w"), stderr=subprocess.STDOUT)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    t0, done_at, frames, last_start = time.time(), None, 0, None
    cache = {}
    print("錄影中 → %s（%.0f 倍速；Ctrl+C 結束）" % (out, args.speed))
    while not stop["flag"]:
        tick = time.time()
        st = get_json("/state")
        log = get_json("/agent_log")
        if task and log and log.get("task") != task:
            log = {"task": task, "steps": [], "plans": [], "done": None}   # 舊任務的紀錄還沒被新任務清掉
        for k, p in (("front", "/frame_live.jpg"), ("map", "/map.png")):
            img = get_img(p)
            if img is not None:
                cache[k] = img
        if frames % max(1, int(2.0 / args.interval)) == 0 or "overhead" not in cache:   # 俯視圖約 2 秒一張就夠
            img = get_img("/overhead.jpg")
            if img is not None:
                cache["overhead"] = img
        writer.write(compose(st, log, cache, int(tick - t0), args.speed))
        frames += 1
        if log and log.get("done") and done_at is None:
            done_at = time.time()
            print("agent 結束：success=%s" % log["done"].get("success"))
        if done_at and time.time() - done_at >= args.tail:
            break
        if agent and agent.poll() is not None and done_at is None and time.time() - t0 > 10:
            done_at = time.time()   # agent 程序結束但沒送 done（例如當掉）：也收尾
            print("agent 程序已結束（exit %s）" % agent.returncode)
        if time.time() - t0 > args.max_minutes * 60:
            print("超過 --max-minutes，停止錄影")
            break
        time.sleep(max(0.0, args.interval - (time.time() - tick)))
    writer.release()
    if agent and agent.poll() is None:
        agent.terminate()
    print("完成：%s（%d 張畫面，%.0f 秒實際時間 → 影片約 %.0f 秒）" % (out, frames, time.time() - t0, frames / fps))
    # OpenCV 只能寫 mp4v，很多播放器、瀏覽器、簡報軟體不認；有 ffmpeg 就轉成 H.264 取代原檔
    import shutil
    if shutil.which("ffmpeg"):
        tmp = out.with_name(out.stem + ".h264.mp4")
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(tmp)])
        if r.returncode == 0:
            tmp.replace(out)
            print("已轉成 H.264（一般播放器、瀏覽器、簡報都能播）")
        else:
            print("ffmpeg 轉檔失敗，保留 mp4v 原檔")
    else:
        print("沒有 ffmpeg：檔案是 mp4v 編碼，裝 ffmpeg（sudo apt install ffmpeg）後可轉成 H.264")


if __name__ == "__main__":
    main()
