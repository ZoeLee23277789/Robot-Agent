# RoboMaster LLM Agent

用 LLM 當 DJI RoboMaster EP 的高階控制器。這個 repo 是從 Adobe Express GUI Agent 精簡而來：保留它的「觀察、思考、行動」迴圈、LLM 層 (`browser_use/llm`) 和 milestone 式評估，把觀察來源換成相機與遙測，動作空間換成底盤、手臂、夾爪。所有瀏覽器與 Adobe 專屬的程式碼都已移除。

## 為什麼拆成兩個程序

RoboMaster SDK 只支援 Python 3.6 到 3.8，這個 repo 的 LLM 層需要 3.11 以上，兩者無法共存於同一個環境。所以機器人端跑一個輕量的 `robot_server.py`，Agent 端用 HTTP 呼叫它。機器人放在遠端時，架構完全不用改。

```
[ Agent 端  Python 3.11+ ]                      [ 機器人端  rmenv, Python 3.8 ]
RobotAgent.py                                   robot_server.py
  robot_agent/service.py   觀察/思考/行動迴圈       安全限幅、動作鎖、緊急停止
  robot_agent/views.py     輸出 schema            rm_connection.py   (你原本的檔案，不用改)
  robot_agent/client.py  ── HTTP (JSON + JPEG) ─▶  patch_ftp.py       (你原本的檔案)
  browser_use/llm/*        原本的 LLM 層              └─ RoboMaster SDK ─▶ CoppeliaSim 或實體 EP
```

## 檔案對照

| 原本 (Adobe Express) | 現在 (Robot) |
| --- | --- |
| `Agent.py` | `RobotAgent.py` |
| `browser_use.Agent` | `robot_agent/service.py` 的 `RobotAgent` |
| browser state (DOM + screenshot) | `/state` 遙測 JSON 加 `/frame.jpg` 相機畫面 |
| `Tools` 與 `@tools.action` | `robot_server.py` 的 `_act_*` 方法 |
| `ask_human_to_pick` (HITL) | `ask_human` 動作，加上 real 模式的逐步確認 |
| `AgentHistoryList` | `runs/<timestamp>/history.json` 與每一步的 `step_XXX.jpg` |
| `evaluation/` | `robot_eval/` |

## 執行步驟

1. Agent 端安裝相依套件：`pip install -r requirements.txt`，再把 `.env.example` 複製成 `.env` 填入 API key。機器人端把你的 `patch_ftp.py` 放到 `robot_server.py` 旁邊 (這個 repo 沒有附，因為 `rm_connection.py` 會 import 它)。
2. 啟動 CoppeliaSim 並開始模擬，然後在 rmenv 裡啟動 server：
   ```bash
   source ~/rmenv/bin/activate
   cd <這個 repo>
   python robot_server.py --mode sim
   ```
3. 另開一個終端機，在 Python 3.11+ 的環境裡執行 Agent：
   ```bash
   python RobotAgent.py --provider openai --task "Rotate to find the red box, then drive up to it and stop about 30 cm away"
   python RobotAgent.py --loop        # 互動模式
   ```

沒有模擬器也想先測 Agent 迴圈的話，用 `python robot_server.py --mode mock`，它會模擬一台 2D 機器人和一個紅色目標，不需要安裝 SDK。

## 遠端機器人

Server 預設只聽 127.0.0.1。要從另一台電腦控制時，建議用 SSH tunnel 或 Tailscale，不要直接把 port 開到公網：

```bash
# 在 Agent 這台電腦上
ssh -N -L 8765:127.0.0.1:8765 user@robot-host
python RobotAgent.py --task "..."                      # 照樣連 127.0.0.1:8765
```

如果一定要讓 server 聽在區網上，請加 token：`python robot_server.py --mode real --host 0.0.0.0 --token <secret>`，Agent 端設定環境變數 `ROBOT_SERVER_TOKEN` 或用 `--robot-token`。

## 安全設計

所有數值在 server 端限幅 (單次最多 1 m、180 度、手臂 100 mm，速度 0.3 m/s)，LLM 給再誇張的數字也一樣。底盤預設用 SDK 的位置控制 `chassis.move`，位移量由機器人自己跑完，Agent 端中途斷線也不會一直往前衝。一次只執行一個動作，`/stop` 不受鎖限制。Agent 結束、出例外或被 Ctrl+C 時都會先送 stop。real 模式預設每一步實體動作前要人按 Enter，可用 `--no-confirm` 關掉。

## 第一次接上模擬器時請先確認的兩件事

1. 方向符號。SDK 裡 `chassis.move` 的 z 為正是左轉，`drive_speed` 的 z 為正是右轉，server 已經依官方範例處理好。請先下一個任務 "turn left 90 degrees then stop" 用眼睛確認方向正確。
2. 模擬器是否支援 `chassis.move`。如果 server 印出 translation timed out，代表模擬器沒有實作位置控制，改用 `--chassis-backend speed`，它會用跟 `test_movement.py` 相同的 `drive_speed` 加計時方式。

## 評估 (checkpoint / milestone)

```bash
python -m robot_eval.run_evaluation                          # 跑 robot_eval/dataset.json 的全部任務
python -m robot_eval.run_evaluation --task nav_001 --repeat 5    # 同一個任務重複五次，看成功率的變異
python -m robot_eval.run_evaluation --rejudge robot_eval/results/<session>/nav_002 --task nav_002   # 不動機器人，重評舊紀錄
```

Dataset 的格式跟原本一樣 (id, difficulty, category, description, milestones)。機器人版多了一個能力：milestone 可以加上 `telemetry` 條件，直接用遙測數值判定，例如 `{"field": "yaw_deg", "abs_min": 75, "abs_max": 105, "when": "final"}`。支援 `equals`、`in`、`min`/`max`、`abs_min`/`abs_max`，`when` 可以是 `final` (只看結束狀態) 或 `any` (任何一步成立即可)。如果機器人沒有回報那個欄位，會自動退回給 judge LLM 判。沒有寫 `telemetry` 的 milestone 和整體成敗，由 judge LLM 看取樣的相機畫面加上動作紀錄一次判完。

成功判準沿用原本的設計：judge 的裁決優先，judge 失敗時才退回「milestone 全過」。報告另外列出 `agent_overclaimed_success`，也就是 Agent 自評成功但實際失敗的任務，以及 `total_distance_m` 供比較路徑效率。評估期間 `ask_human` 會自動回覆「沒有人可以問」，並且關閉逐步確認，所以用實體機器人跑評估時請留在旁邊，隨時可以對 server 按 Ctrl+C。

任務之間，mock 模式會自動重置；sim 和 real 模式只會收手臂、開夾爪，然後停下來等你把場景擺回原位再按 Enter。

## 新增動作

在 `robot_server.py` 的 backend 加一個 `_act_<name>` 方法，在 `robot_agent/views.py` 的 `RobotAction` 加一個同名欄位與參數 model，最後在 `robot_agent/system_prompt.md` 的 `<actions>` 補一行說明。

## 目錄

```
RobotAgent.py            Agent 進入點 (Python 3.11+)
robot_agent/             agent loop、輸出 schema、HTTP client、system prompt、LLM 工廠
robot_eval/              dataset、telemetry 與 LLM 判定、指標、評估 CLI
browser_use/             精簡後只剩 LLM 層：openai、anthropic、google、ollama
robot_server.py          機器人端的 HTTP 服務 (rmenv, Python 3.8)
rm_connection.py         你的連線模組，sim 與 real 的切換只在這裡
test_movement.py         你的 SDK 煙霧測試，懷疑連線有問題時先跑它
```
