# RoboMaster LLM Agent

用 LLM 當 DJI RoboMaster EP 的高階控制器。Agent 端跑的是真的 `browser_use.Agent`（整包 vendor 進 `browser_use/`），把它原本操作瀏覽器的「觀察、思考、行動」迴圈接到機器人身上：觀察來源換成前方相機 + 固定俯視相機 + 遙測，動作空間換成底盤、手臂、夾爪。`robot_eval/` 保留了原本 milestone 式的評估框架，另外加上遙測條件判定。

## 為什麼拆成兩個程序

RoboMaster SDK 只支援 Python 3.6 到 3.8，這個 repo 的 LLM 層需要 3.11 以上，兩者無法共存於同一個環境。所以機器人端跑一個輕量的 `robot_server.py`（在 `rmenv`），Agent 端用 HTTP 呼叫它（在 `.venv`）。機器人放在遠端時，架構完全不用改。

```
[ Agent 端  .venv, Python 3.11+ ]                [ 機器人端  rmenv, Python 3.6-3.8, 檔案在 Test_Robot/ ]
RobotAgentBU.py  (真的 browser_use.Agent)         robot_server.py
  robot_agent/bu_session.py   假裝成 BrowserSession   安全限幅、動作鎖、緊急停止
  robot_agent/bu_tools.py     動作定義 (Tools())      距離感測器 / 俯視攝影機 / 靜態地標
  robot_agent/bu_common.py    world_state / plan hook  rm_connection.py  (你原本的檔案，不用改)
  robot_agent/perception.py   locate/align_to_tunnel   patch_ftp.py      (你原本的檔案)
  robot_agent/client.py  ── HTTP (JSON + JPEG) ──▶      └─ RoboMaster SDK ─▶ CoppeliaSim 或實體 EP
  browser_use/                 完整 vendor 進來的套件
```

`RobotAgent.py` + `robot_agent/service.py` 是更早期自己重寫的主迴圈，還留著可以對照，但目前所有腳本 (`run_agent.sh`/`run_task.sh`) 都已經切換成 `RobotAgentBU.py`，跑的是真的 `browser_use.Agent`。

## 執行步驟

1. Agent 端安裝相依套件：`pip install -r requirements.txt`，再把 `.env.example` 複製成 `.env` 填入 API key。機器人端的程式都在 `Test_Robot/`（`robot_server.py`、它 import 的 `rm_connection.py` 與 `patch_ftp.py`），在 `rmenv` 裡只需要 `robomaster` 和 `opencv-python`。
2. 一個指令把 CoppeliaSim + `robot_server.py` + 距離感測器都準備好：
   ```bash
   ./start_all.sh              # 一般啟動；已經在跑的 robot_server.py 會直接沿用
   ./start_all.sh --restart    # 改過 robot_server.py 的程式碼、或想清掉卡死狀態時，強制重開
   ```
   這一步只重開「控制程式」；如果機器人在模擬裡卡進了什麼東西（例如卡在牆角/球池邊界），要另外在 CoppeliaSim 裡把模擬停止再重新播放一次才會真的重置姿態。
3. 跑任務：
   ```bash
   ./run_task.sh "Rotate to find the red box, then drive up to it and stop about 30 cm away"   # 準備+執行一次到底
   ./run_agent.sh --task "..."      # 前提是 start_all.sh 已經跑過
   ./run_agent.sh --loop            # 互動模式，一直問你下一個任務是什麼
   ```

沒有模擬器也想先測 Agent 迴圈的話，用 `./start_server.sh mock`（等同 `python Test_Robot/robot_server.py --mode mock`），它會模擬一台 2D 機器人和一個紅色目標，不需要安裝 SDK。

## 場景 / 感知輔助腳本（`Env/`）

- `Env/build_playground_new.py`：在 CoppeliaSim 裡建一個「室內兒童遊戲場」場景（球池、隧道、平台+溜滑梯、四根角落柱子、收納箱、可拾取小積木…），機器人固定從 `ROBOT_START = (-0.6, -2.3)` 出發。同目錄的 `build_playground.py`／`build_playroom.py`／`build_office.py`／`build_task_arena.py` 是其他幾種場景。
- `Env/add_overhead_camera.py`：加一台固定在天花板、往下看的正交攝影機，`robot_server.py` 開機時會自動連上，提供 `/overhead.jpg`、`world_xy` 遙測跟 `static_landmarks`。
- `Env/add_proximity_sensors.py`：獨立腳本仍然可用，但**不再需要手動執行**——`robot_server.py` 開機時會自己建立 `prox_front`/`prox_left`/`prox_right` 三個感測器，讀取失敗（例如模擬被重啟清掉了物件）也會自動偵測並重建。

## 機器人端提供的觀察

`/state` 回傳的遙測 JSON，重點欄位：

| 欄位 | 說明 |
| --- | --- |
| `position_m`, `yaw_deg` | RoboMaster SDK 自己的姿態，相對 session 開始時的座標，原點任意、會drift。 |
| `world_xy` | 從俯視攝影機的 zmqRemoteApi 連線查來的**真實**世界座標。跟 `yaw_deg` 不是同一個座標系（y 軸相反），`perception.py`/`robot_server.py` 內部換算時都已經校正過。 |
| `tof_front_mm` / `tof_left_mm` / `tof_right_mm` | 真的距離感測器（CoppeliaSim proximity sensor，繞過永遠回報 0 的 SDK ToF），9999 代表沒偵測到東西。 |
| `camera` | 前方相機**目前**是否真的讀得到新畫面（不是只看有沒有啟動成功——SDK 的 H.264 解碼 thread 可能中途掛掉，這個欄位會誠實反映）。 |
| `static_landmarks` | 場景裡固定不會動的結構（隧道、球池、四根柱子、平台、長椅、兩個收納箱、樓梯、溜滑梯），即時算好的 `{distance_m, turn_left_deg}`，公式跟 `locate_overhead` 一樣，不用呼叫視覺動作就能拿到。 |

前鏡頭 (`/frame.jpg`) 和俯視鏡頭 (`/overhead.jpg`) 會被合成成一張左右並排、各自標好文字的圖 (`FRONT camera` | `OVERHEAD camera`) 送給 LLM 當 screenshot。

## 動作

| 動作 | 用途 |
| --- | --- |
| `move_chassis(forward_m, right_m, turn_left_deg)` | 相對移動，`+turn_left_deg` = 逆時針/左轉。平移先做、旋轉後做，單次上限 1m/180 度。 |
| `move_arm` / `arm_to` / `recenter_arm` | 手臂相對移動 / 絕對姿態 / 回預設姿態。相機掛在手臂上，動手臂也會改變視角。 |
| `gripper(state)` | 開或關夾爪。 |
| `locate(object)` | 只看前鏡頭，回報物體在畫面上的方位角跟該轉幾度置中，不移動機器人。 |
| `face(object)` | 找到物體並自動轉向對準它（最多修正兩次）。 |
| `locate_overhead(object)` | 用固定俯視相機找物體，回報真實世界距離跟該轉幾度面向它；有深度感測器時還會判斷候選物是不是貼地的假目標（地墊/標記）。 |
| `align_to_tunnel(object)` | 專門給隧道/通道這類「有兩端、可以穿過去」的長條結構：找出兩端座標、算出貫通軸線的朝向，並給出應該站的位置——只是定位，不會自動移動，需要 agent 自己接著執行建議的兩步（先開過去、再轉到軸線方向）才算真的對齊。 |
| `remember(fact)` | 把關於機器人本體/感測器的一般性事實存起來，之後每次任務的 system prompt 都會帶上（`<learned_notes>`）。 |
| `ask_human(question)` | 任務不明確、卡住、或動作有風險時詢問人類，逾時 180 秒沒回應會被中斷該步驟。 |
| `stop()` | 立刻停止底盤，不受動作鎖限制。 |

## 遠端機器人

Server 預設只聽 127.0.0.1。要從另一台電腦控制時，建議用 SSH tunnel 或 Tailscale，不要直接把 port 開到公網：

```bash
# 在 Agent 這台電腦上
ssh -N -L 8765:127.0.0.1:8765 user@robot-host
./run_agent.sh --task "..."                      # 照樣連 127.0.0.1:8765
```

如果一定要讓 server 聽在區網上，請加 token：`./start_server.sh real --host 0.0.0.0 --token <secret>`，Agent 端設定環境變數 `ROBOT_SERVER_TOKEN` 或用 `--robot-token`。

## 安全設計

所有數值在 server 端限幅 (單次最多 1 m、180 度、手臂 100 mm)。底盤預設用 `chassis_backend=hybrid`：平移用 SDK 位置控制 `chassis.move`，旋轉用手動輪詢 yaw 的速度控制，兩者都會事後比對里程計，SDK 回報成功但實際沒怎麼動的話會被判定失敗（`rejected after the fact`），而不是照單全收。一次只執行一個動作，`/stop` 不受動作鎖限制。連續多次偵測到「完全沒動、旁邊又沒東西擋」會觸發 `_auto_recover()`：自動停止模擬、重新播放、重連 SDK，救回控制通道卡死的情況（代價是姿態會跳回上次存檔位置）。

**browser_use 的對外連線全部關閉。** vendor 進來的 `browser_use.Agent` 預設會做三件對外的事：posthog 匿名遙測（`ANONYMIZED_TELEMETRY`）、cloud sync（`BROWSER_USE_CLOUD_SYNC`，預設跟遙測一起開）、每次啟動去 PyPI 查新版（`BROWSER_USE_VERSION_CHECK`，這個開關是我們在 `browser_use/utils.py` 加的）。`RobotAgentBU.py` 在 import browser_use 之前就把三個都設成 `false`，`.env.example` 也列出來了；整個系統唯一會對外的連線只剩你選的 LLM provider API。

## 已知狀況 / 排查紀錄

這些是實際跑分中發現、值得知道的行為，不是理論上的邊界情況：

- **前鏡頭方向**：`robot_server.py` 的 `frame_jpeg()` 已經加上水平翻轉修正過去左右相反的問題（實測：送一個遙測證實為真的「左轉」，畫面卻往左滑而不是往右滑）。第一次接上新的模擬器/場景時，建議還是先下一個「turn left 90 degrees then stop」之類的任務，親眼確認方向正確。
- **感測器/攝影機物件會被模擬重啟清掉**：`sim.createProximitySensor()` 建立的距離感測器是「模擬執行期間」才存在的物件，模擬只要被停止過一次就會消失。`robot_server.py` 現在會自動偵測讀取失敗並重建，`state()` 的 `camera` 欄位也會誠實反映前鏡頭是否還在更新，不會停在最後一次成功值就不動了。
- **球池邊界的直角容易讓機器人物理卡死**：出生點在球池旁邊，原地旋轉時偶爾會卡進圍牆的直角，此時旋轉會回報「成功」但實際角度沒變。`_auto_recover()` 通常能自動救回，但這是已知、尚未從場景幾何上根治的問題。
- **同一個顏色可能對應到不只一個物件**：例如「red box」在目前的場景裡同時可能指向可以被夾爪抓取的小積木，也可能指向純裝飾用的大型泡棉方塊；純導航類任務（開過去、看一眼）通常兩個都能接受，但抓取類任務如果認錯，不管導航修得多好都不可能成功。懷疑某個任務一直失敗時，先確認 `locate`/`locate_overhead` 找到的座標到底對應哪個物件，再往下查。
- **`align_to_tunnel`／`locate_overhead` 的結果只是定位，不是移動**：它們的回傳文字裡會明確要求「不要現在就下結論、先執行建議的移動」，但目前的 LLM (`gemini-robotics-er-2-preview`) 偶爾還是會跳過移動直接用文字描述下結論。這是提示詞層級的軟性要求，還沒有做成程式碼層級的強制檢查。

## 評估 (checkpoint / milestone)

```bash
python -m robot_eval.run_evaluation                              # 跑 robot_eval/dataset.json 的全部任務
python -m robot_eval.run_evaluation --task nav_001 --repeat 5     # 同一個任務重複五次，看成功率的變異
python -m robot_eval.run_evaluation --rejudge robot_eval/results/<session>/nav_002 --task nav_002   # 不動機器人，重評舊紀錄
```

Dataset 格式：`id, difficulty, category, description, milestones`。milestone 可以加上 `telemetry` 條件，直接用遙測數值判定，例如 `{"field": "yaw_deg", "abs_min": 75, "abs_max": 105, "when": "final"}`，支援 `equals`、`in`、`min`/`max`、`abs_min`/`abs_max`，`when` 可以是 `final`（只看結束狀態）或 `any`（任何一步成立即可）。沒有回報那個遙測欄位、或沒寫 `telemetry` 的 milestone，會退回給 judge LLM 看取樣的相機畫面加動作紀錄判定。

成功判準：judge 的裁決優先，judge 失敗時才退回「milestone 全過」。報告另外列出 `agent_overclaimed_success`（Agent 自評成功但實際失敗的任務）和 `total_distance_m`。評估期間 `ask_human` 會自動回覆「沒有人可以問」並關閉逐步確認，用實體機器人跑評估時請留在旁邊。`robot_eval/results/` 不進版控（跟 `runs/` 一樣是每次跑分留下的截圖/紀錄）。

任務之間，mock 模式會自動重置；sim 和 real 模式只會收手臂、開夾爪，然後停下來等你把場景擺回原位再按 Enter。

## 新增動作

在 `robot_agent/bu_tools.py` 用 `@tools.action(...)` 加一個新的 async 函式（需要真的移動/查詢機器人的話透過 `_robot`/`_run()` 呼叫 `robot_server.py`），對應的執行邏輯放在 `robot_server.py` 的 backend 加一個方法，最後在 `robot_agent/system_prompt.md` 的 `<actions>` 補一行說明。純感知類動作（不移動機器人）可以參考 `locate`/`locate_overhead`/`align_to_tunnel` 在 `robot_agent/perception.py` 的寫法。

## 目錄

```
RobotAgentBU.py           目前使用中的 Agent 進入點：真的 browser_use.Agent (.venv, Python 3.11+)
RobotAgent.py             更早期自己重寫的主迴圈，保留做對照，不再是預設路徑
robot_agent/
  bu_session.py             假裝成 browser_use 的 BrowserSession，負責合成雙攝影機截圖
  bu_tools.py                動作定義 (真的 Tools()/Registry)
  bu_common.py                world_state/plan/長期摘要的 hook
  perception.py             locate/face/locate_overhead/align_to_tunnel 的視覺+幾何運算
  client.py                 HTTP client
  system_prompt.md          system prompt
  service.py / views.py      更早期自己重寫版本用的觀察/思考/行動迴圈與輸出 schema
robot_eval/                dataset、telemetry 與 LLM 判定、指標、評估 CLI
browser_use/                完整 vendor 進來的 browser_use 套件（llm、agent、tools、screenshots…）
Test_Robot/                機器人端 (rmenv, Python 3.6-3.8)
  robot_server.py            機器人端的 HTTP 服務
  rm_connection.py           你的連線模組，sim 與 real 的切換只在這裡
  patch_ftp.py               模擬器沒有 FTP，讓 SDK 不會卡在等待的補丁 (rm_connection 會 import)
  test_movement.py           SDK 煙霧測試，懷疑連線有問題時先跑它
  camera_check.py / diag_physics.py / reset_robot.py   相機檢查、物理診斷、手動重置姿態的小工具
Env/                       CoppeliaSim 場景與感測器建置腳本 (.venv 執行)
  build_playground_new.py    目前使用的場景（球池/隧道/平台/柱子…）；build_playground.py 等是其他場景
  add_overhead_camera.py     加俯視攝影機
  add_proximity_sensors.py   獨立的距離感測器建立腳本 (robot_server.py 開機會自動做，通常不用手動跑)
start_all.sh / run_task.sh / run_agent.sh / start_server.sh   啟動腳本（都從 repo 根目錄執行）
```
