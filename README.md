# RoboMaster LLM Agent

用 LLM 當 DJI RoboMaster EP 的高階控制器。觀察來源是前方相機 + 固定俯視相機 + 遙測，動作空間是底盤、手臂、夾爪，加上導航／抓取這些一次到位的技能。`RobotAgent.py` → `robot_agent/service.py` 是整個 Agent 端唯一的「觀察、思考、行動」迴圈，自己寫的，不依賴外部 agent 框架；跑分（`robot_eval/`）用的也是這條。

`robot_eval/` 是 milestone 式的評估框架，加上遙測條件判定與 LLM judge。

曾經有第二條改寫自 browser-use 框架的迴圈（`RobotAgentCore.py`），功能跟自己的迴圈重疊（兩邊都有 planner、長期摘要），卻多背了一萬多行第三方程式碼和 7 個額外套件、也從沒進過 `robot_eval` 的跑分驗證，已經拿掉；裡面兩個真的有價值的東西——真實 token/花費統計、完成後用取樣截圖自我查核——已經搬進自己的迴圈，見下面「自我查核與用量統計」。

## 為什麼拆成兩個程序

RoboMaster SDK 只支援 Python 3.6 到 3.8，這個 repo 的 LLM 層需要 3.11 以上，兩者無法共存於同一個環境。所以機器人端跑一個輕量的 `robot_server.py`（在 `rmenv`），Agent 端用 HTTP 呼叫它（在 `.venv`）。機器人放在遠端時，架構完全不用改。

```
[ Agent 端  .venv, Python 3.11+ ]                      [ 機器人端  rmenv, Python 3.6-3.8, 檔案在 Test_Robot/ ]
RobotAgent.py      進入點 (--task / --loop)              robot_server.py
  robot_agent/service.py     觀察→思考→行動、planner、長期摘要、自我查核、token 統計   安全限幅、動作鎖、緊急停止、ToF 防撞
  robot_agent/actions.py     動作註冊表 (參數 schema + handler)                     距離感測器 / 俯視攝影機 / 靜態地標
  robot_agent/skills.py      go_to/approach/drive_through/pick/place  nav_map.py   占據格地圖 + A*（navigate_to）
  robot_agent/perception.py  locate/face/locate_overhead/align_to_tunnel  rm_connection.py / patch_ftp.py
  robot_agent/llm/           LLM 的呼叫包裝（openai/google；其他供應商已拿掉，見 llm/__init__.py 的註解）
  robot_agent/client.py  ── HTTP (JSON + JPEG) ──▶            └─ RoboMaster SDK ─▶ CoppeliaSim 或實體 EP
```

`robot_agent/llm/`（LLM 呼叫包裝與訊息型別）改寫自 browser-use 專案（MIT，見 `THIRD_PARTY_NOTICES.md`）；其他都是本專案自己的程式碼。除了你設定的 LLM API 之外不會連任何外部服務。

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
   ./run_agent.sh --task "..." --no-judge   # 關掉完成後的自我查核（見下）
   ```

沒有模擬器也想先測 Agent 迴圈的話，用 `./start_server.sh mock`（等同 `python Test_Robot/robot_server.py --mode mock`），它會模擬一台 2D 機器人和一個紅色目標，不需要安裝 SDK。

## 場景 / 感知輔助腳本（`Env/`）

- `Env/build_playground_new.py`：在 CoppeliaSim 裡建一個「室內兒童遊戲場」場景（球池、隧道、平台+溜滑梯、四根角落柱子、收納箱、可拾取小積木…），機器人固定從 `ROBOT_START = (-1.1, -2.51)`（靠南牆入口，面向 +x）出發。同目錄的 `build_playground.py`／`build_playroom.py`／`build_office.py`／`build_task_arena.py` 是其他幾種場景。
- `Env/add_overhead_camera.py`：加一台固定在天花板、往下看的正交攝影機，`robot_server.py` 開機時會自動連上，提供 `/overhead.jpg`、`world_xy` 遙測跟 `static_landmarks`。
- `Env/add_proximity_sensors.py`：獨立腳本仍然可用，但**不再需要手動執行**——`robot_server.py` 開機時會自己建立 `prox_front`/`prox_left`/`prox_right` 三個感測器，讀取失敗（例如模擬被重啟清掉了物件）也會自動偵測並重建。

## 導航：地圖與技能

`robot_server.py` 開機時會用 zmqRemoteApi 讀場景幾何建一張占據格地圖（`Test_Robot/nav_map.py`，5 cm 格、±3.5 m）：
靜態且可碰撞的形狀投影到地面，頂面低於 3 cm 的地墊可以壓過、底面高於 30 cm 的平台甲板／隧道頂可以鑽過，
再以 0.22 m 膨脹（5 格；車身半對角線 0.20 m，膨脹只留 4 格時經過凸角或小球旁會被角落擦到）；隧道這種有頂的通道會把內部
重新標成可走並記下軸向。動態物件（積木、球、泡棉）每次規劃前重讀。正常膨脹層找不到路時會用 0.15 m 的「窄縫層」再試一次
（正常層會擋的格子成本加重，只有必要的縫才貼近），走這種路徑在淨空不足 0.30 m 的路段改小步慢走、轉向容忍 3°——
pillar_green 旁泡棉方塊與柱子之間 0.41 m 的縫就是靠這個過的。大角度原地轉身前會先看掃掠淨空，不夠就先直線挪開。
`GET /map.png` 可以看目前的地圖、最後一次規劃的路徑與機器人位置。

伺服器端動作 `navigate_to(x, y, stop_m, face)` 用 A* 規劃、以真實世界姿態（`world_xy` + 由四個輪子算出的
`world_heading_deg`，不用手臂上的相機——手臂抬高時相機視線會翻轉）閉迴路執行，前方 ToF 看到東西就標記重規劃；
在通道裡改用「通道模式」：只沿軸線走、目標在後方就倒車出去、不朝路徑點迴轉，進入通道口窄道時側向偏差超過 5 cm
會先走一段小折線回到中線。每條通道的「中線」不一定是幾何軸線：`compose()` 會在 ±12 cm 內挑整條線離最近障礙最遠的那條
（隧道北口旁邊就是 `bin_balls`，走正中線車身右緣正好碰到桶），窄道也只在離其他障礙夠遠的格子挖通。
`drive_through(name)` 用結構的長軸算出入口與出口（入口站位在窄道外 0.65 m，對齊轉身時掃掠圈才不會碰到旁邊的東西），
穿越後用行進軌跡判定是否真的從內部通過（不是只看終點）。

Agent 端的 `robot_agent/skills.py` 把這些包成 `go_to`／`approach`／`drive_through`／`pick`／`place` 五個技能（由
`service.py` 的動作處理器呼叫）：目標名稱解析成世界座標（地標查表——含三張地墊、其他用俯視相機 pointing），`approach` 抵達後再用前鏡頭精對準
（地標用人話描述辨識、辨識不到不算失敗，因為位置本來就精確已知），最後用前方距離感測器把車頭到目標的距離修到 `stop_m`。
這些技能一次動作到位，LLM 不需要再一步一步下 `move_chassis`，也不需要自己拼湊抓取流程。

**抓取與放置的設計原則**：每一步都有物理驗證，不靠 LLM 看圖猜。`pick` 用距離感測器判斷物件是否在指間、關夾爪後倒退 0.25 m
比對感測器讀值判斷物件有沒有跟著車走（夾爪狀態在模擬裡關上 0.6 秒後就回到 normal，不能拿來判斷）；成功後伺服器記住手上物件
的感測器讀值，導航時忽略它（不然夾著積木走，前方感測器一直讀到 6 cm，會一路標障礙重規劃到無路可走）。實測的物理限制：
夾爪的手指是兩片薄板，手臂放到最低時也只佔離地 0.136-0.152 m（抓取姿態 (180,30)），15 cm 高的積木夾的是頂端那一段；
放在地上的球不管多小都夾不到。所以場景裡的球（橘、紫、粉紅、青四顆）是 6 cm 純球體，放在 0.114 m 高的柱子上，球心對準手指板
中央 0.144 m（`Env/add_graspable_balls.py`）。對球，`pick` 的流程是：底盤送到感測器讀柱子 ≤ 125 mm（底盤一步最少 2 cm，停下的讀值會落在
111-119 mm 之間），再用手臂往前伸補到「等效 110 mm」（實測關夾時讀 111-112 mm 夾得起來、117-119 mm 爪子會往上滑過球）；
放低手臂後先用目標顏色最大的那一塊像素把車頭轉正（柱子只有 3 cm 粗，偏一點感測器就讀不到），送到約 200 mm 再量一次左右偏差，偏 8 mm 以上就轉 2°（約 13 mm）；關上後等 1.5 s 才抬。球抬起後離開感測器光束，所以用手臂相機
驗證：夾住的東西會一直卡在畫面底部的夾爪框裡，倒退後框裡有顏色 ≥ 5% 算夾住，移到搬運姿態後再看一次（真夾住 0.25、球在途中掉了 0.00），
避免「球被夾在抓取高度拖著走、其實沒抬起來」的誤判。2026-10-08 最後一輪四顆各夾一次，夾起 3 顆，回報全部跟模擬器真值一致；
已知還會失敗的情況：從東側接近紫色球時左右偏差 5-7 mm（比轉 2° 能修的還小）會夾不牢，而球掉到地上之後就夾不到了
（`pick` 會直接告訴 agent 不要去追）。

## 機器人端提供的觀察

`/state` 回傳的遙測 JSON，重點欄位：

| 欄位 | 說明 |
| --- | --- |
| `position_m`, `yaw_deg` | RoboMaster SDK 自己的姿態，相對 session 開始時的座標，原點任意、會drift。 |
| `world_xy` | 從俯視攝影機的 zmqRemoteApi 連線查來的**真實**世界座標。跟 `yaw_deg` 不是同一個座標系（y 軸相反），`perception.py`/`robot_server.py` 內部換算時都已經校正過。 |
| `world_heading_deg` | 真實世界車頭方向（度，+x 為 0、逆時針為正），由四個輪子關節位置算出。`static_landmarks` 的 `turn_left_deg` 有這個欄位時就用它算，不再依賴「開機時面向 +x」的假設。 |
| `tof_front_mm` / `tof_left_mm` / `tof_right_mm` | 真的距離感測器（CoppeliaSim proximity sensor，繞過永遠回報 0 的 SDK ToF），9999 代表沒偵測到東西。 |
| `camera` | 前方相機**目前**是否真的讀得到新畫面（不是只看有沒有啟動成功——SDK 的 H.264 解碼 thread 可能中途掛掉，這個欄位會誠實反映）。 |
| `static_landmarks` | 場景裡固定不會動的結構（隧道、球池、四根柱子、平台、長椅、兩個收納箱、樓梯、溜滑梯、三張地墊），即時算好的 `{distance_m, turn_left_deg, world_xy}`，不用呼叫視覺動作就能拿到；`go_to(地標名)` 直接用這裡的 `world_xy`。 |

前鏡頭 (`/frame.jpg`) 和俯視鏡頭 (`/overhead.jpg`) 會被合成成一張左右並排、各自標好文字的圖 (`FRONT camera` | `OVERHEAD camera`) 送給 LLM 當 screenshot。

## 動作

| 動作 | 用途 |
| --- | --- |
| `move_chassis(forward_m, right_m, turn_left_deg, push)` | 相對移動，`+turn_left_deg` = 逆時針/左轉。平移先做、旋轉後做，單次上限 1m/180 度。往前走會自動在前方感測器看到的東西前 0.15 m 停下（回報會說明），要推東西得傳 `push=true`。 |
| `move_arm` / `arm_to` / `recenter_arm` | 手臂相對移動 / 絕對姿態 / 回預設姿態 (89,117)。`arm_to` 與 `recenter_arm` 都會用 `arm_mm` 核對有沒有真的到位。相機掛在手臂上，動手臂也會改變視角。 |
| `gripper(state, power)` | 開或關夾爪（預設 power 50）。 |
| `pick(object)` | 一次完成抓取：導航到物件前 0.3 m → 張開、手臂放到 (180,30) → 用前方距離感測器一步步把物件送進指間（先到 95 mm，沒夾到再深到 80 mm）→ 關夾爪、抬起 → 倒退 0.25 m 驗證（在手上的話感測器讀值不變）→ 回報 GRASPED 或失敗原因。成功後手臂進搬運姿態，伺服器記住手上物件的感測器讀值（導航忽略它）。 |
| `place(target)` | 一次完成放置：導航到目標 → 收納箱用高位釋放（手臂 (160,170)，夾爪離地 0.25 m 高過桶壁，依距離前進到車頭離桶壁 3 cm）；地墊／地面／其他用低位釋放 → 張開 → 倒退、手臂歸位。 |
| `locate(object)` | 只看前鏡頭，回報物體在畫面上的方位角跟該轉幾度置中，不移動機器人。 |
| `face(object)` | 找到物體並自動轉向對準它（最多修正兩次）。 |
| `locate_overhead(object)` | 用固定俯視相機找物體，回報真實世界距離跟該轉幾度面向它；有深度感測器時還會判斷候選物是不是貼地的假目標（地墊/標記）。 |
| `align_to_tunnel(object)` | 專門給隧道/通道這類「有兩端、可以穿過去」的長條結構：找出兩端座標、算出貫通軸線的朝向，並給出應該站的位置——只是定位，不會自動移動，需要 agent 自己接著執行建議的兩步（先開過去、再轉到軸線方向）才算真的對齊。 |
| `go_to(target)` | 自主導航：伺服器端用場景固定結構建的占據格地圖 + A* 規劃路徑，閉迴路執行到目的地（地標名稱、俯視相機找到的物件、或世界座標），一次動作取代整串 `move_chassis`。到地標／物件前約 0.45 m 停下並面向它。 |
| `approach(object, stop_m)` | 接近任務目標：從俯視相機定位、規劃路徑開過去、停在前方 `stop_m`，再用前鏡頭精對準並回報方位與 `tof_front_mm`；前鏡頭沒看到會老實說。 |
| `drive_through(structure)` | 沿長軸穿過隧道或鑽過平台底下，從入口到出口，並用世界座標判定是否真的穿過。 |
| `remember(fact)` | 把關於機器人本體/感測器的一般性事實存起來，之後每次任務的 system prompt 都會帶上（`<learned_notes>`）。 |
| `ask_human(question)` | 任務不明確、卡住、或動作有風險時詢問人類，逾時 180 秒沒回應會被中斷該步驟。 |
| `stop()` | 立刻停止底盤，不受動作鎖限制。 |

## 自我查核與 token 統計

`RobotAgent.py` 預設在任務結束時做兩件事，不用額外設定：

- **自我查核**（`use_judge`，預設開，`--no-judge` 關掉）：只在 agent 自己回報 `success=true` 時才跑，拿整段過程平均取樣的截圖（最多 6 張）加任務描述，另外問一次 LLM「畫面證據真的支持這個結論嗎」，不是看動作回傳的 `ok` 旗標。跟自評不同意時會印出警告並把判定與理由存進 `history.json` 的 `judge_verdict`/`judge_reasoning`。這跟 `robot_eval/` 事後用完整 state_trace、里程碑遙測做的批次判定是兩回事——那邊更完整，這裡是給沒有經過 `robot_eval`、直接互動跑任務時，也有一道「別只信自評」的檢查。`robot_eval` 呼叫 `RobotAgent` 時沒有打開這個選項，不影響跑分批次的行為或數字。
- **token/花費統計**：每次真的呼叫 LLM（主迴圈、planner、長期摘要、自我查核都算），從回應本身的 `usage` 欄位累加 prompt/completion/total token 數，不是用字元數估計。跑完印一行 `📊 N 次 LLM 呼叫｜... tokens`，同時存進 `history.json` 的 `llm_calls`/`total_tokens` 等欄位。

## 遠端機器人

Server 預設只聽 127.0.0.1。要從另一台電腦控制時，建議用 SSH tunnel 或 Tailscale，不要直接把 port 開到公網：

```bash
# 在 Agent 這台電腦上
ssh -N -L 8765:127.0.0.1:8765 user@robot-host
./run_agent.sh --task "..."                      # 照樣連 127.0.0.1:8765
```

如果一定要讓 server 聽在區網上，請加 token：`./start_server.sh real --host 0.0.0.0 --token <secret>`，Agent 端設定環境變數 `ROBOT_SERVER_TOKEN` 或用 `--robot-token`。

## 安全設計

所有數值在 server 端限幅 (單次最多 1 m、180 度、手臂 100 mm)。底盤預設用 `chassis_backend=hybrid`：平移用 SDK 位置控制 `chassis.move`，旋轉用手動輪詢 yaw 的速度控制，兩者都會事後比對里程計，SDK 回報成功但實際沒怎麼動的話會被判定失敗（`rejected after the fact`），而不是照單全收；反過來，平移的 SDK 完成訊號逾時但里程計已走到要求距離八成以上時，會改判成功（`accepted as completed by odometry`），因為模擬裡的「到達」訊號常常比車身晚，逾時不代表沒走到（實測要求 0.5 m、里程計 0.50 m 仍回 timeout，agent 誤以為沒動再重下，連續三次就被中止）。一次只執行一個動作，`/stop` 不受動作鎖限制。連續多次偵測到「完全沒動、旁邊又沒東西擋」會觸發 `_auto_recover()`：自動停止模擬、重新播放、重連 SDK，救回控制通道卡死的情況（代價是姿態會跳回上次存檔位置）。

**對外連線**：Agent 只會連你在 `.env` 設定的 LLM API（agent 與 judge 可以是不同家），以及本機的 robot server（8765）與 CoppeliaSim（23000）。沒有遙測、沒有雲端同步、沒有版本檢查。

## 已知狀況 / 排查紀錄

這些是實際跑分中發現、值得知道的行為，不是理論上的邊界情況：

- **場景裡的三張地墊原本是看不見的（2026-10-08 已補）**：`mat_red`／`mat_green`／`mat_blue` 在存檔的場景裡只是 dummy，沒有任何形狀，h02／h05／h06 放對座標也看不到墊子。`Env/add_mats.py` 補上 0.6 m 見方、1.2 cm 厚的彩色墊子（不碰撞、感測器偵測不到，物理與導航不受影響）；同時把橘球柱子從紅墊上移到 (0.15, 0.00)，黃積木從綠墊正中央移到 (0.00, -2.20)（原本 h06「放到綠墊上」一開始就成立）。藍墊跟藍色收納箱相隔 17.5 cm，俯視圖裡是兩塊並排的藍色，靠地標名稱區分。
- **導航地圖快取在相對路徑下永遠不會失效（2026-10-07 修正）**：headless 用 `./coppeliaSim -h scenes/xxx.ttt` 啟動時，模擬器回報的場景路徑是相對於安裝目錄的，從 robot_server 的工作目錄找不到檔案，修改時間一直是 0，改過場景存檔後地圖仍從舊快取載入（新柱子、搬走的收納箱都不在地圖上）。現在會補成完整路徑，找不到檔案就不用快取。
- **sim 模式的前方相機改成直接讀模擬器的視覺感測器（2026-10-07）**：SDK 的 H.264 串流有解碼延遲，`_fresh_image()` 用固定時鐘上限（共 4 秒）等新畫面，模擬跑得慢（例如用 GUI 版跑批次）時會在新畫面到之前放棄、默默回傳轉向前的舊畫面。10/6 的批次裡 e07 連續四步畫面都慢一個轉向，agent 因此回報錯的柱子順序；整批 21 次大角度旋轉有 6 次轉完畫面幾乎沒變。現在 `frame_jpeg()` 在 sim 模式用 `getVisionSensorImg` 讀當下影像（上下左右翻轉後跟 SDK 穩定畫面差異 3.4，即 JPEG 雜訊），跟模擬速度無關；讀失敗會退回 SDK 串流，`ROBOT_FRONT_CAMERA_SOURCE=sdk` 可以強制用舊路徑。實體機器人仍走 SDK。

- **前鏡頭方向**：`robot_server.py` 的 `frame_jpeg()` 已經加上水平翻轉修正過去左右相反的問題（實測：送一個遙測證實為真的「左轉」，畫面卻往左滑而不是往右滑）。第一次接上新的模擬器/場景時，建議還是先下一個「turn left 90 degrees then stop」之類的任務，親眼確認方向正確。
- **感測器/攝影機物件會被模擬重啟清掉**：`sim.createProximitySensor()` 建立的距離感測器是「模擬執行期間」才存在的物件，模擬只要被停止過一次就會消失。`robot_server.py` 現在會自動偵測讀取失敗並重建，`state()` 的 `camera` 欄位也會誠實反映前鏡頭是否還在更新，不會停在最後一次成功值就不動了。
- **球池邊界的直角容易讓機器人物理卡死**：出生點在球池旁邊，原地旋轉時偶爾會卡進圍牆的直角，此時旋轉會回報「成功」但實際角度沒變。`_auto_recover()` 通常能自動救回，但這是已知、尚未從場景幾何上根治的問題。
- **同一個顏色可能對應到不只一個物件**：例如「red box」在目前的場景裡同時可能指向可以被夾爪抓取的小積木，也可能指向純裝飾用的大型泡棉方塊；純導航類任務（開過去、看一眼）通常兩個都能接受，但抓取類任務如果認錯，不管導航修得多好都不可能成功。懷疑某個任務一直失敗時，先確認 `locate`/`locate_overhead` 找到的座標到底對應哪個物件，再往下查。
- **場景裡不能有兩台 `RoboMaster`**：`Capstone_Playground.ttt` 曾經同時存在一台完整的 EP 模型跟一台被拆過（沒有 `GyroSensor`、沒有 visual 連桿）的複製品。`simRobomaster` 外掛只會驅動完整的那台，但 `sim.getObject('/RoboMaster')` 會拿到先出現的那台，結果距離感測器、`world_xy`、靜態地標全掛在不會動的車上，agent 看到的相機畫面卻來自另一台——`world_xy` 永遠停在起點、`tof` 永遠 9999、回報的距離是錯的車量的。`robot_server.py`／`run_batch.py`／`reset_robot.py` 現在都會優先挑有 `GyroSensor` 的模型並在 log 警告，`start_all.sh` 也會提示 `[DUP_ROBOT]`，但正確做法仍是把多餘那台從場景刪掉並存檔。
- **距離感測器的方向不能用 SDK 的 `yaw_deg` 來定**：`robot_server.py` 自建的三個近接感測器以前用啟動當下的 SDK yaw 決定「前方」，但這個模擬器的 yaw 遙測跟世界車頭正好反號（跟 `TURN_SIGN = -1` 是同一件事），車頭在 −65° 時「前方」感測器實際指向左後方 130°。第八輪回歸測試 110 筆 ToF 讀值只有 8 筆對得到前方的障礙，之前所有「ToF 看到東西就縮步／標障礙／通道裡被擋住」的行為都是被側牆騙的。現在改用四個輪子算出的車頭定向；`start_all.sh` 之後可以用 zmq 讀 `prox_front` 的物件矩陣跟輪子車頭比，差值應該是 0°。
- **2 cm 厚的牆會從地圖上消失**：`nav_map.py` 用 `cv2.fillConvexPoly` 光柵化形狀投影，收納箱的側壁四個角落落在同一欄時凸包退化成線段、什麼都不畫，`bin_balls` 的西壁就這樣不在地圖上，A* 把路排進桶裡。現在會再補畫一格粗的輪廓線。
- **車身尺寸與 `stop_m` 的意思**：模擬模型的根物件 `/RoboMaster` 座標系是 −z 朝車頭、y 側向、x 朝上（不要拿 root 的 x/y 當平面姿態）。相對 root：底盤前後各 0.166 m、含輪子寬 0.30 m，角落離中心 0.224 m；手臂歸位時夾爪凸出到前方 0.231 m；前方距離感測器在前方 0.182 m、高 0.117 m。`navigate_to`／`approach` 的 `stop_m` 是「車頭前緣到目標表面」的距離：站位沿每個方向先掃出目標本體的邊緣（記最後一個占據格，球池這種中空結構才不會在內部就停），再從 邊緣 + stop_m + 0.166（扣回格子量化的 0.10）起找可站、淨空 ≥ 0.30 的點，1 m 內找不到就把 stop_m 逐步縮短（角落的口袋太淺時），到達容忍 0.08 m，所以 `approach(x, 0.3)` 停下來時 `tof_front_mm` 約 280–300，落在跑分「停在它前面」檢查的 150–450 範圍；以前以中心算，停 0.3 m 時物件已經在感測器後面、讀 9999。`move_chassis` 往前走會自動在感測器看到的東西前 0.15 m 停下，要推東西得傳 `push=true`。
- **距離感測器的高度與錐角**：以前裝在離地 0.185 m、20° 錐，0.42 m 內看不到 10 cm 高的收納箱壁（中等批次 m06 停在桶前 0.31 m 讀 9999）。現在離地 0.10 m、16° 錐：0.15–0.6 m 都看得到 ≥ 7 cm 的東西（收納箱壁、8 cm 的球、小積木），錐底在 0.6 m 處仍離地 6 mm，不會掃到地板；實測收納箱 0.30 m 讀 297、球 310、球池牆 200、空曠處 9999。
- **距離感測器會掃到自己的夾爪**：前方感測器在車頭前 0.18 m，夾爪與相機連桿在大多數手臂姿態下都在感測錐裡，感測器回報的第一個物件是自己、被過濾成 9999，前面 0.3 m 的積木就看不到（中等批次 nav_002／m03／m06 的「ToF 讀 9999」全是這個）。`robot_server.py` 建感測器時會把車身所有零件設成「不可偵測」，感測器直接穿過自己；模擬重播會把旗標還原，所以每次重建感測器都會再設一次。
- **不要用 SDK 的 `robotic_arm.recenter()`**：模擬裡它會把手臂帶到 (50,−73) 這種在底盤下方的姿態，之後 `moveto` 回報「到了」手臂卻沒動。`recenter_arm` 現在直接 `moveto(89,117)` 並用 `arm_mm` 核對（差超過 15 mm 就回失敗），`arm_to` 也一樣會核對。
- **夾著的物件會被模擬器掛進機器人的物件樹**：server 建感測器時把「車身零件」設成不可偵測，若當時夾爪裡有積木，積木會一起被設成不可偵測、放回地上後 ToF 從此看不到它。現在 `small_`／`ball_`／`foam_` 開頭的零散物件一律不算車身，而且每次重建感測器都會把車身以外、不可偵測的形狀恢復成可偵測（自我修復）。
- **server 連著的時候不要用 `reset_robot.py` 傳送機器人**：底盤控制會失效（指令回報成功但車不動），要重開 server 才恢復。跑分的 `prepare()` 是「停止模擬 → 重播 → 重開 server」，這條路是安全的；測試腳本要放回零散物件請用 `reset_robot.py --objects --no-robot`。
- **`align_to_tunnel`／`locate_overhead` 的結果只是定位，不是移動**：它們的回傳文字裡會明確要求「不要現在就下結論、先執行建議的移動」，但目前的 LLM (`gemini-robotics-er-2-preview`) 偶爾還是會跳過移動直接用文字描述下結論。這是提示詞層級的軟性要求，還沒有做成程式碼層級的強制檢查。

## 導航層回歸測試（不用 LLM）

```bash
.venv/bin/python Test_Robot/reset_robot.py --objects    # 先把零散物件（泡棉、積木、球）放回建場景的位置並擺正
.venv/bin/python tools/nav_selftest.py                  # 約 35 分鐘：旋轉精度、12 個隨機目標、12 個地標、兩個通道、6 個物件接近幾何
.venv/bin/python tools/nav_selftest.py --only rotate,passages
```

`--objects` 那步不能省：`start_all.sh --restart` 只重開 server、不會重播模擬，連跑幾輪回歸測試後零散物件會被推得越來越歪
（實測連跑五輪之後綠色泡棉樑偏了 1.08 m 還轉了 90°、黃色泡棉圓柱 0.47 m、小黃積木 0.44 m），下一輪的地圖跟接近測試就跟
上一輪不同條件。跑分批次（`run_batch.py`）每題都會停止並重播模擬，物件會回到場景檔存的位置，不需要這步。

用模擬器真值量到達率、終點誤差、旋轉誤差與**碰撞次數**（背景輪詢機器人零件跟場景物件的接觸），並在每一節前後比對所有零散物件的位置、列出被推動 ≥ 5 cm 的物件（接觸輪詢 0.2 s 一次會漏掉短暫的推擠，位移不會漏）。地標的到達判定是「離中心 ≤ 2 m 或離結構外框 ≤ 1 m」（平台長 2.5 m，停在入口外離中心就 2 m）。結果存 `robot_eval/results/nav_selftest_<時間>.json`。
改了 `nav_map.py`／`navigate_to` 的任何參數就跑一次、跟上一份比，這樣改動影響到整張地圖的哪個角落都看得到，不會只看考題經過的路。

跑分或回歸測試建議用 headless 的 CoppeliaSim（記憶體約 400 MB，GUI 版連跑幾批會到 7 GB 以上而且當過）：

```bash
cd ~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04
tail -f /dev/null | ./coppeliaSim -h scenes/Capstone_Playground.ttt > ~/coppeliasim.log 2>&1 &
```

前面的 `tail -f /dev/null |` 不能省：CoppeliaSim 的 Commander 附加元件會讀 stdin，stdin 一關（例如接 `/dev/null`）它就直接「Leaving...」結束。

## 目前的跑分（2026-09-28，agent = Gemini Robotics-ER，judge = gpt-4o）

| 難度 | 通過 | 失敗的題目與原因 |
| --- | --- | --- |
| easy | 12 題（先前批次 75%，導航層重做後未整批重跑） | — |
| medium | 10/11 | m01：藍色球桶北側被長椅、泡棉樑、隧道圍住，站到桶前 30 cm 時離長椅只剩 0.19 m，小於車身角落半徑 0.224 m，對這台車不可行（2026-10-07 已把收納箱搬到 (-1.4, 0.7) 的空地） |
| hard | 9/11 | h05、h09：球（8 cm）在夾爪最低點下方，這個模擬模型抓不到（2026-10-07 已改成 6 cm 球放在柱子上，見「抓取與放置的設計原則」）。積木類的 h03／h06／h08 已改用 pick／place 技能，2026-09-29 乾淨重跑三題全過（里程碑 11/11） |

導航類的題目已沒有失敗。上面數字是用 `robot_eval/run_batch.py` 分批跑出來的，每題都停止並重播模擬、重開 server；
批次中若 LLM API 額度用盡（連續呼叫失敗且一步都沒成功）會自動中止，剩下的題不算分。

## 評估 (checkpoint / milestone)

```bash
python -m robot_eval.run_evaluation                              # 跑 robot_eval/dataset.json 的全部任務
python -m robot_eval.run_evaluation --task nav_001 --repeat 5     # 同一個任務重複五次，看成功率的變異
python -m robot_eval.run_evaluation --rejudge robot_eval/results/<session>/nav_002 --task nav_002   # 不動機器人，重評舊紀錄
```

Dataset 格式：`id, difficulty, category, description, milestones`。milestone 可以加上 `telemetry` 條件，直接用遙測數值判定，例如 `{"field": "yaw_deg", "abs_min": 75, "abs_max": 105, "when": "final"}`，支援 `equals`、`in`、`min`/`max`、`abs_min`/`abs_max`，`when` 可以是 `final`（只看結束狀態）或 `any`（任何一步成立即可）。沒有回報那個遙測欄位、或沒寫 `telemetry` 的 milestone，會退回給 judge LLM 看取樣的相機畫面加動作紀錄判定。

成功判準：judge 的裁決優先，judge 失敗時才退回「milestone 全過」。報告另外列出 `agent_overclaimed_success`（Agent 自評成功但實際失敗的任務）和 `total_distance_m`。評估期間 `ask_human` 會自動回覆「沒有人可以問」並關閉逐步確認，用實體機器人跑評估時請留在旁邊。`robot_eval/results/` 不進版控（跟 `runs/` 一樣是每次跑分留下的截圖/紀錄）。

任務之間，mock 模式會自動重置；sim 和 real 模式只會收手臂、開夾爪，然後停下來等你把場景擺回原位再按 Enter。

### 跨任務筆記污染與 `--notes-mode`

`remember(note)` 原本不管互動模式還是評估都寫同一個檔案 `memory/body_notes.md`，而評估時每題依序建立新的 `RobotAgent`，後面的題目會讀到前面題目剛寫的筆記。這會讓題目順序影響分數、讓模型之間的比較不公平（某個模型贏可能只是因為批次前段剛好寫了有用的筆記，不是控制得比較好）、也讓重現性沒了（clone 這個 repo 的人、或你自己重跑同一個批次，起點都不是乾淨的）。

`--notes-mode`（`run_evaluation.py` 和 `run_batch.py` 都有）解決這個問題：

- `fixed`（預設）：固定讀 `robot_eval/fixed_notes.md`（人工審核過，進版控），所有題目、所有被比較的模型拿到完全一樣的先驗；`remember()` 照常回應，但不寫回磁碟。跑分數字應該用這個模式比較。
- `accumulate`：筆記在這次批次執行內真的跨題累積（整個批次共用一個檔案，存在 `<batch_dir>/accumulated_notes.md`），每次重跑批次都從空白開始。當作「允許跨任務學習」的對照組——report.json 的 `config.notes_mode` 會記下用的是哪一種，兩組數字可以直接對照報告。
- `off`：完全不給筆記、`remember()` 也不寫入，最乾淨的基準線。

互動模式（`RobotAgent.py`，不經過 `robot_eval`）不受影響，繼續預設寫 `memory/body_notes.md`，讓機器人真的能跨 session 累積對自己身體的認識；這個檔案不進版控（見 `.gitignore`），避免你的本機狀態變成別人 clone 這個 repo 的起點。

## 新增動作

在 `robot_agent/actions.py` 的 `ACTION_SPECS` 加一個項目（Pydantic 參數模型 + 一句給 LLM 看的說明 + handler 名稱），在 `robot_agent/service.py` 實作對應的 `_h_<name>` handler；需要移動或查詢機器人就透過 `self.robot.act(...)`／`self.robot.state()`。需要多步閉迴路的動作（導航、抓取）寫在 `robot_agent/skills.py`，handler 只包裝結果。`system_prompt.md` 的動作說明也要同步補一行。

## 目錄

```
RobotAgent.py             進入點 (.venv, Python 3.11+)：--task 單次任務、--loop 互動模式
robot_agent/
  service.py                主迴圈：觀察→思考→行動、planner、長期摘要、自我查核、token 統計、動作 handler
  actions.py                動作註冊表（參數 schema + 說明 + handler 名稱）
  skills.py                 go_to/approach/drive_through/pick/place（伺服器閉迴路 + 感測器驗證）
  llm/                      各家 LLM 的呼叫包裝與訊息型別（改寫自 browser-use 的 llm 子套件，MIT）
  perception.py             locate/face/locate_overhead/align_to_tunnel 的視覺+幾何運算
  client.py                 HTTP client
  system_prompt.md          system prompt
  views.py                  LLM 輸出 schema 與歷史紀錄
robot_eval/                dataset、telemetry 與 LLM 判定、指標、評估 CLI
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

## 動作空間之外：讓 LLM 自己推理

`system_prompt.md` 的 `<task_interpretation>` 區塊明講使用者的任務可能是目標或需求（「我好口渴」），不只是逐步指令——LLM 要先想清楚物理上要做什麼，再決定用現成技能還是自己組合原語。`go_to`/`approach`/`pick`/`place`/`drive_through` 仍然是建議的預設選項（已經把導航、抓取踩過的坑都處理好了），但不再是唯一合法選項：任務吻合不到任何一個技能時，LLM 被要求自己用 `move_chassis`/`arm_to`/`move_arm`/`gripper` 推理出一條路，而不是卡住或硬套錯的技能。
