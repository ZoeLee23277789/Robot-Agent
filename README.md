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

4. 想即時看 agent 在做什麼：瀏覽器開 `http://127.0.0.1:8765/live`（2026-10-21 新增）。三欄分別是占據格地圖（紅箭頭 = 機器人、綠線 = 規劃路徑、橘色 = 零散物件）、俯視鏡頭（已轉 180°，方位跟地圖一致）、前鏡頭 + 遙測表格；下方整排是目前任務與 agent 每一步的 evaluation / thinking / memory / goal / 動作結果，以及每 5 步的 plan。圖片每秒更新、俯視圖每 3 秒、agent 紀錄每 1.5 秒；`?ms=500` 可以改圖片更新間隔。最後一次 navigate_to 鎖定的目標會用洋紅框標在地圖（地標畫地面範圍、零散物件畫 16 cm 方框）和前鏡頭上（`GET /frame_live.jpg`：把目標的 3D 方框用相機世界矩陣與視角投影到畫面；送給 LLM 的 `/frame.jpg` 不加框）。agent 端是用 `POST /agent_event` 把每一步推給 server（`GET /agent_log` 可直接拿 JSON），跑分批次每一題重啟 server 時紀錄會清掉。

5. 錄一段展示影片（2026-10-23 新增）：`tools/record_demo.py` 直接從 server 的端點擷取地圖、俯視圖、前鏡頭（含鎖定目標框）、遙測與 agent 每一步的思考，合成 1920x1080 mp4，任務結束疊上 SUCCESS / FAILED。不需要螢幕錄影軟體，無視窗模擬器也能錄。
   ```bash
   .venv/bin/python tools/record_demo.py --task-id r11                 # 先重置場景，再跑 dataset 的 r11 並錄影
   .venv/bin/python tools/record_demo.py --task "Pick up the lime ball from the table and put it into the blue bin."
   .venv/bin/python tools/record_demo.py --no-reset                    # 只錄影，任務你自己跑；Ctrl+C 結束
   ```
   預設 3 倍速（模擬器只跑約 0.55 倍即時），輸出在 `videos/`（已加進 .gitignore），agent 的終端輸出存在同名 `.agent.log`。有 ffmpeg 時錄完會自動轉成 H.264。

沒有模擬器也想先測 Agent 迴圈的話，用 `./start_server.sh mock`（等同 `python Test_Robot/robot_server.py --mode mock`），它會模擬一台 2D 機器人和一個紅色目標，不需要安裝 SDK。

## 指令速查（模擬 / 實體機器人）

模擬和真機的 server 都聽 8765，一次只能開一個。切換前先關掉舊的：`kill $(lsof -t -i:8765)`。

### 模擬

```bash
# 啟動（擇一）。同時開著視窗版和無視窗版時 server 只會連到其中一個，先確定只剩一個
pkill -x coppeliaSim && ./start_sim_headless.sh && ./start_all.sh --restart   # 無視窗，跑分用
./start_sim_headless.sh --stop && ./start_all.sh --restart                    # 只用已開著的視窗版

./run_agent.sh --task "..."              # 單一任務
./run_agent.sh --loop                    # 互動模式

./run_all.sh                             # 背景跑全部任務（自己開無視窗模擬、當掉自動重開）
./run_all.sh --task p08 --task h05       # 指定題目；--difficulty hard 指定難度
./run_all.sh --resume robot_eval/results/all_XXXX   # 接續跑到一半的批次
./run_all.sh --stop                      # 中止
tail -f /tmp/run_batch.log               # 進度
.venv/bin/python -m robot_eval.run_batch --report-only robot_eval/results/all_XXXX   # 重新彙整成績

./start_sim_headless.sh --stop           # 關掉無視窗模擬
```

### 實體 EP

只有基本動作能用：`move_chassis`、`move_arm`、`arm_to`、`recenter_arm`、`gripper`、`wait`、`stop`。
地圖、真實座標、俯視相機、距離感測器、升降柱都來自模擬器，所以 `navigate_to`、`go_to_landmark`、`drive_through`、`lift_arm`、
approach/pick/place/scan_overhead 在真機上不能用，模擬的分數不能直接套到真機。真機模式下 agent 每個動作都會先問你（`--no-confirm` 關掉）。

連線方式用環境變數 `RM_CONN_TYPE` 選（`Test_Robot/rm_connection.py`）：

| 值 | 接法 | 備註 |
| --- | --- | --- |
| `ap` | 機器人開關撥「直連」，電腦 Wi-Fi 連 `RMEP-xxxxxx`（機器人 IP 192.168.2.1） | Wi-Fi 被機器人佔走，LLM 要靠網路線上網 |
| `sta`（預設） | 開關撥「組網」，用 RoboMaster App 讓機器人連上跟電腦同一個 Wi-Fi | |
| `rndis` | USB 資料線接機器人主控（智能中控）的 micro-USB 孔 | 電腦 Wi-Fi 可以留著上網 |

```bash
export RM_CONN_TYPE=ap
ping -c 2 192.168.2.1                                      # 直連模式先確認通
(cd Test_Robot && ~/rmenv/bin/python test_movement.py real) # 煙霧測試：前進約 0.4 m、轉一下、夾爪、手臂、相機
./start_server.sh real --chassis-backend move              # 終端機 A，開著不要關
./run_agent.sh --task "往前走 0.3 公尺，然後左轉 90 度"      # 終端機 B
unset RM_CONN_TYPE                                         # 收尾，回模擬前
```

第一次用真機請先用下面的 curl 下 `turn_left_deg: 45`，確認從上面看是逆時針（轉向符號是照模擬器調的）。

### 兩邊都能用：直接下單一動作

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/state
curl http://127.0.0.1:8765/actions       # 所有動作與參數（以這個為準）
curl -X POST http://127.0.0.1:8765/action -H 'Content-Type: application/json' \
  -d '{"name":"move_chassis","params":{"forward_m":0.2,"right_m":0,"turn_left_deg":0}}'
curl -X POST http://127.0.0.1:8765/action -H 'Content-Type: application/json' \
  -d '{"name":"gripper","params":{"state":"open"}}'
curl -X POST http://127.0.0.1:8765/action -H 'Content-Type: application/json' \
  -d '{"name":"move_arm","params":{"forward_mm":20,"up_mm":0}}'
curl -X POST http://127.0.0.1:8765/stop  # 緊急停止，不受動作鎖限制
```

### 真機相機（2026-11-07）

韌體 01.01.1008、直連模式：連線、底盤、夾爪、手臂正常，但開影像串流一律失敗。
SDK 送的 cmdset 0x3f / cmdid 0xd2 等不到回應；純文字協定（40923 埠）`stream on;` 直接回 `fail;`；RoboMaster App 也沒有畫面。
所以是硬體（相機、排線或主控接口）的問題，不是程式。沒有相機時 server 照樣能跑，只是 agent 看不到前方。
相機只能接機器人主控，不要直接插電腦（插 Type-C 會出現 USB `error -71`，無法辨識）。

### NexArm 機械臂（Hiwonder，2026-11-09）

桌上的 NexArm 用 USB Type-C 接電腦（CH340 → `/dev/ttyUSB0`），server 直接講它的串列協定（`Test_Robot/nexarm.py`，協定與實測紀錄都在檔頭），
不經過 RoboMaster SDK。手臂、夾爪動作（`move_arm`、`arm_to`、`recenter_arm`、`gripper`）改由它執行，agent 端完全不用改。

```bash
# 一次性：讓這顆 USB 晶片插上就有權限，並多一個固定名稱 /dev/nexarm
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE="0666", SYMLINK+="nexarm"' | sudo tee /etc/udev/rules.d/99-nexarm.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

~/rmenv/bin/python Test_Robot/nexarm_probe.py                      # 只讀：六顆伺服位置 + 末端座標，確認手臂在聽
~/rmenv/bin/python Test_Robot/nexarm_nudge.py --to 150,0,250 --go  # 單一動作測試（不加 --go 只印封包）

kill $(lsof -t -i:8765) 2>/dev/null
./start_server.sh armonly                 # 只有桌上的 NexArm，沒有底盤、沒有相機；move_chassis 會回「不可用」
./start_server.sh real --arm nexarm       # 之後裝上 EP：底盤用 EP、手臂用 NexArm
./run_agent.sh --task "把夾爪打開，手臂抬高 5 公分，再把夾爪合起來"   # armonly 跟 real 一樣每步都會先問
```

可達區域不是方的：`Test_Robot/nexarm_workspace.py` 用韌體「只算 IK、不動」的指令掃過（夾爪水平 / 斜 45° / 朝下三種角度），再用擬合的正運動學複驗，結果存在 `Test_Robot/nexarm_workspace.json/.png`，實機走過 4 個邊緣點驗證（誤差 0–11 mm）。server 移動前查這張表：表外的高度夾到可達範圍，整個前伸距離都到不了就不送；遙測的 `arm_limits` 給目前姿勢還能升降、伸縮多少。夾爪水平時前伸 150 mm 最低只到 200 mm 高，要碰桌面得把夾爪朝下（`arm_to` 的 `pitch_deg: -90`）。換了手臂或改了韌體參數要重掃（server 要先關）。

**模擬裡的 NexArm（2026-11-09 起）**：`Env/add_nexarm.py` 在 EP 模型上加一支用實測尺寸做的 NexArm（簡化外觀、關節與連桿長度準確，跟官方圖面 110/225/145/130 mm 對得上），原本的手臂隱藏不刪（外掛拒絕被拆過的模型），存成 `scenes/Capstone_Playground_nexarm.ttt`，原場景不動。用新場景：`COPPELIASIM_SCENE=scenes/Capstone_Playground_nexarm.ttt ./start_sim_headless.sh`。底座預設坐在原本手臂的升降座 `arm_lift_carriage` 上（車身正中、頂面離地 0.162 m），所以模擬裡 `lift_arm` 照樣能把整支 NexArm 升高；要改掛在底盤別處就給 `--forward/--height`。注意：升降柱只存在模擬（Env/add_arm_lift.py 的虛擬裝置），實體 EP 沒有。手腕相機 `nexarm_wrist_cam` 掛在腕俯仰關節上。

模擬的 NexArm 後端是 `Test_Robot/nexarm_sim.py`：跟實體驅動同一組方法，同一張可達範圍表、同一組 IK 解（手肘朝上朝後），server 的 `--arm auto`（預設）在 sim 模式看到場景裡有 `nexarm_mount` 就自動用它，舊場景照舊用 SDK 的手臂。抓取用掛接：合爪時指間有動態小物件（最長邊 ≤ 12 cm、寬度 ≤ 夾爪開度）就掛到 `nexarm_tcp`，開爪放回去，`gripper_holding` 跟實體一樣由手指停在哪裡判定。手腕相機是手動更新的感測器，`_sim_front_image` 讀之前會先 `handleVisionSensor`。尚未對實體量過：夾爪全開的指間距離（模型假設 35 mm，場景裡的球是 6 cm！）、安裝位置。

外觀用的是 Hiwonder 官方 ROS 2 倉庫（github.com/Hiwonder/NexArm，`src/simulations/nexarm_description`）的 URDF 網格，STL 在 Git LFS 上要從 media.githubusercontent.com 抓，放在 `~/下載/nexarm_description/`；`add_nexarm.py --urdf` 會匯入並把網格掛到這裡的骨架上（骨架的運動學是實測擬合、官方 URDF 的關節位置跟它差 < 1 mm）。手腕相機掛在腕俯仰關節、官方 camera_link 往前 3 cm 處。

**模擬跟實體同時跑**：模擬那套照舊（無視窗 23000 + `start_all.sh` 的 8765），實體手臂用 `./start_arm.sh`（8767），agent 下給實體時加 `--robot-url http://127.0.0.1:8767`。`start_arm.sh` 預設還會把實體的動作即時映到視窗版 CoppeliaSim（23001）裡的 NexArm 模型（數位分身，只讀實體狀態不送指令；`--mirror-sim` 選項），所以模擬畫面、實體手臂、視窗裡的分身三個可以同時看。

對應關係：agent 的 `arm_mm = {x, y}`（前伸、高度）對到 NexArm 的水平距離和 z；`arm_to` 多了選用的 `pitch_deg`（夾爪俯仰）、`yaw_deg`（底座旋轉，0 = 正前、正 = 向左）、`roll_deg`（手腕滾轉）；`move_arm` 多了相對的 `turn_left_deg`（轉底座，底盤不動）、`roll_deg`。遙測 `arm_pose` 有 `reach_mm`、`yaw_deg`。

**手臂走 Wi-Fi（不用 USB 線，2026-11-10 實測）**：韌體開機就開熱點 SSID `NexArm` / 密碼 `hiwonder`，手臂是 192.168.4.1、TCP 8080，封包格式跟 USB 完全一樣（狀態廣播也會送到 TCP 客戶端，一次只收一個客戶端）。驅動的 port 給 `tcp://主機:埠` 就走 socket：
```bash
nmcli con up NexArm                                     # 電腦 Wi-Fi 連手臂熱點（有線網路不受影響）
# 第一次沒有設定檔時：nmcli dev wifi connect NexArm password hiwonder（之後再用這句會報 key-mgmt 找不到屬性，改用 con up）
NEXARM_PORT=tcp://192.168.4.1:8080 ./start_arm.sh        # 其餘一樣；--with-ep 也可以
nmcli con up <原本的 Wi-Fi>                              # 用完切回
```
實測延遲跟 USB 差不多（抬 30 mm 含等伺服停穩 2.1 s）。電腦只有一張 Wi-Fi 卡，連了手臂熱點就不能同時連 EP 的熱點：底盤和手臂都要無線時，EP 要改成路由器模式（App 裡設定、連到跟電腦有線同一個路由器）再用 `RM_CONN_TYPE=sta`。

**實測過的極限（2026-11-10，腳本都要先關 8767 再跑、結果各自存 json）**：
- `Test_Robot/nexarm_fold_test.py --go --mirror 23001`：收攏極限。夾爪水平只能縮到前伸 90 mm（再近伺服停在 12 mm 外）；夾爪朝下可以縮到底座正上方 x=0、z 130–150 誤差 0。所以 `recenter_arm(pose="stow")` = (0, 150, −90)：整支手臂收在底座上方、最高點是肘約 29 cm、肘在底座後方 14.5 cm（裝上車要確認不撞升降柱）。HOME (150, 0, 250) 只是「準備姿勢」，不是最緊的。
- `Test_Robot/nexarm_rotate_test.py --go --mirror 23001`：底座 ±117°（伺服 1 停 712/3382）、手腕 ±90°（伺服 5 停 3083/1013）。方向已目視確認：+yaw 實體向左、分身同向；分身的 j5 要反號才跟實體手腕同向。
遙測多了 `arm_pose`（x/y/z/pitch/roll）、`gripper_claw_deg`、`gripper_holding`。夾爪 open = −60°、close = 50°；
有沒有夾到東西看夾爪伺服停在離目標幾度（空夾爪每次都轉到位、誤差 ≤ 0.3°，停在 4° 以上就是被東西擋住）。
動作結束一律等六顆伺服真的停下來才讀位置（夾爪 −60→50 在名義時間 800 ms 後還在轉，固定等待會誤判成夾到東西）；
座標差 4 mm 以上會補送一次修正（伺服在負載下會停短，往上動時 +10 實測只走 +2，修正後 +12）。
手臂剛上電時曾經不理任何指令（韌體收到指令會印 `[CB] ...`，沒有就是沒在聽），關機再開機即可，不用重燒韌體。
USB 傳輸出錯一次（kernel log `ch341-uart ttyUSB0: urb stopped: -32`）之後，開著的串列埠會永遠讀不到東西又不報錯；
nexarm.py 有看門狗：2 秒沒狀態先送 CMD 11 問一次，再沒有就關掉重開串列埠（第一次實測 agent 跑完 2 秒就遇到，手臂接在 hub 上；建議直接插電腦本體）。

### 鏡射模式：模擬當眼睛，實體照做（2026-11-10）

原本以為 EP 的相機壞了（2026-11-11 確認沒壞：是相機線沒接、或機器人開著才插線，中控只在開機時找相機，關機接好再開機就有畫面），加上沒辦法掃 QR code 設路由器模式，所以實體那邊看不到東西。做法：agent 照常對**模擬** server（8765）下任務，模擬 server 每執行一個底盤/手臂/夾爪動作，就把同一個動作同步送到**實體** server（8767），實體跟著做一樣的動作（不需要真的夾到東西）。

一鍵版（2026-11-11）：`./run_mirror.sh --task r01`（或 `--difficulty easy`、`--resume DIR`）會依序確認／開啟無視窗模擬器（NexArm 場景）、視窗版分身、sim_twin、手臂熱點、實體 server 8767（EP 路由器模式＋手臂熱點，背景跑、EP 斷線自動重開），最後前景跑批次；`--check` 只開環境不跑批次，`--stop` 關掉 8767／sim_twin／視窗版。已經開著的東西直接沿用。手動版如下：

```bash
./start_arm.sh --with-ep                                     # 實體：EP 走熱點、NexArm 走 USB（先開）；預設會開 EP 相機，8767 的 /live 看得到實體畫面
# 模擬的前鏡頭預設是 EP 原廠那顆（Vision_sensor，車頭前方固定）；要用 NexArm 手腕相機：ROBOT_SIM_CAMERA=wrist ./start_all.sh --restart
ROBOT_MIRROR_REAL=http://127.0.0.1:8767 ./start_all.sh --restart   # 模擬 server 重開並掛上鏡射
./run_agent.sh --task "..."                                  # 照常對 8765 下任務，實體會跟著動
```

- 鏡射的動作：move_chassis、move_arm、arm_to、gripper、recenter_arm、stop。導航（go_to / navigate_to / drive_through）拆出來的每一小段也會送，實體走同一條路。`lift_arm` 不送（實體沒升降柱）。
- 底盤送的是模擬**實際量到**的距離與轉角大小（方向沿用指令）：模擬被擋只走一半，實體也只走一半；模擬沒動就不送。
- 實體那邊失敗會寫進動作回報（`real robot mirror FAILED: ...`）與 `result["real"]`，模擬 server 的 log 也會印 `[mirror] ...`。
- 只想鏡射手臂：`ROBOT_MIRROR_ACTIONS=move_arm,arm_to,gripper,recenter_arm`。
- 實體 EP 是盲開的：地上要清出跟模擬場地差不多的空間（場地約 5.3 m 見方），起點朝向跟模擬一致。
- 升降柱已從 `Capstone_Playground_nexarm.ttt` 拆掉（2026-11-10）：手臂直接坐在車身上的固定底座塊 `nexarm_pedestal`，模擬沒有 `lift_arm`，跟實體一致；要回到有升降柱的版本用 `scenes/*.backup_20261110_182927.ttt`。
- 想在視窗版 CoppeliaSim 裡看到底盤在跑：`nohup ~/rmenv/bin/python tools/sim_twin.py > /tmp/sim_twin.log 2>&1 &`（模擬分身：把無視窗那台的底盤位姿與升降柱即時複製到視窗那台；視窗那台要保持停止，手臂關節由實體分身同步）。視窗版開法：`cd ~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04 && ./coppeliaSim.sh scenes/Capstone_Playground_nexarm.ttt &`。
- 每題做完會自動走回出發點（2026-11-11）：run_batch 在 `--mirror-real` 時預設帶 `--home-after`（`--no-home-after` 關掉），run_evaluation 抓完真值後呼叫 `return_to_origin`。模擬裡這個動作用占據格地圖導航回出發點（server 啟動時記下的世界座標與朝向，會繞障礙），再轉回出發朝向；鏡射時實體跟著走同一條路，之後 reset_real 再用實體里程計補殘差。agent 也有 `return_to_origin()` 可用，任務說「回到出發點」就會用。
- 批次沒開實體 server 或 EP 斷線時會停下來等（最多 10 分鐘），重啟 `./start_arm.sh --with-ep` 就會自己接著跑；中途停掉用 `--resume <batch_dir>` 續跑。

## 2026-10-22 的一批修正（批次 all_20261021 之後）

批次跑完 100 題（67%）後對照真值逐題追查，改了下面這些。改動都附在各檔案的註解裡，這裡只列結論：

- **轉向方向與前鏡頭鏡像。** `move_chassis` 的 `turn_left_deg` 現在為正＝真正的左轉（從上方看逆時針、`world_heading_deg` 增加）。以前是直接把它送給 SDK 的 z（SDK 的 z 和 `yaw_deg` 都是順時針為正），所以「turn_left」其實會右轉，整套系統靠 `TURN_SIGN=-1` 和 perception 裡的 y 翻轉互相抵消；數字對、名稱反，LLM 一用常識推理（球池在左邊 → 左轉 → 正值）就轉錯邊，e04 / e10 / nav_002 三題簡單題就是這樣丟的。同一時間前鏡頭多翻了一次左右（依據是用反號的 yaw 判定的「左轉」），用相機世界矩陣投影地標驗證過畫面是鏡像的；現在 `_sim_front_image` 只做上下翻，`static_landmarks` / `locate_overhead` / `scan_overhead` / `align_to_tunnel` 的 `turn_left_deg` 一律是 `bearing − world_heading_deg`。驗證：起點 ball_pit 的 turn_left 從 −113 變 +113；`move_chassis(turn_left_deg=90)` 讓世界航向 +87.7；投影框落在實物上。實體機器人的 SDK 串流翻轉也拿掉了，但沒有重新驗證。
- **夾爪要等「模擬時間」。** 視窗版 CoppeliaSim 只跑約 0.55 倍即時，`_act_gripper` 原本牆鐘 sleep 1.5 s 就 `pause()`，手指還沒合到球就停住，夾爪腳本「兩指都碰到才掛接」永遠不成立，抬手臂時把球撥掉——批次 11 題球掉落（r02/r03/r07/r10/r11/p07/t20 桌上萊姆球、h09/p06 紫球、p01 青球、p03 粉紅球）全是這個。現在用 `_sleep_sim(1.5)` 等模擬時間。
- **球的相機對準。** `_ball_offset_mm` 以前取畫面下半部「最大」的同色色塊，背景的橘色隧道比球大，算出球偏左 12 cm、車子轉 18° 離開柱子（h05/p08 共 13 次放棄）。改成取「最低」（最近）的色塊、單次修正最多 6°、偏差超過 12 cm 視為認錯；對準後 ToF 讀不到就重掃。感測器還是讀不到時，用世界座標面向目標小步逼近（`_creep_to_target`）。
- **pick 不再盲推。** 進給迴圈每走一步 ToF 應該變近，連續兩步沒變近就停（h01 把積木推了 1.03 m 推到溜滑梯邊）。
- **approach 的相機確認。** 問 VLM 前把描述裡的座標拿掉（帶座標時常回「看不到」），並先 `recenter_arm`（agent 自己動過手臂時相機角度不對）。
- **align_to_tunnel 對已知結構不靠 VLM。** 新增唯讀動作 `landmark_axis(name)`，tunnel / platform 直接用 server 的軸線與入口。
- **scan_overhead 過濾 1 cm 高的同色碎片**（綠墊子被列成兩個 green block）。
- **系統提示。** done(success=true) 前要逐項核對任務句子裡的每個要求，面向／距離類要求以遙測為證據；說明 turn_left_deg 的方向慣例與「不要用 yaw_deg 算方位」。批次 32 個失敗裡 15 個是 agent 假報成功。
- **基礎設施。** server 對模擬器的 zmq 連線加 15 s socket 逾時與 REQ_RELAXED/REQ_CORRELATE（模擬器崩潰時動作幾秒內失敗、放鎖，不再像 r04 那樣卡死 20 分鐘）；`_nav_pose` 失敗重試三次；`move_chassis` 夾過上限的距離會寫回參數（要求 3.5 m 被夾成 1.0 m、走滿卻被判「被擋住」）；新增 `./start_sim_headless.sh` 無視窗啟動模擬器（視窗版一天崩四次，且只跑 0.55 倍即時）。

## 動作格式：每格要宣告 `name`（2026-10-21）

`robot_agent/llm/schema.py` 給 Gemini 的 response_schema 會把 `RobotAction` 的每個動作欄位都標成 required，gemini-robotics-er 因此把一格裡的 move_chassis、move_arm、…、pick、place **全部填值**；舊的 `unpack()`「取第一個非空欄位」幾乎永遠執行到 move_chassis（記憶寫「用 pick」、實際送出零移動或亂轉，r05／r12 推理正確卻用滿步數）。現在 `RobotAction` 多一個必填的 `name` 列舉欄位，`unpack()` 只執行 `name` 宣告的動作、其他欄位一律忽略；`service.py` 會印 `⚠ LLM filled N action objects … executing X` 當診斷。提示詞 `<output>` 的範例同步改為 `{"name": "move_chassis", "move_chassis": {...}}`。

## 垃圾與收拾類題目（2026-10-21）

`Env/add_litter.py` 在場景放六件垃圾與兩個垃圾桶：`trash_bottle_1/2`（天藍方形水壺 7×7×20 cm）、`trash_can_1/2`（桃紅方罐 6.5×6.5×20 cm）、`trash_cup_1`（淺灰方杯 7×7×16 cm）、`trash_carton_1`（深棕紙盒 7×7×18 cm），80 g（杯 40 g）；6 cm 寬的瓶子夾不穩、24 cm 高的會被推倒，這組尺寸是實測後定的。瓶罐比柱子上的球高一截、顏色都挑成俯視能分辨的：`scan_overhead` 的種類由「顏色 + 深度高度」的場景規則決定（粉紅 ≥ 0.19 m 是罐否則是球、青 ≥ 0.19 m 是瓶否則是球、棕＝盒、白/灰＝杯、積木色 ≥ 0.20 m 是泡棉），拼貼圖只負責顏色詞，全部 ≥ 16 cm 高、≤ 7 cm 寬、**都是方柱**：夾爪腳本只在兩根手指都碰到時才把物件掛接到夾爪上，圓柱只有一邊指墊接觸、掛不上、開車就掉（實測三次都掉），方柱跟小積木一樣可靠。跟小積木同一套 `pick` 流程（扁的紙屑抓不起來所以不放）；`bin_trash`（黑，一般垃圾：杯、盒）與 `bin_recycle`（白，回收：瓶、罐）是 0.35 m 的開口箱，並排在東南角 (1.85, −1.95)、(2.45, −1.95)——整個房間只有這裡放得下。一瓶一罐放在平台底下，俯視相機看不到，要開進去用前相機找，再用 `pick("<物件> in front of me")`（不導航、不問俯視相機，直接把正前方的東西送進指間）。
題庫 `category: tidy` 共 20 題（t01–t20）：撿單件、分類回收、計數、檢查平台下、陷阱題（黃色杯子不存在）、目的式（「讓遊戲場整潔」）。總題數 100：navigation 25、reasoning 23、tidy 20、manipulation 18、grounding 14；今天新增的 66 題 easy/medium/hard 各 22。難度由題目內容的特徵算出（每題的 `difficulty_basis` 欄位）：**easy** = 不夾取、目標直接指名或一次觀察就能答；**medium** = 夾取一件直接指名的物件，或要比較候選才知道目標但不夾取，或多步驟導航；**hard** = 夾取的目標要先推理、夾兩件以上、夾取後還有後續步驟、要升降柱、物件藏在平台下、陷阱題。加題時照這套規則標。
真值條件新增 `robot_at`、`robot_near_object`、`x_min/x_max/y_min/y_max`。

**無視窗模式跑批次**：有視窗的 CoppeliaSim 今天三次異常（兩次當掉、一次模擬迴圈卡死到即時比 0.00），無視窗 `./coppeliaSim -h scenes/Capstone_Playground.ttt` 穩定（即時比約 0.6）。注意它是互動式主控台程式，stdin 一關（EOF）就會自己結束——從終端機跑沒事，從腳本背景啟動要用 `sleep infinity | ./coppeliaSim -h …` 之類的方式維持 stdin。`start_all.sh` 等伺服器的上限改為 150 s（啟動時上千次 zmq 呼叫會暫時把模擬迴圈佔住）。

## 推理類題目與真值判定（2026-10-21）

`robot_eval/dataset.json` 多了 13 題 `category: reasoning`（r01–r13）：目標用位置、比較、排除、屬性或目的來描述（「不在地上而在家具上的那顆球」「離藍桶最近的球」「兩塊藍積木裡離紅墊較近的那塊」「清空桌子」），r13 是前提錯誤的陷阱題（「桌上的粉紅球」，粉紅球其實在地上）——正確行為是什麼都不碰、回報前提不成立。
agent 端對應的能力：`scan_overhead` 列候選、`system_prompt.md` 的 `<task_interpretation>` 要求先列候選、說明選了哪一個再具體指名；`resolve_target` 從任務句子推出物件所在的桌面並只在那一塊裁切放大找。
評分多了 **truth 型里程碑**：`robot_eval/truth.py` 在任務前後直接向模擬器抓零散物件的真實座標與地標範圍（存成 `truth.json`），judge 用 `inside` / `on` / `moved_max_m` / `moved_min_m` / `min_z` / `any_of` / `robot_near` 判定「東西最後真的在桶子裡」「沒有去動別的東西」，不再只靠畫面；真值摘要也會放進 judge LLM 的證據裡。

## 場景 / 感知輔助腳本（`Env/`）

- `Env/build_playground_new.py`：在 CoppeliaSim 裡建一個「室內兒童遊戲場」場景（球池、隧道、平台+溜滑梯、四根角落柱子、收納箱、可拾取小積木…），機器人固定從 `ROBOT_START = (-1.1, -2.51)`（靠南牆入口，面向 +x）出發。同目錄的 `build_playground.py`／`build_playroom.py`／`build_office.py`／`build_task_arena.py` 是其他幾種場景。
- `Env/add_overhead_camera.py`：加一台固定在天花板、往下看的正交攝影機（2026-10-21 起 1024 px：512 px 時 6 cm 的球只有 4 px，俯視指向模型分不出球跟積木；每一步給 LLM 看的那張由 `service.py` 縮回 512，token 不變），`robot_server.py` 開機時會自動連上，提供 `/overhead.jpg`、`world_xy` 遙測跟 `static_landmarks`。
- `Env/add_arm_lift.py`：在底盤跟手臂底座之間插一個直線關節 `arm_lift_joint`（行程 0–0.40 m），整組手臂、手臂相機、前方距離感測器可以一起升高；同時放一張 0.26 m 見方、0.33 m 高的矮桌 `table`（地標；桌面刻意小，夾爪中心只在車頭前約 0.15 m，球在桌子正中央才夠得到）跟桌上的柱子 + 萊姆綠球 `ball_lime`（球心離地 0.474 m）。手臂本身的 (x,y) 座標、外掛的逆運動學完全不變，伺服器用 zmq 直接控制這個關節（`lift_arm`），不經過 SDK。改完要 `./start_all.sh --restart`。
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

**夾爪重量感測（2026-10-20）**：模型裡夾爪跟手臂末端之間本來就有一個力感測器 `Force_sensor`，伺服器把合力放進遙測 `gripper_force_n`：
空夾爪 1.50 N、夾著 50 g 的球或積木 1.99 N，靜止時沒雜訊、跟手臂姿態無關。`pick` 有這個讀值時以它為主要判據（多 0.25 N 以上算夾著），
倒退驗證後、收到搬運姿態後各讀一次；原本只看相機夾爪窗口的顏色，從東側夾桌上的球時球在收手臂的瞬間掉到車前地上，相機仍看到 22% 的顏色而誤報
GRASPED，力感測器則直接回到空夾爪的 1.50 N。`place` 釋放後也用它確認東西真的離手。真機沒有這個感測器，讀值不存在時退回原本的 ToF／相機判定。

**俯視圖找桌上的小物件（2026-10-20）**：整張俯視圖 512 px 涵蓋 7 m，6 cm 的球只有 4 px，Gemini 的指向三次有兩次把「lime green ball」指到西北角的綠色小積木，agent 夾了積木丟進桶子還宣稱成功。現在 `skills.resolve_target` 會從目標字串或整句任務（`set_task_context`，「… ball from the table」）推出物件在哪個桌面上，呼叫 `perception.locate_overhead(..., roi_world=…)` 只把桌子那一塊裁切放大到約 512 px 再問（空結果再問一次），找到的點不在桌子範圍內就回報「桌上沒有」而不去夾別的東西。`system_prompt.md` 也要求 LLM 把任務裡的位置保留在 pick 的 object 參數裡。修完同一題 3 步 122 s 成功、真值確認球在桶內、積木沒動。

**桌上的東西（手臂升降柱，2026-10-20）**：手臂最高也只到離地約 0.25 m，桌面高度的東西靠手臂本身夠不到。`Env/add_arm_lift.py` 在底盤跟手臂底座之間
插一個直線關節，整組手臂可以升 0–0.40 m，前方距離感測器掛在升降座上跟著升，所以升到桌面高度之後「送物件進指間」的幾何跟在地上一模一樣。
伺服器把每個靜態地標的頂面高度 `top_m` 與地面範圍 `bbox` 放進 `static_landmarks`；`skills.resolve_target` 看到目標座標落在桌子（`SURFACE_LANDMARKS`）
範圍內就回報它在桌上，`pick` 導航到位後先 `lift_arm(top_m)`、照常抓、結束時不管成敗都降回 0；`place(table)` 升到桌面高度、前進到車頭離桌緣 3 cm 再低位釋放。
測試目標是桌上柱子上的萊姆綠球（跟地上的球同一套幾何，球心離桌面 0.144 m），`_BALL_HUE` 多了 `lime`。

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
| `lift_arm(height_mm)` | 手臂升降柱的絕對高度 0–400 mm（`Env/add_arm_lift.py` 加的 prismatic 關節）。整組手臂、手臂相機、前方距離感測器一起升，`arm_mm` 不變；遙測多一欄 `lift_mm`。伺服器會輪詢關節位置確認到位（差 > 8 mm 回失敗），`/reset` 會先降到 0。`pick`／`place` 對桌上的目標會自己升降，LLM 只在其他高處目標才需要直接呼叫。 |
| `move_arm` / `arm_to` / `recenter_arm` | 手臂相對移動 / 絕對姿態 / 回預設姿態 (89,117)。`arm_to` 與 `recenter_arm` 都會用 `arm_mm` 核對有沒有真的到位。相機掛在手臂上，動手臂也會改變視角。 |
| `gripper(state, power)` | 開或關夾爪（預設 power 50）。 |
| `pick(object)` | 一次完成抓取：導航到物件前 0.3 m → 張開、手臂放到 (180,30) → 用前方距離感測器一步步把物件送進指間（先到 95 mm，沒夾到再深到 80 mm）→ 關夾爪、抬起 → 倒退 0.25 m 驗證（在手上的話感測器讀值不變）→ 回報 GRASPED 或失敗原因。成功後手臂進搬運姿態，伺服器記住手上物件的感測器讀值（導航忽略它）。 |
| `place(target)` | 一次完成放置：導航到目標 → 收納箱用高位釋放（手臂 (160,170)，夾爪離地 0.25 m 高過桶壁，依距離前進到車頭離桶壁 3 cm）；地墊／地面／其他用低位釋放 → 張開 → 倒退、手臂歸位。 |
| `locate(object)` | 只看前鏡頭，回報物體在畫面上的方位角跟該轉幾度置中，不移動機器人。 |
| `face(object)` | 找到物體並自動轉向對準它（最多修正兩次）。 |
| `scan_overhead(kind)` | 列出俯視圖裡某一類的**所有**物件：世界座標、距離與方位、深度高度、在哪裡（地上／桌上／桶裡／墊上）。兩種問法取聯集找候選 → 每個候選裁切放大成拼貼圖再問一次顏色與種類 → 球／積木最後由深度高度仲裁（球在柱子上頂點 0.164 m、積木 0.151 m）。題目間接指涉目標（「最遠的球」「離球池最近的積木」「不在地上的那顆」「清空桌子」）時先用它列出候選再推理。 |
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

所有數值在 server 端限幅 (單次最多 1 m、180 度、手臂 100 mm)。底盤預設用 `chassis_backend=hybrid`：實體的平移用 SDK 位置控制 `chassis.move`，模擬的平移（2026-11-11 起）用速度控制加里程計回授（`_sim_translate`，最後 15 cm 減速，1 秒沒前進就當被擋），旋轉用手動輪詢 yaw 的速度控制，兩者都會事後比對里程計，SDK 回報成功但實際沒怎麼動的話會被判定失敗（`rejected after the fact`），而不是照單全收；反過來，平移的 SDK 完成訊號逾時但里程計已走到要求距離八成以上時，會改判成功（`accepted as completed by odometry`），因為模擬裡的「到達」訊號常常比車身晚，逾時不代表沒走到（實測要求 0.5 m、里程計 0.50 m 仍回 timeout，agent 誤以為沒動再重下，連續三次就被中止）。一次只執行一個動作，`/stop` 不受動作鎖限制。連續多次偵測到「完全沒動、旁邊又沒東西擋」會觸發 `_auto_recover()`：自動停止模擬、重新播放、重連 SDK，救回控制通道卡死的情況（代價是姿態會跳回上次存檔位置）。

**對外連線**：Agent 只會連你在 `.env` 設定的 LLM API（agent 與 judge 可以是不同家），以及本機的 robot server（8765）與 CoppeliaSim（23000）。沒有遙測、沒有雲端同步、沒有版本檢查。

## 已知狀況 / 排查紀錄

這些是實際跑分中發現、值得知道的行為，不是理論上的邊界情況：

- **場景裡的三張地墊原本是看不見的（2026-10-08 已補）**：`mat_red`／`mat_green`／`mat_blue` 在存檔的場景裡只是 dummy，沒有任何形狀，h02／h05／h06 放對座標也看不到墊子。`Env/add_mats.py` 補上 0.6 m 見方、1.2 cm 厚的彩色墊子（不碰撞、感測器偵測不到，物理與導航不受影響）；同時把橘球柱子從紅墊上移到 (0.15, 0.00)，黃積木從綠墊正中央移到 (0.00, -2.20)（原本 h06「放到綠墊上」一開始就成立）。藍墊跟藍色收納箱相隔 17.5 cm，俯視圖裡是兩塊並排的藍色，靠地標名稱區分。
- **導航地圖快取在相對路徑下永遠不會失效（2026-10-07 修正）**：headless 用 `./coppeliaSim -h scenes/xxx.ttt` 啟動時，模擬器回報的場景路徑是相對於安裝目錄的，從 robot_server 的工作目錄找不到檔案，修改時間一直是 0，改過場景存檔後地圖仍從舊快取載入（新柱子、搬走的收納箱都不在地圖上）。現在會補成完整路徑，找不到檔案就不用快取。
- **平移速度（2026-11-11）**：`LIMITS["xy_speed"]`（0.5）是真實的 m/s。SDK `chassis.move` 的 xy_speed 其實是刻度不是 m/s：低於 0.5 會被夾成 0.5，再編成 `int(160*v-70)` 一個位元組，實體跑出來約是「位元組 ÷ 100」m/s（0.5 → 0.1 m/s、0.75 → 0.5 m/s）。之前 0.3/0.4/0.5 看起來一樣慢又逾時就是這個原因；現在實體用 `_sdk_xy_speed()` 反推刻度。模擬外掛不理這個位元組，所以模擬平移改成回授式。無視窗模擬本身只有約 0.5 倍即時（俯視相機已改成讀圖時才算圖，剩下的是球池等 103 個動態物件的物理），所以模擬裡 0.5 m/s 牆鐘上約 0.25 m/s。測模擬但不想轉送實體：動作參數加 `"mirror": false`。球池的 60 顆球改成靜態（沒有任務會拿它們，動態時佔掉一半物理算力），場景開了即時模式鎖在 1.0 倍；每段停穩 0.15 s；導航步長上限 0.3/0.5/1.0 m（窄縫/靠近障礙/空曠）。到桌子的導航實測 53 s → 29 s。
- **sim 模式的前方相機改成直接讀模擬器的視覺感測器（2026-10-07）**：SDK 的 H.264 串流有解碼延遲，`_fresh_image()` 用固定時鐘上限（共 4 秒）等新畫面，模擬跑得慢（例如用 GUI 版跑批次）時會在新畫面到之前放棄、默默回傳轉向前的舊畫面。10/6 的批次裡 e07 連續四步畫面都慢一個轉向，agent 因此回報錯的柱子順序；整批 21 次大角度旋轉有 6 次轉完畫面幾乎沒變。現在 `frame_jpeg()` 在 sim 模式用 `getVisionSensorImg` 讀當下影像（上下左右翻轉後跟 SDK 穩定畫面差異 3.4，即 JPEG 雜訊），跟模擬速度無關；讀失敗會退回 SDK 串流，`ROBOT_FRONT_CAMERA_SOURCE=sdk` 可以強制用舊路徑。實體機器人仍走 SDK。

- **導航「lost world pose」（2026-10-21）**：`navigate_to` 讀世界姿態用的是導航專用的 zmq 連線（`_nav_sim`）。`RemoteAPIClient` 是一個 REQ socket，兩個執行緒同時呼叫會讓 pyzmq 丟 EFSM（錯誤訊息 `Operation cannot be accomplished in current state`），`_nav_pose()` 讀一次失敗整段導航就回 `lost world pose` 放棄。觸發條件：瀏覽器開著 `/live`，它每秒抓一次 `/map.png`，舊版 `map_png()` 從 HTTP 執行緒呼叫 `_nav_pose()`。批次 all_20261021 的 m01/m05 就是這樣失敗的（開著儀表板、單獨跑 go_to 也能重現，15 次失敗）。修法：`map_png()` 改用輪詢執行緒已經讀好的 `world_xy`/`world_heading_deg`，並把導航連線的 `call` 包一層鎖（撞到會印 `[zmq-nav]`）。修後同樣條件 0 次錯誤。
- **前鏡頭方向（2026-10-22 已推翻）**：下面這段是舊結論，當時用順時針為正的 yaw 判定「左轉」所以判反了；現在不再翻轉，見上面的 2026-10-22 修正。原文：`robot_server.py` 的 `frame_jpeg()` 已經加上水平翻轉修正過去左右相反的問題（實測：送一個遙測證實為真的「左轉」，畫面卻往左滑而不是往右滑）。第一次接上新的模擬器/場景時，建議還是先下一個「turn left 90 degrees then stop」之類的任務，親眼確認方向正確。
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
start_sim_headless.sh                                        無視窗啟動 CoppeliaSim 並載入場景（跑分批次用；--stop 關閉）
run_all.sh                                                   一鍵背景跑全部任務（無視窗模擬 + run_batch.py --restart-sim，模擬器當掉自動重開、該題重跑；--stop 中止）；
                                                             成績即時顯示在 /live 的 batch results
```

## 動作空間之外：讓 LLM 自己推理

`system_prompt.md` 的 `<task_interpretation>` 區塊明講使用者的任務可能是目標或需求（「我好口渴」），不只是逐步指令——LLM 要先想清楚物理上要做什麼，再決定用現成技能還是自己組合原語。`go_to`/`approach`/`pick`/`place`/`drive_through` 仍然是建議的預設選項（已經把導航、抓取踩過的坑都處理好了），但不再是唯一合法選項：任務吻合不到任何一個技能時，LLM 被要求自己用 `move_chassis`/`arm_to`/`move_arm`/`gripper` 推理出一條路，而不是卡住或硬套錯的技能。
