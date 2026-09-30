"""
RoboMaster Robot Server
=======================
跑在「機器人那一端」的小型 HTTP 服務，請在 rmenv (Python 3.6 到 3.8) 裡執行。
它負責三件事：
    1. 透過 rm_connection.connect() 連上模擬器或實體 EP
    2. 把相機畫面與遙測數值整理成 Agent 看得懂的觀察
    3. 接收 Agent 送來的高階動作，做完安全限幅後才真的下指令

Agent 端 (Python 3.11+, robot_agent 的 LLM 層) 只會透過 HTTP 跟這支程式講話，
所以兩邊的 Python 版本、相依套件完全互不干擾，機器人也可以放在遠端。

執行方式：
    source ~/rmenv/bin/activate
    python robot_server.py --mode sim                 # CoppeliaSim 模擬器
    python robot_server.py --mode real                # 實體 EP (router 模式)
    python robot_server.py --mode mock                # 不需要 SDK，拿來測 Agent 迴圈

API：
    GET  /health            存活檢查
    GET  /state             遙測 JSON (位置、姿態、手臂、夾爪、距離、電量)
    GET  /frame.jpg         最新一張相機畫面 (JPEG)
    GET  /overhead.jpg      房間正上方俯視圖 (JPEG)，要先跑 add_overhead_camera.py
    GET  /overhead_depth.png 俯視攝影機的深度圖 (16-bit PNG，每個像素是攝影機到該點的距離，單位 mm)
    GET  /actions           動作清單與安全上限
    POST /action            {"name": "...", "params": {...}}  一次只會執行一個
    POST /stop              緊急停止，不受動作鎖限制
    POST /reset             把機器人回到初始姿態 (mock 會完整重置；sim 和 real 只會收手臂、開夾爪，位置要自己擺回去)

注意：這個檔案刻意只用 Python 3.8 也能跑的語法，請不要在這裡用 X | None 或 list[str]。
"""

import argparse
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

try:
    import cv2
    import numpy as np
except Exception:  # mock 模式下沒有 cv2 也能跑，只是沒有畫面
    cv2 = None
    np = None


# ============================================================================
# 安全上限：LLM 給的任何數值都會先被夾進這個範圍
# ============================================================================
LIMITS = {
    "forward_m": 1.0,       # 單一動作最多前後 1 公尺
    "right_m": 1.0,         # 單一動作最多左右 1 公尺
    "turn_left_deg": 180.0,  # 單一動作最多轉 180 度
    "xy_speed": 0.3,        # 平移速度 m/s
    "z_speed": 45.0,        # 旋轉速度 deg/s
    "arm_mm": 100.0,        # 手臂單次相對位移上限 mm
    "wait_s": 10.0,
    "nav_time_s": 300.0,    # navigate_to 單次時間預算 (s)：橫跨全場約 7 m 的路徑實測要 200-250 s（agent 端逾時 320 s）
}


# 每次取畫面前要先等幾張新影格，用來沖掉解碼器裡的舊畫面。用 tools/measure_camera_lag.py 量過再調整。
FRESH_FRAMES = int(os.environ.get("ROBOT_FRESH_FRAMES", "3"))

try:
    from nav_map import NavMap, world_to_cell  # 占據格地圖 + A*，跟這支檔案同目錄（機器人端）
except Exception as _nav_import_err:  # 沒有 numpy/cv2 時其他功能照常，只是不能導航
    NavMap = None
    world_to_cell = None
    print("[server] nav_map 載入失敗（%s），navigate_to/drive_through 不可用" % _nav_import_err)


def _wrap_deg(deg):
    return (deg + 180.0) % 360.0 - 180.0


RECOVERY_MARKER = "auto-recovered:"  # _note_stall 在自動重播模擬之後附在訊息裡的標記，agent 端也認這個字串


def _cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm3(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / n for c in v)


def _clamp(value, limit):
    try:
        v = float(value)
    except Exception:
        v = 0.0
    if math.isnan(v) or math.isinf(v):
        v = 0.0
    return max(-limit, min(limit, v))


def _fix_int32(v):
    """RoboMaster SDK 在手臂高度是負值的某些姿態下，會把封包誤判成無符號 32 位元整數，
    回傳一個接近 2^32 的巨大正數而不是真正的負值（實測過兩次，同一個模式：
    4294967223、4294967257，換算回去剛好是 -73、-39 這種合理的手臂座標）。
    這裡偵測到這種典型的溢位模式就轉回正確的有符號值，不然 agent 看到「4294967257mm」
    這種不可能的數字會整個被搞混，浪費好幾步想「修正」一個原本沒壞的東西。"""
    try:
        v = int(v)
    except Exception:
        return v
    if v > 0x7FFFFFFF:
        return v - 0x100000000
    return v


ACTION_DOCS = [
    {"name": "move_chassis", "params": {"forward_m": "float", "right_m": "float", "turn_left_deg": "float"},
     "doc": "relative chassis motion; translation is executed first, then rotation"},
    {"name": "move_arm", "params": {"forward_mm": "float", "up_mm": "float"},
     "doc": "relative arm end-effector motion"},
    {"name": "arm_to", "params": {"x_mm": "float", "y_mm": "float"}, "doc": "absolute arm pose; also sets the camera viewpoint"},
    {"name": "recenter_arm", "params": {}, "doc": "move arm back to its default pose"},
    {"name": "gripper", "params": {"state": "open|close"}, "doc": "open or close the gripper"},
    {"name": "set_held", "params": {"tof_mm": "int|null"},
     "doc": "tell the server an object is held in the gripper at this front-sensor distance (navigation then ignores that reading); null when released"},
    {"name": "wait", "params": {"seconds": "float"}, "doc": "do nothing, then re-observe"},
    {"name": "stop", "params": {}, "doc": "stop the chassis"},
    {"name": "navigate_to", "params": {"x": "float", "y": "float", "stop_m": "float", "tolerance_m": "float", "face": "bool"},
     "doc": "plan a collision-free path on the occupancy map and drive to world (x, y); stop_m stops that far before the point and faces it"},
    {"name": "drive_through", "params": {"name": "tunnel|platform"},
     "doc": "drive through a pass-through structure along its long axis, entrance to exit"},
    {"name": "go_to_landmark", "params": {"name": "landmark", "stop_m": "float"},
     "doc": "navigate to a fixed landmark: pass-through structures -> their entrance, others -> their front edge, facing it"},
]


# ============================================================================
# Backend：真的機器人 / 模擬器
# ============================================================================
def _reconcile_timeout(result, requested_m, travelled_m, accept_ratio=0.8):
    """平移逾時後用里程計補判。

    SDK 的 wait_for_completed 在模擬裡常常等不到「到達」訊號，但車身其實已經走到了
    （實測要求 0.5 m、里程計 0.50 m 仍回 timeout）。以前直接判失敗，agent 以為沒動又重下一次，
    連續三次失敗就被整個任務中止。現在：走到要求距離的八成以上 → 改判成功，訊息說清楚是靠里程計
    判定的；走不到 → 維持失敗（真的被擋住）。回傳 True 代表有改判。"""
    t = result.get("timed_out")
    if not t or requested_m <= 0.05 or travelled_m < requested_m * accept_ratio:
        return False
    result["ok"] = True
    note = ""
    if abs(result.get("turn_skipped") or 0) > 1e-3:
        note = " The rotation part of this command was NOT executed; request it again."
    result["message"] = ("moved %.2fm of %.2fm requested (SDK completion signal timed out after %.1fs, "
                         "accepted as completed by odometry).%s" % (travelled_m, requested_m, t, note))
    return True


def _find_robot_root(sim):
    """找出場景裡「真的」被 simRobomaster 外掛驅動的那台 RoboMaster。

    場景可能有不只一台 alias 叫 RoboMaster 的模型（實測踩過：一台是被拆掉 GyroSensor 跟 visual
    連桿的複製品，外掛的 create_ep 對它回 -1 不收，但它剛好排在前面，sim.getObject('/RoboMaster')
    就選到它），這時距離感測器、world_xy、靜態地標全部掛在一台不會動的車上，agent 看到的相機畫面
    卻來自另一台——每個「找 X 並回報距離」的任務都會拿到錯的數字，導航任務則整批失效。
    這裡改成：列出所有頂層的 RoboMaster 模型，優先挑有 GyroSensor 子物件的（完整的 EP 模型，
    外掛才會註冊），有多台就在 log 大聲警告。"""
    roots = [o for o in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2)
             if sim.getObjectAlias(o) == "RoboMaster"]
    if not roots:
        raise RuntimeError("找不到 /RoboMaster")

    def complete(h):
        return any(sim.getObjectAlias(o) == "GyroSensor" for o in sim.getObjectsInTree(h, sim.handle_all, 0))

    good = [h for h in roots if complete(h)]
    chosen = (good or roots)[0]
    if len(roots) > 1:
        print("[server] !! 場景裡有 %d 台 RoboMaster 模型 (handles %s)，其中有 GyroSensor 的完整模型是 %s；"
              "感測器與 world_xy 掛到 handle %d。請把多餘的那台從場景刪掉並存檔，否則起點校驗與姿態都會錯。"
              % (len(roots), roots, good, chosen))
    elif not good:
        print("[server] !! /RoboMaster (handle %d) 沒有 GyroSensor，可能是被拆過的模型，外掛未必能驅動它" % chosen)
    return chosen


class RoboMasterBackend(object):
    def __init__(self, mode, chassis_backend="move"):
        self.mode = mode
        self.chassis_backend = chassis_backend
        self.telemetry = {}
        self._tlock = threading.Lock()
        self._stall_count = 0  # 連續幾次「完全沒動、旁邊又沒東西擋」，用來偵測控制腳本整個卡死

        self._connect_robot()
        self._start_prox_sensors()
        self._start_overhead_cam()

    def _connect_robot(self):
        """建立/重新建立跟 SDK 的連線，開相機串流，訂閱所有 telemetry。
        獨立成方法是因為 _auto_recover() 也要用同一套流程重新連一次。"""
        from rm_connection import connect  # 連線設定全部留在你原本的檔案裡

        print("[server] 連線中 (mode=%s) ..." % self.mode)
        self.ep = connect(self.mode)
        try:
            print("[server] 連線成功，韌體版本：", self.ep.get_version())
        except Exception as e:
            print("[server] 取不到韌體版本 (可忽略)：", e)

        self._camera_ok = False
        self._camera_fail_streak = 0
        self._camera_live_ok = False
        try:
            self.ep.camera.start_video_stream(display=False)
            self._camera_ok = True
            self._camera_live_ok = True
        except Exception as e:
            print("[server] 相機串流啟動失敗，Agent 會在沒有畫面的情況下運作：", e)

        self._subscribe_all()

    def _auto_recover(self):
        """實測發現這個 CoppeliaSim 的 RoboMaster 控制腳本，不需要撞到東西，跑一段時間、
        做過幾次動作之後就會自己整個卡死（旋轉、手臂全部沒反應，但 telemetry 還在推送）；
        目前找到唯一有效的救法是「停止再啟動模擬」讓腳本重新初始化，光是重連 SDK 沒有用。
        代價：模擬重啟後機器人的姿態會回到上次存檔的樣子，正在進行的任務等於要重新來過，
        但總比整個卡死、後面所有動作都失敗要好。"""
        print("[server] !! 偵測到控制通道卡死（連續 %d 次完全沒動、旁邊又沒東西擋），"
              "嘗試自動重啟模擬..." % self._stall_count)
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            sim = RemoteAPIClient().require("sim")
            sim.stopSimulation()
            time.sleep(2)
            sim.startSimulation()
            time.sleep(2)
        except Exception as e:
            print("[server] 自動重啟模擬失敗：%s，這條自救路徑放棄，需要人工處理" % e)
            self._stall_count = 0
            return False
        try:
            self.ep.close()
        except Exception:
            pass
        try:
            self._connect_robot()
        except Exception as e:
            print("[server] 重新連線 SDK 失敗：%s" % e)
            self._stall_count = 0
            return False
        self._stall_count = 0
        print("[server] 自動恢復完成，模擬已重啟、SDK 已重新連線")
        return True

    def _note_stall(self, stalled):
        """跟 move_chassis 共用同一個卡死計數器：手臂動作 timeout、或回報完成但實際沒有移動，
        一樣是控制腳本卡死的症狀 (實測過：手臂卡死時 arm_to/move_arm/recenter_arm 全部一起罷工)。
        回傳非空字串代表這次呼叫剛好觸發了自動恢復，呼叫端要把它附加到回應訊息裡。"""
        if not stalled:
            self._stall_count = 0
            return ""
        self._stall_count += 1
        if self._stall_count >= 3 and self._auto_recover():
            return (" | auto-recovered: the control channel was stuck (arm/chassis stopped responding "
                     "even though nothing was blocking it), so the simulation was automatically restarted. "
                     "Your arm and chassis pose may have jumped back to the last saved position — "
                     "re-check telemetry and the camera image before continuing.")
        return ""

    # ---- 真的距離感測器：繞過 RoboMaster SDK (它的 sensor.sub_distance 在這個模擬器裡
    # 永遠回報 0，沒有實作)，直接用 CoppeliaSim 的 zmqRemoteApi 建立三個 proximity sensor
    # (prox_front/prox_left/prox_right) 掛在機器人身上。只有 sim 模式才有意義；
    # 找不到套件就只印一行訊息、不影響其他功能。----
    #
    # 實測抓到的 bug：sim.createProximitySensor() 是「模擬執行期間」建立的物件，模擬只要
    # 被停止過一次（使用者手動、start_all.sh 偵測到停止幫忙按播放、或 _auto_recover() 自己
    # 卡死自救時做的「停止再重啟模擬」）就會被整個清掉，但 _poll_prox_sensors() 以前是
    # except Exception: pass，讀不到就當沒發生，tof_*_mm 永遠凍結在最後一次讀到的值
    # （通常是 9999 = 沒偵測到東西）——機器人真的撞上/卡進東西時完全不會被發現，這正是
    # 之前好幾次「log 顯示 tof 全部 9999、機器人卻明明卡死在球池邊界」的真正原因。
    # 現在建立邏輯直接內建在 server 裡（不用再手動跑 add_proximity_sensors.py），
    # 且偵測到連續讀取失敗就自動整批重建，作法跟 world_xy／前方相機是同一套。
    # 高度是相對 root（root 離地 0.065 m）：以前 0.12 → 離地 0.185 m，20° 錐在 0.42 m 內看不到 10 cm 高的收納箱壁
    # （中等批次 m06：停在桶前 0.31 m 讀 9999，0.48 m 才讀得到）。改成離地 0.10 m、16° 錐：0.15-0.6 m 都看得到 ≥ 7 cm
    # 的東西，錐底在 0.6 m 處仍離地 6 mm，不會掃到地板。
    _PROX_SPECS = [
        ("prox_front", "fwd", 0.18, 0.00, 0.035, 0.60, 16),
        ("prox_left", "left", 0.05, -0.14, 0.035, 0.35, 16),
        ("prox_right", "right", 0.05, 0.14, 0.035, 0.35, 16),
    ]

    def _create_prox_sensors(self, sim, root):
        """(重）建立三個距離感測器並掛到機器人身上。初次啟動跟 _poll_prox_sensors
        發現物件被模擬重啟清掉時的重建，都走這條路徑。"""
        yaw = None
        t0 = time.time()
        while time.time() - t0 < 3.0:
            with self._tlock:
                yaw = self.telemetry.get("yaw_deg")
            if yaw is not None:
                break
            time.sleep(0.05)
        if yaw is None:
            yaw = 0.0
            print("[server] 距離感測器校正：3 秒內沒等到 yaw telemetry，先用 0 度校正")
        # 感測器要跟「真實車頭」對齊，SDK 的 yaw 遙測跟世界座標系沒有關係：第八輪回歸測試 server 啟動時
        # yaw=65.2 而車頭其實是世界 -65.2°（兩者正好反號），三個感測器全部歪了 130°，110 筆 ToF 讀值只有 8 筆
        # 對得到前方的障礙，其他全是側牆/側邊物件，害導航一路縮步、亂標障礙、在隧道裡誤判「被擋住」。
        # 優先用輪子幾何算車頭，讀不到才退回 yaw。
        try:
            wh = self._find_wheel_handles(sim, root)
            hd = self._heading_from_wheels(sim, wh) if wh else None
        except Exception as e:
            hd = None
            print("[server] 距離感測器校正：輪子幾何讀取失敗（%s），退回 SDK yaw" % e)
        if hd is not None:
            print("[server] 距離感測器校正：用輪子幾何算出的車頭 %.1f°（SDK yaw=%.1f 不採用）" % (hd, yaw))
            yaw = hd

        for h in sim.getObjectsInTree(root, sim.handle_all, 0):
            if sim.getObjectAlias(h) in ("prox_front", "prox_left", "prox_right"):
                sim.removeObjects([h])

        rad = math.radians(yaw)
        fwd = (math.cos(rad), math.sin(rad), 0.0)
        right = (math.sin(rad), -math.cos(rad), 0.0)
        up = (0.0, 0.0, 1.0)
        root_pos = sim.getObjectPosition(root, sim.handle_world)

        handles = {}
        for name, dir_kind, along_fwd, along_right, height, rng, angle_deg in self._PROX_SPECS:
            direction = {"fwd": fwd, "left": tuple(-v for v in right), "right": right}[dir_kind]
            pos = [root_pos[i] + fwd[i] * along_fwd + right[i] * along_right + up[i] * height
                   for i in range(3)]
            z_axis = direction
            x_axis = _norm3(_cross3(up, z_axis))
            y_axis = _cross3(z_axis, x_axis)
            m = [x_axis[0], y_axis[0], z_axis[0], pos[0],
                 x_axis[1], y_axis[1], z_axis[1], pos[1],
                 x_axis[2], y_axis[2], z_axis[2], pos[2]]
            angle = math.radians(angle_deg)
            far_size = 2 * rng * math.tan(angle / 2)
            h = sim.createProximitySensor(
                sim.proximitysensor_pyramid, 16, 4,
                [4, 4, 4, 4, 0, 0, 0, 0],
                [0.0, rng, 0.02, 0.02, far_size, far_size, 0.0, 0.0, 0.0, angle, 0.0, 0.0, 0.005, 0.0, 0.0])
            sim.setObjectMatrix(h, m, sim.handle_world)
            sim.setObjectAlias(h, name)
            sim.setObjectParent(h, root, True)
            handles[name] = h
        # 車身零件全部設成「不可偵測」：距離感測器裝在車頭前 0.18 m、高 0.12 m，夾爪/相機連桿在大多數手臂姿態下
        # 都掃進感測錐裡，感測器回報的第一個物件是自己、被過濾成 9999，前面 0.3 m 的積木就這樣看不到
        # （中等批次 nav_002/m03/m06 的「ToF 讀 9999 → 沒停在物件前」全是這個）。設成不可偵測後感測器直接穿過自己。
        try:
            det = getattr(sim, "objectspecialproperty_detectable_all", None) or sim.objectspecialproperty_detectable
            n_fix = 0
            robot_shapes = set()
            for sh in sim.getObjectsInTree(root, sim.object_shape_type, 0):
                alias = sim.getObjectAlias(sh)
                # 夾爪夾著的物件在模擬裡會被掛到機器人樹下：實測 server 重啟時 small_red 在夾爪裡，被一起設成
                # 不可偵測，放回地上後 ToF 從此看不到它。零散物件（積木/球/泡棉）一律不算車身。
                if alias.startswith(("small_", "ball_", "foam_", "pit_ball")):
                    continue
                robot_shapes.add(sh)
                sp = sim.getObjectSpecialProperty(sh)
                if sp & det:
                    sim.setObjectSpecialProperty(sh, sp & ~det)
                    n_fix += 1
            # 自我修復：車身以外的形狀都應該可偵測（真實世界的 ToF 什麼都看得到）
            n_heal = 0
            for sh in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0):
                if sh in robot_shapes:
                    continue
                sp = sim.getObjectSpecialProperty(sh)
                if not (sp & det):
                    sim.setObjectSpecialProperty(sh, sp | det)
                    n_heal += 1
            if n_fix or n_heal:
                print("[server] 距離感測器可偵測旗標：車身 %d 個零件設成不可偵測，%d 個場景物件恢復可偵測" % (n_fix, n_heal))
        except Exception as e:
            print("[server] 設定車身不可偵測失敗：%s" % e)
        print("[server] 距離感測器已建立（校正用 yaw=%.1f）：front/left/right" % yaw)
        return handles

    def _start_prox_sensors(self):
        self._prox_handles = None
        if self.mode != "sim":
            return
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            sim = RemoteAPIClient().require("sim")
            root = _find_robot_root(sim)
            handles = self._create_prox_sensors(sim, root)
            tree = sim.getObjectsInTree(root, sim.handle_all, 0)
        except Exception as e:
            print("[server] 距離感測器建立失敗（%s），tof_*_mm 不會出現在 /state" % e)
            return
        self._sim_for_prox = sim
        self._prox_handles = handles
        self._robot_tree = set(tree) | {root}
        threading.Thread(target=self._poll_prox_sensors, daemon=True).start()
        print("[server] 真實距離感測器已連上：front/left/right")

    def _poll_prox_sensors(self):
        field = {"prox_front": "tof_front_mm", "prox_left": "tof_left_mm", "prox_right": "tof_right_mm"}
        fail_streak = 0
        while True:
            sim = self._sim_for_prox
            any_fail = False
            for alias, h in list(self._prox_handles.items()):
                try:
                    res, dist, _point, obj, _n = sim.readProximitySensor(h)
                    # 手臂/夾爪會動，姿勢不一樣時可能剛好掃到自己的手臂而不是真的外部障礙物，
                    # 偵測到的物件如果是機器人自己身上的部件就當作沒偵測到。
                    if res and obj in self._robot_tree:
                        res = 0
                    self._set(field[alias], int(dist * 1000) if res else 9999)
                except Exception:
                    any_fail = True
            if not any_fail:
                fail_streak = 0
                time.sleep(0.1)
                continue
            fail_streak += 1
            if fail_streak == 1 or fail_streak % 20 == 0:
                print("[server] 距離感測器讀取失敗（連續第 %d 次，物件可能被模擬重啟清掉了）" % fail_streak)
            if fail_streak >= 20:
                print("[server] 距離感測器連續失敗 %d 次，嘗試重建..." % fail_streak)
                try:
                    root = _find_robot_root(sim)
                    handles = self._create_prox_sensors(sim, root)
                    tree = sim.getObjectsInTree(root, sim.handle_all, 0)
                    self._prox_handles = handles
                    self._robot_tree = set(tree) | {root}
                    print("[server] 距離感測器已重建")
                    fail_streak = 0
                except Exception as e:
                    print("[server] 距離感測器重建失敗：%s（2 秒後再試）" % e)
                    time.sleep(2.0)
                    continue
            time.sleep(0.1)

    # ---- 俯視攝影機：add_overhead_camera.py 建立的正交攝影機，給 agent 一個
    # locate_overhead(object) 動作，一次看到整個場地算方位角跟距離，不用靠自轉一步步搜索。
    # 同時把機器人「真的」世界座標 (world_xy) 放進 /state，因為 position_m 是相對 session
    # 開始時的座標、原點是任意的，跟俯視圖的世界座標系對不起來，agent 端要用 world_xy 換算。
    def _connect_overhead_sim(self):
        """(重）建立俯視攝影機用的 zmqRemoteApi 連線跟物件 handle。初次啟動跟
        _poll_world_xy 發現連線壞掉時的重連，都走這條路徑，避免兩邊各寫一份。"""
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        sim = RemoteAPIClient().require("sim")
        h = sim.getObject("/overhead_cam", {"noError": True})
        if h == -1:
            raise RuntimeError("找不到 overhead_cam，先跑 add_overhead_camera.py")
        root = _find_robot_root(sim)
        return sim, h, root

    # 場景裡不會動的主要結構地標——名字是 build_playground_new.py 建場景時取的別名，
    # 直接查詢目前這個場景的真實世界座標（不是照抄腳本裡的 POS 字典），這樣不管實際
    # 載入的是哪個變體、座標有沒有跟腳本不一樣，都保證跟眼前這個場景一致。查不到的
    # 就跳過，不影響其他地標。刻意只列大型結構（隧道/球池/柱子/平台/長椅/收納箱），
    # 不含地墊、裝飾泡棉、可拾取的小積木/球——後者位置會被任務移動，寫死進地圖只會
    # 誤導 agent；地墊則太扁、太容易跟任務目標搞混。
    _STATIC_LANDMARK_ALIASES = [
        "tunnel", "ball_pit", "bench", "bin_balls", "bin_blocks", "platform",
        "pillar_red", "pillar_blue", "pillar_yellow", "pillar_green", "stairs", "slide",
        # 地墊：放置任務的目標（難題 h05/h06「放到紅墊/綠墊」），俯視視覺模型找不到扁平的地墊，改當固定地標
        "mat_red", "mat_green", "mat_blue",
    ]

    def _query_static_landmarks(self, sim):
        landmarks = {}
        self._static_landmark_handles = {}
        all_objs = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
        by_alias = {}
        for h in all_objs:
            try:
                by_alias.setdefault(sim.getObjectAlias(h), h)
            except Exception:
                pass
        for name in self._STATIC_LANDMARK_ALIASES:
            h = by_alias.get(name)
            if h is None:
                continue
            try:
                p = sim.getObjectPosition(h, sim.handle_world)
                landmarks[name] = (round(p[0], 2), round(p[1], 2))
                self._static_landmark_handles[name] = h
            except Exception:
                pass
        if landmarks:
            print("[server] 靜態地標已建立：%s" % ", ".join(sorted(landmarks)))
        return landmarks

    def _start_overhead_cam(self):
        self._overhead_handle = None
        self._static_landmarks_xy = {}
        if self.mode != "sim":
            return
        try:
            sim, h, root = self._connect_overhead_sim()
        except Exception as e:
            print("[server] 略過俯視攝影機（%s），/overhead.jpg 跟 world_xy 不會出現" % e)
            return
        self._sim_for_overhead = sim
        self._sim_for_overhead_lock = threading.Lock()
        self._overhead_handle = h
        self._overhead_root = root
        self._static_landmarks_xy = self._query_static_landmarks(sim)
        self._robot_cam = self._robot_camera_handle(sim, root)
        self._wheel_handles = self._find_wheel_handles(sim, root)
        if not self._wheel_handles:
            print("[server] 找不到四個輪子關節，車頭方向退回用相機視線（手臂抬高時可能不準）")
        self._start_nav(root)
        threading.Thread(target=self._poll_world_xy, daemon=True).start()
        print("[server] 俯視攝影機已連上：/overhead.jpg")

    # ---- 導航：占據格地圖 + A* + 閉迴路執行（見 nav_map.py）----
    # turn_left_deg 為正時，世界朝向（相機 z 軸方位角）會「減少」：實測 +20° 讓朝向從 0.8 變成 -23.9。
    # 這台車/模擬器的座標系是鏡像的（README 說的「y 軸相反」）。第一次大角度轉向會再核對一次，反了就翻轉。
    TURN_SIGN = -1.0

    @staticmethod
    def _robot_camera_handle(sim, root):
        cams = sim.getObjectsInTree(root, sim.object_visionsensor_type, 0)
        return cams[0] if cams else -1

    @staticmethod
    def _find_wheel_handles(sim, root):
        """四個輪子關節的 handle：車頭方向 = 前輪中點 - 後輪中點。相機掛在手臂上，手臂一抬高相機就可能
        仰過垂直、視線投影到地面整個反 180 度（實測 approach 抬過手臂後，導航就往反方向走）；輪子不會。"""
        want = {"front_left_wheel_joint": None, "front_right_wheel_joint": None,
                "rear_left_wheel_joint": None, "rear_right_wheel_joint": None}
        for o in sim.getObjectsInTree(root, sim.object_joint_type, 0):
            a = sim.getObjectAlias(o)
            if a in want and want[a] is None:
                want[a] = o
        return want if all(v is not None for v in want.values()) else None

    @staticmethod
    def _heading_from_wheels(sim, wh):
        """車頭方向（度）＝前輪中點 − 後輪中點 的方向。"""
        fl = sim.getObjectPosition(wh["front_left_wheel_joint"], sim.handle_world)
        fr = sim.getObjectPosition(wh["front_right_wheel_joint"], sim.handle_world)
        rl = sim.getObjectPosition(wh["rear_left_wheel_joint"], sim.handle_world)
        rr = sim.getObjectPosition(wh["rear_right_wheel_joint"], sim.handle_world)
        fx, fy = (fl[0] + fr[0]) / 2.0 - (rl[0] + rr[0]) / 2.0, (fl[1] + fr[1]) / 2.0 - (rl[1] + rr[1]) / 2.0
        return math.degrees(math.atan2(fy, fx))

    def _world_heading(self, sim):
        """真實世界車頭方向（度）。優先用輪子幾何；沒有輪子資訊才退回相機視線方向。"""
        wh = getattr(self, "_wheel_handles", None)
        if wh:
            return self._heading_from_wheels(sim, wh)
        m = sim.getObjectMatrix(self._robot_cam, sim.handle_world)
        return math.degrees(math.atan2(m[6], m[2]))

    def _start_nav(self, root):
        self._nav = None
        self._nav_sim = None
        self._nav_last_path = None
        self._nav_last_goal = None
        if NavMap is None:
            return
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            self._nav_sim = RemoteAPIClient().require("sim")  # 導航專用連線：只在動作執行緒用，不跟輪詢搶
            t0 = time.time()
            self._nav = NavMap(self._nav_sim, root)
            print("[server] 占據格地圖已建立：%d 個靜態形狀、%d 個通道，%.1fs（GET /map.png 可以看）"
                  % (self._nav.static_shapes, len(self._nav.passages), time.time() - t0))
            for pg in self._nav.passages:
                print("[server] 通道 (%.2f, %.2f) 軸 %.0f°：中線側向偏移 %+.2f m，沿線最小淨空 %.2f m"
                      % (pg["center"][0], pg["center"][1], pg["axis_deg"], pg.get("lateral", 0.0), pg.get("min_clearance", 0.0)))
        except Exception as e:
            self._nav = None
            print("[server] 占據格地圖建立失敗（%s），navigate_to/drive_through 不可用" % e)

    def _nav_pose(self):
        """機器人真實世界姿態 (x, y, heading_deg)，用導航專用的 zmq 連線讀。"""
        try:
            sim = self._nav_sim
            p = sim.getObjectPosition(self._overhead_root, sim.handle_world)
            return p[0], p[1], self._world_heading(sim)
        except Exception as e:
            print("[server] 讀取世界姿態失敗：%s" % e)
            return None

    def _nav_mark_blocked(self, pose, dist_m):
        """把前方 dist_m 處一小塊標成動態障礙，讓下一次規劃繞開（ToF 看到東西、或指令沒走到時用）。"""
        try:
            x = pose[0] + dist_m * math.cos(math.radians(pose[2]))
            y = pose[1] + dist_m * math.sin(math.radians(pose[2]))
            self._nav.mark_blocked(x, y)
        except Exception as e:
            print("[server] 標記障礙失敗：%s" % e)

    def map_png(self):
        if not getattr(self, "_nav", None):
            return None
        try:
            return self._nav.render_png(robot_pose=self._nav_pose(), path=self._nav_last_path,
                                        goal=self._nav_last_goal)
        except Exception as e:
            print("[server] 地圖繪製失敗：%s" % e)
            return None

    def _turn_to(self, target_heading, pose, log, tol=5.0, max_tries=3):
        """轉到世界朝向 target_heading（度），量測後不夠準就再修，最多 max_tries 次。回傳 (ok, new_pose)。
        轉向方向跟要求相反（多半是頂到東西被帶著轉）→ 回 False 讓呼叫端當作轉向被拒處理，不再自動翻轉符號。"""
        for _ in range(max_tries):
            delta = _wrap_deg(target_heading - pose[2])
            if abs(delta) <= tol:
                return True, pose
            r = self.do("move_chassis", {"forward_m": 0, "right_m": 0,
                                         "turn_left_deg": _clamp(self.TURN_SIGN * delta, LIMITS["turn_left_deg"])})
            new_pose = self._nav_pose()
            if not r.get("ok") or new_pose is None:
                return False, (new_pose or pose)
            remaining = _wrap_deg(target_heading - new_pose[2])
            print("[nav] turn %+.0f° -> remaining %+.0f° (tol %.0f)" % (delta, remaining, tol))
            # 「轉錯邊」的判準是：轉完之後離目標反而更遠。不能看量測值的正負號——接近 180 度的轉向，
            # 量到的 wrap 值正負隨機（要求 -176 量到 +178 是同一個旋轉），舊判準會誤判成反向。
            if abs(delta) > 25 and abs(remaining) > abs(delta) + 20:
                log.append("turn made things worse (requested %+.0f, still %+.0f to go) — probably in contact" % (delta, remaining))
                print("[nav] !! turn went wrong: requested %+.0f, remaining %+.0f at (%.2f, %.2f)" % (delta, remaining, new_pose[0], new_pose[1]))
                return False, new_pose
            pose = new_pose
        return abs(_wrap_deg(target_heading - pose[2])) <= tol * 2, pose

    def _act_navigate_to(self, p):
        if not getattr(self, "_nav", None):
            return {"ok": False, "message": "navigation map unavailable (needs sim mode with the overhead camera)"}
        try:
            target = (float(p["x"]), float(p["y"]))
        except Exception:
            return {"ok": False, "message": "navigate_to needs x and y in world metres"}
        stop_m = max(0.0, float(p.get("stop_m") or 0.0))
        # 站位要準：跑分的 ToF 檢查窗只有 150-450 mm，到達容忍 0.15 + 最後一步少走 3 cm 會讓 stop 0.30 停成 0.43
        tol = max(0.05, float(p.get("tolerance_m") or (0.08 if stop_m > 0 else 0.15)))
        face = bool(p.get("face", stop_m > 0))
        budget = float(p.get("max_time_s") or LIMITS["nav_time_s"])
        t0 = time.time()
        counters = {"prims": 0, "replans": 0}
        log = []
        state = {"goal": None, "tight": False}

        def result(ok, why=""):
            pose_now = self._nav_pose() or (float("nan"), float("nan"), float("nan"))
            g = state["goal"]
            d_goal = math.hypot(g[0] - pose_now[0], g[1] - pose_now[1]) if g else float("nan")
            d_target = math.hypot(target[0] - pose_now[0], target[1] - pose_now[1])
            msg = ("navigate_to (%.2f, %.2f)%s: %s at world (%.2f, %.2f) heading %.0f deg; %.2f m from goal, "
                   "%.2f m from target; %d moves, %d replans, %.0fs%s%s"
                   % (target[0], target[1], (" stop_m=%.2f" % stop_m) if stop_m else "",
                      "arrived" if ok else "stopped", pose_now[0], pose_now[1], pose_now[2], d_goal, d_target,
                      counters["prims"], counters["replans"], time.time() - t0,
                      (" — " + why) if why else "", (" [" + "; ".join(log) + "]") if log else ""))
            return {"ok": ok, "message": msg, "final_world_xy": [round(pose_now[0], 3), round(pose_now[1], 3)],
                    "heading_deg": round(pose_now[2], 1), "distance_to_target_m": round(d_target, 2)}

        pose = self._nav_pose()
        if pose is None:
            return {"ok": False, "message": "cannot read the robot's world pose"}
        self._nav_trajectory = [pose]  # 每個 primitive 之後的姿態，drive_through 用來判定有沒有真的穿過結構內部
        goal = target
        self._nav.refresh_dynamic()  # 目標物也算障礙：它的膨脹會把終點擋在合理距離外，排除它反而讓車撞上去（回歸測試 small_green）
        if stop_m > 0:
            # 終點 = 目標往機器人這一側退 stop_m；退到的點還在障礙/膨脹層裡（長椅、收納箱、球池這種大結構的
            # 中心）就沿同一條線繼續往外推，直到第一個可走的格子，這樣「go_to 長椅」會停在長椅前緣。
            d = math.hypot(target[0] - pose[0], target[1] - pose[1])
            if d > stop_m:
                # 沿「目標→機器人」的方向往外推到第一個可走格；那條線被別的東西擋住時（回歸測試：small_blue_2
                # 旁邊的紫球），換左右各 30/60/90 度的方向再試，取最靠近 stop_m 的那個。多留 0.15 m 給轉身。
                base = math.atan2(pose[1] - target[1], pose[0] - target[0])
                # stop_m 的語意是「車頭前緣到目標表面的距離」（人講的「停在它前面 30 cm」）。沿每個方向先找目標本體的邊緣
                # （從中心往外第一個沒被占據的格子），站位從 邊緣 + stop_m + 車身半長 0.166 起算；以前從中心算 stop_m，
                # 停 0.3 m 時物件已在 ToF 感測器（車頭前 0.18 m）後面、讀 9999，「停在它前面」的檢查永遠不過。
                # 轉身用的一般淨空維持 0.30（不能把牆算進去：角落的柱子會 2 m 內找不到站位）。
                def find_cands(sm):
                    cands = []
                    for off in (0, 30, -30, 60, -60, 90, -90, 120, -120):
                        a = base + math.radians(off)
                        dx, dy = math.cos(a), math.sin(a)
                        # 目標本體的邊緣：沿射線往外掃，記最後一個被占據的格子；碰到占據格之後連續 0.35 m 沒東西就停。
                        # 球池這種中空結構射線先經過空的內部再碰到牆，不能看到第一個空格就停。
                        edge, last_occ, free_run, t = 0.0, None, 0.0, 0.0
                        while t < 1.5:
                            ex, ey = world_to_cell(target[0] + dx * t, target[1] + dy * t)
                            if self._nav.occ[ey, ex]:
                                last_occ, free_run = t, 0.0
                            elif last_occ is not None:
                                free_run += 0.05
                                if free_run >= 0.35:
                                    break
                            t += 0.05
                        if last_occ is not None:
                            edge = last_occ + 0.05
                        back = round(edge + sm + 0.166 - 0.10, 2)  # 邊緣掃描以 5 cm 格子多估半格，扣回來（實測 stop 0.45 → ToF 0.60）
                        while back < 2.0:
                            cand = (target[0] + dx * back, target[1] + dy * back)
                            # 終點要能轉身：淨空 ≥ 掃掠半徑（在通道裡的話不要求，本來就只沿軸走）
                            if self._nav.is_free(*world_to_cell(*cand)) and (
                                    self._nav.clearance_at(*cand) >= 0.30 or self._nav.passage_axis_at(*cand) is not None):
                                break  # 不再多退 5 cm：轉身空間已由淨空 ≥ 0.30 保證
                            back += 0.05
                        if back < 2.0:
                            cands.append((back, target[0] + dx * back, target[1] + dy * back, abs(off)))
                    # 先取最靠近 stop_m 的，其次偏離直線最少的；隧道口窄道裡的站位排最後（第八輪 bin_balls：
                    # 舊的 sort() 在 back 相同時比 x，把車引到隧道口去對齊）
                    cands.sort(key=lambda c: (self._nav.passage_axis_at(c[1], c[2]) is not None, round(c[0], 2), c[3]))
                    return cands
                cands = []
                for sm in (stop_m, round(max(0.2, stop_m - 0.15), 2), round(max(0.2, stop_m - 0.25), 2)):
                    cands = find_cands(sm)
                    if any(c[0] <= 1.0 for c in cands):
                        break  # 要求的距離找得到 1 m 內的站位就用；找不到就縮短 stop_m 再找（角落口袋太淺時）
                if not cands:
                    # 目標周圍找不到可站的點：老實回報失敗，不要把終點設成原地（回歸測試出現過「停在 4.79 m 外卻成功」）
                    return {"ok": False, "message": "navigate_to (%.2f, %.2f): no reachable standoff point around the target"
                            % (target[0], target[1])}
                goal_candidates = [(c[1], c[2]) for c in cands]
            else:
                goal_candidates = [(pose[0], pose[1])]
        else:
            goal_candidates = [goal]
        # 候選終點依距離逐一試規劃：最近的那個可能落在被圍住的小口袋裡（綠柱子旁的泡棉方塊），
        # 規劃不到就換下一個方向，不要直接放棄。
        wps = actual_goal = None
        info = "no candidate goal"
        best_normal = best_tight = None
        for goal in goal_candidates:
            w_, g_, i_ = self._nav.plan((pose[0], pose[1]), goal)
            if w_ is None:
                info = i_
                continue
            back = math.hypot(goal[0] - target[0], goal[1] - target[1])
            if isinstance(i_, dict) and i_.get("tight"):
                if best_tight is None:
                    best_tight = (back, w_, g_, i_)
            elif best_normal is None:
                best_normal = (back, w_, g_, i_)
                if best_tight is None or back <= best_tight[0] + 0.5:
                    break
        # 正常路徑到得了、而且站位不比窄縫站位遠超過 0.5 m → 走正常的；不然才走窄縫（pillar_green：正常 1.80 m vs 窄縫 0.60 m）
        pick = best_normal if best_normal and (best_tight is None or best_normal[0] <= best_tight[0] + 0.5) else best_tight
        if pick is None:
            return {"ok": False, "message": "navigate_to (%.2f, %.2f): %s" % (target[0], target[1], info)}
        _, wps, actual_goal, info = pick
        state["tight"] = bool(isinstance(info, dict) and info.get("tight"))
        if state["tight"]:
            print("[nav] tight path (0.15 m inflation): small steps, 3° heading tolerance")
        state["goal"] = actual_goal
        self._nav_last_path, self._nav_last_goal = list(wps), actual_goal
        print("[nav] from (%.2f, %.2f, %.0f°) to goal (%.2f, %.2f) [target (%.2f, %.2f)]: %d waypoints %s"
              % (pose[0], pose[1], pose[2], actual_goal[0], actual_goal[1], target[0], target[1], len(wps),
                 [(round(x, 2), round(y, 2)) for x, y in wps]))

        def replan(reason):
            counters["replans"] += 1
            if counters["replans"] > 4:
                return None, "blocked repeatedly (%s)" % reason
            pose_r = self._nav_pose()
            if pose_r is None:
                return None, "lost world pose"
            w, g, i = self._nav.plan((pose_r[0], pose_r[1]), goal)
            if w is None:
                return None, "no path after %s: %s" % (reason, i)
            state["goal"] = g
            state["tight"] = bool(isinstance(i, dict) and i.get("tight"))
            self._nav_last_path = list(w)
            return w, ""

        while True:
            if time.time() - t0 > budget:
                self.stop()
                return result(False, "time budget %.0fs exceeded" % budget)
            if counters["prims"] >= 60:
                self.stop()
                return result(False, "too many motion primitives")
            pose = self._nav_pose()
            if pose is None:
                return result(False, "lost world pose")
            g = state["goal"]
            if math.hypot(g[0] - pose[0], g[1] - pose[1]) <= tol:
                break
            if not wps:
                wps = [g]  # 路徑點走完但還沒進容許範圍：直接朝終點再修一次
            wx, wy = wps[0]
            d = math.hypot(wx - pose[0], wy - pose[1])
            if d < 0.10 and len(wps) > 1:
                wps.pop(0)
                continue
            bearing = math.degrees(math.atan2(wy - pose[1], wx - pose[0]))
            axis = self._nav.passage_axis_at(pose[0], pose[1])
            if axis is not None and self._nav.passage_axis_at(wx, wy) is None:
                pinfo0 = self._nav.passage_info_at(pose[0], pose[1]) or {}
                if pinfo0.get("center"):
                    a0 = math.radians(pinfo0["axis_deg"])
                    along0 = abs((pose[0] - pinfo0["center"][0]) * math.cos(a0) + (pose[1] - pinfo0["center"][1]) * math.sin(a0))
                    if along0 > pinfo0["half_len"] + 0.05:
                        axis = None  # 已在牆外的窄道區、下一個路徑點又不在通道裡：一般模式（不然會沿軸追路徑點倒車穿回整條隧道）
            if axis is not None:
                # 通道模式（隧道內）：只沿軸線走、不朝路徑點轉（窄處迴轉會被牆帶著轉）。
                # 目標在後方就倒車出去。ToF 看到東西不標障礙（多半是牆），先對齊軸線，還是擋著才算真的堵住。
                travel = min((axis, axis + 180.0), key=lambda a: abs(_wrap_deg(a - bearing)))
                ta = math.radians(travel)
                along = (wx - pose[0]) * math.cos(ta) + (wy - pose[1]) * math.sin(ta)
                if abs(along) <= 0.10:
                    # 沿軸已經到這個路徑點（側向偏差在通道裡修不掉，不要為它來回振盪）
                    if len(wps) > 1:
                        wps.pop(0)
                        continue
                    break
                d = abs(along)
                # 卡住判定只看「沿軸距離有沒有縮短」：轉向的迭代不算，連續 4 次移動都沒靠近才算卡住
                # （第八輪：折線後轉回軸線要轉三次，被舊的「同一格出現 4 次」誤判成振盪）。
                if counters.get("best_wp") != (wx, wy):
                    counters["best_wp"], counters["best_along"], counters["noprog"] = (wx, wy), d + 1.0, 0
                if d < counters["best_along"] - 0.02:
                    counters["best_along"], counters["noprog"] = d, 0
                reverse = abs(_wrap_deg(travel - pose[2])) > 150.0
                align = travel + 180.0 if reverse else travel
                # 還在通道口的窄道區（牆還沒開始）而且側向偏離中線 > 5 cm：先做一個小折線把車拉回中線，
                # 不然 13 cm 的側向餘裕一進牆就用光（第七輪回歸測試：偏 9 cm 進去，ToF 一路掃到側牆）。
                pinfo = self._nav.passage_info_at(pose[0], pose[1]) or {}
                if pinfo.get("center") and not reverse and counters.get("dogleg", 0) < 2:
                    cxp, cyp = pinfo.get("center_line", pinfo["center"])  # 側向偏移過的中線
                    along_c = (pose[0] - cxp) * math.cos(ta) + (pose[1] - cyp) * math.sin(ta)
                    perp = -(pose[0] - cxp) * math.sin(ta) + (pose[1] - cyp) * math.cos(ta)  # 左為正
                    room = min(-pinfo["half_len"] - along_c - 0.15, d - 0.05)                   # 到牆（留車身半徑）／到目標點，取短的
                    if along_c < -pinfo["half_len"] and abs(perp) > 0.05 and room >= 0.15:
                        theta = min(20.0, math.degrees(math.atan2(abs(perp), room)))
                        cmd = travel - math.copysign(theta, perp)                                 # 偏左就往右斜一點
                        ok, pose = self._turn_to(cmd, pose, log, tol=3.0, max_tries=2)
                        counters["prims"] += 1
                        counters["dogleg"] = counters.get("dogleg", 0) + 1
                        if ok:
                            dl = min(room, abs(perp) / max(math.sin(math.radians(theta)), 0.05))
                            r = self.do("move_chassis", {"forward_m": dl, "right_m": 0, "turn_left_deg": 0})
                            counters["prims"] += 1
                            print("[nav] passage centering: perp %+.2f m -> heading %+.0f°, fwd %.2f ok=%s" % (perp, cmd, dl, r.get("ok")))
                        continue
                if abs(_wrap_deg(align - pose[2])) > 2.0:
                    ok, pose = self._turn_to(align, pose, log, tol=2.0, max_tries=3)
                    counters["prims"] += 1
                    if not ok:
                        counters["turn_fail"] = counters.get("turn_fail", 0) + 1
                        if counters["turn_fail"] > 3:
                            self.stop()
                            return result(False, "cannot align inside the passage")
                    continue
                step = min(d, 0.4)
                if not reverse:
                    tof, tof_m = self._tof_front()
                    # 通道裡 ToF 的錐狀光束會斜掃到側牆（偏離中線幾公分就讀到 15 cm），不能拿來縮步，
                    # 只有真的貼到車頭（< 10 cm）才算被擋。
                    if tof_m is not None and tof_m < 0.10:
                        self.stop()
                        return result(False, "blocked inside the passage (tof_front=%s mm)" % tof)
                if d >= counters["best_along"] - 0.02:
                    counters["noprog"] += 1
                    if counters["noprog"] > 3:
                        self.stop()
                        return result(False, "no progress inside the passage near (%.2f, %.2f)" % pose[:2])
                r = self.do("move_chassis", {"forward_m": -step if reverse else step, "right_m": 0, "turn_left_deg": 0})
                counters["prims"] += 1
                p_after = self._nav_pose()
                if p_after is not None:
                    self._nav_trajectory.append(p_after)
                    print("[nav] passage %s %.2f -> (%.2f, %.2f, %.0f°) ok=%s" % (
                        "rev" if reverse else "fwd", step, p_after[0], p_after[1], p_after[2], r.get("ok")))
                if RECOVERY_MARKER in str(r.get("message", "")):
                    return result(False, "the simulation was auto-restarted mid-navigation (robot pose reset)")
                if not r.get("ok"):
                    counters["turn_fail"] = counters.get("turn_fail", 0) + 1
                    if counters["turn_fail"] > 3:
                        self.stop()
                        return result(False, "stuck inside the passage: %s" % r.get("message", "")[:80])
                continue
            delta = _wrap_deg(bearing - pose[2])
            clear_here = self._nav.clearance_at(pose[0], pose[1])
            tight = bool(state.get("tight")) and clear_here < 0.30  # 窄縫路徑只在真的窄的路段小步慢走
            near = tight or clear_here < 0.35
            step_cap = 0.2 if tight else (0.3 if near else 0.5)
            turn_thr = 3.0 if tight else (5.0 if near else 10.0)
            if abs(delta) > 60.0 and self._nav.clearance_at(pose[0], pose[1]) < 0.24 and counters.get("nudge", 0) < 3:
                # 原地轉身的掃掠半徑約 0.2-0.25 m：離最近障礙不到 0.24 m 就先直線挪開（先試往後——多半是剛倒車出通道，
                # 再試往前），找 0.6 m 內第一個淨空 ≥ 0.26 的點；找不到就照原樣轉。
                hx, hy = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
                nudge = None
                for sgn in (-1.0, 1.0):
                    for dist in (0.15, 0.25, 0.35, 0.45, 0.6):
                        px, py = pose[0] + sgn * dist * hx, pose[1] + sgn * dist * hy
                        line_ok = all(not self._nav.occ[world_to_cell(pose[0] + sgn * t * hx, pose[1] + sgn * t * hy)[::-1]]
                                      for t in (dist * k / 4.0 for k in range(1, 5)))
                        if line_ok and self._nav.clearance_at(px, py) >= 0.26:
                            nudge = sgn * dist
                            break
                    if nudge is not None:
                        break
                if nudge is not None:
                    counters["nudge"] = counters.get("nudge", 0) + 1
                    r = self.do("move_chassis", {"forward_m": nudge, "right_m": 0, "turn_left_deg": 0})
                    counters["prims"] += 1
                    print("[nav] no room to spin at (%.2f, %.2f) (clearance %.2f): moved %+.2f m along the heading first, ok=%s"
                          % (pose[0], pose[1], self._nav.clearance_at(pose[0], pose[1]), nudge, r.get("ok")))
                    continue
            if abs(delta) > turn_thr:  # 空曠處 10 度以內不轉；靠近障礙時 5 度、步長也縮短；窄縫路徑 3 度、步長 0.2
                ok, new_pose = self._turn_to(bearing, pose, log, tol=3.0 if tight else 5.0)
                counters["prims"] += 1
                if ok:
                    pose = new_pose
                    continue
                # 旋轉被判「沒轉動」，通常是在隧道這種窄處被牆卡住。小角度就不修、直接往前走（牆會導引）；
                # 大角度先退一點再試一次，還是不行才放棄。
                counters["turn_fail"] = counters.get("turn_fail", 0) + 1
                st_now = self.state()
                tof_now = st_now.get("tof_front_mm")
                touching = any(v is not None and v != 9999 and v < 250
                               for v in (st_now.get("tof_front_mm"), st_now.get("tof_left_mm"), st_now.get("tof_right_mm")))
                if abs(delta) < 30.0 and not touching:
                    log.append("turn %+.0f rejected but front is clear, driving on" % delta)
                elif counters["turn_fail"] <= 3:
                    # 轉向被拒不代表前方有障礙，不標記地圖；有接觸就先退一點，然後重試轉向。
                    log.append("turn %+.0f rejected (tof=%s), %s and retrying" % (delta, tof_now, "backing up" if touching else "waiting"))
                    print("[nav] turn %+.0f rejected at (%.2f, %.2f, %.0f°), touching=%s -> retry" % (delta, pose[0], pose[1], pose[2], touching))
                    if touching:
                        self.do("move_chassis", {"forward_m": -0.2, "right_m": 0, "turn_left_deg": 0})
                        counters["prims"] += 1
                    continue
                else:
                    self.stop()
                    return result(False, "turn failed repeatedly (delta %+.0f deg)" % delta)
            step = min(d, step_cap) if d > step_cap else max(0.05, d - 0.03)  # 最後一段少走 3 cm，免得衝進膨脹區再轉身掃到東西
            # 前方 ToF 裝在車頭前緣（離中心 0.16 m）。規劃器本來就允許路徑離障礙 0.19 m（以中心算），所以經過障礙旁邊
            # 時 ToF 讀到 20-30 cm 是正常的，不能當成「被擋」（第五輪回歸測試就是這樣把整張圖標成障礙、規劃無路）。
            # 正確做法：讀值比這一步短就把步長縮到它前面 5 cm；只有東西已經貼到車頭（< 10 cm）才標障礙、重規劃。
            tof, tof_m = self._tof_front()
            if tof_m is not None and tof_m < step + 0.05:
                if tof_m > 0.10:
                    step = max(0.05, tof_m - 0.05)
                else:
                    self._nav_mark_blocked(pose, 0.16 + tof_m)
                    wps, why = replan("tof_front=%s mm" % tof)
                    if wps is None:
                        self.stop()
                        return result(False, why)
                    continue
            r = self.do("move_chassis", {"forward_m": step, "right_m": 0, "turn_left_deg": 0})
            counters["prims"] += 1
            p_after = self._nav_pose()
            if p_after is not None:
                self._nav_trajectory.append(p_after)
                print("[nav] fwd %.2f -> (%.2f, %.2f, %.0f°) ok=%s tof=%s" % (
                    step, p_after[0], p_after[1], p_after[2], r.get("ok"), self.state().get("tof_front_mm")))
            if RECOVERY_MARKER in str(r.get("message", "")):
                # 伺服器剛自動重播模擬救卡死，機器人姿態已經跳回存檔位置：這趟導航不能再繼續
                return result(False, "the simulation was auto-restarted mid-navigation (robot pose reset)")
            if not r.get("ok"):
                self._nav_mark_blocked(pose, 0.16 + 0.10)  # 指令沒走到：障礙大概貼在車頭前
                wps, why = replan("move failed: %s" % r.get("message", "")[:80])
                if wps is None:
                    self.stop()
                    return result(False, why)
        if face:
            pose = self._nav_pose()
            if pose is not None:
                bearing = math.degrees(math.atan2(target[1] - pose[1], target[0] - pose[0]))
                if abs(_wrap_deg(bearing - pose[2])) > 4.0:
                    self._turn_to(bearing, pose, log)
                    counters["prims"] += 1
        return result(True)

    def _pick_entrance_side(self, pose, c, u, standoff):
        """通道兩端的入口站位各規劃一次，回傳「從入口進去」方向的單位軸向量：挑正常路徑到得了、路徑最短、
        入口點沒被別的東西擋住（nearest_free 沒把它推走）的那端；兩端都到不了就退回離機器人近的那端。"""
        best = None
        for sgn in (1.0, -1.0):
            ux, uy = u[0] * sgn, u[1] * sgn
            ent = (c[0] - ux * standoff, c[1] - uy * standoff)
            w, g, info = self._nav.plan((pose[0], pose[1]), ent)
            if w is None:
                continue
            pts = [(pose[0], pose[1])] + list(w)
            length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
            tight = bool(isinstance(info, dict) and info.get("tight"))
            off = math.hypot(g[0] - ent[0], g[1] - ent[1])
            score = length + (3.0 if tight else 0.0) + 5.0 * off
            if best is None or score < best[0]:
                best = (score, ux, uy)
        if best is None:
            ux, uy = u
            if (pose[0] - c[0]) * ux + (pose[1] - c[1]) * uy > 0:
                ux, uy = -ux, -uy
            return ux, uy
        return best[1], best[2]

    def _act_go_to_landmark(self, p):
        """去一個固定地標：隧道/平台這種可穿越的結構 → 停在入口外面面向它（不是鑽進去）；其他地標 →
        停在它前緣 stop_m 處面向它（navigate_to 會把終點往外推到第一個可走格）。"""
        if not getattr(self, "_nav", None):
            return {"ok": False, "message": "navigation map unavailable (needs sim mode with the overhead camera)"}
        name = str(p.get("name", "")).strip().lower()
        stop_m = max(0.15, float(p.get("stop_m") or 0.45))
        handles = getattr(self, "_static_landmark_handles", {})
        h = handles.get(name)
        if h is None:
            return {"ok": False, "message": "go_to_landmark: unknown landmark %r (known: %s)" % (name, ", ".join(sorted(handles)))}
        geo = self._nav.group_axis(h)
        pose = self._nav_pose()
        if geo is None:
            # 讀不到幾何（只有位置的地標）：當成以位置為中心、半徑 0.5 m 的東西處理，不要直接放棄
            lm = (getattr(self, "_static_landmarks_xy", None) or {}).get(name)
            if lm:
                geo = ((float(lm[0]), float(lm[1])), (1.0, 0.0), 0.5, 0.5)
        if geo is None or pose is None:
            return {"ok": False, "message": "go_to_landmark: cannot read geometry/pose for %r" % name}
        (cx, cy), (ux, uy), half_len, half_w = geo
        passage = self._nav.passage_axis_at(cx, cy) is not None or name == "platform"
        if passage:
            pg = self._nav.passage_line_near(cx, cy)
            if pg:
                cx, cy = pg["center_line"]  # 側向偏移過的中線（見 _act_drive_through）
            stop_m = max(stop_m, 0.65)  # 窄道外才有空間轉身對齊
            ux, uy = self._pick_entrance_side(pose, (cx, cy), (ux, uy), half_len + stop_m)
            entrance = (cx - ux * (half_len + stop_m), cy - uy * (half_len + stop_m))
            r = self._act_navigate_to({"x": entrance[0], "y": entrance[1], "tolerance_m": 0.08, "face": False})
            pose2 = self._nav_pose()
            if r.get("ok") and pose2 is not None:
                self._turn_to(math.degrees(math.atan2(uy, ux)), pose2, [], tol=3.0, max_tries=3)
            r["message"] = ("go_to_landmark %s: this is a pass-through structure, so the robot stops at its entrance "
                            "(%.2f, %.2f) facing along its axis; use drive_through(%s) to go through. %s"
                            % (name, entrance[0], entrance[1], name, r.get("message", "")))
            return r
        r = self._act_navigate_to({"x": cx, "y": cy, "stop_m": stop_m, "face": True})
        r["message"] = "go_to_landmark %s (centre %.2f, %.2f): %s" % (name, cx, cy, r.get("message", ""))
        return r

    def _act_drive_through(self, p):
        if not getattr(self, "_nav", None):
            return {"ok": False, "message": "navigation map unavailable (needs sim mode with the overhead camera)"}
        name = str(p.get("name", "")).strip().lower()
        handles = getattr(self, "_static_landmark_handles", {})
        h = handles.get(name)
        if h is None:
            return {"ok": False, "message": "drive_through: unknown structure %r (known: %s)"
                    % (name, ", ".join(sorted(handles)))}
        geo = self._nav.group_axis(h)
        if geo is None:
            return {"ok": False, "message": "drive_through: cannot read the geometry of %r" % name}
        (cx, cy), (ux, uy), half_len, half_w = geo
        pg = self._nav.passage_line_near(cx, cy)
        if pg:
            cx, cy = pg["center_line"]  # 側向偏移過的中線：隧道北口貼著球桶，正中線車身右緣正好碰到桶
        pose = self._nav_pose()
        if pose is None:
            return {"ok": False, "message": "cannot read the robot's world pose"}
        ux, uy = self._pick_entrance_side(pose, (cx, cy), (ux, uy), half_len + 0.65)  # 挑到得了、路最短的那一端進
        # 入口站位在窄道外 0.65 m（窄道長 0.40）：在這裡對齊轉身，掃掠圈才不會碰到隧道口旁的球桶
        # （第八輪：在窄道裡轉身，bin_balls 被撞 12 次）；進窄道後由通道模式的折線置中。
        entrance = (cx - ux * (half_len + 0.65), cy - uy * (half_len + 0.65))
        exit_pt = (cx + ux * (half_len + 0.3), cy + uy * (half_len + 0.3))      # 出口留在挖空區裡，不被牆端膨脹封住
        # 出入口一律放在軸線正中：側向偏移過（想避開隧道北口旁的球桶）反而讓車偏著進通道、擦到牆（第六輪回歸測試）。
        # 入口的到達容忍要嚴：進通道時的側向誤差 = 到達入口時的位置誤差，隧道內每側只有 13 cm 餘裕。
        axis_deg = math.degrees(math.atan2(uy, ux))
        r1 = self._act_navigate_to({"x": entrance[0], "y": entrance[1], "tolerance_m": 0.06, "face": False})
        if not r1.get("ok"):
            return {"ok": False, "message": "drive_through %s: could not reach the entrance — %s" % (name, r1.get("message"))}
        log = []
        pose = self._nav_pose()
        if pose is not None:
            _ok, pose = self._turn_to(axis_deg, pose, log, tol=2.0, max_tries=4)
        r2 = self._act_navigate_to({"x": exit_pt[0], "y": exit_pt[1], "tolerance_m": 0.15, "face": False})
        pose = self._nav_pose() or pose
        along = (pose[0] - cx) * ux + (pose[1] - cy) * uy
        # 「真的穿過」＝終點在另一側，而且行進軌跡裡至少有一個姿態落在結構內部（長軸範圍內、側向在半寬內）：
        # 只看終點會把「繞過去」誤判成「穿過去」。
        inside = 0
        for q in getattr(self, "_nav_trajectory", []):
            a = (q[0] - cx) * ux + (q[1] - cy) * uy
            b = abs(-(q[0] - cx) * uy + (q[1] - cy) * ux)
            if abs(a) < half_len and b < half_w * 0.8:
                inside += 1
        passed = along > half_len and inside > 0
        msg = ("drive_through %s: axis %.0f deg, entrance (%.2f, %.2f) -> exit (%.2f, %.2f); now at (%.2f, %.2f), "
               "%.2f m past the centre along the axis (half-length %.2f m), %d trajectory poses inside the structure -> %s. %s"
               % (name, axis_deg, entrance[0], entrance[1], exit_pt[0], exit_pt[1], pose[0], pose[1], along,
                  half_len, inside, "PASSED THROUGH" if passed else "did NOT go through it", r2.get("message", "")))
        return {"ok": bool(r2.get("ok")) and passed, "message": msg, "passed": passed,
                "final_world_xy": [round(pose[0], 3), round(pose[1], 3)]}

    def _poll_world_xy(self):
        # 這裡以前是 except Exception: pass：sim.getObjectPosition() 只要噴一次例外，
        # world_xy 就會從此凍結在最後一次成功讀到的值，thread 不會死、也不會印任何東西，
        # 表面上一切正常，agent 端卻整段任務都在用過時座標算方位角/距離。實測真的發生過，
        # 一次跑分 42 步裡有 30 幾步 world_xy 完全沒變。現在失敗會印出來，連續失敗夠多次
        # （0.1s 一次、20 次≈2 秒）就整個重建連線跟 handle，不是原地卡死等下一次奇蹟。
        fail_streak = 0
        while True:
            try:
                with self._sim_for_overhead_lock:
                    sim = self._sim_for_overhead
                    p = sim.getObjectPosition(self._overhead_root, sim.handle_world)
                    hd = None
                    if getattr(self, "_wheel_handles", None) or getattr(self, "_robot_cam", -1) != -1:
                        hd = self._world_heading(sim)  # 真實世界車頭方向（輪子幾何）
                self._set("world_xy", [round(p[0], 3), round(p[1], 3)])
                if hd is not None:
                    self._set("world_heading_deg", round(hd, 1))
                fail_streak = 0
            except Exception as e:
                fail_streak += 1
                if fail_streak == 1 or fail_streak % 20 == 0:
                    print("[server] world_xy 讀取失敗（連續第 %d 次）：%s" % (fail_streak, e))
                if fail_streak >= 20:
                    print("[server] world_xy 連續失敗 %d 次，嘗試重建俯視攝影機連線..." % fail_streak)
                    try:
                        sim, h, root = self._connect_overhead_sim()
                        with self._sim_for_overhead_lock:
                            self._sim_for_overhead = sim
                            self._overhead_handle = h
                            self._overhead_root = root
                            self._robot_cam = self._robot_camera_handle(sim, root)
                            self._wheel_handles = self._find_wheel_handles(sim, root)
                        print("[server] 俯視攝影機連線已重建，world_xy 恢復更新。")
                        fail_streak = 0
                    except Exception as reconnect_err:
                        print("[server] 重建俯視攝影機連線失敗：%s（2 秒後再試）" % reconnect_err)
                        time.sleep(2.0)
                        continue
            time.sleep(0.1)

    def overhead_jpeg(self, quality=80):
        # world_xy 的背景 polling 跟這裡都用同一條 zmqRemoteApi 連線，
        # 這個 client 不是 thread-safe 的，兩邊同時呼叫會讓其中一邊拿到壞掉的回應
        # （實測過：獨立呼叫都正常，兩個 thread 一起搶同一條連線就會噴例外、被這裡的
        # except 吃掉變成 503）。用 lock 把兩邊串行化，成本很低（都是很快的呼叫）。
        if not self._overhead_handle or cv2 is None:
            return None
        try:
            with self._sim_for_overhead_lock:
                img, res = self._sim_for_overhead.getVisionSensorImg(self._overhead_handle)
            arr = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
            arr = cv2.flip(arr, 0)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    def overhead_depth_png(self):
        # sim.getVisionSensorDepth 的 options bit0=1 代表直接回傳「公尺」，不用自己拿
        # near/far clipping plane 去反推真實距離——這樣就算之後改了 add_overhead_camera.py
        # 的攝影機高度或視野範圍，這裡也不用跟著改。編碼成 16-bit PNG、單位 mm：
        # uint16 上限 65535mm (65.5m) 遠超過場地範圍，1mm 解析度綽綽有餘，跟專案裡其他
        # 量測 (arm_mm、tof_*_mm) 用同一種單位，agent 端不用再做單位轉換。
        # 翻轉方向跟 overhead_jpeg 的 cv2.flip(arr, 0) 保持一致，兩張圖的像素座標系才會對得上。
        if not self._overhead_handle or cv2 is None:
            return None
        try:
            with self._sim_for_overhead_lock:
                buf, res = self._sim_for_overhead.getVisionSensorDepth(self._overhead_handle, 1)
            arr = np.frombuffer(buf, dtype=np.float32).reshape(res[1], res[0])
            arr = cv2.flip(arr, 0)
            mm = np.clip(arr * 1000.0, 0, 65535).astype(np.uint16)
            ok, out = cv2.imencode(".png", mm)
            return out.tobytes() if ok else None
        except Exception:
            return None

    # ---- 遙測訂閱：每一項都獨立 try，模擬器不支援的就自動略過 ----
    def _set(self, key, value):
        with self._tlock:
            self.telemetry[key] = value
            self._last_rx = time.time()

    def link(self):
        """跟機器人的連線還活著嗎？模擬一按停止，SDK 連線就斷了，但這支 server 不會自己知道。
        判斷方式：遙測是持續推送的，超過 3 秒沒收到任何一筆，就當作斷線。"""
        last = getattr(self, "_last_rx", None)
        if last is None:
            return {"robot_link": "unknown", "detail": "no telemetry has ever arrived"}
        age = time.time() - last
        if age > 3.0:
            return {"robot_link": "lost", "detail": "no telemetry for %.0fs. The simulation was probably stopped or "
                    "restarted. Press play in CoppeliaSim, then restart robot_server.py." % age}
        return {"robot_link": "ok", "detail": "telemetry %.1fs ago" % age}

    def _subscribe_all(self):
        ep = self.ep
        subs = [
            ("chassis position", lambda: ep.chassis.sub_position(
                freq=5, callback=lambda p: self._set("position_m", {"x": round(p[0], 3), "y": round(p[1], 3)}))),
            ("chassis attitude", lambda: ep.chassis.sub_attitude(
                freq=20, callback=lambda a: self._set("yaw_deg", round(a[0], 1)))),  # 20 Hz：45 deg/s 一個取樣 2 度
            ("arm position", lambda: ep.robotic_arm.sub_position(
                freq=5, callback=lambda p: self._set("arm_mm", {"x": _fix_int32(p[0]), "y": _fix_int32(p[1])}))),
            ("gripper status", lambda: ep.gripper.sub_status(
                freq=5, callback=lambda s: self._set("gripper", str(s)))),
            ("distance sensor", lambda: ep.sensor.sub_distance(
                freq=5, callback=lambda d: self._set("tof_distance_mm", int(d[0])))),
            ("battery", lambda: ep.battery.sub_battery_info(
                freq=1, callback=lambda b: self._set("battery_percent", int(b)))),
        ]
        for name, fn in subs:
            try:
                fn()
            except Exception as e:
                print("[server] 略過遙測訂閱 %s：%s" % (name, e))

    def state(self):
        with self._tlock:
            data = dict(self.telemetry)
        if self._prox_handles:
            # 這個模擬器的 SDK ToF 永遠回報 0，有真的 prox 讀值時就不要一起丟出去混淆。
            data.pop("tof_distance_mm", None)
        data["mode"] = self.mode
        # _camera_ok 只代表「啟動時初始化成功」；_camera_live_ok 才是「最近真的抓得到新畫面」——
        # SDK 的解碼 thread 死掉時前者不會變，後者才會誠實反映前方相機其實已經瞎了。
        data["camera"] = self._camera_ok and self._camera_live_ok
        # 場景裡不會動的地標，換算成跟 locate_overhead()/align_to_tunnel() 同一種「距離 +
        # 該轉幾度面向它」格式，讓 agent 不用每次都呼叫 locate_overhead 才知道隧道/球池
        # 在哪——省下的是「感知這種東西在哪」的步驟，不是叫它跳過移動或對齊的驗證。
        # 公式（world_xy 的 y 軸要反過來才跟 yaw_deg 同一個座標系）跟 perception.py 的
        # locate_overhead 完全一樣，已經對過真實移動資料校正過。
        world_xy = data.get("world_xy")
        yaw = data.get("yaw_deg")
        heading = data.get("world_heading_deg")
        if self._static_landmarks_xy and world_xy is not None and (heading is not None or yaw is not None):
            rx, ry = world_xy
            landmarks = {}
            for name, (lx, ly) in self._static_landmarks_xy.items():
                dist = math.hypot(lx - rx, ly - ry)
                if heading is not None:
                    # 有真實世界朝向就直接用：turn_left 為正會讓世界朝向減少（TURN_SIGN），所以取負號。
                    # 這樣不管機器人一開始面向哪裡都正確；舊公式假設 SDK yaw=0 對應世界 +x。
                    bearing_w = math.degrees(math.atan2(ly - ry, lx - rx))
                    turn_left = -_wrap_deg(bearing_w - heading)
                else:
                    vx = lx - rx
                    vy = -(ly - ry)
                    bearing = math.degrees(math.atan2(vy, vx))
                    turn_left = (bearing - yaw + 180) % 360 - 180
                landmarks[name] = {"distance_m": round(dist, 2), "turn_left_deg": round(turn_left, 1),
                                   "world_xy": [lx, ly]}
            data["static_landmarks"] = landmarks
        return data

    def _fresh_image(self, raw=False):
        """取得「現在」的畫面。
        實測發現：動作一結束就去讀，拿到的是動作開始前的舊畫面，整整慢一步。原因是影像走 H.264，
        解碼器要等後面幾張進來才會吐出前面那張，模擬器的影格率又低，延遲是以「張數」計而不是以秒計。
        所以這裡先清空佇列，再連續等 N 張新畫面進來，取最後一張。"""
        img = self.ep.camera.read_cv2_image(strategy="newest", timeout=3)
        if raw:
            return img
        t0 = time.time()
        for _ in range(FRESH_FRAMES):
            if time.time() - t0 > 4.0:
                break
            try:
                img = self.ep.camera.read_cv2_image(strategy="pipeline", timeout=1.5)
            except Exception:
                break
        return img

    def frame_jpeg(self, width=640, quality=80, raw=False):
        if not self._camera_ok or cv2 is None:
            return None
        try:
            img = self._fresh_image(raw)
        except Exception:
            img = None
        if img is None:
            self._note_camera_fail()
            return None
        self._camera_fail_streak = 0
        self._camera_live_ok = True
        # 實測證實：這個模擬器的前方相機串流是左右鏡像的——送一個遙測證實為真的
        # 「往左轉 20 度」（yaw telemetry 確實 +22.4 度），畫面裡的東西卻往左滑、
        # 新東西從右邊冒出來，物理上正確的左轉應該是東西往右滑。這個鏡像會讓
        # perception.py 算出來的 bearing_right_deg／turn_left_deg 全部反著指，
        # locate/face 每次「轉過去對準」都會往錯的方向轉，正是好幾次任務卡在
        # 反覆對不準、東西轉一轉就不見的根本原因。這裡翻正，之後 perception.py
        # 的公式不用再改，直接假設畫面已經是正常方向就對了。
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, int(h * width / float(w))))
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

    def _note_camera_fail(self):
        # 實測發現：跑分一開始連續呼叫 /stop 會打斷 SDK 正在解碼中的 H.264 串流，讓底層
        # av 解碼器噴 InvalidDataError，直接把 robomaster SDK 自己的 _video_decoder_task
        # thread 弄死，而且沒有任何重啟機制——一次跑分裡 48 步只有第 1 步有前方畫面，
        # 剩下 47 步全部悄悄退化成只剩俯視圖，state() 的 camera 欄位卻全程回報 true，
        # agent 完全不知道自己「瞎了」。現在連續失敗夠多次就重啟 video stream，
        # 且 camera 欄位會誠實反映「最近有沒有真的抓到新畫面」，不是只看有沒有初始化成功。
        self._camera_fail_streak += 1
        if self._camera_fail_streak == 1 or self._camera_fail_streak % 5 == 0:
            print("[server] 前方相機讀不到新畫面（連續第 %d 次）" % self._camera_fail_streak)
        if self._camera_fail_streak >= 3:
            self._camera_live_ok = False
        if self._camera_fail_streak == 3 or (self._camera_fail_streak > 3 and self._camera_fail_streak % 10 == 0):
            print("[server] 前方相機連續失敗 %d 次，嘗試重啟 video stream..." % self._camera_fail_streak)
            try:
                self.ep.camera.stop_video_stream()
            except Exception:
                pass
            try:
                self.ep.camera.start_video_stream(display=False)
                print("[server] video stream 已重啟，等下一次讀取確認是否恢復")
                self._camera_fail_streak = 0
            except Exception as e:
                print("[server] video stream 重啟失敗：%s" % e)

    # ---- 動作 ----
    def stop(self):
        try:
            self.ep.chassis.drive_speed(x=0, y=0, z=0)
        except Exception as e:
            print("[server] stop 失敗：", e)

    def do(self, name, params):
        fn = getattr(self, "_act_" + name, None)
        if fn is None:
            return {"ok": False, "message": "unknown action %r" % name}
        if name != "move_chassis":
            return fn(params or {})
        # 底盤動作前後各讀一次里程計，把「實際量到的變化」附在回報裡。
        # 這樣 log 裡就看得出指令和實際運動有沒有對上 (方向相反、打滑、撞到東西都會現形)。
        before = self.state()
        result = fn(params or {})
        time.sleep(0.6)  # 等車身完全停穩，下一張相機畫面才不會是晃動中的
        after = self.state()
        try:
            dyaw = (after["yaw_deg"] - before["yaw_deg"] + 180) % 360 - 180
            dx = after["position_m"]["x"] - before["position_m"]["x"]
            dy = after["position_m"]["y"] - before["position_m"]["y"]
            travelled = math.hypot(dx, dy)
            requested = math.hypot(params.get("forward_m", 0) or 0, params.get("right_m", 0) or 0)
            # 逾時不等於沒走到：先用里程計補判（見 _reconcile_timeout），再附上量測值。
            _reconcile_timeout(result, requested, travelled)
            result["message"] += " | odometry: travelled %.2fm, yaw telemetry %+.0fdeg -> %+.0fdeg" % (
                travelled, before["yaw_deg"], after["yaw_deg"])
            result["measured"] = {"dyaw_deg": round(dyaw, 1), "distance_m": round(travelled, 3)}
            # SDK 的 wait_for_completed 有時候在卡住的當下也回 True (例如頂著東西完全推不動、
            # 位置誤差一開始就已經在容許範圍內)，光看 ok 看不出來，這裡用實際走的距離再核對一次。
            if result.get("ok") and requested > 0.05 and travelled < requested * 0.3:
                result["ok"] = False
                result["message"] += (" | rejected after the fact: SDK reported success but the robot barely moved "
                                       "(%.2fm of %.2fm requested) — it is almost certainly blocked by something." %
                                       (travelled, requested))
            # 旋轉一樣要核對：hybrid/speed 這兩種背景用計時的 drive_speed 轉，SDK 端完全沒有回饋，
            # 不管轉成什麼樣子都回 ok=True。這裡拿量到的 yaw 變化跟要求的角度比對，兜不起來就改判失敗，
            # 不然 agent 會以為自己轉了、其實原地沒動，白白浪費步數。
            req_turn = abs(params.get("turn_left_deg", 0) or 0)
            if result.get("ok") and req_turn > 3 and abs(dyaw) < req_turn * 0.3:
                result["ok"] = False
                result["message"] += (" | rejected after the fact: SDK reported success but the chassis barely "
                                       "rotated (%.0fdeg of %.0fdeg requested) — treat it as if it did not turn." %
                                       (dyaw, req_turn))

            # 卡死偵測：這次指令有實質內容 (不是 0,0,0)、完全沒動、而且三個真感測器都淨空
            # (代表不是被什麼東西物理擋住)，這種「該動卻完全不動」的組合連續出現幾次，
            # 幾乎可以肯定是控制腳本本身卡死了，不是這次指令的問題，重試也沒用。
            requested_any = requested > 0.05 or req_turn > 3
            tof_vals = [after.get(k) for k in ("tof_front_mm", "tof_left_mm", "tof_right_mm")]
            all_clear = all(v == 9999 for v in tof_vals) if all(v is not None for v in tof_vals) else False
            stalled = requested_any and travelled < 0.02 and abs(dyaw) < 1 and all_clear
            result["message"] += self._note_stall(stalled)
        except Exception:
            pass

        # 診斷用（先不動任何行為）：懷疑 timeout 當下 self.stop() 雖然有發，但車身撞到東西
        # 之後的殘餘動能／SDK 位置控制指令沒被速度控制乾淨蓋掉，還要再幾秒才會真的停穩——
        # 實測發現：這裡回報「已經停了」之後，下一步就算是純感知動作 (不會再送任何底盤指令)，
        # 姿態還是繼續在變。這裡不改變 0.6 秒的既有邏輯，只是在 timeout 之後多花幾秒把「是不是
        # 真的還在動」的過程印出來，確認了再決定要拉長 sleep 還是改成輪詢等穩定。
        if not result.get("ok") and "timed out" in result.get("message", ""):
            try:
                print("[server] move_chassis timeout，量測停穩過程（是不是還在自己動）...")
                last = after
                for i in range(5):
                    time.sleep(0.5)
                    cur = self.state()
                    dyaw2 = (cur["yaw_deg"] - last["yaw_deg"] + 180) % 360 - 180
                    dpos2 = math.hypot(cur["position_m"]["x"] - last["position_m"]["x"],
                                        cur["position_m"]["y"] - last["position_m"]["y"])
                    print("[server]   +%.1fs 後：yaw %+.1f -> %+.1f (Δ%.1f 度)，position 變化 %.3fm" %
                          ((i + 1) * 0.5, last["yaw_deg"], cur["yaw_deg"], dyaw2, dpos2))
                    last = cur
            except Exception as e:
                print("[server] 停穩過程量測失敗（不影響本次動作結果）：%s" % e)
        return result

    def _act_move_chassis(self, p):
        fwd = _clamp(p.get("forward_m", 0), LIMITS["forward_m"])
        right = _clamp(p.get("right_m", 0), LIMITS["right_m"])
        turn_left = _clamp(p.get("turn_left_deg", 0), LIMITS["turn_left_deg"])
        guard_note = ""
        if fwd > 0.02 and not p.get("push"):
            # 前方距離感測器看得到東西而且比這一步還近：只走到它前面 0.15 m。LLM 想推東西（推泡棉樑的任務）
            # 要明講 push=true。中等批次 m02 就是硬開 0.8 m 出隧道口，把泡棉樑推走後卡死在樑/球桶/長椅之間。
            tof, tof_m = self._tof_front()
            if tof_m is not None and tof_m < fwd + 0.15:
                capped = max(0.0, round(tof_m - 0.15, 2))
                guard_note = (" | front sensor sees something %.2f m ahead: drove %.2f m instead of the requested %.2f m to stay "
                              "0.15 m clear of it (pass push=true only if you really mean to push it)" % (tof_m, capped, fwd))
                fwd = capped
                p["forward_m"] = capped  # do() 的里程計比對也用縮短後的距離，不然會被判成「幾乎沒動＝被擋住」
        v, w = LIMITS["xy_speed"], LIMITS["z_speed"]

        backend = self.chassis_backend
        move_translation = backend in ("move", "hybrid")
        move_rotation = backend == "move"

        if move_translation:
            # 位置控制：位移量交給機器人自己跑完，就算 Agent 端斷線也不會一直往前衝。
            # 這個模式下，如果路徑上真的有障礙物擋住，SDK 會等不到「到達」就回傳 timeout，
            # 等同於一種碰撞偵測：卡住了就不會繼續硬頂。
            # SDK 的 chassis.move 裡，z 為正是「左轉」。
            if abs(fwd) > 1e-3 or abs(right) > 1e-3:
                # 時間預算：模擬裡車身的實際平均速度只有 xy_speed 的一半左右（實測 1.0 m 在 6.3 s 內
                # 只走了 0.87 m），照標稱速度算會讓正常的移動也被判逾時，所以距離那一項給兩倍。
                t = math.hypot(fwd, right) / v * 2 + 3
                done = self.ep.chassis.move(x=fwd, y=right, z=0, xy_speed=v).wait_for_completed(timeout=t)
                if not done:
                    self.stop()
                    # 先回報逾時並附上秒數；外層 do() 量完里程計後，走到八成以上會改判成功
                    # （見 _reconcile_timeout）。turn_skipped 讓改判時能提醒 agent 旋轉沒做。
                    return {"ok": False, "timed_out": t, "turn_skipped": turn_left,
                            "message": "translation timed out after %.1fs (blocked by something?)" % t}
        else:
            # 速度控制：跟 test_movement.py 同一種寫法，給速度、等時間、再停。這裡沒有中途碰撞偵測，
            # 全靠 LLM 自己判斷該不該走這一步；只在位置控制的完成訊號不可靠時才用。
            dist = math.hypot(fwd, right)
            if dist > 1e-3:
                t = dist / v
                self.ep.chassis.drive_speed(x=v * fwd / dist, y=v * right / dist, z=0, timeout=t + 1)
                time.sleep(t)
                self.stop()
                time.sleep(0.3)

        if move_rotation:
            if abs(turn_left) > 0.5:
                t = abs(turn_left) / w + 3
                done = self.ep.chassis.move(x=0, y=0, z=turn_left, z_speed=w).wait_for_completed(timeout=t)
                if not done:
                    self.stop()
                    return {"ok": False, "message": "rotation timed out after %.1fs" % t}
        else:
            # speed / hybrid：這個模擬器對 chassis.move 的「轉到定值」不會正確回報完成，改用 drive_speed。
            # 一開始是算好時間單純 sleep，但模擬器不一定跑在即時速度（機器負載重時會變慢），
            # 算出來的時間跟實際轉到的角度對不上：量到的 yaw 變化有時候是 0、有時候轉過頭。
            # 改成邊轉邊看真實的 yaw_deg 遙測，轉到差不多了就停，不是用猜的時間停。
            # 官方文件說 drive_speed 的 z 為正是「右轉」、跟 chassis.move 相反，理論上要加負號，
            # 但實測發現這個模擬器的 yaw_deg 遙測跟這個假設對不起來：z 給負的，量到的 yaw 卻往
            # 反方向跑，等於每次轉都轉錯邊。這裡改成直接用 turn_left 本身的正負（不加負號），
            # 已經拿 20 度、90 度分別測過，量到的 yaw 變化方向跟要求的一致。
            if abs(turn_left) <= 45.0:
                # 小角度用慢速：慣性 + 取樣延遲造成的過頭量跟角速度成正比（第八輪 rotate：15 度轉向多轉 7.5 度）。
                # 速度 = 角度 × 1 deg/s（最少 12），15 度就用 15 deg/s，停得住；通道裡對齊只差 2-4 度時尤其重要。
                w = max(12.0, min(w, abs(turn_left) * 1.0))
            if abs(turn_left) > 0.5:
                start_yaw = self.state().get("yaw_deg")
                timeout_t = abs(turn_left) / w + 4
                self.ep.chassis.drive_speed(x=0, y=0, z=math.copysign(w, turn_left), timeout=timeout_t + 1)
                if start_yaw is None:
                    # 沒有 yaw 遙測可以核對，退回原本算好時間的做法
                    time.sleep(abs(turn_left) / w)
                else:
                    deadline = time.time() + timeout_t
                    prev_yaw = start_yaw
                    turned = 0.0
                    while time.time() < deadline:
                        time.sleep(0.05)
                        cur_yaw = self.state().get("yaw_deg")
                        if cur_yaw is None:
                            break
                        # 累積每次取樣之間的小變化，不要用「跟起點的 wrap 差值」：要求 180 度時，wrap 過的差值
                        # 在 178 → -178 之間跳過去，停止條件永遠碰不到，會一路轉到逾時（實測 e02 轉了 227 度）。
                        turned += (cur_yaw - prev_yaw + 180) % 360 - 180
                        prev_yaw = cur_yaw
                        # 保險：如果方向又不對或轉過頭一大截，不要傻傻等滿 timeout，先停下來。
                        if abs(turned) > abs(turn_left) + 20:
                            break
                        # 提早停：遙測 5 Hz、旋轉 45 deg/s，從「看到到了」到真的停下來會再多轉約 8-9 度
                        # （tools/nav_selftest.py 量到 45~135 度一致多轉 8.8 度）。小角度慣性小，留 2 度就好。
                        # 20 Hz 遙測後大角度的延遲量剩 ~2 度 + 慣性 ~3 度；小角度已經降速，留 1 度
                        margin = 5.0 if abs(turn_left) > 25 else 1.0
                        if turn_left > 0 and turned >= turn_left - margin:
                            break
                        if turn_left < 0 and turned <= turn_left + margin:
                            break
                self.stop()
                time.sleep(0.3)
        return {"ok": True, "message": "moved forward=%.2fm right=%.2fm turn_left=%.0fdeg%s" % (fwd, right, turn_left, guard_note)}

    def _act_move_arm(self, p):
        x = _clamp(p.get("forward_mm", 0), LIMITS["arm_mm"])
        y = _clamp(p.get("up_mm", 0), LIMITS["arm_mm"])
        before = self.state().get("arm_mm")
        done = self.ep.robotic_arm.move(x=x, y=y).wait_for_completed(timeout=6)
        if not done:
            note = self._note_stall(True)
            return {"ok": False, "message": "arm move timed out (probably at its mechanical limit)" + note}
        time.sleep(0.4)  # 等下一筆遙測進來，再確認手臂是不是真的動了
        after = self.state().get("arm_mm")
        if before and after:
            dx, dy = after["x"] - before["x"], after["y"] - before["y"]
            if abs(dx) < 2 and abs(dy) < 2 and (abs(x) >= 5 or abs(y) >= 5):
                # 這句話可能是真的頂到物理極限，也可能是控制腳本卡死裝出來的假象 (實測過兩者長
                # 得一模一樣)，交給 _note_stall 累計，連續好幾次才會被當真的卡死處理。
                note = self._note_stall(True)
                return {"ok": False, "message": "arm did NOT move (still at x=%s y=%s mm); it is at a limit in that direction"
                                                 % (after["x"], after["y"]) + note}
            self._note_stall(False)
            return {"ok": True, "message": "arm moved by forward=%+.0fmm up=%+.0fmm, now at x=%s y=%s mm"
                                           % (dx, dy, after["x"], after["y"])}
        return {"ok": True, "message": "arm command sent forward=%.0fmm up=%.0fmm (no arm telemetry to verify)" % (x, y)}

    def _act_arm_to(self, p):
        """手臂移到絕對座標 (mm)。相機裝在手臂上，所以這同時也是在調整相機的高度和俯仰角。"""
        x = max(60.0, min(220.0, float(p.get("x_mm", 100))))
        y = max(-30.0, min(160.0, float(p.get("y_mm", 60))))
        done = self.ep.robotic_arm.moveto(x=x, y=y).wait_for_completed(timeout=6)
        time.sleep(0.3)
        if done and self._arm_near(x, y) is None:
            done = False  # SDK 說到了但 arm_mm 差超過 15 mm：手臂卡住了（實測放低後「reached」卻停在 (71,-54)）
        time.sleep(0.4)
        now = self.state().get("arm_mm")
        note = self._note_stall(not done)
        return {"ok": bool(done), "message": "arm target x=%.0f y=%.0f mm, %s, now at %s" % (
            x, y, "reached" if done else "timed out (limit?)", now) + note}

    def _arm_near(self, x, y, tol=15):
        now = self.state().get("arm_mm") or {}
        return now if (abs((now.get("x") or 0) - x) <= tol and abs((now.get("y") or 0) - y) <= tol) else None

    def _act_recenter_arm(self, p):
        # SDK 的 recenter 在模擬裡常回報完成但手臂沒動、或從放低的姿態回不來（實測：(180,30) 之後 recenter 逾時，
        # 手臂停在 (72,-52)）。做完一律用 arm_mm 核對，沒到就再用 moveto(89,117) 補一次，還是沒到才回失敗。
        # 不呼叫 SDK 的 recenter()：模擬裡它會把手臂帶到 (50,-73)/(67,-58) 這種在底盤下方的姿態，之後 moveto 也回不來；
        # 直接 moveto(89,117)（實測從任何姿態、包含放低的 (180,30) 都到得了），沒到就等 0.5 s 再補一次。
        now = None
        for _ in range(2):
            self.ep.robotic_arm.moveto(x=89, y=117).wait_for_completed(timeout=6)
            time.sleep(0.5)
            now = self._arm_near(89, 117)
            if now is not None:
                break
        note = self._note_stall(now is None)
        if now is None:
            return {"ok": False, "message": "arm recenter failed: arm is at %s, not near (89,117)%s" % (self.state().get("arm_mm"), note)}
        return {"ok": True, "message": "arm recentered (now at %s)%s" % (now, note)}

    def _tof_front(self):
        """(原始讀值 mm, 給導航用的公尺數)：手上夾著東西時（set_held），跟那個距離差不到 6 cm 的讀值是自己夾的
        物件、不是障礙，導航忽略它（實測夾著積木走，ToF 一直讀 62 mm，一路標障礙重規劃到「無路可走」）。"""
        tof = self.state().get("tof_front_mm")
        if tof is None or tof == 9999:
            return tof, None
        held = getattr(self, "_held_tof_mm", None)
        if held is not None and abs(int(tof) - held) <= 60:
            return tof, None
        return tof, tof / 1000.0

    def _act_set_held(self, p):
        v = p.get("tof_mm")
        self._held_tof_mm = int(v) if v is not None else None
        self._set("held_tof_mm", self._held_tof_mm)
        return {"ok": True, "message": ("holding an object at %d mm on the front sensor; navigation will ignore that reading"
                                        % self._held_tof_mm) if self._held_tof_mm is not None else "not holding anything"}

    def _act_gripper(self, p):
        state = str(p.get("state", "")).lower()
        if state not in ("open", "close"):
            return {"ok": False, "message": "gripper state must be 'open' or 'close'"}
        power = int(max(1, min(100, float(p.get("power") or 50))))
        if state == "open":
            self._held_tof_mm = None
            self._set("held_tof_mm", None)
            self.ep.gripper.open(power=power)
        else:
            self.ep.gripper.close(power=power)
        time.sleep(1.5)
        try:
            self.ep.gripper.pause()
        except Exception:
            pass
        return {"ok": True, "message": "gripper %s" % state}

    def _act_wait(self, p):
        s = abs(_clamp(p.get("seconds", 1), LIMITS["wait_s"]))
        time.sleep(s)
        return {"ok": True, "message": "waited %.1fs" % s}

    def _act_stop(self, p):
        self.stop()
        return {"ok": True, "message": "chassis stopped"}

    def reset(self):
        """實體世界沒有「重置」這回事，這裡只把手臂和夾爪回到預設，底盤位置要由人或模擬器重置。
        之前這裡完全沒管 recenter_arm 的結果，永遠回報「arm recentered」，結果手臂卡死的時候
        還是照樣騙人說重置成功——才會發生任務一開始手臂就卡在怪姿態、相機看到的是自己的殼，
        agent 卻完全不知道發生了什麼事。現在會真的檢查、失敗就老實說，且失敗時再試一次
        (常常是暫時性的、recenter_arm 本身也有可能觸發自動恢復)。"""
        self.stop()
        self._act_gripper({"state": "open"})
        r = self._act_recenter_arm({})
        if not r.get("ok"):
            r = self._act_recenter_arm({})  # 失敗重試一次，可能是暫時性的、或中間已經自動恢復了
        ok = bool(r.get("ok"))
        return {"ok": ok, "message": ("gripper opened; " + r.get("message", "") +
                                       "; chassis pose NOT reset") if ok else
                          ("gripper opened, but arm did NOT recenter (" + r.get("message", "") +
                           ") — check the arm before starting the task; chassis pose NOT reset")}

    def close(self):
        self.stop()
        try:
            if self._camera_ok:
                self.ep.camera.stop_video_stream()
        except Exception:
            pass
        try:
            self.ep.close()
        except Exception:
            pass
        print("[server] 已關閉機器人連線")


# ============================================================================
# Backend：Mock (沒有 SDK 也能測整個 Agent 迴圈)
# ============================================================================
class MockBackend(object):
    """一個 2D 的假機器人：會累積位置，畫面上畫出自己的座標和一顆假的紅色目標。"""

    def __init__(self):
        self.mode = "mock"
        self.target = (1.5, 0.0)  # 世界座標中的紅色方塊
        self.reset()
        print("[server] Mock backend 啟動，不會連線到任何機器人")

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0          # 左轉為正
        self.arm = [80.0, 100.0]
        self.grip = "opened"
        return {"ok": True, "message": "mock world reset"}

    def state(self):
        dx, dy = self.target[0] - self.x, self.target[1] - self.y
        return {
            "mode": "mock", "camera": cv2 is not None,
            "position_m": {"x": round(self.x, 3), "y": round(self.y, 3)},
            "yaw_deg": round(self.yaw, 1),
            "arm_mm": {"x": self.arm[0], "y": self.arm[1]},
            "gripper": self.grip,
            "tof_distance_mm": int(math.hypot(dx, dy) * 1000),
            "battery_percent": 88,
        }

    def frame_jpeg(self, width=640, quality=80, raw=False):
        if cv2 is None:
            return None
        img = np.full((360, 640, 3), 225, dtype=np.uint8)
        cv2.rectangle(img, (0, 200), (640, 360), (170, 170, 170), -1)  # 地板
        # 把目標投影到畫面上：用相對方位角決定水平位置，用距離決定大小
        dx, dy = self.target[0] - self.x, self.target[1] - self.y
        dist = max(math.hypot(dx, dy), 0.05)
        bearing_left = math.degrees(math.atan2(-dy, dx)) - self.yaw  # y 為右，所以取負
        bearing_left = (bearing_left + 180) % 360 - 180
        if abs(bearing_left) < 45:
            cx = int(320 - bearing_left / 45.0 * 320)
            size = int(min(150, 40 / dist))
            cv2.rectangle(img, (cx - size, 220 - size), (cx + size, 220 + size), (40, 40, 220), -1)
        cv2.putText(img, "MOCK x=%.2f y=%.2f yaw=%.0f grip=%s" % (self.x, self.y, self.yaw, self.grip),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

    def stop(self):
        pass

    def do(self, name, p):
        p = p or {}
        if name == "move_chassis":
            fwd = _clamp(p.get("forward_m", 0), LIMITS["forward_m"])
            right = _clamp(p.get("right_m", 0), LIMITS["right_m"])
            turn = _clamp(p.get("turn_left_deg", 0), LIMITS["turn_left_deg"])
            th = math.radians(self.yaw)
            # 世界座標：x 朝前，y 朝右；左轉讓車頭往 -y 偏
            self.x += fwd * math.cos(th) + right * math.sin(th)
            self.y += -fwd * math.sin(th) + right * math.cos(th)
            self.yaw = (self.yaw + turn + 180) % 360 - 180
            time.sleep(0.2)
            return {"ok": True, "message": "moved forward=%.2fm right=%.2fm turn_left=%.0fdeg" % (fwd, right, turn)}
        if name == "move_arm":
            self.arm[0] += _clamp(p.get("forward_mm", 0), LIMITS["arm_mm"])
            self.arm[1] += _clamp(p.get("up_mm", 0), LIMITS["arm_mm"])
            return {"ok": True, "message": "arm moved"}
        if name == "arm_to":
            self.arm = [float(p.get("x_mm", 100)), float(p.get("y_mm", 60))]
            return {"ok": True, "message": "arm target reached"}
        if name == "recenter_arm":
            self.arm = [80.0, 100.0]
            return {"ok": True, "message": "arm recentered"}
        if name == "gripper":
            state = str(p.get("state", "")).lower()
            if state not in ("open", "close"):
                return {"ok": False, "message": "gripper state must be 'open' or 'close'"}
            self.grip = "opened" if state == "open" else "closed"
            return {"ok": True, "message": "gripper %s" % state}
        if name == "wait":
            time.sleep(min(abs(_clamp(p.get("seconds", 1), LIMITS["wait_s"])), 0.2))
            return {"ok": True, "message": "waited"}
        if name == "navigate_to":
            self.x, self.y = float(p.get("x", self.x)), float(p.get("y", self.y))
            return {"ok": True, "message": "navigate_to: (mock) teleported to (%.2f, %.2f)" % (self.x, self.y),
                    "final_world_xy": [self.x, self.y]}
        if name == "drive_through":
            return {"ok": True, "message": "drive_through %s: (mock) PASSED THROUGH" % p.get("name"), "passed": True}
        if name == "go_to_landmark":
            return {"ok": True, "message": "go_to_landmark %s: (mock) arrived" % p.get("name")}
        if name == "stop":
            return {"ok": True, "message": "chassis stopped"}
        return {"ok": False, "message": "unknown action %r" % name}

    def close(self):
        pass


# ============================================================================
# HTTP 層
# ============================================================================
class _ThreadingServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(backend, token):
    action_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # 安靜一點，只印動作
            pass

        def _authorized(self):
            if not token:
                return True
            return self.headers.get("X-Robot-Token", "") == token

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._authorized():
                return self._send(401, {"ok": False, "message": "bad token"})
            path = self.path.split("?")[0]
            if path == "/health":
                info = {"ok": True, "mode": backend.mode}
                if hasattr(backend, "link"):
                    info.update(backend.link())
                return self._send(200, info)
            if path == "/state":
                return self._send(200, backend.state())
            if path == "/actions":
                return self._send(200, {"actions": ACTION_DOCS, "limits": LIMITS})
            if path == "/frame.jpg":
                jpg = backend.frame_jpeg(raw="raw=1" in self.path)
                if jpg is None:
                    return self._send(503, {"ok": False, "message": "no camera frame available"})
                return self._send(200, jpg, "image/jpeg")
            if path == "/overhead.jpg":
                jpg = backend.overhead_jpeg() if hasattr(backend, "overhead_jpeg") else None
                if jpg is None:
                    return self._send(503, {"ok": False, "message": "no overhead camera available"})
                return self._send(200, jpg, "image/jpeg")
            if path == "/overhead_depth.png":
                png = backend.overhead_depth_png() if hasattr(backend, "overhead_depth_png") else None
                if png is None:
                    return self._send(503, {"ok": False, "message": "no overhead depth available"})
                return self._send(200, png, "image/png")
            if path == "/map.png":
                png = backend.map_png() if hasattr(backend, "map_png") else None
                if png is None:
                    return self._send(503, {"ok": False, "message": "no navigation map available"})
                return self._send(200, png, "image/png")
            return self._send(404, {"ok": False, "message": "not found"})

        def do_POST(self):
            if not self._authorized():
                return self._send(401, {"ok": False, "message": "bad token"})
            path = self.path.split("?")[0]
            if path == "/stop":
                backend.stop()
                print("[server] !! 緊急停止")
                return self._send(200, {"ok": True, "message": "stopped"})
            if path == "/reset":
                if not action_lock.acquire(False):
                    return self._send(409, {"ok": False, "message": "robot is busy with another action"})
                try:
                    return self._send(200, backend.reset())
                except Exception as e:
                    return self._send(200, {"ok": False, "message": "%s: %s" % (type(e).__name__, e)})
                finally:
                    action_lock.release()
            if path != "/action":
                return self._send(404, {"ok": False, "message": "not found"})

            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                name, params = str(req["name"]), req.get("params") or {}
            except Exception as e:
                return self._send(400, {"ok": False, "message": "bad request: %s" % e})

            if not action_lock.acquire(False):
                return self._send(409, {"ok": False, "message": "robot is busy with another action"})
            try:
                t0 = time.time()
                try:
                    result = backend.do(name, params)
                except Exception as e:
                    backend.stop()  # 動作中途出例外，先讓底盤停下來
                    result = {"ok": False, "message": "%s: %s" % (type(e).__name__, e)}
                result["duration_s"] = round(time.time() - t0, 2)
                print("[server] %s %s -> %s" % (name, json.dumps(params), result))
                return self._send(200, result)
            finally:
                action_lock.release()

    return Handler


def main():
    ap = argparse.ArgumentParser(description="RoboMaster robot server for the LLM agent")
    ap.add_argument("--mode", default="sim", choices=["sim", "real", "mock"])
    ap.add_argument("--host", default="127.0.0.1",
                    help="預設只聽本機。要給遠端 Agent 用，建議走 SSH tunnel 或 Tailscale，不要直接開 0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=os.environ.get("ROBOT_SERVER_TOKEN", ""),
                    help="設了之後，所有請求都要帶 X-Robot-Token 標頭")
    ap.add_argument("--chassis-backend", default="move", choices=["move", "speed", "hybrid"],
                    help="move 用 SDK 的位置控制 (平移+旋轉都有完成偵測，建議先試這個)；"
                         "speed 用 drive_speed 加計時，完全沒有中途碰撞偵測；"
                         "hybrid 平移用位置控制 (保留卡住偵測)、旋轉用計時 (模擬器不回報轉向完成時用這個)")
    args = ap.parse_args()

    backend = MockBackend() if args.mode == "mock" else RoboMasterBackend(args.mode, args.chassis_backend)
    httpd = _ThreadingServer((args.host, args.port), make_handler(backend, args.token))
    print("[server] 就緒：http://%s:%d  (mode=%s, token=%s)" % (
        args.host, args.port, args.mode, "on" if args.token else "off"))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] 收到 Ctrl+C，準備關閉")
    finally:
        # 跟 test_movement.py 一樣：不管怎樣，離開前一定先停車再斷線
        backend.close()
        httpd.server_close()


if __name__ == "__main__":
    main()
