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
import collections
import json
import math
import os
import urllib.request
import signal
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
    "xy_speed": 0.5,        # 平移速度 m/s（真實世界的速度）。2026-11-11 實測：SDK chassis.move 的 xy_speed 不是 m/s——它會把 <0.5 夾成 0.5，
                            # 再編成 int(160*v-70) 這個位元組，實體跑出來約是「位元組/100」m/s（0.5→0.1 m/s、0.75→0.5 m/s），
                            # 所以之前 0.3/0.4/0.5 全是最慢的 0.1 m/s。_sdk_xy_speed() 負責換算；模擬外掛完全不理這個位元組（固定 ~0.12 m/s），
                            # 模擬改用 drive_speed + 里程計回授（_sim_translate）
    "z_speed": 70.0,        # 旋轉速度 deg/s（從 45 調高；模擬旋轉是看 yaw 回授停的，大角度留 5° 緩衝）
    "arm_mm": 100.0,        # 手臂單次相對位移上限 mm
    "lift_mm": 400.0,       # 手臂升降柱的絕對高度上限 mm（Env/add_arm_lift.py 的行程）
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


def _nex_move_verdict(req_dx, req_dz, meas_dx, meas_dz, target, tol_mm=12.0, clamped="", has_lift=False):
    """NexArm 相對移動的裁決：拿「量到的位移」跟要求比，不能只看離（可能被夾過的）目標多遠。
    2026-11-09 實測：從 z=294 要求 +50，目標被夾到上限、只走了 +36，server 卻回 ok=True，agent 就回報「圓滿完成」。
    差超過 tol_mm 就算沒達成；目標被可達區域夾過的話（clamped 非空），原因就是到了那個方向的極限。"""
    note = " | measured forward=%+d up=%+d mm" % (meas_dx, meas_dz)
    ok = True
    for req, meas, label, axis in ((req_dx, meas_dx, "forward", "x"), (req_dz, meas_dz, "up", "z")):
        if abs(req) < 5:
            continue
        # 差在容許內、而且至少走了要求的一半才算達成（要求 +10 只走 +2 差 8 mm 在容許內，但那不是「有動」）
        if abs(meas - req) <= tol_mm and (abs(req) < 10 or (meas * req > 0 and abs(meas) >= 0.5 * abs(req))):
            continue
        ok = False
        if clamped and ("%s " % axis) in clamped:
            edge = {("z", True): "highest", ("z", False): "lowest", ("x", True): "farthest", ("x", False): "nearest"}[(axis, req > 0)]
            why = "that is the %s the gripper can reach at this pose (%s)" % (edge, clamped)
            if axis == "z" and req > 0:
                folded = (math.hypot(target.get("x", 999) or 0, target.get("y", 0) or 0) < 80 and (target.get("pitch", 0) or 0) <= -80)
                if folded:
                    why += ("; the arm is folded over its base (gripper pointing down), which caps the height at 160 mm"
                            " - extend forward (move_arm forward_mm) or recenter_arm(pose=\"home\") first to go higher")
                else:
                    why += "; use lift_arm to raise the whole arm" if has_lift else "; this arm has no lift column"
            elif axis == "z" and req < 0:
                why += "; pointing the gripper down (arm_to with pitch_deg -90) reaches lower"
        else:
            why = "servo load or an unreachable pose"
        note += " | NOT achieved: asked %s %+.0f mm but the arm moved %+d mm; %s" % (label, req, meas, why)
    return ok, note


ACTION_DOCS = [
    {"name": "move_chassis", "params": {"forward_m": "float", "right_m": "float", "turn_left_deg": "float"},
     "doc": "relative chassis motion; translation is executed first, then rotation"},
    {"name": "move_arm", "params": {"forward_mm": "float", "up_mm": "float", "turn_left_deg": "float (optional, NexArm only: rotate the arm's base)",
                                    "roll_deg": "float (optional, NexArm only: rotate the wrist)"},
     "doc": "relative arm end-effector motion"},
    {"name": "arm_to", "params": {"x_mm": "float", "y_mm": "float", "pitch_deg": "float (optional, NexArm only)",
                                  "yaw_deg": "float (optional, NexArm only: base rotation, 0 = straight ahead, + = left)",
                                  "roll_deg": "float (optional, NexArm only: wrist roll)"},
     "doc": "absolute arm pose: x forward reach, y height; also sets the camera viewpoint"},
    {"name": "recenter_arm", "params": {"pose": "home|stow (optional, NexArm only)"},
     "doc": "move arm back to its default pose; pose=stow folds the NexArm over its base with the gripper pointing down"},
    {"name": "lift_arm", "params": {"height_mm": "float"},
     "doc": "absolute height of the arm lift column (0-400 mm): raises the whole arm, its camera and the front distance sensor"},
    {"name": "gripper", "params": {"state": "open|close"}, "doc": "open or close the gripper"},
    {"name": "set_held", "params": {"tof_mm": "int|null"},
     "doc": "tell the server an object is held in the gripper at this front-sensor distance (navigation then ignores that reading); null when released"},
    {"name": "wait", "params": {"seconds": "float"}, "doc": "do nothing, then re-observe"},
    {"name": "stop", "params": {}, "doc": "stop the chassis"},
    {"name": "return_to_origin", "params": {"tol_m": "float (optional)", "tol_deg": "float (optional)"},
     "doc": "real chassis only: drive back to where this SDK session started, by odometry (mirror batches use it between tasks)"},
    {"name": "navigate_to", "params": {"x": "float", "y": "float", "stop_m": "float", "tolerance_m": "float", "face": "bool",
                                       "approach_from_deg": "float|null"},
     "doc": "plan a collision-free path on the occupancy map and drive to world (x, y); stop_m stops that far before the point and faces it; "
            "approach_from_deg (world angle, target -> standing point) forces which side to stand on"},
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


def _new_sim_client(recv_timeout_s=15.0):
    """RemoteAPIClient 加上 socket 逾時與 REQ 容錯。模擬器死掉或卡住時，呼叫在 recv_timeout_s 內丟例外，
    不會把動作執行緒永遠卡在 recv()（2026-10-22 r04：CoppeliaSim 崩潰後 navigate_to 卡住、鎖不放，後面全部 busy）；
    REQ_RELAXED / REQ_CORRELATE 讓逾時之後的下一次 send 不會因為 REQ 狀態機而 EFSM。"""
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    c = RemoteAPIClient()
    try:
        import zmq
        c.socket.setsockopt(zmq.RCVTIMEO, int(recv_timeout_s * 1000))
        c.socket.setsockopt(zmq.REQ_RELAXED, 1)
        c.socket.setsockopt(zmq.REQ_CORRELATE, 1)
    except Exception as e:
        print("[server] zmq socket 選項設定失敗：%s" % e)
    return c


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
    def __init__(self, mode, chassis_backend="move", arm="sdk", arm_port="/dev/nexarm"):
        self.mode = mode
        self.chassis_backend = chassis_backend
        self.telemetry = {}
        self._tlock = threading.Lock()
        self._stall_count = 0  # 連續幾次「完全沒動、旁邊又沒東西擋」，用來偵測控制腳本整個卡死
        self.ep = None
        self.nexarm = None
        # SDK chassis.move / drive_speed 的 z 軸正方向：模擬器的 RoboMaster 外掛是順時針為正（turn_left 要取負），
        # 實體 EP 實測是逆時針為正（2026-11-09：送 z=-90 真機往右轉）。yaw 遙測兩邊都是順時針為正，跟這個無關。
        self._sdk_z_sign = -1.0 if mode == "sim" else 1.0
        # 手臂用誰：armonly / --arm nexarm → USB 接的實體 NexArm；sim 模式下場景裡有 NexArm 模型（Env/add_nexarm.py）→ 模擬的 NexArm；
        # 其他情況用 RoboMaster SDK 自己的手臂。先決定好，_subscribe_all 才知道要不要略過 SDK 的手臂/夾爪遙測。
        self._want_nexarm = (mode == "armonly" or arm == "nexarm"
                             or (arm in ("auto", "nexarm-sim") and mode == "sim" and self._scene_has_nexarm()))

        if mode == "armonly":
            # 只有桌上的 NexArm，沒有 RoboMaster：不連 SDK、沒有底盤、沒有相機。手臂與夾爪動作照常，其他動作會說不可用。
            self._camera_ok = self._camera_live_ok = False
            self._camera_fail_streak = 0
            print("[server] armonly 模式：不連 RoboMaster，只有 NexArm 的手臂與夾爪可用")
        else:
            self._connect_robot()
        self._start_prox_sensors()
        self._start_overhead_cam()
        if self._want_nexarm:
            if mode == "sim":
                self._start_nexarm_sim()
            else:
                self._start_nexarm(arm_port)

    @staticmethod
    def _scene_has_nexarm():
        try:
            sim = _new_sim_client(5.0).require("sim")
            root = _find_robot_root(sim)
            return sim.getObject("./nexarm_mount", {"proxy": root, "noError": True}) != -1
        except Exception as e:
            print("[server] 查不到場景裡有沒有 NexArm（%s），用 SDK 的手臂" % e)
            return False

    def _connect_robot(self):
        """建立/重新建立跟 SDK 的連線，開相機串流，訂閱所有 telemetry。
        獨立成方法是因為 _auto_recover() 也要用同一套流程重新連一次。"""
        from rm_connection import connect, start_camera  # 連線設定全部留在你原本的檔案裡

        print("[server] 連線中 (mode=%s) ..." % self.mode)
        self.ep = connect(self.mode)
        try:
            print("[server] 連線成功，韌體版本：", self.ep.get_version())
        except Exception as e:
            print("[server] 取不到韌體版本 (可忽略)：", e)

        self._camera_ok = False
        self._camera_fail_streak = 0
        self._camera_live_ok = False
        # start_video_stream 失敗只回 False 不丟例外，以前這裡會誤把 _camera_ok 設成 True
        if os.environ.get("RM_NO_CAMERA") == "1":
            print("[server] RM_NO_CAMERA=1：不開 EP 相機串流（相機壞掉時每次重啟都要等 3 次逾時，約 20 秒）")
        elif start_camera(self.ep):
            self._camera_ok = True
            self._camera_live_ok = True
        else:
            print("[server] 相機串流啟動失敗，Agent 會在沒有畫面的情況下運作")
        self._start_ep_link_watchdog()

        self._subscribe_all()
        self._hold_pose()

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
        # 前方感測器掛在手臂升降座上（Env/add_arm_lift.py 加的 arm_lift_carriage）：升降座升到桌面高度去抓東西時，
        # 感測器要跟夾爪一起升，「送物件進指間」的幾何才跟在地上抓一模一樣；沒有升降座的場景就照舊掛在底盤上。
        # 建立當下升降座可能不在最低點（server 中途重開），位置要先加上目前的行程，掛上去之後相對升降座才是對的。
        lift_car = sim.getObject("./arm_lift_carriage", {"proxy": root, "noError": True})
        lift_q = 0.0
        if lift_car != -1:
            try:
                lift_q = sim.getJointPosition(sim.getObject("./arm_lift_joint", {"proxy": root}))
            except Exception:
                lift_q = 0.0

        handles = {}
        for name, dir_kind, along_fwd, along_right, height, rng, angle_deg in self._PROX_SPECS:
            direction = {"fwd": fwd, "left": tuple(-v for v in right), "right": right}[dir_kind]
            on_lift = name == "prox_front" and lift_car != -1
            pos = [root_pos[i] + fwd[i] * along_fwd + right[i] * along_right + up[i] * (height + (lift_q if on_lift else 0.0))
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
            sim.setObjectParent(h, lift_car if on_lift else root, True)
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
                if alias.startswith(("small_", "ball_", "foam_", "pit_ball", "trash_")):
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
        print("[server] 距離感測器已建立（校正用 yaw=%.1f）：front/left/right%s"
              % (yaw, "；前方那顆掛在手臂升降座上（目前升 %.0f mm）" % (lift_q * 1000) if lift_car != -1 else ""))
        return handles

    def _start_prox_sensors(self):
        self._prox_handles = None
        if self.mode != "sim":
            return
        try:
            sim = _new_sim_client().require("sim")
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
        sim = _new_sim_client().require("sim")
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
        # 矮桌（Env/add_arm_lift.py）：桌上的東西要把手臂升到桌面高度才夾得到，skills.pick/place 靠 top_m 決定升多高
        "table",
        # 垃圾桶（Env/add_litter.py）：黑色一般垃圾、白色回收，收拾類任務的目的地
        "bin_trash", "bin_recycle",
    ]

    @staticmethod
    def _landmark_extent(sim, h):
        """地標整棵樹的世界座標範圍：{"top_m": 頂面高度, "bbox": [xmin, xmax, ymin, ymax]}；沒有形狀就 None。
        top_m 取「地面投影面積最大的那個形狀」的頂面（桌面、長椅的座面），不是整棵樹的最高點——椅背、放在桌上的
        柱子/物件都比桌面高，拿最高點當桌面會把手臂升到物件上方。"""
        lo, hi = [1e9] * 3, [-1e9] * 3
        shapes = sim.getObjectsInTree(h, sim.object_shape_type, 0)
        # 只算靜態的結構件：球池底下還掛著 100 多顆會動的池球，算進去不但超過上限、範圍也會跟著球跑
        shapes = [sh for sh in shapes if sim.getObjectInt32Param(sh, sim.shapeintparam_static)]
        if len(shapes) > 80:
            # 每個形狀要 8 次 zmq 呼叫。球池底下有 299 個形狀（底板、40 多片牆、之後是 200 多顆固定的池球 pit_fixed_*），
            # 全算太慢；建場景時底板跟牆先建，取前 60 個就涵蓋整個外框。俯視掃描要靠球池的範圍排除池裡的球（2026-10-21）。
            shapes = shapes[:60]
        best_area, top = -1.0, None
        for sh in shapes:
            # 每個形狀只做 2 次 zmq 來回（以前 15 次：6 個 bbox 參數 + 矩陣 + 8 個角點各一次 multiplyVector；
            # 球池 60 個形狀就 900 次，有視窗的 CoppeliaSim 上慢到啟動超過 90 秒）。場景的地標都是基本形狀，
            # bbox 以物件原點為中心，角點用矩陣在本地算就好。
            size = sim.getShapeBB(sh)[0]   # 回傳 (尺寸, 姿態)
            m = sim.getObjectMatrix(sh, sim.handle_world)
            slo, shi = [1e9] * 3, [-1e9] * 3
            for cx in (-size[0] / 2, size[0] / 2):
                for cy in (-size[1] / 2, size[1] / 2):
                    for cz in (-size[2] / 2, size[2] / 2):
                        w = [m[0] * cx + m[1] * cy + m[2] * cz + m[3],
                             m[4] * cx + m[5] * cy + m[6] * cz + m[7],
                             m[8] * cx + m[9] * cy + m[10] * cz + m[11]]
                        slo = [min(a, c) for a, c in zip(slo, w)]
                        shi = [max(a, c) for a, c in zip(shi, w)]
            lo = [min(a, c) for a, c in zip(lo, slo)]
            hi = [max(a, c) for a, c in zip(hi, shi)]
            area = (shi[0] - slo[0]) * (shi[1] - slo[1])
            if area > best_area:
                best_area, top = area, shi[2]
        if not shapes:
            return None
        return {"top_m": round(top, 3), "bbox": [round(lo[0], 2), round(hi[0], 2), round(lo[1], 2), round(hi[1], 2)]}

    def _query_static_landmarks(self, sim):
        landmarks = {}
        self._static_landmark_handles = {}
        self._static_landmark_info = {}
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
                continue
            try:
                self._static_landmark_info[name] = self._landmark_extent(sim, h)
            except Exception:
                self._static_landmark_info[name] = None
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
        with self._sim_for_overhead_lock:
            self._render_overhead()   # 從此只在讀圖時算圖（見 _render_overhead）
        self._static_landmarks_xy = self._query_static_landmarks(sim)
        self._robot_cam = self._robot_camera_handle(sim, root)
        self._wheel_handles = self._find_wheel_handles(sim, root)
        if not self._wheel_handles:
            print("[server] 找不到四個輪子關節，車頭方向退回用相機視線（手臂抬高時可能不準）")
        self._gripper_fs = self._find_gripper_force_sensor(sim, root)
        if self._gripper_fs == -1:
            print("[server] 找不到夾爪的 Force_sensor，/state 不會有 gripper_force_n")
        self._lift_joint = self._find_lift_joint(sim, root)
        if self._lift_joint == -1:
            print("[server] 這台沒有手臂升降柱（arm_lift_joint），lift_arm 不可用、/state 不會有 lift_mm")
        else:
            print("[server] 手臂升降柱已連上（arm_lift_joint handle %d，行程 0-%.0f mm）" % (self._lift_joint, LIMITS["lift_mm"]))
        self._start_nav(root)
        threading.Thread(target=self._poll_world_xy, daemon=True).start()
        print("[server] 俯視攝影機已連上：/overhead.jpg")

    # ---- 導航：占據格地圖 + A* + 閉迴路執行（見 nav_map.py）----
    # 2026-10-22 起 move_chassis 的 turn_left_deg 為正＝真正的左轉（從上方看逆時針、世界朝向增加）。
    # 以前是 +1 送給 SDK 的 z（SDK 的 z/yaw 順時針為正），所以「turn_left」其實會右轉，整套系統用 TURN_SIGN=-1
    # 和公式裡的負號互相抵消；數字對、名稱反，LLM 一用常識推理（球池在左邊→左轉→正值）就轉錯邊（批次 e04/e10/nav_002）。
    # 現在 _act_move_chassis 自己把符號翻給 SDK，所有「世界幾何 → turn_left」的換算都直接用 bearing - heading。
    TURN_SIGN = 1.0

    @staticmethod
    def _find_gripper_force_sensor(sim, root):
        """RoboMasterEP 模型裡夾爪跟手臂末端之間的力感測器（alias Force_sensor）。讀出來的合力就是「掛在夾爪上的重量」：
        實測空夾爪 1.50 N、夾著 50 g 的球 1.99 N（正好多 0.49 N），靜止時完全沒雜訊、跟手臂姿態無關——
        比相機看夾爪窗口裡有沒有顏色可靠得多（球掉在車前地上時相機一樣看得到顏色，2026-10-20 從東側夾球就這樣誤報）。"""
        try:
            return sim.getObject("./Force_sensor", {"proxy": root, "noError": True})
        except Exception:
            return -1

    @staticmethod
    def _find_lift_joint(sim, root):
        """Env/add_arm_lift.py 加在底盤跟手臂之間的直線關節；沒有就 -1。"""
        if os.environ.get("ROBOT_NO_LIFT") == "1":
            # 鏡射批次：實體 NexArm 沒有升降柱，模擬也把它關掉，兩邊能力才一樣（不然 agent 在模擬抬 400 mm、實體抬不了）
            print("[server] ROBOT_NO_LIFT=1：模擬的升降柱停用，lift_arm 不可用（跟實體一致）")
            return -1
        try:
            return sim.getObject("./arm_lift_joint", {"proxy": root, "noError": True})
        except Exception:
            return -1

    @staticmethod
    def _robot_camera_handle(sim, root):
        """agent 在模擬裡用哪顆相機當眼睛，環境變數 ROBOT_SIM_CAMERA 選：
          ep（預設，2026-11-11）：EP 原廠的前鏡頭 Vision_sensor（1280×720、視角 102°），裝在夾爪總成下緣、腕部 roll 那一節
                                 （實體相機修好後就裝在那裡），所以手臂動、腕部轉，視角都跟著動
          wrist：NexArm 的手腕相機 nexarm_wrist_cam（2026-11-09～11-10 用的那顆，跟著夾爪看）"""
        want = os.environ.get("ROBOT_SIM_CAMERA", "ep").strip().lower()
        cams = sim.getObjectsInTree(root, sim.object_visionsensor_type, 0)
        alias = {sim.getObjectAlias(c): c for c in cams}
        if want == "wrist" and "nexarm_wrist_cam" in alias:
            print("[server] ROBOT_SIM_CAMERA=wrist：用 NexArm 手腕相機")
            return alias["nexarm_wrist_cam"]
        if "Vision_sensor" in alias:
            print("[server] 前鏡頭用 EP 原廠的 Vision_sensor（ROBOT_SIM_CAMERA=wrist 可改用 NexArm 手腕相機）")
            return alias["Vision_sensor"]
        return alias.get("nexarm_wrist_cam", cams[0] if cams else -1)

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
        self._nav_target = None     # /live 用：最後一次 navigate_to 鎖定的目標（座標、名稱、地面範圍）
        self._cam_fov = None
        if NavMap is None:
            return
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            # 導航專用連線。RemoteAPIClient 是一個 zmq REQ socket，不是執行緒安全的：兩個執行緒同時呼叫會讓
            # pyzmq 丟 EFSM（"Operation cannot be accomplished in current state"），_nav_pose() 讀一次失敗整段導航就
            # 放棄（2026-10-21 批次 m01/m05 的 lost world pose）。這裡把 call 包一層鎖，並記錄是誰撞到誰。
            client = _new_sim_client()
            _orig_call, _lk, _owner = client.call, threading.Lock(), [None]

            def _guarded_call(func, args):
                me = threading.current_thread().name
                if not _lk.acquire(False):
                    print("[zmq-nav] %s 要呼叫 %s，但連線正被 %s 使用，等它釋放" % (me, func, _owner[0]))
                    _lk.acquire()
                _owner[0] = me
                try:
                    return _orig_call(func, args)
                except Exception as e:
                    print("[zmq-nav] %s 呼叫 %s 失敗：%s" % (me, func, e))
                    raise
                finally:
                    _owner[0] = None
                    _lk.release()
            client.call = _guarded_call
            self._nav_sim = client.require("sim")
            t0 = time.time()
            self._nav = NavMap(self._nav_sim, root)
            print("[server] 占據格地圖已建立：%d 個靜態形狀、%d 個通道，%.1fs（GET /map.png 可以看）"
                  % (self._nav.static_shapes, len(self._nav.passages), time.time() - t0))
            # 出發點（世界座標 + 車頭朝向）：return_to_origin 在模擬裡用地圖導航回這裡（2026-11-11，使用者要「做完任務走回出發點」）
            self._home_world = self._nav_pose()
            if self._home_world:
                print("[server] 出發點：世界 (%.2f, %.2f) 朝向 %.0f°（return_to_origin 會回這裡）" % self._home_world)
            for pg in self._nav.passages:
                print("[server] 通道 (%.2f, %.2f) 軸 %.0f°：中線側向偏移 %+.2f m，沿線最小淨空 %.2f m"
                      % (pg["center"][0], pg["center"][1], pg["axis_deg"], pg.get("lateral", 0.0), pg.get("min_clearance", 0.0)))
        except Exception as e:
            self._nav = None
            print("[server] 占據格地圖建立失敗（%s），navigate_to/drive_through 不可用" % e)

    def _nav_pose(self):
        """機器人真實世界姿態 (x, y, heading_deg)，用導航專用的 zmq 連線讀。"""
        last = None
        for attempt in range(3):   # 偶發的 zmq 錯誤不該讓整段導航放棄：重試三次再說
            try:
                sim = self._nav_sim
                p = sim.getObjectPosition(self._overhead_root, sim.handle_world)
                return p[0], p[1], self._world_heading(sim)
            except Exception as e:
                last = e
                time.sleep(0.15)
        print("[server] 讀取世界姿態失敗（重試 3 次）：%s" % last)
        return None

    def _nav_mark_blocked(self, pose, dist_m):
        """把前方 dist_m 處一小塊標成動態障礙，讓下一次規劃繞開（ToF 看到東西、或指令沒走到時用）。"""
        try:
            x = pose[0] + dist_m * math.cos(math.radians(pose[2]))
            y = pose[1] + dist_m * math.sin(math.radians(pose[2]))
            self._nav.mark_blocked(x, y)
        except Exception as e:
            print("[server] 標記障礙失敗：%s" % e)

    # 前鏡頭畫面上的框：相機 x 軸朝車的左邊，畫面已經不是鏡像（2026-10-22 修正），所以 x = 中心 - u。
    FRAME_MIRROR = -1.0

    def frame_live_jpeg(self, width=640):
        """/live 專用：frame_jpeg 加上「鎖定目標」的框（洋紅）。送給 LLM 的 /frame.jpg 不受影響。"""
        jpg = self.frame_jpeg(width=width)
        tgt = getattr(self, "_nav_target", None)
        if jpg is None or tgt is None or cv2 is None:
            return jpg
        try:
            sim = getattr(self, "_sim_for_overhead", None)
            cam = getattr(self, "_robot_cam", -1)
            if sim is None or cam is None or cam == -1:
                return jpg
            with self._sim_for_overhead_lock:
                M = sim.getObjectMatrix(cam, sim.handle_world)
                if getattr(self, "_cam_fov", None) is None:
                    self._cam_fov = sim.getObjectFloatParam(cam, sim.visionfloatparam_perspective_angle)
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            H, W = img.shape[:2]
            f = (max(W, H) / 2.0) / math.tan(self._cam_fov / 2.0)
            R = [[M[0], M[1], M[2]], [M[4], M[5], M[6]], [M[8], M[9], M[10]]]
            t = (M[3], M[7], M[11])
            bb = tgt.get("bbox")
            if bb:
                xs, ys = (bb[0], bb[1]), (bb[2], bb[3])
            else:
                x0, y0 = tgt["xy"]
                xs, ys = (x0 - 0.08, x0 + 0.08), (y0 - 0.08, y0 + 0.08)
            zs = (0.0, tgt.get("top_m") or 0.25)
            us, vs = [], []
            for x in xs:
                for y in ys:
                    for z in zs:
                        d = (x - t[0], y - t[1], z - t[2])
                        pc = [sum(R[i][k] * d[i] for i in range(3)) for k in range(3)]   # R^T d
                        if pc[2] <= 0.05:
                            continue
                        us.append(f * pc[0] / pc[2])
                        vs.append(f * pc[1] / pc[2])
            if len(us) < 4:
                return jpg   # 目標在背後／視野外
            x0 = int(round(W / 2.0 + self.FRAME_MIRROR * min(us))); x1 = int(round(W / 2.0 + self.FRAME_MIRROR * max(us)))
            y0 = int(round(H / 2.0 - max(vs))); y1 = int(round(H / 2.0 - min(vs)))
            x0, x1 = sorted((x0, x1))
            if x1 < 0 or x0 >= W or y1 < 0 or y0 >= H:
                return jpg
            x0, x1 = max(0, x0), min(W - 1, x1); y0, y1 = max(0, y0), min(H - 1, y1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 255), 2)
            cv2.putText(img, "%s %.1fs ago" % (tgt.get("label", "target"), time.time() - tgt["t"]) if time.time() - tgt["t"] > 90
                        else str(tgt.get("label", "target")), (x0 + 2, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return buf.tobytes() if ok else jpg
        except Exception as e:
            print("[server] frame_live 畫框失敗：%s" % e)
            return jpg

    def map_png(self):
        if not getattr(self, "_nav", None):
            return None
        try:
            # 姿態用輪詢執行緒已經讀好的遙測，不呼叫 _nav_pose()：那條 zmq 連線只能給動作執行緒用。
            # /live 每秒抓一次 /map.png，HTTP 執行緒跟導航同時用同一個 socket 會讓模擬器回
            # 「Operation cannot be accomplished in current state」，navigate_to 讀到一次失敗就整段放棄
            # （2026-10-21 批次 m01/m05 的 lost world pose 就是這樣來的，開著儀表板就能重現）。
            st = self.state()
            xy, hd = st.get("world_xy"), st.get("world_heading_deg")
            pose = (float(xy[0]), float(xy[1]), float(hd)) if xy and hd is not None else None
            return self._nav.render_png(robot_pose=pose, path=self._nav_last_path,
                                        goal=self._nav_last_goal, target=getattr(self, "_nav_target", None))
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
        label = str(p.get("label") or "").strip()
        info = (getattr(self, "_static_landmark_info", None) or {}).get(label.lower())
        self._nav_target = {"xy": target, "label": label or "target", "t": time.time(),
                            "bbox": info["bbox"] if info else None, "top_m": info["top_m"] if info else None}
        # 站位要準：跑分的 ToF 檢查窗只有 150-450 mm，到達容忍 0.15 + 最後一步少走 3 cm 會讓 stop 0.30 停成 0.43
        tol = max(0.05, float(p.get("tolerance_m") or (0.08 if stop_m > 0 else 0.15)))
        face = bool(p.get("face", stop_m > 0))
        approach_from = p.get("approach_from_deg")   # 呼叫端指定從哪一側接近（目標→站位的世界角度）：桌子要垂直桌緣接近
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
                if approach_from is not None:
                    # 桌上的東西：斜著接近時車頭的角會先頂到桌緣，車頭中心停不到離桌緣 2 cm；skills.pick/place 指定
                    # 垂直桌緣的方向，這裡就只沿那個方向（±15°）找站位，找不到才退回全方位掃描
                    base = math.radians(float(approach_from))
                # stop_m 的語意是「車頭前緣到目標表面的距離」（人講的「停在它前面 30 cm」）。沿每個方向先找目標本體的邊緣
                # （從中心往外第一個沒被占據的格子），站位從 邊緣 + stop_m + 車身半長 0.166 起算；以前從中心算 stop_m，
                # 停 0.3 m 時物件已在 ToF 感測器（車頭前 0.18 m）後面、讀 9999，「停在它前面」的檢查永遠不過。
                # 轉身用的一般淨空維持 0.30（不能把牆算進去：角落的柱子會 2 m 內找不到站位）。
                def find_cands(sm, offs=(0, 30, -30, 60, -60, 90, -90, 120, -120)):
                    cands = []
                    for off in offs:
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
                    # 呼叫端指定了接近方向（桌子要垂直桌緣）時反過來：偏角 0 的站位優先，距離其次——2026-10-21 r11
                    # 偏 15° 的站位只因為近 2 cm 被選中，車身斜 16° 進桌子，把桌上的球撞下去
                    if approach_from is not None:
                        # c[3] = |off|：0/90/180 都是「正對某一邊」，15 是斜的；正對的一律優先，再比離桌緣的距離（15 cm 一級），
                        # 同一級裡選離機器人現在位置最近的那一側（少走路；2026-10-21 r03 四邊打平時照候選順序選到東側，
                        # 從東側夾球一直會把球撞掉——光照讓顏色對準偏 5-7 mm，README 的紫球紀錄也是東側）
                        cands.sort(key=lambda c: (self._nav.passage_axis_at(c[1], c[2]) is not None, c[3] % 90 != 0,
                                                  int(c[0] / 0.15), math.hypot(c[1] - pose[0], c[2] - pose[1])))
                    else:
                        cands.sort(key=lambda c: (self._nav.passage_axis_at(c[1], c[2]) is not None, round(c[0], 2), c[3]))
                    return cands
                cands = []
                for sm in (stop_m, round(max(0.2, stop_m - 0.15), 2), round(max(0.2, stop_m - 0.25), 2)):
                    # 指定接近方向時：四個正對邊的站位（偏好的那一側 + 其他三側的法線）都算，再加偏好側 ±15°；
                    # 排序時正對的優先、其次離桌緣最近。2026-10-21 r11：桌子南側正前方 0.3 m 就是橘球的柱子，只看南側軸線
                    # 會把站位推到 1.95 m 外（送料最多走 1 m，pick 讀 9999），西側軸線其實 0.6 m 就能站
                    cands = find_cands(sm, (0, 90, -90, 180, 15, -15)) if approach_from is not None else find_cands(sm)
                    if not cands and approach_from is not None:
                        cands = find_cands(sm)   # 指定的那一側站不了，退回全方位（總比沒站位好，呼叫端會看到斜著站）
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
                step = min(d, 1.0)   # 原 0.4：直線段放長、段數變少（每段都有停穩+里程計+等實體的固定開銷）；前方 ToF 比步長短就會自動縮步
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
            # 2026-11-11：0.2/0.3/0.5 → 0.3/0.5/1.0。每段都有固定開銷（停穩、里程計、等實體），到桌子 22 段裡 14 段是直走；
            # 前方 ToF 比步長短時會自動縮步，規劃器本來就留 0.19 m 淨空，步長放大不影響避障
            step_cap = 0.3 if tight else (0.5 if near else 1.0)
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
            r = self._act_navigate_to({"x": entrance[0], "y": entrance[1], "tolerance_m": 0.08, "face": False, "label": name})
            pose2 = self._nav_pose()
            if r.get("ok") and pose2 is not None:
                self._turn_to(math.degrees(math.atan2(uy, ux)), pose2, [], tol=3.0, max_tries=3)
            r["message"] = ("go_to_landmark %s: this is a pass-through structure, so the robot stops at its entrance "
                            "(%.2f, %.2f) facing along its axis; use drive_through(%s) to go through. %s"
                            % (name, entrance[0], entrance[1], name, r.get("message", "")))
            return r
        r = self._act_navigate_to({"x": cx, "y": cy, "stop_m": stop_m, "face": True, "label": name})
        r["message"] = "go_to_landmark %s (centre %.2f, %.2f): %s" % (name, cx, cy, r.get("message", ""))
        return r

    def _act_landmark_axis(self, p):
        """唯讀：通道類地標（tunnel / platform）的軸線幾何，給 agent 端 align_to_tunnel 用。server 本來就知道這些結構
        的位置和軸向，不必靠指向模型在俯視圖上找兩端（gpt-4o 指不準，批次 m02/m05 都從這裡開始出錯）。"""
        if not getattr(self, "_nav", None):
            return {"ok": False, "message": "navigation map unavailable (needs sim mode with the overhead camera)"}
        name = str(p.get("name", "")).strip().lower()
        h = (getattr(self, "_static_landmark_handles", {}) or {}).get(name)
        if h is None:
            return {"ok": False, "message": "landmark_axis: unknown landmark %r" % name}
        geo = self._nav.group_axis(h)
        if geo is None:
            return {"ok": False, "message": "landmark_axis: cannot read the geometry of %r" % name}
        (cx, cy), (ux, uy), half_len, half_w = geo
        pg = self._nav.passage_line_near(cx, cy)
        if pg:
            cx, cy = pg["center_line"]
        pose = self._nav_pose()
        if pose is None:
            return {"ok": False, "message": "cannot read the robot's world pose"}
        ux, uy = self._pick_entrance_side(pose, (cx, cy), (ux, uy), half_len + 0.65)
        standoff = half_len + 0.65
        axis_deg = math.degrees(math.atan2(uy, ux))
        ent = (cx - ux * standoff, cy - uy * standoff)
        return {"ok": True, "name": name, "centre": [round(cx, 2), round(cy, 2)], "axis_bearing_deg": round(axis_deg, 1),
                "half_len_m": round(half_len, 2), "half_w_m": round(half_w, 2),
                "near_end": [round(cx - ux * half_len, 2), round(cy - uy * half_len, 2)],
                "far_end": [round(cx + ux * half_len, 2), round(cy + uy * half_len, 2)],
                "entrance": [round(ent[0], 2), round(ent[1], 2)],
                "exit": [round(cx + ux * (half_len + 0.3), 2), round(cy + uy * (half_len + 0.3), 2)],
                "message": "%s: through-axis bearing %.0f deg, entrance (%.2f, %.2f)" % (name, axis_deg, ent[0], ent[1])}

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
        r1 = self._act_navigate_to({"x": entrance[0], "y": entrance[1], "tolerance_m": 0.06, "face": False, "label": name})
        if not r1.get("ok"):
            return {"ok": False, "message": "drive_through %s: could not reach the entrance — %s" % (name, r1.get("message"))}
        log = []
        pose = self._nav_pose()
        if pose is not None:
            _ok, pose = self._turn_to(axis_deg, pose, log, tol=2.0, max_tries=4)
        r2 = self._act_navigate_to({"x": exit_pt[0], "y": exit_pt[1], "tolerance_m": 0.15, "face": False, "label": name})
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
                    lift_q = None
                    if getattr(self, "_lift_joint", -1) != -1:
                        lift_q = sim.getJointPosition(self._lift_joint)
                    grip_f = None
                    if getattr(self, "_gripper_fs", -1) != -1:
                        res_f, f, _t = sim.readForceSensor(self._gripper_fs)
                        if res_f & 1:
                            grip_f = math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
                self._set("world_xy", [round(p[0], 3), round(p[1], 3)])
                if grip_f is not None:
                    self._set("gripper_force_n", round(grip_f, 2))
                if hd is not None:
                    self._set("world_heading_deg", round(hd, 1))
                if lift_q is not None:
                    self._set("lift_mm", int(round(lift_q * 1000)))
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
                            self._lift_joint = self._find_lift_joint(sim, root)
                            self._gripper_fs = self._find_gripper_force_sensor(sim, root)
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
                self._render_overhead()
                img, res = self._sim_for_overhead.getVisionSensorImg(self._overhead_handle)
            arr = np.frombuffer(img, dtype=np.uint8).reshape(res[1], res[0], 3)
            arr = cv2.flip(arr, 0)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    def _render_overhead(self):
        """俯視相機改成「手動更新」：1024×1024 每個物理步（20 Hz）都算圖，無視窗模擬只剩 0.37 倍即時，
        改成只在真的要讀圖時才算，模擬變 0.54 倍（2026-11-11 量，剩下的是 103 個動態物件的物理）。
        旗標在模擬停止/重播後會被還原，所以每次讀圖前都重設一次（兩個很便宜的呼叫）。要在 _sim_for_overhead_lock 裡呼叫。"""
        sim = self._sim_for_overhead
        try:
            sim.setExplicitHandling(self._overhead_handle, 1)
            sim.handleVisionSensor(self._overhead_handle)
        except Exception as e:
            print("[server] 俯視相機手動更新失敗（%s），改回自動" % e)
            try:
                sim.setExplicitHandling(self._overhead_handle, 0)
            except Exception:
                pass

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
                self._render_overhead()
                buf, res = self._sim_for_overhead.getVisionSensorDepth(self._overhead_handle, 1)
            arr = np.frombuffer(buf, dtype=np.float32).reshape(res[1], res[0])
            arr = cv2.flip(arr, 0)
            mm = np.clip(arr * 1000.0, 0, 65535).astype(np.uint16)
            ok, out = cv2.imencode(".png", mm)
            return out.tobytes() if ok else None
        except Exception:
            return None

    EP_LINK_LOST_EXIT_S = 20.0

    def _start_ep_link_watchdog(self):
        """實體 EP 的 SDK 連線死掉時（Wi-Fi 還通、ping 正常，但遙測停了；2026-11-10 兩批鏡射都在開車途中遇到），
        卡在 SDK 裡的動作永遠不回來、動作鎖放不開，之後每個指令都 409。SDK 沒有可靠的重連方法，
        最乾淨的是整個程序退出（exit code 3），start_arm.sh 看到 3 就自動重開；run_batch 的 reset_real 會等它回來。"""
        if self.mode != "real" or getattr(self, "_ep_watchdog", None) is not None:
            return

        def loop():
            while True:
                time.sleep(2.0)
                last = getattr(self, "_last_rx", None)
                if last is None:
                    continue
                age = time.time() - last
                if age > self.EP_LINK_LOST_EXIT_S:
                    print("[server] !! EP 遙測已停 %.0f 秒（Wi-Fi 可能還通，是 SDK 連線死了），程序退出讓 start_arm.sh 重開" % age, flush=True)
                    # stop() 走 SDK，連線死掉時可能也卡住：丟到執行緒裡最多等 2 秒就直接退出
                    threading.Thread(target=lambda: self.stop(), daemon=True).start()
                    time.sleep(2.0)
                    os._exit(3)

        self._ep_watchdog = threading.Thread(target=loop, daemon=True)
        self._ep_watchdog.start()

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
        arm_last = getattr(self, "_last_arm_rx", None)
        arm_age = (time.time() - arm_last) if arm_last else None
        if self.nexarm is not None and self.mode != "sim" and arm_age is not None and arm_age > 3.0:
            return {"robot_link": "lost", "detail": "no status from the NexArm for %.0fs. Check its power and the link "
                    "(USB: plug it straight into the PC, not a hub; Wi-Fi: the PC must stay on the NexArm hotspot); "
                    "the server keeps reopening the link by itself." % arm_age}
        if age > 3.0:
            if self.ep is not None:
                return {"robot_link": "lost", "detail": "no telemetry from the RoboMaster EP for %.0fs: its Wi-Fi link is gone "
                        "(PC switched network, or the robot is off). Reconnect and restart the server." % age}
            return {"robot_link": "lost", "detail": "no telemetry for %.0fs. The simulation was probably stopped or "
                    "restarted. Press play in CoppeliaSim, then restart robot_server.py." % age}
        detail = "telemetry %.1fs ago" % age
        if arm_age is not None and self.ep is not None:
            detail += ", arm %.1fs ago" % arm_age
        return {"robot_link": "ok", "detail": detail}

    def _subscribe_all(self):
        ep = self.ep
        subs = [
            ("chassis position", lambda: ep.chassis.sub_position(
                freq=5, callback=lambda p: self._set("position_m", {"x": round(p[0], 3), "y": round(p[1], 3)}))),
            ("chassis attitude", lambda: ep.chassis.sub_attitude(
                freq=20, callback=lambda a: self._on_yaw(round(a[0], 1)))),  # 20 Hz：45 deg/s 一個取樣 2 度
            ("arm position", lambda: ep.robotic_arm.sub_position(
                freq=5, callback=lambda p: self._set("arm_mm", {"x": _fix_int32(p[0]), "y": _fix_int32(p[1])}))),
            ("gripper status", lambda: ep.gripper.sub_status(
                freq=5, callback=lambda s: self._set("gripper", str(s)))),
            ("distance sensor", lambda: ep.sensor.sub_distance(
                freq=5, callback=lambda d: self._set("tof_distance_mm", int(d[0])))),
            ("battery", lambda: ep.battery.sub_battery_info(
                freq=1, callback=lambda b: self._set("battery_percent", int(b)))),
        ]
        # 用 NexArm 時，SDK 的手臂/夾爪遙測是舊手臂（模擬裡被隱藏了但還在）的值，會跟 NexArm 的 arm_mm 互相覆蓋
        skip = {"arm position", "gripper status"} if getattr(self, "_want_nexarm", False) else set()
        for name, fn in subs:
            if name in skip:
                continue
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
        if self.nexarm is not None:
            data.pop("gripper_force_n", None)   # 舊手臂的腕部力感測器：手臂藏起來了還是會讀到固定值，會誤導 agent
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
                bearing_w = math.degrees(math.atan2(ly - ry, lx - rx))
                if heading is not None:
                    # turn_left 為正＝左轉（逆時針）：直接用世界幾何，不管機器人一開始面向哪裡都正確
                    turn_left = _wrap_deg(bearing_w - heading)
                else:
                    # 沒有世界朝向（實體機）：SDK yaw 順時針為正，世界朝向 ≈ -yaw（以開機朝向為 0）
                    turn_left = (bearing_w + yaw + 180) % 360 - 180
                landmarks[name] = {"distance_m": round(dist, 2), "turn_left_deg": round(turn_left, 1),
                                   "world_xy": [lx, ly]}
                info = (getattr(self, "_static_landmark_info", None) or {}).get(name)
                if info:
                    landmarks[name]["top_m"] = info["top_m"]     # 頂面高度：桌子類地標要把手臂升到這個高度
                    landmarks[name]["bbox"] = info["bbox"]       # 地面範圍 [xmin, xmax, ymin, ymax]：判斷物件是不是在它上面
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

    def _sim_front_image(self):
        """sim 模式：直接向模擬器讀機器人自己的視覺感測器，不走 SDK 的 H.264 串流。

        2026-10-07 實測：SDK 串流在 headless 模擬器下旋轉完第一張就是新畫面，但 10/6 那批用 GUI 版
        跑時，e07 連續四步每一步的畫面都停在「上一次轉向之前」（yaw 174.9 時紅柱子該在正前方，畫面
        卻是牆；下一步才出現紅柱子），整批 21 次大角度旋轉裡也有 6 次轉完畫面幾乎一模一樣。_fresh_image
        用固定的時鐘上限（總共 4 秒、每張 1.5 秒）等新畫面，模擬一慢就在新畫面到之前放棄、默默回傳舊的，
        agent 看著舊畫面寫出「看到綠柱子」之類的錯誤回報。這裡讀的是模擬器當下的影像，跟模擬速度無關。
        跟穩定後的 SDK 畫面比對：上下、左右都翻轉後差異 3.4（JPEG 雜訊等級），其他方向都在 47 以上。
        實體機器人沒有這條路，照舊走 SDK。"""
        if self.mode != "sim" or os.environ.get("ROBOT_FRONT_CAMERA_SOURCE", "sim") == "sdk":
            return None
        sim = getattr(self, "_sim_for_overhead", None)
        cam = getattr(self, "_robot_cam", -1)
        if sim is None or cam is None or cam == -1:
            return None
        with self._sim_for_overhead_lock:
            # 先叫感測器重新拍一張再讀：NexArm 的手腕相機是「手動更新」的感測器（不每步都算，省算力），不先 handle 永遠是舊圖
            # （2026-11-09 實測手臂轉到朝下、畫面還是水平時那張）。對自動更新的感測器多 handle 一次也無害。
            try:
                sim.handleVisionSensor(cam)
            except Exception:
                pass
            buf, res = sim.getVisionSensorImg(cam)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)
        # 視覺感測器的緩衝區是由下往上存，只需要上下翻（flip 0）。以前翻 -1（上下＋左右）是為了跟 SDK 串流一致，
        # 但那條串流本身就是鏡像的：2026-10-22 用相機世界矩陣投影地標驗證，物理上在車右側的平台出現在畫面左邊。
        return cv2.flip(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0)

    def frame_jpeg(self, width=640, quality=80, raw=False):
        if cv2 is None:
            return None
        if not raw:
            try:
                img = self._sim_front_image()
            except Exception as e:
                print("[server] 直接讀模擬器前方相機失敗，退回 SDK 串流：%s" % e)
                img = None
            if img is not None:
                self._camera_fail_streak = 0
                self._camera_live_ok = True
                h, w = img.shape[:2]
                if w > width:
                    img = cv2.resize(img, (width, int(h * width / float(w))))
                ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                return buf.tobytes() if ok else None
        if not self._camera_ok:
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
        # 2026-10-22：這裡以前再做一次水平翻轉，依據是「遙測證實的左轉畫面卻往左滑」；但那個「左轉」是用順時針為正的
        # SDK yaw 判定的，其實是右轉，右轉時景物往左滑才是對的。翻轉拿掉。實體機器人的串流沒有重新驗證過。
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
        if self.ep is None:
            return
        try:
            self.ep.chassis.drive_speed(x=0, y=0, z=0)
        except Exception as e:
            print("[server] stop 失敗：", e)
        self._hold_pose()

    def _sdk_purge_actions(self, why=""):
        """清掉 SDK 動作表裡卡住的動作。robomaster SDK 的 bug（action.py send_action）：表裡還有同一個目標的動作時，
        它在「握著 _in_progress_mutex」的狀態下 raise，鎖永遠不放 → 收遙測的執行緒跟著卡死 → 20 秒後看門狗重開 server。
        2026-11-10 三次 EP「連線死掉」都是這條路：某個 move 逾時或 0 位移的 hold 沒被機器人 ack，留在表裡（state idle），
        下一個 chassis.move 就踩到。這裡：鎖若被卡著就先放掉，把表清空，再把那些動作標成 failed。"""
        d = getattr(self.ep, "_action_dispatcher", None)
        if d is None:
            return 0
        m = d._in_progress_mutex
        got = m.acquire(timeout=0.5)
        if not got and m.locked():
            try:
                m.release()   # 被 SDK 的 raise 卡住的鎖（threading.Lock 允許別的執行緒放）
                got = m.acquire(timeout=0.5)
            except Exception:
                got = False
        stale = list(d._in_progress.items())
        d._in_progress.clear()
        if got:
            m.release()
        for _k, act in stale:
            try:
                import robomaster.action as _ra
                act._changeto_state(_ra.ACTION_FAILED)
            except Exception:
                pass
        if stale:
            print("[server] 清掉 %d 個卡在 SDK 動作表裡的動作%s" % (len(stale), ("（%s）" % why) if why else ""))
        return len(stale)

    def _sdk_move(self, **kw):
        """chassis.move 的安全版：先清掉卡住的動作；SDK 因為「already performing」raise 時放掉它沒放的鎖、再送一次。"""
        self._sdk_purge_actions()
        try:
            return self.ep.chassis.move(**kw)
        except Exception as e:
            if "already performing" in str(e) or "already running" in str(e):
                self._sdk_purge_actions("already performing: %s" % str(e)[:60])
                return self.ep.chassis.move(**kw)
            raise

    @staticmethod
    def _sdk_xy_speed(v_mps):
        """真實速度 (m/s) → SDK chassis.move 的 xy_speed 刻度。SDK 編碼 byte=int(160*xy_speed-70)，實體跑出來 ≈ byte/100 m/s
        （2026-11-11 量：xy_speed 0.5→0.09 m/s、0.625→0.25、0.75→0.34 均速/0.5 峰值），反推 xy_speed=(100*v+70)/160，夾在 SDK 的 [0.5, 2.0]。"""
        return max(0.5, min(2.0, (100.0 * float(v_mps) + 70.0) / 160.0))

    def _sim_translate(self, fwd, right, v):
        """模擬的平移：drive_speed 給速度、邊走邊看里程計（position_m），到了就停；最後 15 cm 降到 0.15 m/s 免得衝過頭。
        1 秒內里程計沒前進就當被擋住，停下回報。回傳 {"ok":..., "message":...}。"""
        dist = math.hypot(fwd, right)
        ux, uy = fwd / dist, right / dist
        p0 = (self.state().get("position_m") or {})
        if "x" not in p0:
            # 沒有里程計可回授：退回位置控制
            t = dist / v * 2 + 3
            done = self._sdk_move(x=fwd, y=right, z=0, xy_speed=self._sdk_xy_speed(v)).wait_for_completed(timeout=t)
            return {"ok": True} if done else {"ok": False, "timed_out": t, "message": "translation timed out after %.1fs (blocked by something?)" % t}
        x0, y0 = p0["x"], p0["y"]
        budget = dist / v * 2 + 3
        deadline = time.time() + budget
        slow_v = min(v, 0.15)
        cur_v = None
        travelled = 0.0
        last_progress_t, last_progress_d = time.time(), 0.0
        blocked = False
        while time.time() < deadline:
            remaining = dist - travelled
            want_v = slow_v if remaining < 0.15 else v
            if cur_v != want_v:
                self.ep.chassis.drive_speed(x=want_v * ux, y=want_v * uy, z=0, timeout=budget + 1)
                cur_v = want_v
            time.sleep(0.05)
            p = self.state().get("position_m") or {}
            if "x" not in p:
                break
            travelled = math.hypot(p["x"] - x0, p["y"] - y0)
            if travelled >= dist - 0.02:
                break
            if travelled > last_progress_d + 0.01:
                last_progress_t, last_progress_d = time.time(), travelled
            elif time.time() - last_progress_t > 1.0:
                blocked = True
                break
        self.stop()
        time.sleep(0.15)
        if blocked:
            return {"ok": False, "message": "translation blocked after %.2fm of %.2fm (odometry stopped advancing)" % (travelled, dist)}
        return {"ok": True}

    def _hold_pose(self):
        """實體 EP 停下來之後會自己慢慢滑（2026-11-09 實測：沒人碰、沒指令，12 秒內 yaw 變 17°、位置晃幾公分，
        任務之間方向因此每次差十幾度）。速度控制歸零不會鎖住輪子；再送一個 0 位移的位置控制指令，讓底盤用
        位置迴路守住現在的姿態。只在實體模式做：模擬器不會滑，而且外掛對 0 位移的 move 回報不可靠。
        0 位移的 move 機器人有時不會 ack，會一直留在 SDK 動作表裡（見 _sdk_purge_actions）：等它最多 1.5 秒，沒完成就清掉。"""
        if self.mode != "real" or self.ep is None:
            return
        try:
            act = self._sdk_move(x=0, y=0, z=0, xy_speed=0.3, z_speed=30)
            if act is not None and not act.wait_for_completed(timeout=1.5):
                self._sdk_purge_actions("hold pose 沒有 ack")
        except Exception as e:
            print("[server] hold pose 失敗：%s" % e)

    # 鏡射到實體 server（--mirror-real http://127.0.0.1:8767）：模擬當眼睛和腦，實體照著做同樣的動作。
    # 掛在 do() 這一層是因為導航（navigate_to / go_to_landmark / drive_through）拆出來的每一小段 move_chassis 也走這裡，
    # 實體就會跟著走同一條路。底盤鏡射的是模擬「實際量到」的位移/轉角（被擋住只走一半，實體也只走一半），手臂/夾爪送同樣的目標。
    MIRROR_ACTIONS_DEFAULT = ("move_chassis", "move_arm", "arm_to", "gripper", "recenter_arm", "stop")
    mirror_real = None
    mirror_actions = MIRROR_ACTIONS_DEFAULT

    def _mirror_post(self, name, p):
        """把一個動作送到實體 server，傳輸層失敗（手臂 USB 瞬斷等）會等 2 秒重送，最多 3 次。回 dict。"""
        url = self.mirror_real
        real = {"ok": False, "message": "not sent"}
        for attempt in range(3):
            try:
                req = urllib.request.Request(url.rstrip("/") + "/action", data=json.dumps({"name": name, "params": p}).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=90) as resp:
                    real = json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                real = {"ok": False, "message": "real server unreachable: %s: %s" % (type(e).__name__, e)}
            msg = str(real.get("message", ""))
            transient = (not real.get("ok")) and any(k in msg for k in ("PortNotOpenError", "port that is not open", "no status from the arm",
                                                                        "unreachable", "串列", "device disconnected"))
            if not transient:
                break
            print("[mirror] %s 傳輸層失敗（%s），%d 秒後重送" % (name, msg[:80], 2))
            time.sleep(2.0)
        return real

    def do(self, name, params):
        url = getattr(self, "mirror_real", None)
        # params 帶 "mirror": false 可以只動模擬、不轉送實體（測模擬本身的速度對應時用；agent 不會送這個）
        if not url or name not in (self.mirror_actions or ()) or (params or {}).get("mirror") is False:
            return self._do_local(name, params)
        # 同時送：實體跟模擬一起開始做「要求的動作」（2026-11-10 使用者：等模擬做完再送，實體慢了十幾秒）。
        # 模擬做完後若它其實沒做到（被擋、被夾），再補一個修正把實體拉回跟模擬一樣的地方。
        p = dict(params or {})
        if name == "move_chassis":
            p = {"forward_m": float(p.get("forward_m") or 0), "right_m": float(p.get("right_m") or 0),
                 "turn_left_deg": float(p.get("turn_left_deg") or 0)}
        holder = {}
        th = threading.Thread(target=lambda: holder.__setitem__("real", self._mirror_post(name, p)), daemon=True)
        t0 = time.time()
        th.start()
        result = self._do_local(name, params)
        th.join(timeout=120)
        real = holder.get("real") or {"ok": False, "message": "real server did not answer within 120 s"}
        real_msg = str(real.get("message", ""))
        print("[mirror] %s %s -> %s %s" % (name, json.dumps(p, ensure_ascii=False), "ok" if real.get("ok") else "FAIL", real_msg[:160]))
        fix_note = ""
        if name == "move_chassis" and real.get("ok"):
            # 模擬實際走多少？差太多就補送差額（方向沿用指令、大小用量到的）
            m = result.get("measured") or {}
            fix = {"forward_m": 0.0, "right_m": 0.0, "turn_left_deg": 0.0}
            dist = m.get("distance_m")
            if dist is not None:
                for k in ("forward_m", "right_m"):
                    if abs(p[k]) > 0.01:
                        done = math.copysign(min(dist, abs(p[k])), p[k])
                        if abs(p[k] - done) > 0.05:
                            fix[k] = round(done - p[k], 3)
            if m.get("dyaw_deg") is not None and abs(p["turn_left_deg"]) > 0.5:
                done = math.copysign(min(abs(float(m["dyaw_deg"])), abs(p["turn_left_deg"])), p["turn_left_deg"])
                if abs(p["turn_left_deg"] - done) > 5.0:
                    fix["turn_left_deg"] = round(done - p["turn_left_deg"], 1)
            trans_fix = abs(fix["forward_m"]) > 0 or abs(fix["right_m"]) > 0
            if trans_fix and abs(p["turn_left_deg"]) > 0.5:
                # 平移＋旋轉合在一起的指令：平移是在「轉之前」的朝向做的，補平移要先轉回去、補完再轉到模擬實際的朝向
                done_turn = p["turn_left_deg"] + fix["turn_left_deg"]
                steps = [{"forward_m": 0, "right_m": 0, "turn_left_deg": round(-done_turn, 1)},
                         {"forward_m": fix["forward_m"], "right_m": fix["right_m"], "turn_left_deg": 0},
                         {"forward_m": 0, "right_m": 0, "turn_left_deg": round(done_turn, 1)}]
            elif trans_fix:
                steps = [{"forward_m": fix["forward_m"], "right_m": fix["right_m"], "turn_left_deg": 0}]
            elif abs(fix["turn_left_deg"]) > 0:
                steps = [{"forward_m": 0, "right_m": 0, "turn_left_deg": fix["turn_left_deg"]}]
            else:
                steps = []
            for st in steps:
                r2 = self._mirror_post("move_chassis", st)
                fix_note += " | real corrected by %s: %s" % (json.dumps(st), "ok" if r2.get("ok") else str(r2.get("message", ""))[:100])
                print("[mirror] 模擬只做了一部分，實體補送 %s -> %s" % (st, "ok" if r2.get("ok") else "FAIL"))
                if not r2.get("ok"):
                    break
        result["real"] = {"ok": bool(real.get("ok")), "message": real_msg + fix_note, "params": p,
                          "measured": real.get("measured"), "duration_s": round(time.time() - t0, 2)}
        if not real.get("ok"):
            result["message"] = str(result.get("message", "")) + " | real robot mirror FAILED: " + real_msg[:200]
        # 永久紀錄（append，跨 server 重啟）：事後才能回答「哪一題實體沒做完」。agent 的紀錄只存模擬回報，導航小段也不印。
        try:
            with open(os.environ.get("ROBOT_MIRROR_LOG", "/tmp/mirror.log"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "name": name, "params": p, "sim_ok": result.get("ok"),
                                    "sim_measured": result.get("measured"), "sim_msg": str(result.get("message", ""))[:160],
                                    "real_ok": bool(real.get("ok")), "real_measured": real.get("measured"),
                                    "real_msg": (real_msg + fix_note)[:220], "real_s": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return result

    def _do_local(self, name, params):
        fn = getattr(self, "_act_" + name, None)
        if fn is None:
            return {"ok": False, "message": "unknown action %r" % name}
        if self.ep is None and name in ("move_chassis", "stop"):
            return {"ok": False, "message": "no chassis in armonly mode: this server only drives the NexArm on the desk"}
        if name != "move_chassis":
            return fn(params or {})
        # 底盤動作前後各讀一次里程計，把「實際量到的變化」附在回報裡。
        # 這樣 log 裡就看得出指令和實際運動有沒有對上 (方向相反、打滑、撞到東西都會現形)。
        before = self.state()
        result = fn(params or {})
        time.sleep(0.15)  # 等車身停穩，下一張相機畫面才不會是晃動中的（原 0.6→0.3→0.15；每段路都要付一次，導航段數多時很可觀）
        after = self.state()
        self._hold_pose()
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
        # 夾過上限的值寫回去：do() 事後拿 params 跟里程計比對，不然要求 3.5 m 被夾成 1.0 m、走滿 0.99 m
        # 會被判成「幾乎沒動＝被擋住」（批次 m02 第 2 步，之後 LLM 整題都在「我卡住了」模式）
        p["forward_m"], p["right_m"], p["turn_left_deg"] = fwd, right, turn_left
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
        # 調速用：單次指定速度（測實體穩定上限時不用重啟），夾在 0.1–1.0 m/s、10–120 deg/s
        if p.get("xy_speed"):
            v = max(0.1, min(1.0, float(p["xy_speed"])))
        if p.get("z_speed"):
            w = max(10.0, min(120.0, float(p["z_speed"])))

        backend = self.chassis_backend
        move_translation = backend in ("move", "hybrid")
        move_rotation = backend == "move"

        if move_translation:
            # 位置控制：位移量交給機器人自己跑完，就算 Agent 端斷線也不會一直往前衝。
            # 這個模式下，如果路徑上真的有障礙物擋住，SDK 會等不到「到達」就回傳 timeout，
            # 等同於一種碰撞偵測：卡住了就不會繼續硬頂。
            # SDK 的 chassis.move 裡，z 為正是「左轉」。
            if (abs(fwd) > 1e-3 or abs(right) > 1e-3) and self.mode == "sim":
                # 模擬：外掛的位置控制固定只有 ~0.12 m/s 而且完成訊號不可靠（會把上一個逾時動作的完成算到下一個頭上，
                # 2026-11-11 實測第二段 0.3 秒就回「完成」但沒動），改用速度控制 + 里程計回授，到了就停
                r = self._sim_translate(fwd, right, v)
                if not r.get("ok"):
                    return r
            elif abs(fwd) > 1e-3 or abs(right) > 1e-3:
                # 實體：位置控制，速度先換算成 SDK 的刻度（見 _sdk_xy_speed）。時間預算用真實速度算，距離那一項給兩倍，
                # 加減速與完成訊號延遲各留餘裕（0.5 m @ 0.5 m/s 實測 1.8 s 回報）。
                t = math.hypot(fwd, right) / v * 2 + 3
                done = self._sdk_move(x=fwd, y=right, z=0, xy_speed=self._sdk_xy_speed(v)).wait_for_completed(timeout=t)
                if not done:
                    self._sdk_purge_actions("translation timed out")
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
                # turn_left 逆時針為正；送給 SDK 的符號依模式（見 _sdk_z_sign：模擬取負、實體不取負）
                done = self._sdk_move(x=0, y=0, z=self._sdk_z_sign * turn_left, z_speed=w).wait_for_completed(timeout=t)
                if not done:
                    self._sdk_purge_actions("rotation timed out")
                if not done:
                    self.stop()
                    return {"ok": False, "message": "rotation timed out after %.1fs" % t}
        else:
            # speed / hybrid：這個模擬器對 chassis.move 的「轉到定值」不會正確回報完成，改用 drive_speed。
            # 一開始是算好時間單純 sleep，但模擬器不一定跑在即時速度（機器負載重時會變慢），
            # 算出來的時間跟實際轉到的角度對不上：量到的 yaw 變化有時候是 0、有時候轉過頭。
            # 改成邊轉邊看真實的 yaw_deg 遙測，轉到差不多了就停，不是用猜的時間停。
            # SDK 的 drive_speed z 與 yaw_deg 遙測同一個座標系：順時針為正。turn_left 逆時針為正，
            # 所以送給 SDK 的是 z_cmd = -turn_left，而邊轉邊累積的 yaw 變化要跟 z_cmd 比，不是跟 turn_left 比。
            if abs(turn_left) <= 45.0:
                # 小角度用慢速：慣性 + 取樣延遲造成的過頭量跟角速度成正比（第八輪 rotate：15 度轉向多轉 7.5 度）。
                # 速度 = 角度 × 1 deg/s（最少 12），15 度就用 15 deg/s，停得住；通道裡對齊只差 2-4 度時尤其重要。
                w = max(12.0, min(w, abs(turn_left) * 1.0))
            if abs(turn_left) > 0.5:
                start_yaw = self.state().get("yaw_deg")
                timeout_t = abs(turn_left) / w + 4
                z_cmd = self._sdk_z_sign * turn_left   # 送給 SDK 的符號（依模式）
                yaw_goal = -turn_left                   # yaw 遙測順時針為正：左轉時 yaw 要減少。邊轉邊累積的變化跟這個比
                self.ep.chassis.drive_speed(x=0, y=0, z=math.copysign(w, z_cmd), timeout=timeout_t + 1)
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
                        if abs(turned) > abs(yaw_goal) + 20:
                            break
                        # 提早停：遙測 5 Hz、旋轉 45 deg/s，從「看到到了」到真的停下來會再多轉約 8-9 度
                        # （tools/nav_selftest.py 量到 45~135 度一致多轉 8.8 度）。小角度慣性小，留 2 度就好。
                        # 20 Hz 遙測後大角度的延遲量剩 ~2 度 + 慣性 ~3 度；小角度已經降速，留 1 度
                        margin = 5.0 if abs(turn_left) > 25 else 1.0
                        if yaw_goal > 0 and turned >= yaw_goal - margin:
                            break
                        if yaw_goal < 0 and turned <= yaw_goal + margin:
                            break
                self.stop()
                time.sleep(0.15)
        return {"ok": True, "message": "moved forward=%.2fm right=%.2fm turn_left=%.0fdeg%s" % (fwd, right, turn_left, guard_note)}

    def _act_move_arm(self, p):
        if self.nexarm is not None:
            return self._nex_move_arm(p)
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
        if self.nexarm is not None:
            return self._nex_arm_to(p)
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
        if self.nexarm is not None:
            return self._nex_recenter_arm(p)
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

    def _act_lift_arm(self, p):
        """手臂升降柱：Env/add_arm_lift.py 在底盤跟手臂底座之間加的直線關節，不經過 RoboMaster SDK（SDK 沒有這個東西），
        伺服器直接用 zmq 設關節目標、輪詢關節位置確認有沒有到。整組手臂、手臂相機、前方距離感測器都跟著升，
        手臂自己的 arm_mm 座標不變。height_mm 是絕對高度（0 = 最低、平常開車的姿態）。"""
        if self.nexarm is not None and self.mode != "sim":
            # 實體 NexArm 沒有升降柱（模擬裡的升降柱是 Env/add_arm_lift.py 加的虛擬裝置，模擬的 NexArm 底座就坐在它上面）
            return {"ok": False, "message": ("this arm (NexArm) has no lift column; nothing else can raise it. How much higher the "
                                             "gripper can still go is arm_limits.can_raise_mm in the telemetry (a horizontal gripper, "
                                             "pitch 0, reaches the highest). Report what was achieved instead of retrying.")}
        sim = getattr(self, "_nav_sim", None)     # 導航專用連線：只在動作執行緒用，不跟輪詢搶
        root = getattr(self, "_overhead_root", None)
        if sim is None or root is None:
            return {"ok": False, "message": "arm lift unavailable: no simulator connection"}
        j = getattr(self, "_lift_joint", -1)
        if j == -1:
            j = self._find_lift_joint(sim, root)
            self._lift_joint = j
        if j == -1:
            return {"ok": False, "message": ("this robot has no arm lift column (arm_lift_joint not in the scene); "
                                             "the arm cannot be raised beyond its own workspace")}
        try:
            target_mm = max(0.0, min(LIMITS["lift_mm"], float(p.get("height_mm", 0) or 0)))
            before = sim.getJointPosition(j)
            sim.setJointTargetPosition(j, target_mm / 1000.0)
            t0 = time.time()
            last, stuck = before, 0
            q = before
            while time.time() - t0 < 10.0:
                time.sleep(0.1)
                q = sim.getJointPosition(j)
                if abs(q * 1000.0 - target_mm) <= 5.0:
                    break
                stuck = stuck + 1 if abs(q - last) < 0.0005 else 0   # 2 秒完全沒動：卡住或到極限了
                last = q
                if stuck >= 20:
                    break
            time.sleep(0.3)   # 等位置控制收斂、手臂停止晃
            q = sim.getJointPosition(j)
        except Exception as e:
            return {"ok": False, "message": "arm lift command failed: %s" % e}
        self._set("lift_mm", int(round(q * 1000)))
        ok = abs(q * 1000.0 - target_mm) <= 8.0
        msg = "arm lift at %.0f mm (target %.0f mm, was %.0f mm)" % (q * 1000, target_mm, before * 1000)
        if not ok:
            msg += " — did NOT reach the target (blocked, or the lift is at its limit)"
        return {"ok": ok, "message": msg}

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
        if self.nexarm is not None:
            return self._nex_gripper(state)
        power = int(max(1, min(100, float(p.get("power") or 50))))
        if state == "open":
            self._held_tof_mm = None
            self._set("held_tof_mm", None)
            self.ep.gripper.open(power=power)
        else:
            self.ep.gripper.close(power=power)
        # 等的是模擬時間：視窗版模擬器常只跑 0.5 倍即時，牆鐘 1.5 s 只等於 0.8 s 模擬時間，手指還沒合到球就被
        # pause() 停住，夾爪腳本「兩指都碰到才掛接」的條件永遠不成立 → 抬手臂時把球撥掉（2026-10-22 批次 11 題球掉落）
        self._sleep_sim(1.5, cap_s=6.0)
        try:
            self.ep.gripper.pause()
        except Exception:
            pass
        return {"ok": True, "message": "gripper %s" % state}

    # ---- NexArm（Hiwonder 六軸手臂，USB 串列，見 nexarm.py）：取代 SDK 的 robotic_arm / gripper ----
    # 座標對應：RoboMaster 手臂是 2D (x 前伸, y 高度)，agent 的 arm_mm / arm_to / move_arm 都用這兩個字；
    # NexArm 的 x 一樣是前伸，高度是它的 z，側向 y 固定 0（左右靠底盤轉），所以 arm_mm = {x: x, y: z}。
    def _start_nexarm_sim(self):
        """模擬器裡的 NexArm（Env/add_nexarm.py 的模型）：介面跟實體驅動一樣，見 nexarm_sim.py。"""
        from nexarm_sim import NexArmSim
        self.nexarm = None
        self.nexarm = NexArmSim(on_status=self._on_nexarm_status)
        st, _ = self.nexarm.status()
        print("[server] 模擬器裡的 NexArm 已接上：x=%s y=%s z=%s pitch=%.1f claw=%s" % (st["x"], st["y"], st["z"], st["pitch"], st["claw"]))
        self._set("gripper_holding", False)

    def _start_nexarm(self, port):
        from nexarm import NexArm
        self.nexarm = None   # NexArm() 建構時背景執行緒就開始回呼 _on_nexarm_status，先讓 self.nexarm 存在
        self.nexarm = NexArm(port, on_status=self._on_nexarm_status)
        if getattr(self, "mirror_sim_port", None):
            # 數位分身：實體手臂的伺服位置即時套到視窗版 CoppeliaSim 裡的 NexArm 模型，看得到實體在做什麼
            from nexarm_sim import SimMirror
            self._mirror = SimMirror(lambda: self.nexarm.status()[0], port=self.mirror_sim_port)
        st, _ = self.nexarm.status()
        print("[server] NexArm 已連線 (%s)：x=%s y=%s z=%s pitch=%.1f claw=%s servos=%s" % (
            port, st["x"], st["y"], st["z"], st["pitch"], st["claw"], st["servos"]))
        self._set("gripper_holding", False)

    def _on_nexarm_status(self, st):
        with self._tlock:
            import math
            from nexarm import yaw_of
            reach = math.hypot(st["x"], st["y"])
            self.telemetry["arm_mm"] = {"x": int(round(reach)), "y": st["z"]}
            self.telemetry["arm_pose"] = {"x_mm": st["x"], "y_mm": st["y"], "z_mm": st["z"], "reach_mm": int(round(reach)),
                                          "yaw_deg": round(yaw_of(st["x"], st["y"]), 1) if reach > 1 else 0.0,
                                          "pitch_deg": st["pitch"], "roll_deg": st["roll"]}
            # 可達範圍跟「每個方向還剩多少」：agent 每步都看得到，不用等撞到上限才知道（2026-11-09 實測：
            # 手臂停在頂端 z=327 時 agent 照樣下「抬高 5 公分」，只動了 +1 mm）。arm_mm 的 y 就是高度（NexArm 的 z）。
            # 可達範圍不是方的，跟夾爪俯仰角有關（nexarm_workspace.py 掃出來、實機驗證過）：給「目前這個前伸距離 / 高度 /
            # 俯仰角」下還能動多少，加一句怎麼到更低的地方（夾爪水平時前伸 150 最低只到 200 mm，朝下才碰得到桌面）
            lim = self.nexarm.limits_here(st) if self.nexarm is not None else None
            if lim:
                lim["note"] = (("NexArm on the lift column (lift_arm raises the whole arm; these ranges are relative to the lift). "
                                if getattr(self, "_lift_joint", -1) != -1 else "NexArm, no lift column. ")
                               + "Ranges are for the CURRENT gripper pitch: y_mm_here is the height range at "
                               "this forward reach, x_mm_here the forward range at this height. Pointing the gripper down "
                               "(arm_to pitch_deg -90) reaches much lower; horizontal (0) reaches higher.")
                self.telemetry["arm_limits"] = lim
            if getattr(self, "_origin_yaw", None) is not None:
                self.telemetry["origin_yaw_deg"] = self._origin_yaw   # 連線時的朝向：return_to_origin 的 0 度、里程計 x 軸
            self.telemetry["gripper_claw_deg"] = st["claw"]
            self.telemetry["gripper"] = "opened" if st["claw"] <= -40 else ("closed" if st["claw"] >= 40 else "normal")
            if st.get("voltage_mv"):
                # 手臂主板量到的輸入電壓：變壓器約 12.0 V；接 EP 電池 / 3S 鋰電時 10.8–12.6 V，伺服規格下限 9 V
                self.telemetry["arm_voltage_v"] = round(st["voltage_mv"] / 1000.0, 2)
            # 手臂的心跳另外記：底盤也在的時候（real --arm nexarm）不能讓手臂的 5 Hz 狀態把 EP 斷線蓋掉
            # （2026-11-10 實測：Wi-Fi 切走、EP ping 不到，/health 還是 "telemetry 0.1s ago"）
            self._last_arm_rx = time.time()
            if self.ep is None:
                self._last_rx = time.time()

    @staticmethod
    def _nex_result(r, what):
        pose = r.get("pose") or {}
        if not pose:
            return {"ok": False, "message": "%s: %s" % (what, r.get("message", "no status from the arm"))}
        where = "arm now at x=%s y=%s z=%s mm, pitch %.1f deg" % (pose["x"], pose["y"], pose["z"], pose["pitch"])
        if r.get("unreachable"):   # 可達區域表說到不了，指令根本沒送，手臂沒動
            return {"ok": False, "message": "%s: NOT sent, %s; arm did not move (%s)" % (what, r["message"], where),
                    "arm_pose": pose}
        if r["ok"]:
            msg = "%s: reached; %s" % (what, where)
        else:
            msg = ("%s: did NOT reach the target (off by %.0f mm / %.1f deg: out of reach or blocked); %s"
                   % (what, r["error_mm"], r["pitch_error_deg"], where))
        if r.get("clamped"):
            msg += " | target clamped to the arm's workspace: " + r["clamped"]
        return {"ok": bool(r["ok"]), "message": msg, "arm_pose": pose}

    def _nex_move_arm(self, p):
        import math
        from nexarm import yaw_of
        dx = _clamp(p.get("forward_mm", 0), LIMITS["arm_mm"])
        dz = _clamp(p.get("up_mm", 0), LIMITS["arm_mm"])
        dyaw = _clamp(p.get("turn_left_deg", 0) or 0, 180.0)
        droll = _clamp(p.get("roll_deg", 0) or 0, 180.0)
        r = self.nexarm.move_by(dx=dx, dz=dz, dyaw=dyaw, droll=droll)
        what = "arm move forward=%+.0f up=%+.0f mm" % (dx, dz)
        if dyaw:
            what += " base turn_left=%+.0f deg" % dyaw
        if droll:
            what += " wrist roll=%+.0f deg" % droll
        out = self._nex_result(r, what)
        before, after = r.get("before"), r.get("pose")
        if before and after:
            # 前伸量看水平距離（底座轉了之後 x 本身會變），轉角看 atan2
            mx = math.hypot(after["x"], after["y"]) - math.hypot(before["x"], before["y"])
            mz = after["z"] - before["z"]
            myaw = yaw_of(after["x"], after["y"]) - yaw_of(before["x"], before["y"])
            myaw = (myaw + 180) % 360 - 180
            mroll = after["roll"] - before["roll"]
            out["measured"] = {"forward_mm": round(mx), "up_mm": mz, "turn_left_deg": round(myaw, 1), "roll_deg": mroll}
            ok, note = _nex_move_verdict(dx, dz, mx, mz, r.get("target") or {}, clamped=r.get("clamped", ""),
                                         has_lift=(self.mode == "sim"))
            if dyaw and abs(myaw - dyaw) > 5:
                ok = False
                note += " | base turned %+.0f deg of the requested %+.0f" % (myaw, dyaw)
            if droll and abs(mroll - droll) > 5:
                ok = False
                note += " | wrist rolled %+.0f deg of the requested %+.0f" % (mroll, droll)
            out["ok"] = out["ok"] and ok
            out["message"] += note
        return out

    def _nex_arm_to(self, p):
        from nexarm import HOME, rotate_xy
        reach = float(p.get("x_mm", HOME[0]))
        z = float(p.get("y_mm", HOME[2]))
        pitch = p.get("pitch_deg")
        yaw = float(p.get("yaw_deg") or 0.0)
        roll = p.get("roll_deg")
        x, y = rotate_xy(reach, 0.0, yaw)
        r = self.nexarm.move_to(x=x, y=y, z=z, pitch=float(pitch) if pitch is not None else 0.0,
                                roll=float(roll) if roll is not None else None)
        what = "arm target reach=%.0f height=%.0f mm" % (reach, z)
        if yaw:
            what += " base yaw=%.0f deg" % yaw
        if roll is not None:
            what += " wrist roll=%.0f deg" % float(roll)
        return self._nex_result(r, what)

    def _nex_recenter_arm(self, p):
        if (p or {}).get("pose") == "stow":
            return self._nex_result(self.nexarm.stow(), "arm stow (gripper pointing down, folded over the base)")
        return self._nex_result(self.nexarm.home(), "arm recenter")

    def _nex_gripper(self, state):
        from nexarm import CLAW_CLOSE, CLAW_OPEN
        if state == "open":
            self._held_tof_mm = None
            self._set("held_tof_mm", None)
        r = self.nexarm.set_claw(CLAW_OPEN if state == "open" else CLAW_CLOSE)
        if not r.get("pose"):
            return {"ok": False, "message": "gripper %s: no status from the arm" % state}
        holding = bool(r.get("holding"))
        self._set("gripper_holding", holding)
        if state == "open":
            msg = "gripper opened"
        elif holding:
            # 空夾爪實測每次都轉到位（誤差 ≤ 0.3°）；停在離目標好幾度之外只有一種解釋：有東西擋在兩指之間
            msg = ("gripper closed on something: the finger servo stopped %.1f deg short of fully closed, "
                   "so an object is between the fingers" % r["shortfall_deg"])
        else:
            msg = "gripper closed fully: nothing is between the fingers"
        return {"ok": True, "message": msg, "holding": holding}

    def _sleep_sim(self, sim_seconds, cap_s=6.0):
        """等模擬時間過 sim_seconds 秒（牆鐘上限 cap_s）；沒有模擬器連線時退回牆鐘 sleep。"""
        sim = getattr(self, "_sim_for_overhead", None)
        lock = getattr(self, "_sim_for_overhead_lock", None)
        t_wall = time.time()
        if sim is None or lock is None:
            time.sleep(sim_seconds)
            return
        try:
            with lock:
                t0 = sim.getSimulationTime()
            while time.time() - t_wall < cap_s:
                time.sleep(0.1)
                with lock:
                    t = sim.getSimulationTime()
                if t - t0 >= sim_seconds:
                    return
        except Exception:
            remaining = sim_seconds - (time.time() - t_wall)
            if remaining > 0:
                time.sleep(remaining)

    def _act_wait(self, p):
        s = abs(_clamp(p.get("seconds", 1), LIMITS["wait_s"]))
        time.sleep(s)
        return {"ok": True, "message": "waited %.1fs" % s}

    _origin_yaw = None   # 這條 SDK 連線第一筆 yaw：里程計 position_m 的 x 軸就是這個朝向（真機 yaw 不會在連線時歸零）

    def _on_yaw(self, yaw):
        if self._origin_yaw is None:
            self._origin_yaw = yaw
            print("[server] 連線時的朝向 yaw=%+.1f deg（return_to_origin 以它為 0）" % yaw)
        self._set("yaw_deg", yaw)

    def _act_return_to_origin(self, p):
        """實體底盤用里程計走回「這條 SDK 連線開始時」的位置與朝向（鏡射批次每題之間用；模擬那邊是重載場景瞬移回起點）。
        里程計座標：position_m 是連線起點座標系（x 前、y 右），yaw_deg 順時針為正（真機實測）。
        做法：轉向原點 → 直走距離（切成 ≤1 m 段）→ 轉回朝向 0；每段都走 do("move_chassis") 所以會附實測里程。
        最後回報殘差；里程計會累積誤差，殘差超過 0.3 m 會 ok=False 讓批次 log 看得到。"""
        tol_m = float(p.get("tol_m", 0.08))
        tol_deg = float(p.get("tol_deg", 6.0))
        if self.mode == "sim" and getattr(self, "_nav", None) and getattr(self, "_home_world", None):
            return self._sim_return_home(tol_m, tol_deg)
        notes = []
        origin = self._origin_yaw or 0.0

        def rel_yaw(absolute):
            return (absolute - origin + 180) % 360 - 180
        for attempt in range(1, 3):
            st = self.state()
            x, y, yaw = st["position_m"]["x"], st["position_m"]["y"], rel_yaw(st["yaw_deg"])
            dist = math.hypot(x, y)
            if dist > tol_m:
                phi = math.degrees(math.atan2(-y, -x))          # 原點相對目前位置的方向（起點座標系，順時針為正）
                turn_cw = (phi - yaw + 180) % 360 - 180
                r = self.do("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(-turn_cw, 1)})
                notes.append("turn %+.0f: %s" % (-turn_cw, "ok" if r.get("ok") else r.get("message", "")[:80]))
                left = dist
                while left > 0.02:
                    seg = min(LIMITS["forward_m"], left)
                    r = self.do("move_chassis", {"forward_m": round(seg, 3), "right_m": 0, "turn_left_deg": 0})
                    notes.append("drive %.2f: %s" % (seg, "ok" if r.get("ok") else r.get("message", "")[:80]))
                    if not r.get("ok"):
                        break
                    left -= seg
            st = self.state()
            yaw = rel_yaw(st["yaw_deg"])
            if abs((yaw + 180) % 360 - 180) > tol_deg:
                r = self.do("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(((yaw + 180) % 360 - 180), 1)})
                notes.append("face 0: %s" % ("ok" if r.get("ok") else r.get("message", "")[:80]))
            st = self.state()
            x, y, yaw = st["position_m"]["x"], st["position_m"]["y"], rel_yaw(st["yaw_deg"])
            resid = math.hypot(x, y)
            if resid <= tol_m * 2 and abs((yaw + 180) % 360 - 180) <= tol_deg * 1.5:
                break
        ok = resid <= 0.3
        return {"ok": ok, "message": "return_to_origin: residual %.2f m, yaw %+.0f deg from start heading (odometry) | %s" % (resid, yaw, "; ".join(notes) or "already at origin"),
                "residual_m": round(resid, 3), "yaw_deg": round(yaw, 1)}

    def _sim_return_home(self, tol_m, tol_deg):
        """模擬：用占據格地圖導航回出發點（繞障礙，不是直線），再轉回出發時的車頭朝向。
        每一小段都走 do("move_chassis")，鏡射模式下實體會跟著走同一條路；實體自己的里程計歸位（批次 reset_real）之後再補殘差。"""
        hx, hy, hh = self._home_world
        notes = []
        r = self._act_navigate_to({"x": hx, "y": hy, "tolerance_m": max(0.1, tol_m), "face": False, "label": "start"})
        notes.append("navigate: %s" % ("ok" if r.get("ok") else str(r.get("message", ""))[:120]))
        pose = self._nav_pose()
        if pose:
            dh = (hh - pose[2] + 180) % 360 - 180
            if abs(dh) > tol_deg:
                r2 = self.do("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(dh, 1)})
                notes.append("face start heading %+.0f: %s" % (dh, "ok" if r2.get("ok") else str(r2.get("message", ""))[:80]))
            pose = self._nav_pose() or pose
        self._nav_target = None
        if not pose:
            return {"ok": False, "message": "return_to_origin: cannot read world pose | " + "; ".join(notes)}
        resid = math.hypot(hx - pose[0], hy - pose[1])
        dh = (hh - pose[2] + 180) % 360 - 180
        ok = resid <= 0.3
        return {"ok": ok, "message": "return_to_origin: %s, residual %.2f m, heading %+.0f deg from start (world pose) | %s"
                                     % ("back at the start" if ok else "did NOT reach the start", resid, -dh, "; ".join(notes)),
                "residual_m": round(resid, 3), "yaw_deg": round(-dh, 1)}

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
        lift_note = ""
        if getattr(self, "_lift_joint", -1) != -1:
            lift_note = "; " + self._act_lift_arm({"height_mm": 0}).get("message", "")   # 升降柱降到底，每個任務從同一個姿態開始
        self._act_gripper({"state": "open"})
        r = self._act_recenter_arm({})
        if not r.get("ok"):
            r = self._act_recenter_arm({})  # 失敗重試一次，可能是暫時性的、或中間已經自動恢復了
        ok = bool(r.get("ok"))
        return {"ok": ok, "message": ("gripper opened; " + r.get("message", "") + lift_note +
                                       "; chassis pose NOT reset") if ok else
                          ("gripper opened, but arm did NOT recenter (" + r.get("message", "") +
                           ") — check the arm before starting the task" + lift_note + "; chassis pose NOT reset")}

    def close(self):
        self.stop()
        if getattr(self, "_mirror", None) is not None:
            self._mirror.close()
        if self.nexarm is not None:
            self.nexarm.close()
        if self.ep is None:
            print("[server] 已關閉 NexArm 連線")
            return
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
# agent 端每一步 POST /agent_event 進來的紀錄（任務、思考、評估、記憶、目標、動作與結果、規劃器建議），給 /live 顯示
AGENT_LOG = {"task": None, "run_dir": None, "started": None, "steps": collections.deque(maxlen=80), "plans": collections.deque(maxlen=20), "done": None}
AGENT_LOG_LOCK = threading.Lock()


def agent_event(ev):
    with AGENT_LOG_LOCK:
        kind = ev.get("event")
        if kind == "start":
            AGENT_LOG["task"], AGENT_LOG["run_dir"], AGENT_LOG["started"], AGENT_LOG["done"] = ev.get("task"), ev.get("run_dir"), time.time(), None
            AGENT_LOG["steps"].clear()
            AGENT_LOG["plans"].clear()
        elif kind == "step":
            AGENT_LOG["steps"].append(ev)
        elif kind == "plan":
            AGENT_LOG["plans"].append(ev)
        elif kind == "done":
            AGENT_LOG["done"] = ev


def agent_log_snapshot():
    with AGENT_LOG_LOCK:
        return {"task": AGENT_LOG["task"], "run_dir": AGENT_LOG["run_dir"], "started": AGENT_LOG["started"],
                "steps": list(AGENT_LOG["steps"]), "plans": list(AGENT_LOG["plans"]), "done": AGENT_LOG["done"]}


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot_eval", "results")


def batch_status():
    """/live 的批次成績表：最近一個 run_batch.py 批次（有 driver.log 的資料夾）每題的結果。只讀檔案、不碰 zmq。"""
    try:
        dirs = [os.path.join(RESULTS_DIR, d) for d in os.listdir(RESULTS_DIR)]
        dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "driver.log"))]
    except OSError:
        dirs = []
    if not dirs:
        return {"batch": None}
    bd = max(dirs, key=lambda d: os.path.getmtime(os.path.join(d, "driver.log")))
    results = {}
    for session in sorted(os.listdir(bd)):       # session 是時間戳，後面的蓋過前面的（重跑的那次）
        sp = os.path.join(bd, session)
        if not os.path.isdir(sp):
            continue
        for tid in os.listdir(sp):
            ev = os.path.join(sp, tid, "evaluation.json")
            if os.path.isfile(ev):
                try:
                    with open(ev, encoding="utf-8") as f:
                        m = json.load(f)["metrics"]
                except (OSError, ValueError, KeyError):
                    continue
                results[m["task_id"]] = {"ok": m["is_successful"], "milestones": "%d/%d" % (m["completed_milestones"], m["total_milestones"]),
                                         "steps": m["total_steps"], "seconds": round(m["total_duration_seconds"]),
                                         "reason": (m.get("judge_failure_reason") or ("" if m["is_successful"] else m.get("judge_reasoning")) or "")[:160]}
    try:
        with open(os.path.join(bd, "batch_tasks.json"), encoding="utf-8") as f:
            tasks = json.load(f)
    except (OSError, ValueError):
        tasks = [{"id": t, "difficulty": None} for t in results]
    try:
        with open(os.path.join(bd, "driver.log"), encoding="utf-8", errors="replace") as f:
            log_tail = f.read().splitlines()[-12:]
    except OSError:
        log_tail = []
    current = next((ln.split("] ", 1)[-1].strip("= ") for ln in reversed(log_tail) if "=====" in ln), None)
    finished = any("批次結束" in ln for ln in log_tail)
    return {"batch": os.path.basename(bd), "tasks": tasks, "results": results, "current": None if finished else current,
            "finished": finished, "log_tail": log_tail}


LIVE_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>RoboMaster live</title>
<style>
body{margin:0;background:#1e1e1e;color:#ddd;font:13px/1.4 monospace}
.grid{display:grid;grid-template-columns:repeat(3,minmax(320px,1fr));gap:10px;padding:10px;align-items:start}
.card{background:#2a2a2a;padding:8px;border-radius:6px}
.card img{display:block;width:100%;height:auto;background:#111}
h3{margin:0 0 6px;font-size:13px;color:#9cf;font-weight:normal}
table{border-collapse:collapse;width:100%}td{padding:1px 6px;vertical-align:top}td:first-child{color:#9cf;white-space:nowrap}
.agent{grid-column:1 / -1}
#task{color:#fff;font-weight:bold;margin-bottom:6px}
#plans{color:#fc9;margin-bottom:6px;max-height:130px;overflow:auto}#plans div{margin-bottom:4px}
#steps{max-height:460px;overflow:auto;border-top:1px solid #444;padding-top:4px}
.step{border-bottom:1px solid #3a3a3a;padding:6px 0}.step b{color:#9cf}.act{margin-left:14px}.err{color:#f88}
.ov{transform:rotate(180deg)}
#bsum{color:#fff;margin-bottom:6px}#btab{max-height:320px;overflow:auto}#btab td{border-bottom:1px solid #3a3a3a}.ok{color:#8f8}.bad{color:#f88}.run{color:#fc9}.pend{color:#777}#blog{color:#999;white-space:pre-wrap;margin-top:6px;max-height:140px;overflow:auto}
</style></head>
<body><div class="grid">
<div class="card"><h3>occupancy map &mdash; red arrow = robot, green = planned path, red circle = goal, magenta box = locked target, orange = loose objects</h3><img id="map"></div>
<div class="card"><h3>overhead camera (rotated 180&deg; so it matches the map)</h3><img id="ov" class="ov"></div>
<div>
  <div class="card"><h3>front camera (on the arm) &mdash; magenta box = locked target</h3><img id="fr"></div>
  <div class="card" style="margin-top:10px"><h3>telemetry</h3><table id="st"><tr><td>loading...</td></tr></table></div>
</div>
<div class="card agent"><h3>batch results (robot_eval/results, newest run_batch.py batch)</h3><div id="bsum">no batch yet</div><div id="btab"></div><div id="blog"></div></div>
<div class="card agent"><div id="task">task: (no agent running)</div><div id="plans"></div><div id="steps"></div></div>
</div>
<script>
const ms = parseInt(new URLSearchParams(location.search).get("ms") || "1000");
const esc = s => String(s == null ? "" : s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const num = v => (typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(2)) : v);
let ovLoaded = false;
function tick(){
  const t = Date.now();
  document.getElementById("map").src = "/map.png?t=" + t;
  document.getElementById("fr").src = "/frame_live.jpg?t=" + t;
  if (!ovLoaded || t % 3000 < ms) { document.getElementById("ov").src = "/overhead.jpg?t=" + t; ovLoaded = true; }
  fetch("/state").then(r => r.json()).then(d => {
    const rows = [
      ["position (x, y)", d.world_xy ? d.world_xy.map(num).join(", ") : "-"],
      ["heading", d.world_heading_deg != null ? num(d.world_heading_deg) + "°" : "-"],
      ["tof front / left / right", [d.tof_front_mm, d.tof_left_mm, d.tof_right_mm].map(v => v == null ? "-" : v).join(" / ") + " mm"],
      ["arm (x, y)", d.arm_mm ? d.arm_mm.x + ", " + d.arm_mm.y + " mm" : "-"],
      ["lift", d.lift_mm != null ? d.lift_mm + " mm" : "-"],
      ["gripper", (d.gripper || "-") + (d.gripper_force_n != null ? "  (load " + num(d.gripper_force_n - 1.5) + " N)" : "")
                  + (d.gripper_claw_deg != null ? "  (claw " + d.gripper_claw_deg + "°" + (d.gripper_holding ? ", holding something" : "") + ")" : "")],
      ["held object", d.held_tof_mm != null ? "yes, " + d.held_tof_mm + " mm on the sensor" : "no"],
      ["camera", d.camera ? "ok" : "no frames"]];
    if (d.mode) rows.push(["mode", d.mode + (d.mode !== "sim" ? "  (real robot: no map / no camera here; the simulation's view is at " + simLink() + ")" : "")]);
    if (d.arm_voltage_v != null) rows.push(["arm supply", d.arm_voltage_v + " V"]);
    document.getElementById("st").innerHTML = rows.map(r => "<tr><td>" + r[0] + "</td><td>" + (r[0] === "mode" ? r[1] : esc(r[1])) + "</td></tr>").join("");
    if (d.mode && d.mode !== "sim" && !document.getElementById("nocam")) {
      // 實體模式沒有地圖、俯視相機、手臂相機（EP 相機壞、NexArm 沒相機）：三格黑的不是壞掉，畫面在模擬那邊
      const note = document.createElement("div"); note.id = "nocam";
      note.style.cssText = "margin:8px 0 0 8px;padding:10px;background:#2a2a1a;color:#ffd866;border:1px solid #665;font-size:14px";
      note.innerHTML = "This is the <b>real robot</b> page: it has no occupancy map and no camera (the EP camera is dead, the NexArm has none), so the three panels stay black. The simulation is the one that sees: open " + simLink() + " for the map and cameras. Mirrored moves from the simulation show up in the telemetry below.";
      document.body.insertBefore(note, document.body.firstChild);
    }
  }).catch(() => {});
}
function simLink(){ const u = "http://" + location.hostname + ":8765/live"; return '<a href="' + u + '" style="color:#8cf">' + u + '</a>'; }
let lastKey = "";
function agent(){
  fetch("/agent_log").then(r => r.json()).then(d => {
    const key = (d.task || "") + "|" + d.steps.length + "|" + (d.done ? 1 : 0) + "|" + d.plans.length;
    if (key === lastKey) return; lastKey = key;
    document.getElementById("task").textContent = "task: " + (d.task || "(no agent running)") + (d.done ? ("   → done(success=" + d.done.success + "): " + d.done.text) : "");
    const pl = document.getElementById("plans");
    pl.innerHTML = d.plans.map(p => "<div>🗺️ plan @step " + p.step + ": " + esc(p.text) + "</div>").join("");
    pl.scrollTop = pl.scrollHeight;
    const box = document.getElementById("steps");
    box.innerHTML = d.steps.map(s => {
      const acts = (s.actions || []).map(a => "<div class='act'>" + (a.ok ? "✅" : "❌") + " <b>" + esc(a.name) + "</b>(" + esc(JSON.stringify(a.params)) + ") → " + esc(a.message) + "</div>").join("");
      return "<div class='step'><b>step " + s.step + "</b> (" + s.llm_seconds + "s)" +
        (s.error ? "<div class='err'>❌ " + esc(s.error) + "</div>" : "") +
        "<div>👍 <i>eval:</i> " + esc(s.evaluation) + "</div><div>💭 <i>thinking:</i> " + esc(s.thinking) + "</div>" +
        "<div>🧠 <i>memory:</i> " + esc(s.memory) + "</div><div>🎯 <i>goal:</i> " + esc(s.next_goal) + "</div>" + acts + "</div>";
    }).join("");
    box.scrollTop = box.scrollHeight;
  }).catch(() => {});
}
function batch(){
  fetch("/batch_status").then(r => r.json()).then(d => {
    if (!d.batch) return;
    const res = d.results, mine = d.tasks.filter(t => res[t.id]);   // 只算這個批次的題目（重跑時資料夾裡還有舊結果）
    const done = mine.length, ok = mine.filter(t => res[t.id].ok).length;
    const by = {};
    d.tasks.forEach(t => { const k = t.difficulty || "?"; by[k] = by[k] || [0, 0]; if (res[t.id]) { by[k][1]++; if (res[t.id].ok) by[k][0]++; } });
    document.getElementById("bsum").textContent = d.batch + (d.finished ? "  (finished)" : "  (running: " + (d.current || "-") + ")") +
      "  |  evaluated " + done + "/" + d.tasks.length + "  |  success " + ok + "/" + done + (done ? " (" + Math.round(100 * ok / done) + "%)" : "") +
      "  |  " + Object.entries(by).map(([k, v]) => k + " " + v[0] + "/" + v[1]).join("  ");
    document.getElementById("btab").innerHTML = "<table>" + d.tasks.map(t => {
      const r = res[t.id], running = !d.finished && d.current && d.current.endsWith(t.id);
      const st = r ? (r.ok ? "<span class='ok'>✓</span>" : "<span class='bad'>✗</span>") : (running ? "<span class='run'>▶</span>" : "<span class='pend'>·</span>");
      return "<tr><td>" + st + " " + esc(t.id) + "</td><td>" + esc(t.difficulty || "") + "</td><td>" +
        (r ? esc(r.milestones) + "</td><td>" + r.steps + " steps</td><td>" + r.seconds + "s</td><td>" + esc(r.reason) : "</td><td></td><td></td><td>") + "</td></tr>";
    }).join("") + "</table>";
    const bl = document.getElementById("blog"); bl.textContent = d.log_tail.join("\n"); bl.scrollTop = bl.scrollHeight;
  }).catch(() => {});
}
tick(); setInterval(tick, ms); agent(); setInterval(agent, 1500); batch(); setInterval(batch, 5000);
</script></body></html>
"""


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
            if path == "/":   # 瀏覽器直接打 host:port 時帶去儀表板，不要回 not found
                self.send_response(302)
                self.send_header("Location", "/live")
                self.end_headers()
                return
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
            if path == "/frame_live.jpg":
                jpg = backend.frame_live_jpeg() if hasattr(backend, "frame_live_jpeg") else backend.frame_jpeg()
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
            if path == "/agent_log":
                return self._send(200, agent_log_snapshot())
            if path == "/batch_status":
                return self._send(200, batch_status())
            if path == "/live":
                # 即時儀表板：占據格地圖、俯視圖、前相機每秒重抓一次，遙測一起顯示（純 HTML+JS，不用額外套件）
                return self._send(200, LIVE_HTML.encode("utf-8"), "text/html; charset=utf-8")
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
            if path == "/agent_event":   # agent 的旁路紀錄，不走動作鎖
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    agent_event(json.loads(self.rfile.read(n).decode("utf-8") or "{}"))
                except Exception as e:
                    return self._send(400, {"ok": False, "message": "%s: %s" % (type(e).__name__, e)})
                return self._send(200, {"ok": True})
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
    try:
        import faulthandler
        import signal
        faulthandler.register(signal.SIGUSR1, all_threads=True)   # 卡住時：kill -USR1 <pid> 印出所有執行緒在哪
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="RoboMaster robot server for the LLM agent")
    ap.add_argument("--mode", default="sim", choices=["sim", "real", "armonly", "mock"],
                    help="armonly：沒有 RoboMaster，只控制桌上用 USB 接的 NexArm 手臂（等於 --arm nexarm，底盤動作不可用）")
    ap.add_argument("--arm", default="auto", choices=["auto", "sdk", "nexarm", "nexarm-sim"],
                    help="手臂/夾爪用誰：auto（預設）＝ sim 模式下場景裡有 NexArm 模型就用模擬的 NexArm，否則用 RoboMaster SDK 的手臂；"
                         "sdk 強制用 RoboMaster 的手臂；nexarm 是 USB 接的實體 Hiwonder NexArm（nexarm.py）；nexarm-sim 強制用模擬的")
    ap.add_argument("--arm-port", default=os.environ.get("NEXARM_PORT", "/dev/nexarm"),
                    help="NexArm 的串列埠（udev 規則 99-nexarm.rules 會建 /dev/nexarm；沒有就用 /dev/ttyUSB0）")
    ap.add_argument("--mirror-sim", type=int, default=None, metavar="ZMQ_PORT",
                    help="實體 NexArm 的數位分身：把它的動作即時顯示在這個埠的 CoppeliaSim 裡（視窗版通常是 23001，"
                         "場景要先載 Capstone_Playground_nexarm.ttt：python Test_Robot/nexarm_sim.py --load-nexarm-scene）")
    ap.add_argument("--mirror-real", default=os.environ.get("ROBOT_MIRROR_REAL") or None, metavar="URL",
                    help="把這個（模擬）server 執行的底盤/手臂/夾爪動作同步送到實體 server（例如 http://127.0.0.1:8767）：模擬當眼睛，"
                         "實體照做同樣的動作；ROBOT_MIRROR_ACTIONS=move_arm,arm_to,gripper,recenter_arm 可以只鏡射手臂")
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

    if args.mode == "mock":
        backend = MockBackend()
    else:
        RoboMasterBackend.mirror_sim_port = args.mirror_sim   # _start_nexarm 在 __init__ 裡會讀，所以先掛在類別上
        backend = RoboMasterBackend(args.mode, args.chassis_backend, arm=args.arm, arm_port=args.arm_port)
        if args.mirror_real and args.mode != "sim":
            # 鏡射只有模擬 server 做；實體 server 繼承到 ROBOT_MIRROR_REAL 會把動作再送給自己（2026-11-11 r01：每段 409 busy 重試 3 次多等 6 秒）
            print("[server] mode=%s 忽略鏡射設定 %s（只有 --mode sim 會鏡射）" % (args.mode, args.mirror_real))
            args.mirror_real = ""
        if args.mirror_real:
            backend.mirror_real = args.mirror_real
            acts = os.environ.get("ROBOT_MIRROR_ACTIONS", "").strip()
            if acts:
                backend.mirror_actions = tuple(a.strip() for a in acts.split(",") if a.strip())
            try:
                with urllib.request.urlopen(args.mirror_real.rstrip("/") + "/health", timeout=3) as resp:
                    h = json.loads(resp.read().decode("utf-8"))
                print("[server] 鏡射到實體 server %s（mode=%s, link=%s）：%s" % (
                    args.mirror_real, h.get("mode"), h.get("robot_link"), ", ".join(backend.mirror_actions)))
            except Exception as e:
                print("[server] 鏡射目標 %s 現在連不上（%s），動作會在執行時再試、失敗會寫在回報裡" % (args.mirror_real, e))
    httpd = _ThreadingServer((args.host, args.port), make_handler(backend, args.token))
    print("[server] 就緒：http://%s:%d  (mode=%s, token=%s)" % (
        args.host, args.port, args.mode, "on" if args.token else "off"))

    def _on_term(signum, frame):
        # 用 nohup/背景啟動的 server 收不到 Ctrl+C（非互動 shell 的背景工作會忽略 SIGINT），
        # 大家都用 kill（SIGTERM）關它；把 SIGTERM 也走同一條乾淨關閉的路，離開前一樣先停車、關串列埠。
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _on_term)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] 收到 Ctrl+C / SIGTERM，準備關閉")
    finally:
        # 跟 test_movement.py 一樣：不管怎樣，離開前一定先停車再斷線
        backend.close()
        httpd.server_close()


if __name__ == "__main__":
    main()
