"""
RoboMaster Robot Server
=======================
跑在「機器人那一端」的小型 HTTP 服務，請在 rmenv (Python 3.6 到 3.8) 裡執行。
它負責三件事：
    1. 透過 rm_connection.connect() 連上模擬器或實體 EP
    2. 把相機畫面與遙測數值整理成 Agent 看得懂的觀察
    3. 接收 Agent 送來的高階動作，做完安全限幅後才真的下指令

Agent 端 (Python 3.11+, browser_use 的 LLM 層) 只會透過 HTTP 跟這支程式講話，
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
}


# 每次取畫面前要先等幾張新影格，用來沖掉解碼器裡的舊畫面。用 tools/measure_camera_lag.py 量過再調整。
FRESH_FRAMES = int(os.environ.get("ROBOT_FRESH_FRAMES", "3"))


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
    {"name": "wait", "params": {"seconds": "float"}, "doc": "do nothing, then re-observe"},
    {"name": "stop", "params": {}, "doc": "stop the chassis"},
]


# ============================================================================
# Backend：真的機器人 / 模擬器
# ============================================================================
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
        try:
            self.ep.camera.start_video_stream(display=False)
            self._camera_ok = True
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
    # 永遠回報 0，沒有實作)，直接用 CoppeliaSim 的 zmqRemoteApi 讀 add_proximity_sensors.py
    # 建立的三個 proximity sensor (prox_front/prox_left/prox_right)。只有 sim 模式才有意義；
    # 找不到套件或找不到感測器都只是印一行訊息、不影響其他功能。----
    def _start_prox_sensors(self):
        self._prox_handles = None
        if self.mode != "sim":
            return
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            sim = RemoteAPIClient().require("sim")
            root = sim.getObject("/RoboMaster", {"noError": True})
            if root == -1:
                raise RuntimeError("找不到 /RoboMaster")
            tree = sim.getObjectsInTree(root, sim.handle_all, 0)
            handles = {}
            for h in tree:
                alias = sim.getObjectAlias(h)
                if alias in ("prox_front", "prox_left", "prox_right"):
                    handles[alias] = h
            missing = {"prox_front", "prox_left", "prox_right"} - set(handles)
            if missing:
                raise RuntimeError("缺少 %s，先跑 add_proximity_sensors.py" % sorted(missing))
        except Exception as e:
            print("[server] 略過真實距離感測器（%s），tof_*_mm 不會出現在 /state" % e)
            return
        self._sim_for_prox = sim
        self._prox_handles = handles
        self._robot_tree = set(tree) | {root}
        threading.Thread(target=self._poll_prox_sensors, daemon=True).start()
        print("[server] 真實距離感測器已連上：front/left/right")

    def _poll_prox_sensors(self):
        field = {"prox_front": "tof_front_mm", "prox_left": "tof_left_mm", "prox_right": "tof_right_mm"}
        sim = self._sim_for_prox
        while True:
            for alias, h in self._prox_handles.items():
                try:
                    res, dist, _point, obj, _n = sim.readProximitySensor(h)
                    # 手臂/夾爪會動，姿勢不一樣時可能剛好掃到自己的手臂而不是真的外部障礙物，
                    # 偵測到的物件如果是機器人自己身上的部件就當作沒偵測到。
                    if res and obj in self._robot_tree:
                        res = 0
                    self._set(field[alias], int(dist * 1000) if res else 9999)
                except Exception:
                    pass
            time.sleep(0.1)

    # ---- 俯視攝影機：add_overhead_camera.py 建立的正交攝影機，給 agent 一個
    # locate_overhead(object) 動作，一次看到整個場地算方位角跟距離，不用靠自轉一步步搜索。
    # 同時把機器人「真的」世界座標 (world_xy) 放進 /state，因為 position_m 是相對 session
    # 開始時的座標、原點是任意的，跟俯視圖的世界座標系對不起來，agent 端要用 world_xy 換算。
    def _start_overhead_cam(self):
        self._overhead_handle = None
        if self.mode != "sim":
            return
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            sim = RemoteAPIClient().require("sim")
            h = sim.getObject("/overhead_cam", {"noError": True})
            if h == -1:
                raise RuntimeError("找不到 overhead_cam，先跑 add_overhead_camera.py")
            root = sim.getObject("/RoboMaster", {"noError": True})
            if root == -1:
                raise RuntimeError("找不到 /RoboMaster")
        except Exception as e:
            print("[server] 略過俯視攝影機（%s），/overhead.jpg 跟 world_xy 不會出現" % e)
            return
        self._sim_for_overhead = sim
        self._sim_for_overhead_lock = threading.Lock()
        self._overhead_handle = h
        self._overhead_root = root
        threading.Thread(target=self._poll_world_xy, daemon=True).start()
        print("[server] 俯視攝影機已連上：/overhead.jpg")

    def _poll_world_xy(self):
        sim = self._sim_for_overhead
        while True:
            try:
                with self._sim_for_overhead_lock:
                    p = sim.getObjectPosition(self._overhead_root, sim.handle_world)
                self._set("world_xy", [round(p[0], 3), round(p[1], 3)])
            except Exception:
                pass
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
                freq=5, callback=lambda a: self._set("yaw_deg", round(a[0], 1)))),
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
        data["camera"] = self._camera_ok
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
            return None
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, int(h * width / float(w))))
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

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
            result["message"] += " | odometry: travelled %.2fm, yaw telemetry %+.0fdeg -> %+.0fdeg" % (
                travelled, before["yaw_deg"], after["yaw_deg"])
            result["measured"] = {"dyaw_deg": round(dyaw, 1), "distance_m": round(travelled, 3)}
            # SDK 的 wait_for_completed 有時候在卡住的當下也回 True (例如頂著東西完全推不動、
            # 位置誤差一開始就已經在容許範圍內)，光看 ok 看不出來，這裡用實際走的距離再核對一次。
            requested = math.hypot(params.get("forward_m", 0) or 0, params.get("right_m", 0) or 0)
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
        return result

    def _act_move_chassis(self, p):
        fwd = _clamp(p.get("forward_m", 0), LIMITS["forward_m"])
        right = _clamp(p.get("right_m", 0), LIMITS["right_m"])
        turn_left = _clamp(p.get("turn_left_deg", 0), LIMITS["turn_left_deg"])
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
                t = math.hypot(fwd, right) / v + 3
                done = self.ep.chassis.move(x=fwd, y=right, z=0, xy_speed=v).wait_for_completed(timeout=t)
                if not done:
                    self.stop()
                    return {"ok": False, "message": "translation timed out after %.1fs (blocked by something?)" % t}
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
            if abs(turn_left) > 0.5:
                start_yaw = self.state().get("yaw_deg")
                timeout_t = abs(turn_left) / w + 4
                self.ep.chassis.drive_speed(x=0, y=0, z=math.copysign(w, turn_left), timeout=timeout_t + 1)
                if start_yaw is None:
                    # 沒有 yaw 遙測可以核對，退回原本算好時間的做法
                    time.sleep(abs(turn_left) / w)
                else:
                    deadline = time.time() + timeout_t
                    while time.time() < deadline:
                        time.sleep(0.05)
                        cur_yaw = self.state().get("yaw_deg")
                        if cur_yaw is None:
                            break
                        turned = (cur_yaw - start_yaw + 180) % 360 - 180
                        # 保險：如果方向又不對或轉過頭一大截，不要傻傻等滿 timeout，先停下來。
                        if abs(turned) > abs(turn_left) + 20:
                            break
                        if turn_left > 0 and turned >= turn_left - 2:
                            break
                        if turn_left < 0 and turned <= turn_left + 2:
                            break
                self.stop()
                time.sleep(0.3)
        return {"ok": True, "message": "moved forward=%.2fm right=%.2fm turn_left=%.0fdeg" % (fwd, right, turn_left)}

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
        time.sleep(0.4)
        now = self.state().get("arm_mm")
        note = self._note_stall(not done)
        return {"ok": bool(done), "message": "arm target x=%.0f y=%.0f mm, %s, now at %s" % (
            x, y, "reached" if done else "timed out (limit?)", now) + note}

    def _act_recenter_arm(self, p):
        done = self.ep.robotic_arm.recenter().wait_for_completed(timeout=8)
        note = self._note_stall(not done)
        return {"ok": bool(done), "message": ("arm recentered" if done else "arm recenter timed out") + note}

    def _act_gripper(self, p):
        state = str(p.get("state", "")).lower()
        if state not in ("open", "close"):
            return {"ok": False, "message": "gripper state must be 'open' or 'close'"}
        if state == "open":
            self.ep.gripper.open(power=50)
        else:
            self.ep.gripper.close(power=50)
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
