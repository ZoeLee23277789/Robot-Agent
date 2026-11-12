"""
NexArm（Hiwonder，ESP32 主控 + AT32 伺服控制器 + 六顆匯流排伺服）的 USB 串列驅動，給 robot_server.py 用。

接法：手臂底座的 Type-C 接電腦 → CH340 → /dev/ttyUSB0（udev 規則 99-nexarm.rules 另外給它 /dev/nexarm），1000000 8N1。
協定（韌體原始碼 ~/下載/Nex_Arm：CommProtocol.cpp、Global.h、Robot_Arm.cpp）：
    封包 = FF FF | ID(0xFF) | LEN(=參數長度+2) | CMD | 參數 | CHK(=~(ID+LEN+CMD+參數和) & 0xFF)
    CMD 8  COORDINATE_SET：pitch×10, x, y, z, roll, claw（int16 LE）+ time ms（uint16），手臂自己算逆運動學
    CMD 11 GET_CUR_COORDS：韌體每秒自己送約 5 次，24 bytes = x, y, z, pitch×10, roll, claw + 六顆伺服位置（0–4095）

2026-11-09 實測（出廠韌體 ESP32 1.0.2 / AT32 1.0.2）：
  - 座標單位 mm、pitch 度；同一個目標實際會停在低約 3 mm 的位置（下垂），x 準到 1 mm。
  - claw 就是夾爪伺服（ID6）的角度（度）：伺服位置 = 2048 + claw × 11.38，-89 ~ 50 都轉得到位（空夾爪）。
    夾到東西時伺服會被擋住、到不了目標，拿「目標 − 實際」的差距判斷有沒有夾住。
  - 韌體的增量指令（CMD 50）是加在「上一次下達的目標」上，不是實測位置，一次搆不到之後每步都帶著偏差，所以這裡
    一律用絕對座標（CMD 8），相對移動自己算。
  - 剛上電時曾經所有指令都被忽略；韌體每收到一個指令會往同一個串列埠印 `[CB] id=0xFF cmd=N`，看這行最準。
    沒有 [CB] 就是手臂沒在聽：關機再開機。
"""

import struct
import threading
import time

import serial

BAUD = 1_000_000
SYSTEM_ID = 0xFF
CMD_CHECK_BAT_LEVEL = 2          # 回 uint16 mV：主板量到的輸入電壓（ADC×11，Nex_Arm_Board.cpp）；電池供電時看它夠不夠
CMD_COORDINATE_SET = 8
CMD_GET_CUR_COORDS = 11

COUNTS_PER_DEG = 4096 / 360.0   # 伺服一圈 4096 格
SERVO_CENTER = 2048
CLAW_OPEN = -60     # 韌體 AiLLMControl 的 open_claw 用這個值
CLAW_CLOSE = 50     # 韌體搖桿模式允許的最大值，空夾爪實測轉得到位
HOLD_SHORTFALL_DEG = 4.0   # 夾爪伺服停在離目標這麼多度以上 → 被東西擋住了（空夾實測誤差 ≤ 0.3°）

HOME = (150.0, 0.0, 250.0, 0.0)   # x, y, z, pitch：實測可達、手腕水平，當 recenter 的姿態
# 收攏姿勢（2026-11-10 nexarm_fold_test.py 實測）：夾爪朝下、垂在底座正上方，誤差 0 mm。整支手臂前後都收在底座上方，
# 最高點是肘（約 290 mm，比 HOME 的 307 矮），肘在底座後方約 145 mm。再低 10 mm（z 130）就開始到不了。
STOW = (0.0, 0.0, 150.0, -90.0)
# 外框：各軸的絕對界線。x/z 真正到得了的區域不是方的，看 REACH（nexarm_workspace.py 掃出來的表）。
# z 下限 30：末端再低就碰到桌面（韌體搖桿模式也是 30）。x 下限看 x_min_for_pitch：夾爪水平時縮到 90 以內伺服就停在離目標
# 10 mm 以上（實測 80 差 12 mm），夾爪朝下則一路縮到 0 都精準。
WORKSPACE = {"x": (-460.0, 460.0), "y": (-460.0, 460.0), "z": (30.0, 560.0),
             "pitch": (-90.0, 10.0), "roll": (-90.0, 90.0), "claw": (-90.0, 50.0)}


def x_min_for_pitch(pitch):
    """夾爪朝下（pitch ≤ −80）才允許縮到底座正上方；其他俯仰角只驗證過到 90 mm。"""
    return 0.0 if pitch <= -80.0 else 90.0


# 底座旋轉（ID1）的範圍：nexarm_rotate_test.py 2026-11-10 實測 ±117.3°（伺服 1 停在 712 / 3382，正向轉時數值變小），
# 手腕滾轉（ID5）±90°（伺服 5 停在 3083 / 1013，正向變大）。
YAW_LIMIT_DEG = 117.0


def yaw_of(x, y):
    """底座旋轉角：0 = 正前方，正 = 向左（逆時針，韌體 y 軸正向）。"""
    import math
    return math.degrees(math.atan2(y, x))


def rotate_xy(x, y, yaw_deg):
    """把 (x, y) 的水平距離保留、轉到 yaw_deg。"""
    import math
    r = math.hypot(x, y)
    return r * math.cos(math.radians(yaw_deg)), r * math.sin(math.radians(yaw_deg))


def relative_target(base, dx=0.0, dy=0.0, dz=0.0, dpitch=0.0, dyaw=0.0, droll=0.0):
    """base = [x, y, z, pitch, roll, claw]。前伸 dx 沿目前底座朝向、側移 dy 垂直於它，再把整個水平向量轉 dyaw。
    回傳 move_to 要的 (x, y, z, pitch, roll)。"""
    import math
    r = math.hypot(base[0], base[1])
    yaw = yaw_of(base[0], base[1]) if r > 1e-6 else 0.0
    r2 = max(0.0, r + dx)
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    x, y = r2 * c - dy * s, r2 * s + dy * c
    if abs(dyaw) > 1e-9 and math.hypot(x, y) > 1e-6:
        # 不繞圈：目標角直接夾在 ±YAW_LIMIT_DEG，不然 90°+200° 會被當成 −70° 從另一邊過去
        x, y = rotate_xy(x, y, max(-YAW_LIMIT_DEG, min(YAW_LIMIT_DEG, yaw_of(x, y) + dyaw)))
    return x, y, base[2] + dz, base[3] + dpitch, base[4] + droll


def clamp_yaw(x, y):
    """底座轉角超過 YAW_LIMIT_DEG 就轉回極限；回傳 (x, y, 說明或 None)。"""
    yaw = yaw_of(x, y)
    if abs(yaw) <= YAW_LIMIT_DEG or (abs(x) < 1e-6 and abs(y) < 1e-6):
        return x, y, None
    lim = YAW_LIMIT_DEG if yaw > 0 else -YAW_LIMIT_DEG
    nx, ny = rotate_xy(x, y, lim)
    return nx, ny, "base yaw %.0f->%.0f deg (base rotation limit)" % (yaw, lim)
REACH_TOL_MM = 12.0
REACH_TOL_DEG = 5.0


class ReachMap(object):
    """nexarm_workspace.json：對每個掃過的夾爪俯仰角，記錄 (水平距離 r, 高度 z) 格點裡哪些到得了。
    2026-11-09 用 CMD 7（只算 IK、不動）＋ 自己擬合的 FK 複驗掃出來，再實機走了 4 個邊緣點驗證（誤差 0–11 mm）；
    表外的過渡點 (200, 250, -90°) 實機也確實到不了。查詢時用最接近的俯仰角那一張。"""

    def __init__(self, path=None):
        import json
        import os
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "nexarm_workspace.json")
        data = json.load(open(path, encoding="utf-8"))
        self.step = data["step_mm"]
        self.cells = {float(p): set(map(tuple, v["reachable"])) for p, v in data["pitches"].items()}
        self.pitches = sorted(self.cells)

    def _grid(self, pitch):
        return self.cells[min(self.pitches, key=lambda p: abs(p - pitch))]

    def _snap(self, v):
        return int(round(v / self.step) * self.step)

    # 夾爪朝下、縮在底座正上方時的實測高度上限（2026-11-10 用 server arm_to 掃：x=0 pitch −90 → z 160 到位、170 往前漂 9 mm/4°、
    # 180 差 18 mm/8°）。IK 表說 190 太樂觀，因為伺服到極限時韌體算得出解但實機停不到。r ≤ 10 用 160；10～80 之間沒實測，
    # 線性過渡到表上 r=80 的值（表在 r ≥ 80 有實機驗證過）。
    FOLDED_Z_CAP_MM = 160.0
    FOLDED_R_INNER_MM = 10.0
    FOLDED_R_OUTER_MM = 80.0

    def _z_cap(self, r, pitch):
        if pitch > -80.0 or r >= self.FOLDED_R_OUTER_MM:
            return None
        outer = [z for (x, z) in self._grid(pitch) if x == self._snap(self.FOLDED_R_OUTER_MM)]
        if not outer:
            return self.FOLDED_Z_CAP_MM
        t = max(0.0, r - self.FOLDED_R_INNER_MM) / (self.FOLDED_R_OUTER_MM - self.FOLDED_R_INNER_MM)
        return self.FOLDED_Z_CAP_MM + t * (max(outer) - self.FOLDED_Z_CAP_MM)

    def reachable(self, r, z, pitch):
        cap = self._z_cap(r, pitch)
        return (z >= WORKSPACE["z"][0] and r >= WORKSPACE["x"][0]
                and (cap is None or z <= cap + 1e-6)
                and (self._snap(r), self._snap(z)) in self._grid(pitch))

    def z_span(self, r, pitch):
        """在這個水平距離、這個俯仰角下，高度的 [最低, 最高]（已扣掉桌面）；到不了回 None。
        只取包含連續一段的最外圍；夾爪水平時中間有洞（前伸 150 最低 200）也一併反映在最低值裡。"""
        rs = self._snap(r)
        zs = [z for (x, z) in self._grid(pitch) if x == rs and z >= WORKSPACE["z"][0]]
        if not zs:
            return None
        cap = self._z_cap(r, pitch)
        hi = max(zs) if cap is None else int(min(max(zs), cap))
        return (min(zs), max(min(zs), hi))

    def r_span(self, z, pitch):
        zs = self._snap(z)
        rs = [x for (x, zz) in self._grid(pitch) if zz == zs and x >= WORKSPACE["x"][0]]
        return (min(rs), max(rs)) if rs and z >= WORKSPACE["z"][0] else None


try:
    REACH = ReachMap()
except Exception as _e:   # 沒有掃描檔就退回只用外框（先跑 nexarm_workspace.py）
    REACH = None
    print("[nexarm] 沒有可達區域表（%s），只用外框限制；先跑 Test_Robot/nexarm_workspace.py" % _e)


def build_frame(device_id, cmd, args=b""):
    data = bytes([device_id & 0xFF, (len(args) + 2) & 0xFF, cmd & 0xFF]) + bytes(args)
    return b"\xff\xff" + data + bytes([(~sum(data)) & 0xFF])


def extract_frames(buf):
    """從收到的位元組裡抓出所有校驗和正確的封包，回 ([(id, cmd, args), ...], 剩下還沒收完的尾巴)。
    韌體的除錯文字（[CB]、[FW] ...）跟封包混在同一條線上，校驗和不對的位置就當雜訊跳過。"""
    frames = []
    i = 0
    n = len(buf)
    keep_from = None   # 最早一個「可能還沒收完」的候選位置；後面若出現完整合法封包，它就是雜訊
    while True:
        i = buf.find(b"\xff\xff", i)
        if i < 0:
            break
        if i + 5 > n:
            keep_from = i if keep_from is None else keep_from
            break
        ln = buf[i + 3]
        if ln < 2:
            i += 1
            continue
        end = i + 3 + ln          # CHK 的位置
        if end >= n:
            # 還沒收完——但長度欄位也可能只是雜訊（例如文字裡剛好有 FF FF），不能停在這裡等，繼續往後找
            if keep_from is None:
                keep_from = i
            i += 1
            continue
        dev, cmd, args = buf[i + 2], buf[i + 4], buf[i + 5:end]
        if (~(dev + ln + cmd + sum(args))) & 0xFF == buf[end]:
            frames.append((dev, cmd, args))
            i = end + 1
            keep_from = None
        else:
            i += 1
    return frames, (buf[keep_from:] if keep_from is not None else b"")


def _clamp_to(name, v):
    lo, hi = WORKSPACE[name]
    return max(lo, min(hi, float(v)))


class TcpLink(object):
    """跟 serial.Serial 同樣的 read/write/flush/close 介面，但走手臂自己的 Wi-Fi 熱點。
    韌體（system_task_handle.cpp）開機就起 AP：SSID NexArm / 密碼 hiwonder，TCP 8080（EspNow_Ctrl_t::server(8080)），
    收到的位元組直接丟進同一個協定解析器，所有回覆與 5 Hz 的狀態廣播也都 sendData 到這個客戶端，所以封包格式跟 USB 一模一樣。
    一次只接受一個客戶端（第二個連上來會被 stop）。"""

    def __init__(self, url, timeout=0.05):
        import socket
        host, _, port = url[len("tcp://"):].partition(":")
        self.sock = socket.create_connection((host, int(port or 8080)), timeout=3.0)
        self.sock.settimeout(timeout)
        self._timeout = socket.timeout

    def read(self, n):
        try:
            data = self.sock.recv(n)
        except self._timeout:
            return b""
        if not data:                    # 對方關了連線：丟例外讓 _reader 走重開那條路
            raise IOError("tcp link closed by the arm")
        return data

    def write(self, data):
        self.sock.sendall(data)

    def flush(self):
        pass

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def open_link(port, baud=BAUD):
    """port 是 /dev/... 就開串列埠；是 tcp://主機:埠 就連手臂的 Wi-Fi 熱點（例如 tcp://192.168.4.1:8080）。"""
    if str(port).startswith("tcp://"):
        return TcpLink(port)
    return serial.Serial(port, baud, timeout=0.05, write_timeout=0.3)


class NexArm(object):
    def __init__(self, port="/dev/nexarm", baud=BAUD, on_status=None):
        self.port = port
        self.baud = baud
        self.on_status = on_status      # 每收到一筆狀態就呼叫 on_status(dict)，robot_server 拿來更新遙測
        self._ser = open_link(port, baud)
        self._opened_ts = time.time()
        self._wlock = threading.RLock()    # _send 失敗時會在鎖裡呼叫 _reopen，要可重入
        self._status = None
        self._status_ts = 0.0
        self.voltage_mv = None          # 最近一次 CMD 2 回報的輸入電壓（mV）
        self._last_bat_poll = 0.0
        self._last_poke = 0.0
        self._silent_pokes = 0
        self._target = None             # 上一次「有到」的目標 (x, y, z, pitch, roll, claw)：相對移動以它為基準，不累積下垂
        self.cb_count = 0               # 看過幾個 [CB]（韌體收到指令的回音）
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        if not self.wait_status(2.0):
            self.close()
            raise RuntimeError("NexArm 在 %s 上 2 秒內沒有送出任何狀態封包：電源開了嗎（開機會嗶）？線接對了嗎？" % port)

    # ---- 接收 ----
    def _reader(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(512)
            except Exception as e:
                print("[nexarm] 串列讀取失敗：%s，1 秒後重試" % e)
                time.sleep(1.0)
                self._reopen()
                continue
            if not chunk:
                self._watchdog()
                continue
            self.cb_count += chunk.count(b"[CB]")
            buf += chunk
            frames, buf = extract_frames(buf)
            if len(buf) > 2048:          # 一直收不到合法封包的雜訊，不要讓緩衝區無限長
                buf = buf[-256:]
            for _dev, cmd, args in frames:
                if cmd == CMD_CHECK_BAT_LEVEL and len(args) >= 2:
                    self.voltage_mv = struct.unpack_from("<H", args)[0]
                    continue
                if cmd == CMD_GET_CUR_COORDS and len(args) >= 24:
                    now = time.time()
                    if now - self._last_bat_poll > 5.0:
                        self._last_bat_poll = now
                        try:
                            self._send(CMD_CHECK_BAT_LEVEL, b"")
                        except Exception:
                            pass
                    v = struct.unpack_from("<12h", args)
                    st = {"x": v[0], "y": v[1], "z": v[2], "pitch": v[3] / 10.0, "roll": v[4], "claw": v[5],
                          "servos": list(v[6:12]), "voltage_mv": self.voltage_mv}
                    self._status = st
                    self._status_ts = time.time()
                    if self._silent_pokes:
                        print("[nexarm] 狀態回報恢復")
                        self._silent_pokes = 0
                    if self.on_status is not None:
                        try:
                            self.on_status(st)
                        except Exception:
                            pass

    def _watchdog(self):
        """狀態超過 2 秒沒來：先送一次 CMD 11 問它（韌體自己沒在廣播時會回），再沒有就關掉重開串列埠。
        2026-11-09 實測：CH340 在 kernel log 出現 `urb stopped: -32` 之後，裝置節點還在、read() 不丟例外，
        但永遠讀不到任何東西；只有重新開啟串列埠（重新送出 USB 讀取請求）才會恢復。"""
        now = time.time()
        if now - max(self._status_ts, self._opened_ts) < 2.0 or now - self._last_poke < 1.5:
            return
        self._last_poke = now
        self._silent_pokes += 1
        if self._silent_pokes == 1:
            print("[nexarm] 2 秒沒有狀態回報，送 CMD 11 問一次")
            try:
                self._send(CMD_GET_CUR_COORDS, b"")
            except Exception as e:
                print("[nexarm] 送出失敗：%s" % e)
        else:
            print("[nexarm] 還是沒有回報（第 %d 次），重新開啟串列埠 %s" % (self._silent_pokes, self.port))
            self._reopen()

    def _reopen(self):
        with self._wlock:
            try:
                self._ser.close()
            except Exception:
                pass
            try:
                self._ser = open_link(self.port, self.baud)
                self._opened_ts = time.time()
                print("[nexarm] 連線重新開啟（%s）" % self.port)
            except Exception as e:
                print("[nexarm] 重新開啟失敗：%s（USB 掉了？等它回來再試）" % e)

    def status(self):
        """(最新狀態 dict 或 None, 幾秒前收到的)"""
        st = self._status
        return (dict(st) if st else None), (time.time() - self._status_ts if st else None)

    def wait_status(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._status is not None:
                return True
            time.sleep(0.05)
        return False

    def link_ok(self, max_age=1.5):
        return self._status is not None and time.time() - self._status_ts < max_age

    # ---- 送出 ----
    def _send(self, cmd, payload):
        frame = build_frame(SYSTEM_ID, cmd, payload)
        with self._wlock:
            try:
                self._ser.write(frame)
                self._ser.flush()
            except Exception as e:
                # USB 瞬斷（kernel: usb 1-10.1 USB disconnect → 重新列舉成 ttyUSB1，/dev/nexarm 跟著走）：裝置節點要 1–3 秒才回來，
                # 立刻重開一定失敗、再寫就 PortNotOpenError。等它回來，最多 4 秒；2026-11-10 批次一天內斷了 10 次（hub 那一路不穩）。
                print("[nexarm] 寫入失敗（%s），等串列埠回來再送" % e)
                last = e
                for _ in range(8):
                    time.sleep(0.5)
                    self._reopen()
                    try:
                        self._ser.write(frame)
                        self._ser.flush()
                        break
                    except Exception as e2:
                        last = e2
                else:
                    raise last
        return frame

    def _base_pose(self):
        """相對移動的基準：上一次有到的目標；沒有的話用實測位置。"""
        if self._target is not None:
            return list(self._target)
        st = self._status
        if st is None:
            return None
        return [st["x"], st["y"], st["z"], st["pitch"], st["roll"], st["claw"]]

    def move_to(self, x=None, y=None, z=None, pitch=None, roll=None, claw=None, ms=None, correct=True):
        """絕對座標（mm / 度）。沒給的軸沿用基準姿態。回傳 dict：
        ok（實測停在目標 12 mm / 5° 內）、pose（實測）、before（動作前實測）、target、error_mm、pitch_error_deg、
        clamped（被夾進工作空間的軸）、corrected（有沒有補送一次修正）。"""
        base = self._base_pose()
        if base is None:
            return {"ok": False, "message": "no status from the arm", "pose": None}
        want = [x, y, z, pitch, roll, claw]
        names = ["x", "y", "z", "pitch", "roll", "claw"]
        tgt, clamped = [], []
        for name, b, w in zip(names, base, want):
            v = float(b if w is None else w)
            c = _clamp_to(name, v)
            if abs(c - v) > 1e-6:
                clamped.append("%s %.0f->%.0f" % (name, v, c))
            tgt.append(c)
        xmin = x_min_for_pitch(tgt[3])
        r_h = (tgt[0] ** 2 + tgt[1] ** 2) ** 0.5
        if r_h < xmin:
            nx, ny = rotate_xy(xmin, 0.0, yaw_of(tgt[0], tgt[1]) if r_h > 1e-6 else 0.0)
            clamped.append("reach %.0f->%.0f mm (closest verified reach at pitch %.0f deg; pointing down (-90) can retract to 0)"
                           % (r_h, xmin, tgt[3]))
            tgt[0], tgt[1] = nx, ny
        tgt[0], tgt[1], note = clamp_yaw(tgt[0], tgt[1])
        if note:
            clamped.append(note)
        if REACH is not None:
            # 真正的可達區域不是方的：目標在表外時，先把高度夾到這個前伸距離的可達範圍（「抬到最高」這類要求照樣做到上限），
            # 這個前伸距離根本到不了（任何高度都不行）就不送，直接回報可達範圍，不讓手臂停在一個歪掉的姿勢。
            r = (tgt[0] ** 2 + tgt[1] ** 2) ** 0.5
            if not REACH.reachable(r, tgt[2], tgt[3]):
                span = REACH.z_span(r, tgt[3])
                if span is None:
                    rs = REACH.r_span(tgt[2], tgt[3])
                    return {"ok": False, "pose": dict(self._status) if self._status else None, "unreachable": True,
                            "message": ("target out of reach: at pitch %.0f deg the gripper cannot get to forward %.0f mm at any "
                                        "height%s" % (tgt[3], r, ("; at height %.0f mm the forward reach is %d to %d mm" % (tgt[2], rs[0], rs[1]))
                                                      if rs else ""))}
                zc = min(max(tgt[2], span[0]), span[1])
                clamped.append("z %.0f->%.0f (reachable height at forward %.0f mm, pitch %.0f deg: %d to %d mm)"
                               % (tgt[2], zc, r, tgt[3], span[0], span[1]))
                tgt[2] = float(zc)
        cur = self._status
        dist = (sum((cur[k] - t) ** 2 for k, t in zip(("x", "y", "z"), tgt[:3]))) ** 0.5 if cur else 0.0
        if ms is None:
            ms = int(min(3000, max(500, 500 + dist * 5)))   # 100 mm 大約 1 秒
        payload = struct.pack("<6hH", int(round(tgt[3] * 10)), int(round(tgt[0])), int(round(tgt[1])),
                              int(round(tgt[2])), int(round(tgt[4])), int(round(tgt[5])), int(ms))
        before = dict(cur) if cur else None
        after = self._send_and_settle(payload, ms)

        def errors(st):
            e = (sum((st[k] - t) ** 2 for k, t in zip(("x", "y", "z"), tgt[:3]))) ** 0.5
            return e, abs(st["pitch"] - tgt[3])

        err, perr = errors(after)
        corrected = False
        # 一次閉迴路修正：伺服在負載下會停在離目標幾 mm 的地方（往上動時更明顯，實測 +10 只走 +2），
        # 把差值加回目標再送一次就會更接近。差太多（> 40 mm）代表根本搆不到，補也沒用，不補。
        if correct and (err > 4.0 or perr > 2.0) and err <= 40.0:
            import math
            fix = list(tgt)
            for idx, key, lim in ((2, "z", 15.0), (3, "pitch", 5.0)):
                d = tgt[idx] - after[key]
                fix[idx] = _clamp_to(names[idx], tgt[idx] + max(-lim, min(lim, d)))
            # 水平方向只沿「指令的底座朝向」修正前伸距離，方向本身不動：韌體的底座轉角 = atan2(y, x)，
            # 手臂收在底座正上方（reach≈0）時實測 xy 的方向是雜訊，直接把差值加回 xy 會讓底座甩到隨機方向
            # （2026-11-10 實測：stow 後抬 30 mm，底座甩到 117° 極限）。
            r_t = math.hypot(tgt[0], tgt[1])
            yaw_t = yaw_of(tgt[0], tgt[1]) if r_t > 1e-6 else 0.0
            c, s = math.cos(math.radians(yaw_t)), math.sin(math.radians(yaw_t))
            r_meas = after["x"] * c + after["y"] * s
            r_fix = max(xmin, r_t + max(-15.0, min(15.0, r_t - r_meas)))
            fix[0], fix[1] = rotate_xy(r_fix, 0.0, yaw_t)
            payload2 = struct.pack("<6hH", int(round(fix[3] * 10)), int(round(fix[0])), int(round(fix[1])),
                                   int(round(fix[2])), int(round(fix[4])), int(round(fix[5])), 500)
            after = self._send_and_settle(payload2, 500)
            err, perr = errors(after)
            corrected = True
        ok = err <= REACH_TOL_MM and perr <= REACH_TOL_DEG
        self._target = tuple(tgt) if ok else None
        return {"ok": ok, "pose": dict(after), "before": before, "target": dict(zip(names, tgt)),
                "error_mm": round(err, 1), "pitch_error_deg": round(perr, 1), "clamped": ", ".join(clamped),
                "ms": ms, "corrected": corrected}

    def _send_and_settle(self, payload, ms):
        """送座標指令，等名義上的動作時間過去，再等六顆伺服真的停下來（連續三筆狀態位置都沒變）才回傳狀態。
        實測夾爪從 -60 轉到 50 在 800 ms 的名義時間後還在轉，固定等待會讀到半路的位置、誤判成夾到東西。"""
        t_send = time.time()
        self._send(CMD_COORDINATE_SET, payload)
        time.sleep(ms / 1000.0 + 0.1)
        deadline = time.time() + 3.0
        prev, prev_ts, stable = None, 0.0, 0
        while time.time() < deadline:
            st, ts = self._status, self._status_ts
            if ts > t_send + ms / 1000.0 and ts != prev_ts:
                if prev is not None:
                    stable = stable + 1 if all(abs(a - b) <= 3 for a, b in zip(st["servos"], prev["servos"])) else 0
                    if stable >= 2:
                        return st
                prev, prev_ts = st, ts
            time.sleep(0.03)
        return self._status

    def limits_here(self, st=None):
        """以目前姿勢（同一個前伸距離、同一個俯仰角）看，高度和前伸各還有多少空間。給 agent 每一步看的 arm_limits 用。"""
        st = st or self._status
        if st is None:
            return None
        r = (st["x"] ** 2 + st["y"] ** 2) ** 0.5
        out = {"pitch_deg": round(st["pitch"], 1)}
        if REACH is None:
            zlo, zhi = WORKSPACE["z"]
            xlo, xhi = WORKSPACE["x"]
        else:
            zs = REACH.z_span(r, st["pitch"]) or (st["z"], st["z"])
            xs = REACH.r_span(st["z"], st["pitch"]) or (r, r)
            (zlo, zhi), (xlo, xhi) = zs, xs
        out.update({"y_mm_here": [int(zlo), int(zhi)], "x_mm_here": [int(xlo), int(xhi)],
                    "can_raise_mm": int(max(0, zhi - st["z"])), "can_lower_mm": int(max(0, st["z"] - zlo)),
                    "can_extend_mm": int(max(0, xhi - r)), "can_retract_mm": int(max(0, r - xlo))})
        if REACH is not None:
            down = REACH.z_span(r, -90)
            if down:
                out["lowest_with_gripper_pointing_down_mm"] = int(down[0])
        return out

    def move_by(self, dx=0.0, dy=0.0, dz=0.0, dpitch=0.0, dyaw=0.0, droll=0.0):
        """相對移動。dx 是沿「目前底座朝向」的前伸、dyaw 轉底座（正 = 向左）、droll 轉手腕。"""
        base = self._base_pose()
        if base is None:
            return {"ok": False, "message": "no status from the arm", "pose": None}
        return self.move_to(*relative_target(base, dx, dy, dz, dpitch, dyaw, droll))

    def home(self):
        return self.move_to(*HOME, roll=0.0)

    def stow(self):
        """收攏：夾爪朝下垂在底座正上方（STOW）。先把手腕轉下去再縮，直接從 HOME 送 (0,150,-90) 中途肘會先往後甩。"""
        r = self.move_to(x=120.0, y=0.0, z=200.0, pitch=-90.0, roll=0.0)
        if not r.get("ok"):
            return r
        return self.move_to(*STOW, roll=0.0)

    def set_claw(self, claw, ms=800):
        """只動夾爪（手臂姿態沿用基準）。回傳 move_to 的 dict 再加 holding / shortfall_deg：
        伺服停在離目標 HOLD_SHORTFALL_DEG 以上 → 夾爪被東西擋住 = 夾到了。"""
        r = self.move_to(claw=claw, ms=ms, correct=False)   # 夾爪只看伺服有沒有到位，不做座標修正
        st = r.get("pose")
        if st:
            expected = SERVO_CENTER + claw * COUNTS_PER_DEG
            shortfall = (expected - st["servos"][5]) / COUNTS_PER_DEG   # 正 = 比目標開
            r["shortfall_deg"] = round(shortfall, 1)
            r["holding"] = bool(claw > 0 and shortfall >= HOLD_SHORTFALL_DEG)
        else:
            r["shortfall_deg"] = None
            r["holding"] = False
        return r

    def close(self):
        self._stop.set()
        try:
            self._thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            self._ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    arm = NexArm(sys.argv[1] if len(sys.argv) > 1 else "/dev/nexarm")
    st, age = arm.status()
    print("status (%.2fs ago): %s" % (age, st))
    arm.close()
