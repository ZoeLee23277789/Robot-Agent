"""
RoboMaster 連線輔助模組
========================
把連線設定集中在這裡，模擬器和實體機器人共用同一組呼叫方式。
之後要換成接實體 EP，只需要改這個檔案，呼叫端的程式完全不用動。
"""

import os
import time

import patch_ftp  # noqa: F401  模擬器沒有 FTP 服務，這個補丁讓 SDK 不會卡在等待上
import robomaster.config as config
from robomaster import conn, robot


def _local_ips() -> set:
    """這台電腦自己的所有 IPv4（含有線、Wi-Fi、loopback）。"""
    import socket
    import subprocess
    ips = {"127.0.0.1"}
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=3).stdout
        for line in out.splitlines():
            parts = line.split()
            if "inet" in parts:
                ips.add(parts[parts.index("inet") + 1].split("/")[0])
    except Exception:
        pass
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except Exception:
        pass
    return ips


def find_real_ep(timeout: float = 4.0) -> list:
    """路由器模式：聽 EP 的 UDP 40927 廣播（前 14 bytes 是 SN），回 [(ip, sn), ...]，不含這台電腦自己送的。
    2026-11-10 踩到的坑：視窗版 CoppeliaSim 的 RoboMaster 外掛會在本機模擬一台 EP（TCP 40921、廣播 "RM0001********"），
    SDK 的 scan_robot_ip 第一個聽到誰就連誰，結果 server 連到模擬的那台還以為是實體（韌體版本 00.02.0256、相機正常、電量 88）。"""
    import select
    import socket
    mine = _local_ips()
    found = {}
    socks = []
    # 新韌體廣播到 40927；SDK 沒給 SN 時聽的是舊埠 45678（scan_robot_ip 的「相容舊版」分支），兩個都聽
    for port in sorted({config.ROBOT_BROADCAST_PORT, 40927, 45678}):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            socks.append(s)
        except Exception as e:
            print(f"[rm_connection] 聽 UDP {port} 失敗：{e}")
    t0 = time.time()
    while socks and time.time() - t0 < timeout:
        ready, _, _ = select.select(socks, [], [], 1.0)
        for s in ready:
            try:
                data, addr = s.recvfrom(1024)
            except Exception:
                continue
            if addr[0] in mine:
                continue
            # 40927 的封包前 14 bytes 是明文 SN；45678 的是加密的 STAConnInfo（2026-11-11 實測 EP 兩個埠都會廣播），
            # 硬解會 UnicodeDecodeError，所以解不出 ASCII 就只記 IP、不蓋掉已經拿到的 SN
            try:
                sn = conn.get_sn_form_data(data)
                if not sn.isprintable():
                    raise ValueError
            except Exception:
                sn = found.get(addr[0], "?")
            found[addr[0]] = sn if sn != "?" else found.get(addr[0], "?")
    for s in socks:
        s.close()
    return sorted(found.items())


def pick_real_ep(timeout: float = 4.0) -> str:
    """決定路由器模式要連哪個 IP：RM_ROBOT_IP 指定 > 廣播掃到的唯一一台 > RM_SN 挑；找不到就丟 RuntimeError 說明原因。"""
    forced = os.environ.get("RM_ROBOT_IP", "").strip()
    if forced:
        print(f"[rm_connection] RM_ROBOT_IP={forced}，不掃描")
        return forced
    robots = find_real_ep(timeout)
    want_sn = os.environ.get("RM_SN", "").strip()
    if want_sn:
        robots = [r for r in robots if r[1] == want_sn]
    if not robots:
        raise RuntimeError("這台電腦所在的網段上聽不到任何實體 EP 的廣播（UDP 40927）。EP 的開關要在「路由器」那側並重開機、"
                           "而且要已經用 App 掃過 QR code 加入同一個網段的 Wi-Fi；知道 IP 的話可以 RM_ROBOT_IP=... 直接指定"
                           + (f"（RM_SN={want_sn} 沒對上）" if want_sn else ""))
    if len(robots) > 1:
        raise RuntimeError("網段上有不只一台 EP：" + ", ".join(f"{ip} (SN {sn})" for ip, sn in robots)
                           + "；用 RM_SN=<SN> 或 RM_ROBOT_IP=<ip> 指定要哪一台")
    ip, sn = robots[0]
    print(f"[rm_connection] 實體 EP：{ip}（SN {sn}）")
    return ip


def connect(mode: str = "sim", retries: int = 4, retry_delay: float = 2.0) -> robot.Robot:
    """建立並回傳一個已初始化的 Robot 物件。

    mode="sim"  連到本機的 CoppeliaSim 模擬器 (127.0.0.1)
    mode="real" 連到實體 EP。連線方式用環境變數 RM_CONN_TYPE 選：
                "sta"（預設）router 模式，機器人和這台電腦要在同一個網段（先用 RoboMaster App 掃 QR code 設定）；
                             會自己聽廣播找實體 EP（排除本機模擬器），也可 RM_ROBOT_IP=/RM_SN= 指定
                "ap"         直連模式，這台電腦直接連機器人自己發的 Wi-Fi 熱點（機器人 IP 固定 192.168.2.1）
                "rndis"      USB 線接智能中控的 micro-USB 孔（機器人 IP 192.168.42.2），電腦的 Wi-Fi 可以留著上網

    模擬器這條連線偶爾會在剛啟動模擬、或上一次連線沒乾淨關閉時失敗一次
    （"Robot: Can not connect to robot, check connection please."），
    但通常隔個一兩秒重試就會成功，所以這裡內建重試，不用手動重跑整支程式。
    """
    if mode == "sim":
        config.LOCAL_IP_STR = "127.0.0.1"
        config.ROBOT_IP_STR = "127.0.0.1"
    elif mode == "real":
        config.LOCAL_IP_STR = None
        config.ROBOT_IP_STR = None
    else:
        raise ValueError(f"未知的 mode: {mode!r}，只能是 'sim' 或 'real'")

    # SDK 的 request_connection 用 `conn_type is CONNECTION_WIFI_AP` 比較字串，
    # 從環境變數讀來的 "ap" 不是同一個物件 → 三個分支都不成立 → UnboundLocalError: proxy_addr。
    # 所以一定要換成 SDK 自己的常數再傳進去。
    conn_types = {"sta": conn.CONNECTION_WIFI_STA, "ap": conn.CONNECTION_WIFI_AP, "rndis": conn.CONNECTION_USB_RNDIS}
    name = os.environ.get("RM_CONN_TYPE", "sta").strip().lower() if mode == "real" else "sta"
    if name not in conn_types:
        raise ValueError(f"RM_CONN_TYPE={name!r} 不認得，只能是 sta / ap / rndis")
    conn_type = conn_types[name]
    print(f"[rm_connection] mode={mode} conn_type={name}")
    if mode == "real" and name == "sta":
        # 不讓 SDK 自己掃：它會連到本機模擬器外掛假裝的 EP（見 find_real_ep）
        config.ROBOT_IP_STR = pick_real_ep()
    last_err = None
    for attempt in range(1, retries + 1):
        ep = robot.Robot()
        try:
            ep.initialize(conn_type=conn_type)
            return ep
        except Exception as e:
            last_err = e
            try:
                ep.close()
            except Exception:
                pass
            if attempt < retries:
                print(f"[rm_connection] 連線失敗（第 {attempt}/{retries} 次）：{e}，{retry_delay:.0f}s 後重試...")
                time.sleep(retry_delay)
    raise last_err


def start_camera(ep: robot.Robot, retries: int = 3, resolution: str = "720p") -> bool:
    """開前鏡頭串流，成功回 True。

    SDK 的 start_video_stream 失敗時只會 log error 然後回 False，不會丟例外，呼叫端一定要看回傳值。
    實體 EP（韌體 01.01.1008、直連模式，2026-11-07）送串流控制指令（cmdset 0x3f / cmdid 0xd2）曾連續兩次等不到回應；
    失敗時先退出 SDK 串流模式再重試。
    """
    for attempt in range(1, retries + 1):
        try:
            if ep.camera.start_video_stream(display=False, resolution=resolution) is not False:
                return True
        except Exception as e:
            print(f"[rm_connection] 相機串流啟動例外：{e}")
        print(f"[rm_connection] 相機串流啟動失敗（第 {attempt}/{retries} 次）")
        try:
            ep.camera._stream_sdk(0)
        except Exception:
            pass
        time.sleep(1.0)
    return False
