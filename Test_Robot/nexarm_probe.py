"""
NexArm（Hiwonder，ESP32 + 匯流排伺服）的 USB 串列協定探測：只讀六顆伺服的位置，不送任何會讓手臂動的指令。

封包格式照官方 Hiwonder-official/lerobot-nexarm 的 src/lerobot/motors/nexarm/nexarm.py：
    封包 = FF FF | ID | LEN | CMD | 參數... | CHK
    LEN = 參數長度 + 2，CHK = ~(ID + LEN + CMD + 參數總和) & 0xFF
    1000000 baud 8N1，控制器 ID 0xFF

實測（2026-11-09，出廠韌體、沒進 LeRobot 模式）：手臂不用問就每秒自己送約 5 個 CMD 11（0x0B，Wiki 3.3 的「讀目前座標」）封包，
24 bytes = 12 個 little-endian int16：x, y, z, pitch, roll, claw, 再接六顆伺服位置（0–4095，中位 2048）。
韌體原始碼（Hiwonder 的 Nex_Arm.zip，ESP32 1.0.2 / AT32 1.0.2）：x/y/z 是 mm，pitch 是度 ×10。
韌體每收到一個 USB 指令都會往同一個串列埠印一行 `[CB] id=0xFF cmd=N len=M`，要確認指令有沒有進去就看這行。
CMD 96 是 LeRobot 橋接模式（CMD 68 進入）才有的；平常讀位置就聽 CMD 11。
第一次接上時曾經所有指令都沒有 [CB]，讓 ESP32 重開機（RTS 拉起、DTR 放掉 = EN 拉低）之後才正常。

執行方式（手臂接 12V 變壓器、開關 ON、Type-C 接電腦；Hiwonder App/藍牙/PC 軟體先關掉）：
    ~/rmenv/bin/python Test_Robot/nexarm_probe.py               # 自動找 /dev/ttyUSB* /dev/ttyACM*
    ~/rmenv/bin/python Test_Robot/nexarm_probe.py /dev/ttyUSB0
"""

import glob
import struct
import sys
import time

import serial

BAUD = 1_000_000
SYSTEM_ID = 0xFF
CMD_GET_COORDS = 0x0B
JOINTS = ["ID1 底座左右", "ID2 肩", "ID3 肘", "ID4 腕俯仰", "ID5 腕旋轉", "ID6 夾爪"]


def build_frame(device_id: int, cmd: int, args: bytes = b"") -> bytes:
    data = bytes([device_id & 0xFF, (len(args) + 2) & 0xFF, cmd & 0xFF]) + args
    return b"\xff\xff" + data + bytes([(~sum(data)) & 0xFF])


def parse_reply(rx: bytes):
    """在收到的位元組裡找第一個校驗和正確的封包，回 (id, cmd, args)；找不到回 None。"""
    i = 0
    while (i := rx.find(b"\xff\xff", i)) >= 0 and i + 5 <= len(rx):
        dev, ln, cmd = rx[i + 2], rx[i + 3], rx[i + 4]
        end = i + 3 + ln  # CHK 的位置：標頭 2 + ID + LEN + CMD，再加 LEN-2 個參數
        if ln >= 2 and end < len(rx):
            args = rx[i + 5:end]
            if (~(dev + ln + cmd + sum(args))) & 0xFF == rx[end]:
                return dev, cmd, args
        i += 1
    return None


def main():
    ports = sys.argv[1:] or sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if not ports:
        raise SystemExit("找不到 /dev/ttyUSB* 或 /dev/ttyACM*：手臂的 Type-C 有接上電腦嗎？（lsusb 看有沒有新裝置）")
    port = ports[0]
    print(f"串列埠 {port} @ {BAUD}")
    try:
        ser = serial.Serial(port, BAUD, timeout=0.2, write_timeout=0.2)
    except serial.SerialException as e:
        if "Permission denied" in str(e):
            raise SystemExit(f"沒有權限開 {port}：sudo usermod -aG dialout $USER 之後登出再登入"
                             f"（臨時可以 sudo chmod 666 {port}）")
        raise
    time.sleep(0.1)
    ser.reset_input_buffer()

    # 手臂會自己廣播狀態，聽 0.5 秒就好；順便送一次 0x0B 查詢，沒有廣播的韌體也能拿到
    ser.write(build_frame(SYSTEM_ID, CMD_GET_COORDS))
    rx = b""
    t0 = time.time()
    while time.time() - t0 < 0.5:
        rx += ser.read(256)
    ser.close()
    print(f"0.5 秒收到 {len(rx)} bytes，開頭:", rx[:30].hex(" ") if rx else "(沒有任何回應)")

    reply = parse_reply(rx[-60:]) or parse_reply(rx)
    if reply is None:
        raise SystemExit("沒有解析到合法封包。檢查：電源開了嗎（開機會嗶一聲）、App/藍牙是不是還連著、鮑率")
    dev, cmd, args = reply
    print(f"狀態封包 id=0x{dev:02x} cmd={cmd} 參數 {len(args)} bytes")
    if cmd != CMD_GET_COORDS or len(args) < 24:
        raise SystemExit("不是預期的座標狀態封包（cmd 11、24 bytes）")
    v = struct.unpack_from("<12h", args)
    print(f"  末端 x={v[0]} y={v[1]} z={v[2]}  pitch={v[3]} roll={v[4]} claw={v[5]}")
    for name, pos in zip(JOINTS, v[6:]):
        print(f"  {name:10s} {pos:5d}")
    print("協定通了。")


if __name__ == "__main__":
    main()
