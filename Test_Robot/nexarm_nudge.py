"""
NexArm 小幅度移動測試：送 Wiki 3.3 的 0x32「座標增量」指令，只動你指定的軸，其他軸維持原樣。

    payload = dX, dY, dZ, dPitch, dRoll, dClaw（int16 little-endian）+ 時間 ms（uint16），LEN = 0x10
    就是韌體 CMD_ARM_MOVE_INC（50），格式見 ~/下載/Nex_Arm/Robot_Arm.cpp 的 move_inc：x/y/z 是 mm，dPitch 是度 × 10
    2026-11-09 實測：開機姿態 z=281 時 --dz 10 反而得到 z −6、pitch −3.4°，可能已接近最高點，往上沒有解

注意增量是加在「上一次下達的目標」上，不是加在實測位置上：一次搆不到（目標被夾住）之後的每一步都會帶著那個偏差。
所以接進 server 要用 --to 的絕對座標（CMD_COORDINATE_SET = 8：pitch×10, x, y, z, roll, claw, time）。
實測同一個目標會停在低約 3 mm 的位置（下垂）。

預設只印出要送的封包、不會動；加 --go 才真的送出，送出前後各讀一次狀態封包（CMD 11）比對。
    ~/rmenv/bin/python Test_Robot/nexarm_nudge.py --dz 10            # 只看封包
    ~/rmenv/bin/python Test_Robot/nexarm_nudge.py --dz 10 --go       # 末端往上 10 mm
    ~/rmenv/bin/python Test_Robot/nexarm_nudge.py --dz -10 --go      # 回原位
    ~/rmenv/bin/python Test_Robot/nexarm_nudge.py --to 150,0,250 --go   # 絕對座標，pitch 預設 0，roll/claw 沿用目前值
"""

import argparse
import struct
import time

import serial

from nexarm_probe import BAUD, CMD_GET_COORDS, JOINTS, SYSTEM_ID, build_frame, parse_reply

CMD_COORDINATE_SET = 8
CMD_COORD_INCREMENT = 0x32
LIMIT_MM = 30      # 測試用，單次增量上限
LIMIT_TO_MM = 60   # 測試用，--to 離目前位置的上限


def read_status(ser, seconds=0.5):
    ser.reset_input_buffer()
    rx = b""
    t0 = time.time()
    while time.time() - t0 < seconds:
        rx += ser.read(256)
    reply = parse_reply(rx[-60:]) or parse_reply(rx)
    if reply is None or reply[1] != CMD_GET_COORDS or len(reply[2]) < 24:
        return None
    return struct.unpack_from("<12h", reply[2])


def show(tag, v):
    if v is None:
        print(f"{tag}: 讀不到狀態封包")
        return
    print(f"{tag}: x={v[0]} y={v[1]} z={v[2]} pitch={v[3]} roll={v[4]} claw={v[5]} | 伺服 {list(v[6:])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    for axis in ("dx", "dy", "dz"):
        ap.add_argument(f"--{axis}", type=int, default=0, help="mm")
    ap.add_argument("--dpitch", type=float, default=0, help="度（送出時乘 10）")
    ap.add_argument("--droll", type=int, default=0)
    ap.add_argument("--dclaw", type=int, default=0)
    ap.add_argument("--ms", type=int, default=1000, help="動作時間")
    ap.add_argument("--to", help="絕對座標 x,y,z（mm）；給了就忽略 --dx/--dy/--dz 等增量")
    ap.add_argument("--pitch", type=float, default=0.0, help="--to 用：末端俯仰（度）")
    ap.add_argument("--roll", type=int, default=None, help="--to 用：不給就沿用目前值")
    ap.add_argument("--claw", type=int, default=None, help="--to 用：不給就沿用目前值")
    ap.add_argument("--go", action="store_true", help="真的送出（沒加就只印封包）")
    a = ap.parse_args()

    ser = serial.Serial(a.port, BAUD, timeout=0.1, write_timeout=0.2)
    time.sleep(0.1)
    before = read_status(ser)
    show("之前", before)

    if a.to:
        x, y, z = (int(round(float(v))) for v in a.to.split(","))
        if before is None:
            raise SystemExit("讀不到目前位置，不送絕對座標")
        dist = ((x - before[0]) ** 2 + (y - before[1]) ** 2 + (z - before[2]) ** 2) ** 0.5
        if dist > LIMIT_TO_MM:
            raise SystemExit(f"目標離目前位置 {dist:.0f} mm，超過測試上限 {LIMIT_TO_MM} mm")
        roll = before[4] if a.roll is None else a.roll
        claw = before[5] if a.claw is None else a.claw
        payload = struct.pack("<6hH", round(a.pitch * 10), x, y, z, roll, claw, a.ms)
        frame = build_frame(SYSTEM_ID, CMD_COORDINATE_SET, payload)
        print(f"目標: x={x} y={y} z={z} pitch={a.pitch}° roll={roll} claw={claw} 時間 {a.ms} ms")
    else:
        for name in ("dx", "dy", "dz"):
            if abs(getattr(a, name)) > LIMIT_MM:
                raise SystemExit(f"--{name} 超過測試上限 {LIMIT_MM} mm")
        payload = struct.pack("<6hH", a.dx, a.dy, a.dz, round(a.dpitch * 10), a.droll, a.dclaw, a.ms)
        frame = build_frame(SYSTEM_ID, CMD_COORD_INCREMENT, payload)
    print("要送的封包:", frame.hex(" "))
    if not a.go:
        ser.close()
        print("（沒有加 --go，不送出）")
        return

    ser.write(frame)
    time.sleep(a.ms / 1000 + 0.5)
    after = read_status(ser)
    show("之後", after)
    ser.close()
    if before and after:
        print("變化: " + ", ".join(f"{k}{after[i] - before[i]:+d}" for i, k in enumerate(["x", "y", "z", "pitch", "roll", "claw"]))
              + " | " + ", ".join(f"{JOINTS[i].split()[0]}{after[6 + i] - before[6 + i]:+d}" for i in range(6)))


if __name__ == "__main__":
    main()
