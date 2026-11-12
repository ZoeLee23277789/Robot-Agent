"""
掃描 NexArm 真正到得了的區域，手臂完全不動。

做法：
  1. 對 (前伸 x, 高度 z, 夾爪俯仰 pitch) 的格點送 CMD 7（IKINE_RESULT_GET：只算逆運動學、不動，2026-11-09 實測伺服前後差 0），
     拿回六顆伺服的目標值。韌體對到不了的點也照樣回一組伺服值，不會說做不到，所以要第 2 步。
  2. 用自己的正運動學把伺服值算回末端位置，跟目標差 3 mm / 1° 以內才算到得了。
     FK 模型是用今晚 11 組「伺服值 ↔ AT32 回報的末端位置」擬合的平面三連桿，最大誤差 0.8 mm；
     擬合出的尺寸跟韌體 arm_kin 一致（肩高 110、連桿 227 / 145 / 130 mm）。
  3. 側向 y 靠底座（ID1）旋轉，跟可達範圍無關，所以只掃 y = 0 的平面；水平距離 r = sqrt(x² + y²) 代入同一張表即可。

不包含：碰撞（桌面、底座本身、夾著的東西）。z 低於約 30 mm 末端就碰到桌面（韌體搖桿模式下限 30）。

執行（server 要先關掉，兩個程式不能同時開同一個串列埠）：
    kill $(lsof -t -i:8765)
    ~/rmenv/bin/python Test_Robot/nexarm_workspace.py               # 約 2–3 分鐘，結果存 nexarm_workspace.json + .png
"""

import argparse
import json
import math
import struct
import time
from pathlib import Path

import serial

from nexarm import BAUD, SYSTEM_ID, build_frame, extract_frames

CMD_IKINE_RESULT_GET = 7
COUNTS_PER_DEG = 4096 / 360.0

# 平面 FK：p1 = o1 + d2，p2 = p1 + o2 + d3，pitch = c + d2 + d3 − d4（d_i = 伺服 i 離中位的角度）
FK = {"o1": 80.79, "o2": -80.39, "ax": -0.12, "az": 110.35, "L1": 227.45, "L2": 145.31, "L3": 129.79, "c": -0.04}

OUT = Path(__file__).with_name("nexarm_workspace.json")


def fk(s2, s3, s4):
    d2, d3, d4 = ((s - 2048) / COUNTS_PER_DEG for s in (s2, s3, s4))
    p1 = math.radians(FK["o1"] + d2)
    p2 = p1 + math.radians(FK["o2"] + d3)
    pitch = FK["c"] + d2 + d3 - d4
    p3 = math.radians(pitch)
    x = FK["ax"] + FK["L1"] * math.cos(p1) + FK["L2"] * math.cos(p2) + FK["L3"] * math.cos(p3)
    z = FK["az"] + FK["L1"] * math.sin(p1) + FK["L2"] * math.sin(p2) + FK["L3"] * math.sin(p3)
    return x, z, pitch


class IKClient(object):
    def __init__(self, port):
        self.port = port
        self.ser = None
        self.reconnects = 0
        self._open()

    def _open(self):
        for _ in range(30):     # USB 斷線重連要 2 秒左右（2026-11-09 掃到一半在 hub 上斷過一次）
            try:
                self.ser = serial.Serial(self.port, BAUD, timeout=0.01)
                time.sleep(0.3)
                self.ser.reset_input_buffer()
                self.buf = b""
                return
            except serial.SerialException:
                time.sleep(0.5)
        raise RuntimeError("串列埠 %s 15 秒內都開不起來" % self.port)

    def ik(self, x, z, pitch, timeout=0.15):
        for attempt in range(3):
            try:
                return self._ik(x, z, pitch, timeout)
            except serial.SerialException as e:
                self.reconnects += 1
                print("   串列埠出錯（%s），重新開啟後重試 (%d,%d)" % (e, x, z))
                try:
                    self.ser.close()
                except Exception:
                    pass
                time.sleep(1.0)
                self._open()
        return None

    def _ik(self, x, z, pitch, timeout):
        self.ser.write(build_frame(SYSTEM_ID, CMD_IKINE_RESULT_GET,
                                   struct.pack("<6h", int(round(pitch * 10)), int(x), 0, int(z), 0, 0)))
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.buf += self.ser.read(256)
            frames, self.buf = extract_frames(self.buf)
            for _d, cmd, args in frames:
                if cmd == CMD_IKINE_RESULT_GET and len(args) >= 24:
                    v = struct.unpack_from("<12h", args)
                    if v[0] == int(x) and v[2] == int(z):   # 對到這次的請求，不是上一筆遲到的回覆
                        return v[6:12]
        return None

    def close(self):
        self.ser.close()


def scan(port, pitches, x_range, z_range, step, tol_mm=3.0, tol_deg=1.0):
    cli = IKClient(port)
    result = {"pitches": {}, "step_mm": step, "x_range": x_range, "z_range": z_range, "fk": FK,
              "tol_mm": tol_mm, "scanned": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        for pitch in pitches:
            ok_cells, no_reply = [], 0
            t0 = time.time()
            for x in range(x_range[0], x_range[1] + 1, step):
                for z in range(z_range[0], z_range[1] + 1, step):
                    servos = cli.ik(x, z, pitch)
                    if servos is None:
                        no_reply += 1
                        continue
                    if not all(0 <= s <= 4095 for s in servos[:5]):
                        continue
                    fx, fz, fp = fk(*servos[1:4])
                    if math.hypot(fx - x, fz - z) <= tol_mm and abs(fp - pitch) <= tol_deg:
                        ok_cells.append([x, z])
            zs_by_x = {}
            for x, z in ok_cells:
                zs_by_x.setdefault(x, []).append(z)
            result["pitches"][str(pitch)] = {
                "reachable": ok_cells,
                "z_span_by_x": {str(x): [min(zs), max(zs)] for x, zs in sorted(zs_by_x.items())},
                "no_reply": no_reply,
            }
            print("pitch %+4d°: %4d reachable cells, %d no-reply, %.0fs" % (pitch, len(ok_cells), no_reply, time.time() - t0))
    finally:
        cli.close()
    return result


def draw(result, path):
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    step = result["step_mm"]
    (x0, x1), (z0, z1) = result["x_range"], result["z_range"]
    cell = 4
    pitches = list(result["pitches"].keys())
    w = ((x1 - x0) // step + 1) * cell
    h = ((z1 - z0) // step + 1) * cell
    panels = []
    colors = [(60, 160, 60), (200, 120, 40), (60, 60, 200), (150, 60, 150)]
    for i, p in enumerate(pitches):
        img = np.full((h + 30, w + 50, 3), 255, np.uint8)
        for x, z in result["pitches"][p]["reachable"]:
            cx = 40 + (x - x0) // step * cell
            cy = h - (z - z0) // step * cell
            cv2.rectangle(img, (cx, cy - cell), (cx + cell - 1, cy - 1), colors[i % len(colors)], -1)
        cy30 = h - (30 - z0) // step * cell
        cv2.line(img, (40, cy30), (40 + w, cy30), (0, 0, 0), 1)       # 桌面附近 z = 30
        for xv in range(0, x1 + 1, 100):
            cv2.putText(img, str(xv), (40 + (xv - x0) // step * cell - 8, h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        for zv in range(0, z1 + 1, 100):
            cv2.putText(img, str(zv), (2, h - (zv - z0) // step * cell + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(img, "pitch %s deg" % p, (45, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        panels.append(img)
    cv2.imwrite(str(path), np.hstack(panels))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/nexarm")
    ap.add_argument("--pitches", default="0,-45,-90", help="夾爪俯仰角（度），0 = 水平朝前、-90 = 垂直朝下")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--x", default="0,460")
    ap.add_argument("--z", default="-60,560")
    ap.add_argument("--out", default=str(OUT), help="結果檔；補掃別的範圍時換一個檔名，不要蓋掉 server 在用的表")
    a = ap.parse_args()
    pitches = [int(v) for v in a.pitches.split(",")]
    xr = [int(v) for v in a.x.split(",")]
    zr = [int(v) for v in a.z.split(",")]
    res = scan(a.port, pitches, xr, zr, a.step)
    out = Path(a.out)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    png = draw(res, out.with_suffix(".png"))
    print("寫入 %s%s" % (out, (" 和 %s" % png) if png else ""))
    for p, info in res["pitches"].items():
        spans = info["z_span_by_x"]
        print("pitch %s°：" % p)
        for x in range(0, xr[1] + 1, 50):
            s = spans.get(str(x))
            print("   前伸 x=%3d mm：高度 %s" % (x, ("%d ~ %d mm" % tuple(s)) if s else "到不了"))


if __name__ == "__main__":
    main()
