#!/usr/bin/env python
"""用一圈 ToF（runs/_tof_sweep.json）跟導航占據格（runs/_nav_map.png）比對，估出車在地圖上的位姿。
格點搜尋 x,y ±0.8 m（5 cm）、朝向 ±25°（1°），成本 = 各方向 |量到 − 地圖射線| 的截尾平均。"""
import json, math, sys
import numpy as np, cv2
img = cv2.imread("runs/_nav_map.png"); N = img.shape[0] // 4
small = cv2.resize(img, (N, N), interpolation=cv2.INTER_NEAREST)
gray = small.astype(int).sum(axis=2)
occ = gray < 200                      # 深色 = 牆/家具（外框也是深色）
CELL, HALF = 0.05, N * 0.05 / 2
def cell(x, y): return int((x + HALF) / CELL), int((HALF - y) / CELL)
def ray(x, y, th, rmax=7.0):
    c, s = math.cos(th), math.sin(th); t = 0.0
    while t < rmax:
        ix, iy = cell(x + c * t, y + s * t)
        if not (0 <= ix < N and 0 <= iy < N) or occ[iy, ix]: return t
        t += 0.025
    return rmax
sweep = [d for d in json.load(open("runs/_tof_sweep.json")) if d["tof_mm"] and d["tof_mm"] < 5500]
def cost(x, y, dth):
    errs = []
    for d in sweep:
        th = math.radians(d["heading_deg"] + dth)
        sx, sy = x + 0.16 * math.cos(th), y + 0.16 * math.sin(th)
        errs.append(abs(ray(sx, sy, th) - d["tof_mm"] / 1000.0))
    errs.sort(); k = max(3, int(len(errs) * 0.75))       # 丟掉最差的 25%（箱子、椅子、地圖沒有的東西）
    return sum(errs[:k]) / k
x0, y0 = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (0.0, 0.0)
best = []
for dth in range(-25, 26, 1):
    for ix in range(-16, 17):
        for iy in range(-16, 17):
            x, y = x0 + ix * 0.05, y0 + iy * 0.05
            cx, cy = cell(x, y)
            if occ[cy, cx]: continue
            best.append((cost(x, y, dth), x, y, dth))
best.sort()
print("beams used", len(sweep), "| cost at (0,0,0): %.3f m" % cost(0, 0, 0))
for c, x, y, dth in best[:6]: print("cost %.3f  x %+.2f  y %+.2f  heading offset %+d deg" % (c, x, y, dth))
c, x, y, dth = best[0]
print("per-beam at best:", [(round(d["heading_deg"]), round(d["tof_mm"] / 1000, 2), round(ray(x + 0.16 * math.cos(math.radians(d["heading_deg"] + dth)), y + 0.16 * math.sin(math.radians(d["heading_deg"] + dth)), math.radians(d["heading_deg"] + dth)), 2)) for d in sweep])
