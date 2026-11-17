#!/usr/bin/env python
"""手機 LiDAR 掃描（glTF/GLB/OBJ/PLY，公尺）→ 底面占據格（2026-11-12，使用者：「我是要建立一個底面就好了」）。

做法：找地板高度（頂點在垂直軸上最密的那一層），在離地 slice_heights 的幾個高度把網格切片，切出來的線段投影到地面，
畫進 cell 公尺的格子；再加上「地板以上 0.08–0.90 m 高度帶碰到的所有三角形」的投影：桌面、椅面整片投下來就是實心障礙
（2026-11-13 使用者：「桌椅要變成佔位圖，機器人過不了」——EP 加手臂約 0.45 m 高，桌子底下過不去，所以桌子整塊都算占據）。
輸出：<out>.png（俯視圖，1 m 格線）、<out>.json（占據格中心座標、地板高度、範圍、up 軸）。

用法：python -I tools/scan_floorplan.py scans/2026_11_12/2026_11_12.glb --out scans/2026_11_12/floorplan [--up y] [--cell 0.05]
"""
import argparse
import json
import math
import sys

import numpy as np
import trimesh

try:
    import cv2
except Exception:
    cv2 = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--up", default="auto", choices=["auto", "x", "y", "z"])
    ap.add_argument("--cell", type=float, default=0.05)
    ap.add_argument("--heights", default="0.10,0.20,0.30", help="離地的切片高度（公尺）")
    ap.add_argument("--band", default="0.08,0.90", help="這個離地高度帶碰到的三角形全部投影（桌面 0.75 m 也包進來 → 桌子整塊占據）")
    ap.add_argument("--close", type=int, default=2, help="形態學閉運算的格數（補桌面掃描的小洞），0 關閉")
    a = ap.parse_args()
    sc = trimesh.load(a.mesh, force="scene")
    m = sc.to_geometry() if hasattr(sc, "to_geometry") else sc.dump(concatenate=True)
    V = np.asarray(m.vertices, dtype=np.float64)
    ext = V.max(axis=0) - V.min(axis=0)
    up = {"x": 0, "y": 1, "z": 2}[a.up] if a.up != "auto" else int(np.argmin(ext))   # 最薄的軸 = 高度
    h = [i for i in range(3) if i != up]            # 地面兩軸（保持原順序）
    hist, edges = np.histogram(V[:, up], bins=int(ext[up] / 0.02) + 1)
    floor = float(edges[hist.argmax()] + 0.01)
    print("[floorplan] 單位公尺；up 軸 = %s，地板高度 %.2f，地面範圍 %s..%s" % ("xyz"[up], floor, np.round(V.min(axis=0)[h], 2), np.round(V.max(axis=0)[h], 2)))
    cell = a.cell
    lo = V.min(axis=0)[h] - 0.2
    hi = V.max(axis=0)[h] + 0.2
    W = int(math.ceil((hi[0] - lo[0]) / cell))
    H = int(math.ceil((hi[1] - lo[1]) / cell))
    occ = np.zeros((H, W), dtype=np.uint8)

    def to_cell(p):
        return int((p[0] - lo[0]) / cell), int((p[1] - lo[1]) / cell)

    # 1. 切片：每個高度一組線段
    normal = np.zeros(3); normal[up] = 1.0
    n_seg = 0
    for hz in [float(v) for v in a.heights.split(",")]:
        origin = np.zeros(3); origin[up] = floor + hz
        try:
            lines = trimesh.intersections.mesh_plane(m, normal, origin)
        except Exception as e:
            print("[floorplan] 切片 %.2f 失敗：%s" % (hz, e)); continue
        for seg in lines:
            p0, p1 = to_cell(seg[0][h]), to_cell(seg[1][h])
            if cv2 is not None:
                cv2.line(occ, p0, p1, 1, 1)
            n_seg += 1
    # 2. 高度帶「碰到」的三角形全部投影：桌面/椅面是水平的大片 → 投下來整塊占據；牆是垂直的 → 投成線；
    #    地板本身（max < floor+b0）跟天花板、高櫃上緣（min > floor+b1）不算
    b0, b1 = [float(v) for v in a.band.split(",")]
    F = np.asarray(m.faces)
    tri_h = V[F][:, :, up]
    inband = (tri_h.max(axis=1) >= floor + b0) & (tri_h.min(axis=1) <= floor + b1)
    for f in F[inband]:
        pts = np.array([to_cell(V[i][h]) for i in f], dtype=np.int32)
        if cv2 is not None:
            cv2.fillConvexPoly(occ, pts, 1)
    if a.close > 0 and cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.close + 1, 2 * a.close + 1))
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, k)
    ys, xs = np.nonzero(occ)
    cells = [[round(lo[0] + (x + 0.5) * cell, 3), round(lo[1] + (y + 0.5) * cell, 3)] for x, y in zip(xs, ys)]
    print("[floorplan] 切片線段 %d、高度帶三角形 %d、占據格 %d（%.1f%%）" % (n_seg, int(inband.sum()), len(cells), 100.0 * len(cells) / occ.size))
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"source": a.mesh, "cell_m": cell, "up_axis": "xyz"[up], "ground_axes": ["xyz"[h[0]], "xyz"[h[1]]],
                   "floor_height": floor, "origin": [float(lo[0]), float(lo[1])], "size": [W, H], "occupied": cells}, f)
    if cv2 is not None:
        scale = max(1, int(600 / max(W, H)))
        img = np.full((H, W, 3), 255, np.uint8)
        img[occ > 0] = (40, 40, 40)
        img = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
        # 1 m 格線 + 座標標籤（模型座標）
        for gx in range(int(math.floor(lo[0])), int(math.ceil(hi[0])) + 1):
            px = int((gx - lo[0]) / cell * scale)
            cv2.line(img, (px, 0), (px, H * scale), (200, 200, 255), 1)
            cv2.putText(img, "%d" % gx, (px + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        for gy in range(int(math.floor(lo[1])), int(math.ceil(hi[1])) + 1):
            py = int((gy - lo[1]) / cell * scale)
            cv2.line(img, (0, py), (W * scale, py), (200, 200, 255), 1)
            cv2.putText(img, "%d" % gy, (2, py - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        cv2.putText(img, "axes: %s (right)  %s (down)  floor %.2f" % ("xyz"[h[0]], "xyz"[h[1]], floor), (5, H * scale - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1)
        cv2.imwrite(a.out + ".png", img)
        print("[floorplan] 寫出 %s.png / .json" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
