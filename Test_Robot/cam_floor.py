# -*- coding: utf-8 -*-
"""前鏡頭的「地板邊界」測距（explore 模式，2026-11-12）。

為什麼：一顆 20° 的 ToF 每 10° 一筆，一圈只有 36 個距離，障礙的形狀全靠膨脹猜、錐角邊上的東西會被擺到中線上，
使用者看視窗版：「障礙物位置不對」。相機一張畫面橫跨約 100°，地板是均勻的灰色，把每一欄像素由下往上掃到第一段
「不是地板」的地方，那一列就對應一個地面距離——一張畫面就是幾百筆連續的障礙輪廓，等於一台用相機做的 LiDAR。

刻度不用手算：每一停 ToF 量到畫面正中央那一點的距離，拿 (中央欄的邊界列, ToF 距離) 這些配對擬合「列 → 公尺」的曲線
（d = a / (row - r0) + b，針孔相機對地面的投影就是這個形狀）。擬合結果存檔，之後直接用。

限制：只看得到「站在地板上」的東西（懸空桌面要靠 ToF）；地板上的反光、地毯邊會誤判，所以命中分數比 ToF 低。
"""
import json
import math
import os

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

CALIB_PATH = os.path.expanduser("~/.robot_cam_floor_calib.json")
MIN_RUN_PX = 14          # 連續這麼多像素不是地板才算邊界（反光、接縫會短暫跳出）
LAB_TOL = 14.0           # 跟地板參考色的 Lab 距離（L 佔權重較低，反光才不會整片被當障礙）
EDGE_COLS_STEP = 4       # 每 4 欄取一筆
HALF_FOV_USE_DEG = 28.0  # 只用中央 ±28°（鏡頭邊緣畸變大）
H_FOV_DEG = 104.0        # EP 相機 16:9 的水平視角（官方 120° 對角）約 104°


class FloorRanger(object):
    def __init__(self, calib_path=CALIB_PATH):
        self.calib_path = calib_path
        self.params = None       # (a, r0, b)：d_m = a / (row - r0) + b
        self.floor_lab = None    # 地板參考色（Lab）
        self.pairs = []          # 校正用 (row, d_m)
        self.blur_sigma = 2.0    # 模糊太重的話牆腳板的突變會變成漸變、floodFill 直接穿過去（sigma 5 時整張都是地板）
        self.fill_tol = 9        # 相鄰像素差的門檻
        self.load()

    # ------------------------------------------------------------ 校正
    def load(self):
        try:
            with open(self.calib_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.params = tuple(d["params"]) if d.get("params") else None
            self.floor_lab = np.array(d["floor_lab"], dtype=np.float32) if d.get("floor_lab") else None
            self.pairs = [tuple(p) for p in d.get("pairs", [])]
        except Exception:
            pass

    def save(self):
        try:
            with open(self.calib_path, "w", encoding="utf-8") as f:
                json.dump({"params": self.params, "floor_lab": None if self.floor_lab is None else [float(v) for v in self.floor_lab],
                           "pairs": self.pairs[-200:]}, f)
        except Exception:
            pass

    def add_pair(self, row, d_m):
        """畫面中央欄的地板邊界列 + ToF 量到的距離（從車身中心算）。"""
        if row is None or d_m is None or not (0.25 <= d_m <= 3.5):
            return
        self.pairs.append((float(row), float(d_m)))
        self.pairs = self.pairs[-200:]
        self.fit()

    def fit(self):
        """d = a/(row - r0) + b：r0 掃一遍（地平線在畫面上方某列），a、b 用最小平方。至少 6 筆、距離要有跨度。"""
        if len(self.pairs) < 6:
            return False
        rows = np.array([p[0] for p in self.pairs]); ds = np.array([p[1] for p in self.pairs])
        if ds.max() - ds.min() < 0.8:
            return False
        best = None
        for r0 in np.arange(-200.0, rows.min() - 2.0, 2.0):
            x = 1.0 / (rows - r0)
            A = np.vstack([x, np.ones_like(x)]).T
            sol, res, _, _ = np.linalg.lstsq(A, ds, rcond=None)
            a, b = sol
            if a <= 0:
                continue
            err = float(np.sqrt(np.mean((A @ sol - ds) ** 2)))
            if best is None or err < best[0]:
                best = (err, a, r0, b)
        if best is None:
            return False
        a, r0, b = best[1], best[2], best[3]
        if len(self.pairs) >= 10:
            # 穩健：ToF 常從桌腳之間看到更遠的東西，那種配對（殘差大）丟掉四分之一再擬合一次
            res = np.abs(a / (rows - r0) + b - ds)
            keep = res <= np.percentile(res, 75)
            rows2, ds2 = rows[keep], ds[keep]
            best2 = None
            for r0b in np.arange(-200.0, rows2.min() - 2.0, 2.0):
                x = 1.0 / (rows2 - r0b)
                A = np.vstack([x, np.ones_like(x)]).T
                sol, _, _, _ = np.linalg.lstsq(A, ds2, rcond=None)
                if sol[0] <= 0:
                    continue
                err = float(np.sqrt(np.mean((A @ sol - ds2) ** 2)))
                if best2 is None or err < best2[0]:
                    best2 = (err, sol[0], r0b, sol[1])
            if best2 is not None:
                best = best2
                a, r0, b = best[1], best[2], best[3]
        self.params = (float(a), float(r0), float(b))
        self.rms_m = float(best[0])
        self.save()
        return True

    def row_to_m(self, row):
        if self.params is None:
            return None
        a, r0, b = self.params
        if row <= r0 + 1:
            return None
        return a / (row - r0) + b

    # ------------------------------------------------------------ 地板分割
    def floor_mask(self, img):
        """回傳 bool 遮罩（True = 地板）。兩個條件同時成立才是地板：
        1. 顏色：跟地板參考色的色度（Lab 的 a、b）夠近、亮度在寬鬆範圍內——亮度不能卡太緊，木紋地板從陰影到反光 L 從 80 跑到 150。
        2. 質感：7×7 的局部標準差在 1.2–11 之間——木紋有紋理（3–8），牆面是平的（<1），牆腳板/物件邊緣很高（>12）。
        2026-11-12 用四張實拍量出來的：牆跟地板的色度只差 2–7，單靠顏色分不開；質感把牆、柱子、牆腳板全切掉。"""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = cv2.blur(g, (7, 7))
        sq = cv2.blur(g * g, (7, 7))
        std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        lab = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 3), cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = lab.shape[:2]
        patch = lab[int(h * 0.80):int(h * 0.95), int(w * 0.35):int(w * 0.65)].reshape(-1, 3)
        ref = np.median(patch, axis=0)
        if self.floor_lab is not None:
            if abs(ref[1] - self.floor_lab[1]) > 6 or abs(ref[2] - self.floor_lab[2]) > 6 or abs(ref[0] - self.floor_lab[0]) > 50:
                ref = self.floor_lab          # 正前方那一塊不是地板（貼著東西）：用存檔的參考色
            else:
                self.floor_lab = 0.9 * self.floor_lab + 0.1 * ref
        else:
            self.floor_lab = ref
        colour_ok = (np.abs(lab[:, :, 1] - ref[1]) <= 6) & (np.abs(lab[:, :, 2] - ref[2]) <= 6) & \
                    (lab[:, :, 0] >= ref[0] - 50) & (lab[:, :, 0] <= ref[0] + 45)
        texture_ok = (std >= 1.2) & (std <= 11.0)
        mask = (colour_ok & texture_ok).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # 補木紋裡的小洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))    # 去掉牆上零星像地板的點
        return mask.astype(bool)

    def edges(self, img):
        """每一欄（每 EDGE_COLS_STEP 欄）由下往上掃第一段 ≥ MIN_RUN_PX 的非地板 → 回傳 {col: row}；沒邊界的欄不在裡面。"""
        mask = self.floor_mask(img)
        h, w = mask.shape
        out = {}
        for c in range(0, w, EDGE_COLS_STEP):
            col = mask[:, c]
            run = 0
            edge = None
            for r in range(h - 1, -1, -1):
                if not col[r]:
                    run += 1
                    if run >= MIN_RUN_PX:
                        edge = r + MIN_RUN_PX - 1
                        break
                else:
                    run = 0
            if edge is not None and edge < h - 2:
                out[c] = edge
        return out

    def centre_edge_row(self, img):
        e = self.edges(img)
        w = img.shape[1]
        rows = [r for c, r in e.items() if abs(c - w // 2) <= 12]
        return float(np.median(rows)) if rows else None

    # ------------------------------------------------------------ 距離輪廓
    def profile(self, img):
        """回傳 [(bearing_left_deg, forward_m, lateral_left_m)]：以車身中心為原點、車頭為前。沒校正好就回 []。
        沒找到邊界的欄不回傳（可能是地板一路到地平線、也可能是懸空的東西，交給 ToF）。"""
        if self.params is None:
            return []
        h, w = img.shape[:2]
        fx = (w / 2.0) / math.tan(math.radians(H_FOV_DEG / 2.0))
        out = []
        for c, r in self.edges(img).items():
            bearing = math.degrees(math.atan((c - w / 2.0) / fx))
            if abs(bearing) > HALF_FOV_USE_DEG:
                continue
            fwd = self.row_to_m(r)
            # 0.4 m 內不用相機：畫面底部是車自己的影子跟夾爪尖端，每張都在同一列，會在車旁邊留一圈假障礙；近距離交給 ToF
            if fwd is None or fwd < 0.40 or fwd > 4.0 or r > h * 0.85:
                continue
            lat = -fwd * (c - w / 2.0) / fx     # 畫面右邊 = 車的右邊 = 負的「左」
            out.append((-bearing, fwd, lat))
        return out

    def debug_image(self, img):
        vis = img.copy()
        for c, r in self.edges(img).items():
            cv2.circle(vis, (c, r), 2, (0, 0, 255), -1)
        return vis
