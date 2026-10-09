# -*- coding: utf-8 -*-
"""
占據格地圖 + A* 路徑規劃（機器人端，跟 robot_server.py 一起跑在 rmenv）。

地圖直接從 CoppeliaSim 的場景幾何建：
    靜態層  開機時建一次——所有「靜態且可碰撞」的形狀，取世界座標的 bounding box 投影到地面。
    動態層  每次規劃前重讀——「非靜態且可碰撞」的形狀（泡棉方塊、小積木、球…），位置會被任務改變。
    高度規則  頂面低於 FLAT_MAX_Z 的（地墊、球池底板）可以壓過去；底面高於 CLEARANCE_Z 的
              （平台甲板、隧道頂、護欄）可以從底下鑽過；其餘都是障礙。
    膨脹      以機器人半徑膨脹一圈，規劃時把機器人當成一個點。
    通道      隧道這種「有頂、兩側是牆」的結構，膨脹會把 0.5 m 寬的內部整個封死，所以在
              膨脹後把隧道內部（用 tunnel_top 的投影往內縮牆厚）重新標成可走，牆本身仍是障礙。

真實機器人沒有 zmq 可以查幾何：這一層的介面只有 build_static()/refresh_dynamic() 兩個入口，
之後換成「俯視攝影機深度圖 → 離地高度 → 障礙遮罩」是同一種輸出（bool 陣列），其他部分不用動。
"""
import hashlib
import heapq
import math
import os
import tempfile

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

CELL = 0.05             # 格子邊長 (m)
EXTENT = 3.5            # 地圖涵蓋 ±EXTENT (m)，跟俯視攝影機的 7 m 正交視野一致
ROOM_HALF = 2.95        # 房間內側半寬：牆外的格子一律當障礙（不然南牆的門洞會讓規劃器把終點放到房間外）
N = int(round(2 * EXTENT / CELL))
FLAT_MAX_Z = 0.03       # 頂面低於這個高度 → 地墊/地貼，可壓過
CLEARANCE_Z = 0.30      # 底面高於這個高度 → 可從底下鑽過（EP 含手臂約 0.27 m）
# 兩層膨脹（回歸測試的結論）：單一半徑小了會擦到牆角（0.15：停在球池牆前 4 cm，一轉身就掃到）、
# 大了會把 0.47 m 的真實走廊封死（0.25：綠柱子角落規劃不到路）。
HARD_RADIUS = 0.22      # 硬膨脹（5 格 = 0.25 m）：決定「能不能走」。車身半對角線 0.20 m，4 格（0.19）在凸角與小球旁會被角落擦到；0.14 時到達率 100% 但碰撞暴增（貼著走、步進側偏就擦到）；
                        # 0.19 = 半寬 0.12 + 每步側偏餘裕，綠柱子角落 0.475 m 的走廊仍走得過
SWEEP_RADIUS = 0.32     # 轉身掃掠半徑 0.20 m + 安全邊：終點與轉彎處要求淨空 ≥ 這個值；路徑成本在這個距離內加重
TIGHT_RADIUS = 0.15     # 窄縫備援層：正常層無路時才用（3 格）；車身半寬 0.16，走這種路要小步慢走
ROBOT_RADIUS = HARD_RADIUS  # 舊名稱，快取鍵值等還有引用
PREFER_CLEAR = SWEEP_RADIUS
PASSAGE_LANE_W = 0.30   # 通道口外延伸區只留中線窄道（車寬 + 每側 3 cm）：斜著進隧道口會卡到牆端，先對正才能進
PASSAGE_WALL_T = 0.08   # 通道內部往內縮的量（牆厚 0.05 + 一點餘裕）
PASSAGE_EXTEND = 0.40   # 通道挖空區沿長軸向兩端延伸：不然牆端的膨脹會把 0.5 m 寬的開口夾成一格


def world_to_cell(x, y):
    ix = int(math.floor((x + EXTENT) / CELL))
    iy = int(math.floor((y + EXTENT) / CELL))
    return max(0, min(N - 1, ix)), max(0, min(N - 1, iy))


def cell_to_world(ix, iy):
    return -EXTENT + (ix + 0.5) * CELL, -EXTENT + (iy + 0.5) * CELL


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


class NavMap(object):
    def __init__(self, sim, robot_root):
        self.sim = sim
        self.robot_root = robot_root
        self.robot_tree = set(sim.getObjectsInTree(robot_root, sim.handle_all, 0)) | {robot_root}
        self.static = np.zeros((N, N), dtype=bool)
        self.dynamic = np.zeros((N, N), dtype=bool)
        self.passages = []          # 通道內部矩形（世界座標）與軸向，膨脹後重新標成可走；見 passage_axis_at
        self.passage_id = None      # 每格屬於哪個通道（-1 = 不在通道裡）
        self.inflated = None
        self.cost = None
        self.clearance = None
        self.static_shapes = 0
        self.dynamic_shapes = 0
        self.build_static()

    # ------------------------------------------------------------------ 幾何
    def _footprint(self, h):
        """形狀 h 的世界座標 bounding box：回傳 (地面投影多邊形 [(x,y)...], zmin, zmax)。"""
        sim = self.sim
        lo = [sim.getObjectFloatParam(h, p) for p in (sim.objfloatparam_objbbox_min_x,
                                                      sim.objfloatparam_objbbox_min_y,
                                                      sim.objfloatparam_objbbox_min_z)]
        hi = [sim.getObjectFloatParam(h, p) for p in (sim.objfloatparam_objbbox_max_x,
                                                      sim.objfloatparam_objbbox_max_y,
                                                      sim.objfloatparam_objbbox_max_z)]
        m = sim.getObjectMatrix(h, sim.handle_world)
        pts = []
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    pts.append(sim.multiplyVector(m, [x, y, z]))
        zs = [p[2] for p in pts]
        return [(p[0], p[1]) for p in pts], min(zs), max(zs)

    def _classify(self, zmin, zmax):
        """True = 這個形狀是地面上的障礙；False = 可以壓過去或從底下鑽過。"""
        if zmax < FLAT_MAX_Z:
            return False
        if zmin > CLEARANCE_Z:
            return False
        return True

    @staticmethod
    def _poly_cells(poly):
        return np.array([[world_to_cell(x, y)] for x, y in poly], dtype=np.int32)

    def _rasterize(self, grid, poly):
        cells = self._poly_cells(poly)
        hull = cv2.convexHull(cells)
        canvas = np.zeros((N, N), dtype=np.uint8)
        cv2.fillConvexPoly(canvas, hull, 1)
        # 薄牆（收納箱 2 cm 的側壁）四個角落可能落在同一欄/列，凸包退化成線段，fillConvexPoly 什麼都不畫，
        # 這面牆就從地圖上消失（第八輪回歸測試：bin_balls 西壁不在地圖上，A* 把車排進桶裡撞了 12 次）。
        # 補畫一格粗的輪廓線，保證每個形狀至少一格厚。
        cv2.polylines(canvas, [hull], True, 1, 1)
        grid |= canvas.astype(bool)

    def _shapes(self):
        sim = self.sim
        return [s for s in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0)
                if s not in self.robot_tree]

    def _cache_path(self):
        """靜態層快取檔：依場景路徑 + 場景檔修改時間命名。每個任務前 robot_server 都會重開、重建一次靜態層
        （約 1500 次 zmq 呼叫、10 秒），場景沒改的話直接讀快取，減少對 CoppeliaSim 的負載與啟動時間。"""
        try:
            scene = self.sim.getStringParam(self.sim.stringparam_scene_path_and_name)
            # headless 用相對路徑啟動（./coppeliaSim -h scenes/xxx.ttt）時，這裡拿到的是相對於 CoppeliaSim
            # 安裝目錄的路徑，從 robot_server 的工作目錄找不到檔案，修改時間就一直是 0，快取永遠不會失效：
            # 2026-10-07 改場景（加柱子、搬收納箱）存檔後，地圖照樣從舊快取載入。補成完整路徑；還是找不到就不用快取。
            if scene and not os.path.isabs(scene):
                scene = os.path.join(self.sim.getStringParam(self.sim.stringparam_application_path), scene)
            if not scene or not os.path.exists(scene):
                return None
            mtime = int(os.path.getmtime(scene))
            key = hashlib.md5(("%s|%d|%s|%s|%s|v4" % (scene, mtime, CELL, ROBOT_RADIUS, PASSAGE_EXTEND)).encode()).hexdigest()[:12]
            return os.path.join(tempfile.gettempdir(), "robot_nav_static_%s.npz" % key)
        except Exception:
            return None

    def build_static(self):
        sim = self.sim
        cache = self._cache_path()
        if cache and os.path.exists(cache):
            try:
                z = np.load(cache, allow_pickle=True)
                self.static = z["static"].astype(bool)
                self.passages = list(z["passages"].tolist())
                self.static_shapes = int(z["static_shapes"])
                self._mask_outside_room()
                self.compose()
                print("[nav] 靜態地圖從快取載入：%s" % cache)
                return
            except Exception as e:
                print("[nav] 快取讀取失敗（%s），重建靜態地圖" % e)
        self.static[:] = False
        self.passages = []
        n = 0
        for s in self._shapes():
            try:
                if not sim.getObjectInt32Param(s, sim.shapeintparam_respondable):
                    continue
                if not sim.getObjectInt32Param(s, sim.shapeintparam_static):
                    continue
                poly, zmin, zmax = self._footprint(s)
                alias = sim.getObjectAlias(s)
                if alias.endswith("_top") and zmin > CLEARANCE_Z:
                    # 有頂的通道（隧道）：頂面投影往內縮牆厚，就是通道內部
                    rects, axis_deg, center, half_len, half_w = self._passage_rects(poly, PASSAGE_WALL_T, PASSAGE_EXTEND, PASSAGE_LANE_W)
                    self.passages.append({"polys": rects, "poly": rects[0], "axis_deg": axis_deg,
                                          "center": center, "half_len": half_len, "half_w": half_w})
                if not self._classify(zmin, zmax):
                    continue
                self._rasterize(self.static, poly)
                n += 1
            except Exception:
                continue
        self.static_shapes = n
        self._mask_outside_room()
        self.compose()
        if cache:
            try:
                np.savez(cache, static=self.static, passages=np.array(self.passages, dtype=object),
                         static_shapes=n)
            except Exception as e:
                print("[nav] 快取寫入失敗（%s）" % e)

    def _mask_outside_room(self):
        lo, _ = world_to_cell(-ROOM_HALF, -ROOM_HALF)
        hi, _ = world_to_cell(ROOM_HALF, ROOM_HALF)
        self.static[:lo, :] = True
        self.static[hi + 1:, :] = True
        self.static[:, :lo] = True
        self.static[:, hi + 1:] = True

    def refresh_dynamic(self, exclude_xy=None, exclude_r=0.0):
        """重讀非靜態物件。exclude_xy/exclude_r：這次的目標物本身不算障礙（不然終點會被自己封住）。"""
        sim = self.sim
        self.dynamic[:] = False
        n = 0
        for s in self._shapes():
            try:
                if not sim.getObjectInt32Param(s, sim.shapeintparam_respondable):
                    continue
                if sim.getObjectInt32Param(s, sim.shapeintparam_static):
                    continue
                poly, zmin, zmax = self._footprint(s)
                if not self._classify(zmin, zmax):
                    continue
                if exclude_xy is not None:
                    cx = sum(p[0] for p in poly) / len(poly)
                    cy = sum(p[1] for p in poly) / len(poly)
                    if math.hypot(cx - exclude_xy[0], cy - exclude_xy[1]) <= exclude_r:
                        continue
                self._rasterize(self.dynamic, poly)
                n += 1
            except Exception:
                continue
        self.dynamic_shapes = n
        self.compose()

    @staticmethod
    def _passage_rects(poly, shrink_w, extend_l, lane_w):
        """通道頂面的投影 → 可走區：內部矩形（寬度往內縮牆厚）+ 兩端各一條中線窄道（長 extend_l、寬 lane_w）。"""
        arr = np.array(poly, dtype=np.float64)
        c = arr.mean(axis=0)
        w, v = np.linalg.eigh(np.cov((arr - c).T))
        u = v[:, int(np.argmax(w))]
        n = np.array([-u[1], u[0]])
        half_l = float(np.abs((arr - c) @ u).max())
        half_w = max(0.05, float(np.abs((arr - c) @ n).max()) - shrink_w)

        def rect(center, hl, hw):
            return [tuple(center + su * u * hl + sn * n * hw) for su in (-1, 1) for sn in (-1, 1)]

        inner = rect(c, half_l + 0.02, half_w)
        lane_a = rect(c - u * (half_l + extend_l / 2), extend_l / 2 + 0.02, lane_w / 2)
        lane_b = rect(c + u * (half_l + extend_l / 2), extend_l / 2 + 0.02, lane_w / 2)
        return [inner, lane_a, lane_b], math.degrees(math.atan2(u[1], u[0])), (float(c[0]), float(c[1])), half_l, half_w

    def compose(self):
        occ = self.static | self.dynamic
        r = int(math.ceil(HARD_RADIUS / CELL))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        inflated = cv2.dilate(occ.astype(np.uint8), kernel).astype(bool)
        rt = int(math.ceil(TIGHT_RADIUS / CELL))
        kernel_t = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rt + 1, 2 * rt + 1))
        inflated_tight = cv2.dilate(occ.astype(np.uint8), kernel_t).astype(bool)  # 窄縫備援層
        # 通道內部與兩端窄道重新標成可走（牆本身留在 occ 裡，所以車不會真的撞牆，只是不被膨脹封死）
        passage_id = np.full((N, N), -1, dtype=np.int32)
        dist = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 3) * CELL
        for i, pg in enumerate(self.passages):
            a = math.radians(pg["axis_deg"])
            u = np.array([math.cos(a), math.sin(a)])
            nrm = np.array([-u[1], u[0]])
            c = np.array(pg["center"], dtype=np.float64)
            hl = float(pg["half_len"])
            hw = float(pg.get("half_w", 0.0)) + PASSAGE_WALL_T + 0.05
            # 通道自己的牆不算「別的障礙」，但其他東西（隧道北口旁的球桶 bin_balls，西壁離軸線只有 15.5 cm）
            # 在窄道裡照樣要膨脹：以前整條窄道都挖成可走，A* 把路排到窄道邊緣，車一到就擦到桶
            # （第八輪回歸測試 bin_balls 被撞 12 次）。窄道裡的格子仍標 passage_id（通道模式照常接手），
            # 只是不再把離其他障礙 < HARD_RADIUS 的格子從膨脹層裡挖掉。
            own_poly = [tuple(c + su * u * (hl + 0.05) + sn * nrm * hw) for su in (-1, 1) for sn in (-1, 1)]
            own = np.zeros((N, N), dtype=np.uint8)
            cv2.fillConvexPoly(own, cv2.convexHull(self._poly_cells(own_poly)), 1)
            others = occ & ~own.astype(bool)
            infl_others = cv2.dilate(others.astype(np.uint8), kernel).astype(bool)
            # 中線側向偏移：在 ±12 cm 內挑「整條線（含兩端窄道）離最近真實障礙最遠」的那條。
            # 距離用「點到占據方塊邊緣」算（距離轉換只有格心解析度，偏 1 cm 跟偏 5 cm 算出來一樣，
            # 實際差 4 cm 的餘裕）。隧道北口貼著球桶，正中線車身右緣正好碰到桶；往左偏 2-3 cm 兩側各剩 ~18 cm。
            L = hl + PASSAGE_EXTEND - 0.05
            win = int((L + 0.6) / CELL) + 2
            cx0, cy0 = world_to_cell(float(c[0]), float(c[1]))
            y0w, x0w = max(0, cy0 - win), max(0, cx0 - win)
            ys_, xs_ = np.nonzero(occ[y0w:cy0 + win + 1, x0w:cx0 + win + 1])
            ox = (xs_ + x0w) * CELL - EXTENT   # 方塊左下角
            oy = (ys_ + y0w) * CELL - EXTENT
            samples = np.arange(-L, L + 1e-6, 0.05)
            best = None
            for k in range(-12, 13):
                d = k * 0.01
                pts = c[None, :] + samples[:, None] * u[None, :] + d * nrm[None, :]
                if len(ox) == 0:
                    m = 9.9
                else:
                    ddx = np.maximum(np.maximum(ox[None, :] - pts[:, 0:1], pts[:, 0:1] - (ox[None, :] + CELL)), 0.0)
                    ddy = np.maximum(np.maximum(oy[None, :] - pts[:, 1:2], pts[:, 1:2] - (oy[None, :] + CELL)), 0.0)
                    m = float(np.sqrt(ddx * ddx + ddy * ddy).min())
                if best is None or (m, -abs(d)) > (best[0], -abs(best[1])):
                    best = (m, d)
            pg["lateral"] = best[1]
            pg["min_clearance"] = best[0]
            pg["center_line"] = (float(c[0] + nrm[0] * best[1]), float(c[1] + nrm[1] * best[1]))
            # 沿偏移後中線 ±6 cm 的窄條：一律挖通（不然中線被球桶的膨脹層蓋住，A* 到不了吸附上去的終點）
            cl = c + nrm * best[1]
            Lf = hl + PASSAGE_EXTEND
            strip_poly = [tuple(cl + su * u * Lf + sn * nrm * 0.06) for su in (-1, 1) for sn in (-1, 1)]
            strip = np.zeros((N, N), dtype=np.uint8)
            cv2.fillConvexPoly(strip, cv2.convexHull(self._poly_cells(strip_poly)), 1)
            cv2.polylines(strip, [cv2.convexHull(self._poly_cells(strip_poly))], True, 1, 1)
            strip = strip.astype(bool) & ~occ
            for poly in pg.get("polys", [pg["poly"]]):
                canvas = np.zeros((N, N), dtype=np.uint8)
                cv2.fillConvexPoly(canvas, cv2.convexHull(self._poly_cells(poly)), 1)
                full = canvas.astype(bool) & ~occ
                carve = (full & ~infl_others) | (strip & full)
                inflated &= ~carve
                inflated_tight &= ~carve
                passage_id[full] = i
        self.passage_id = passage_id
        self.occ = occ
        self.inflated = inflated
        self.inflated_tight = inflated_tight
        # 成本看「離真實障礙多遠」（用 occ，不用挖空後的 inflated）：通道挖空區裡牆邊的格子雖然可走，
        # 還是要比中線貴，不然 A* 會貼著隧道牆角切過去（回歸測試量到多次擦到 tunnel_left）。
        self.clearance = dist                  # 每格離最近真實障礙的距離 (m)，通道出入口挑側向偏移時用
        penalty = np.clip((PREFER_CLEAR - dist) / PREFER_CLEAR, 0.0, 1.0)
        self.cost = 1.0 + 10.0 * penalty       # 掃掠半徑以內的格子成本最高 11 倍：能繞就繞，繞不了才擠過去

    def passage_axis_at(self, x, y):
        """(x, y) 在某個通道裡的話回傳該通道的長軸方位角（度，方向不定，可 +180），否則 None。"""
        info = self.passage_info_at(x, y)
        return info["axis_deg"] if info else None

    def passage_info_at(self, x, y):
        """(x, y) 所在通道的完整資訊（axis_deg / center / half_len：牆體本身的半長，不含兩端窄道），否則 None。"""
        ix, iy = world_to_cell(x, y)
        pid = int(self.passage_id[iy, ix]) if self.passage_id is not None else -1
        return self.passages[pid] if pid >= 0 else None

    def snap_to_passage_axis(self, x, y):
        """(x, y) 在通道裡的話，投影到通道的中心軸線上（通道模式只能沿軸走，偏在一側的終點永遠到不了）。"""
        ix, iy = world_to_cell(x, y)
        pid = int(self.passage_id[iy, ix]) if self.passage_id is not None else -1
        if pid < 0:
            return x, y
        pg = self.passages[pid]
        cx, cy = pg.get("center_line", pg["center"])  # 側向偏移過的中線（避開隧道口旁的球桶）
        a = math.radians(pg["axis_deg"])
        ux, uy = math.cos(a), math.sin(a)
        t = (x - cx) * ux + (y - cy) * uy
        return cx + ux * t, cy + uy * t

    def passage_line_near(self, x, y, max_dist_m=0.8):
        """離 (x, y) 最近（中心距離 ≤ max_dist_m）的通道資料（含 center_line / lateral），沒有就 None。"""
        best = None
        for pg in self.passages:
            d = math.hypot(pg["center"][0] - x, pg["center"][1] - y)
            if d <= max_dist_m and (best is None or d < best[0]):
                best = (d, pg)
        return best[1] if best else None

    def clearance_at(self, x, y):
        ix, iy = world_to_cell(x, y)
        return float(self.clearance[iy, ix]) if self.clearance is not None else 0.0

    def best_lateral(self, x, y, ux, uy, max_off=0.15, step=0.05):
        """通道出入口的側向微調：沿垂直於軸的方向在 ±max_off 內挑離真實障礙最遠的點
        （隧道北口正貼著球桶，走正中線只剩 3 cm，往另一側偏 15 cm 就不會擦到）。"""
        nx, ny = -uy, ux
        best = (self.clearance_at(x, y), x, y)
        k = -max_off
        while k <= max_off + 1e-9:
            cx, cy = x + nx * k, y + ny * k
            c = self.clearance_at(cx, cy)
            if c > best[0] + 1e-6:
                best = (c, cx, cy)
            k += step
        return best[1], best[2]

    def mark_blocked(self, x, y, r_cells=2):
        """把 (x, y) 附近一小塊標成動態障礙——但不標通道格：通道裡 ToF 看到的多半是牆，標了就把唯一的路封死。"""
        ix, iy = world_to_cell(x, y)
        y0, y1, x0, x1 = max(0, iy - r_cells), iy + r_cells + 1, max(0, ix - r_cells), ix + r_cells + 1
        block = np.ones((y1 - y0, x1 - x0), dtype=bool)
        if self.passage_id is not None:
            block &= self.passage_id[y0:y1, x0:x1] < 0
        if not block.any():
            return False
        self.dynamic[y0:y1, x0:x1] |= block
        self.compose()
        return True

    # ------------------------------------------------------------------ 規劃
    def is_free(self, ix, iy):
        return 0 <= ix < N and 0 <= iy < N and not self.inflated[iy, ix]

    def nearest_free(self, ix, iy, max_r_m, min_clear=0.0):
        """最近的可走格；min_clear > 0 時優先找淨空 ≥ min_clear 的格（終點要能轉身），找不到再退回任何可走格。"""
        def good(x, y):
            return min_clear <= 0 or self.clearance[y, x] >= min_clear or self.passage_id[y, x] >= 0
        if self.is_free(ix, iy) and good(ix, iy):
            return ix, iy
        R = int(math.ceil(max_r_m / CELL))
        best, best_d, fallback, fb_d = None, None, None, None
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                x, y = ix + dx, iy + dy
                if not self.is_free(x, y):
                    continue
                d = dx * dx + dy * dy
                if good(x, y):
                    if best is None or d < best_d:
                        best, best_d = (x, y), d
                elif fallback is None or d < fb_d:
                    fallback, fb_d = (x, y), d
        return best or fallback

    def plan(self, start_xy, goal_xy, start_snap_m=0.5, goal_snap_m=1.0):
        """回傳 (waypoints [(x,y)...] 不含起點, actual_goal_xy, info) 或 (None, None, reason)。"""
        s = self.nearest_free(*world_to_cell(*start_xy), max_r_m=start_snap_m)
        if s is None:
            return None, None, "robot is boxed in (no free cell within %.1fm)" % start_snap_m
        g = self.nearest_free(*world_to_cell(*goal_xy), max_r_m=goal_snap_m, min_clear=SWEEP_RADIUS)
        if g is None:
            return None, None, "goal is inside an obstacle and no free cell within %.1fm" % goal_snap_m
        g = world_to_cell(*self.snap_to_passage_axis(*cell_to_world(*g)))
        path = self._astar(s, g)
        tight = False
        if path is None:
            path = self._astar(s, g, tight=True)  # 窄縫備援：膨脹只留 0.15 m 再找一次
            tight = path is not None
            if path is None:
                return None, None, "no path found (blocked)"
        self._tight_mode = tight
        try:
            pts = [cell_to_world(*c) for c in self._simplify(path)]
        finally:
            self._tight_mode = False
        if pts and math.hypot(pts[0][0] - start_xy[0], pts[0][1] - start_xy[1]) < CELL:
            pts = pts[1:]
        return pts, cell_to_world(*g), {"cells": len(path), "waypoints": len(pts), "tight": tight}

    def _astar(self, s, g, tight=False):
        sx, sy = s
        gx, gy = g
        cost = self.cost
        inflated = self.inflated_tight if tight else self.inflated
        if tight:
            cost = cost + np.where(self.inflated, 30.0, 0.0)  # 能繞就繞，只有必要的窄縫才用到 0.15 m 的餘裕
        openq = [(0.0, 0, sx, sy)]
        gscore = {s: 0.0}
        came = {}
        counter = 0
        nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
        while openq:
            _, _, x, y = heapq.heappop(openq)
            if (x, y) == g:
                path = [(x, y)]
                while (x, y) in came:
                    x, y = came[(x, y)]
                    path.append((x, y))
                return path[::-1]
            gc = gscore[(x, y)]
            for dx, dy, w in nbrs:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < N and 0 <= ny < N) or inflated[ny, nx]:
                    continue
                if dx and dy and (inflated[y, nx] or inflated[ny, x]):
                    continue  # 不切障礙物的角
                ng = gc + w * cost[ny, nx]
                if ng < gscore.get((nx, ny), float("inf")):
                    gscore[(nx, ny)] = ng
                    came[(nx, ny)] = (x, y)
                    counter += 1
                    h = math.hypot(nx - gx, ny - gy)
                    heapq.heappush(openq, (ng + h, counter, nx, ny))
        return None

    def _line_free(self, a, b, min_clear=0.0):
        """a→b 的直線沒碰到膨脹層，而且（通道外）每一格離真實障礙 ≥ min_clear。"""
        x0, y0 = a
        x1, y1 = b
        n = max(abs(x1 - x0), abs(y1 - y0))
        tight = getattr(self, "_tight_mode", False)
        layer = self.inflated_tight if tight else self.inflated
        for i in range(n + 1):
            t = i / float(n) if n else 0.0
            x = int(round(x0 + (x1 - x0) * t))
            y = int(round(y0 + (y1 - y0) * t))
            if layer[y, x]:
                return False
            if min_clear > 0 and self.passage_id[y, x] < 0 and self.clearance[y, x] < min_clear:
                return False
        return True

    def _simplify(self, path):
        """視線可達就直接連線，把一格一格的路徑縮成幾個轉折點。
        轉折點是機器人原地轉身的地方，優先挑淨空 ≥ 掃掠半徑的格子（貼著障礙的轉角轉身會掃到）。"""
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        last = len(path) - 1
        cl = np.array([self.clearance[y, x] for x, y in path], dtype=np.float64)
        while i < last:
            j = last
            # 直線段要離真實障礙 ≥ HARD_RADIUS + 5 cm；走廊本身就更窄時放寬到走廊的淨空（不然每一格都變轉折點）
            while j > i + 1:
                seg_min = float(cl[i:j + 1].min())
                if self._line_free(path[i], path[j], min_clear=min(HARD_RADIUS + 0.05, seg_min)):
                    break
                j -= 1
            if j != last:
                k = j
                while k > i + 1 and self.clearance[path[k][1], path[k][0]] < SWEEP_RADIUS:
                    k -= 1
                if k > i + 1:
                    j = k
            out.append(path[j])
            i = j
        return out

    # ------------------------------------------------------------------ 結構幾何（drive_through 用）
    def group_axis(self, root_handle):
        """某個地標群組（例如 /tunnel、/platform）的地面投影：回傳 (中心, 長軸單位向量, 半長, 半寬)。"""
        sim = self.sim
        pts = []
        shapes = sim.getObjectsInTree(root_handle, sim.object_shape_type, 0)
        if not shapes:
            # 地墊的表面形狀沒掛在群組底下（建場景腳本的非碰撞零件另外放），用「群組名_」前綴在整個場景找
            prefix = sim.getObjectAlias(root_handle) + "_"
            shapes = [s for s in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0)
                      if sim.getObjectAlias(s).startswith(prefix)]
        for only_respondable in (True, False):   # 地墊這種沒有可碰撞零件的地標退回用全部形狀
            for s in shapes:
                try:
                    if only_respondable and not sim.getObjectInt32Param(s, sim.shapeintparam_respondable):
                        continue
                    poly, _, _ = self._footprint(s)
                    pts.extend(poly)
                except Exception:
                    continue
            if pts:
                break
        if not pts:
            return None
        arr = np.array(pts, dtype=np.float64)
        c = arr.mean(axis=0)
        cov = np.cov((arr - c).T)
        w, v = np.linalg.eigh(cov)
        axis = v[:, int(np.argmax(w))]
        proj = (arr - c) @ axis
        perp = (arr - c) @ np.array([-axis[1], axis[0]])
        return (float(c[0]), float(c[1])), (float(axis[0]), float(axis[1])), \
            float(proj.max() - proj.min()) / 2.0, float(perp.max() - perp.min()) / 2.0

    # ------------------------------------------------------------------ 除錯圖
    def render_png(self, robot_pose=None, path=None, goal=None, scale=4):
        if cv2 is None:
            return None
        img = np.full((N, N, 3), 255, dtype=np.uint8)
        img[self.inflated] = (235, 210, 170)      # 膨脹層：淡藍（BGR）
        img[self.static] = (40, 40, 40)           # 靜態障礙：黑
        img[self.dynamic] = (60, 140, 250)        # 動態障礙：橘
        img = cv2.resize(img, (N * scale, N * scale), interpolation=cv2.INTER_NEAREST)

        def px(x, y):
            ix, iy = world_to_cell(x, y)
            return int((ix + 0.5) * scale), int((iy + 0.5) * scale)

        if path:
            pts = [px(x, y) for x, y in path]
            if robot_pose is not None:
                pts = [px(robot_pose[0], robot_pose[1])] + pts
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, (60, 180, 60), 2)
        if goal is not None:
            cv2.circle(img, px(*goal), 6, (0, 0, 220), 2)
        if robot_pose is not None:
            x, y, hd = robot_pose
            p = px(x, y)
            q = px(x + 0.3 * math.cos(math.radians(hd)), y + 0.3 * math.sin(math.radians(hd)))
            cv2.circle(img, p, 5, (0, 0, 220), -1)
            cv2.arrowedLine(img, p, q, (0, 0, 220), 2, tipLength=0.4)
        img = cv2.flip(img, 0)  # +y 朝上
        ok, buf = cv2.imencode(".png", img)
        return buf.tobytes() if ok else None
