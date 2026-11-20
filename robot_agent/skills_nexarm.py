"""
實體場地（RoboMaster EP + Hiwonder NexArm）專用的技能：pick_from_top / place_on_top / survey_scene / return_home。

跟 skills.py（遊戲場模擬：地圖導航 + 前向 ToF 送進指間的 pick/place）分開放（2026-11-16 整理）：兩邊的做法完全不同——
這裡沒有物件的世界座標，全部靠手臂上的相機、實體 ToF 與手臂回報；遊戲場那邊靠占據格地圖與模擬器真值。
分開之後改實體技能不會動到遊戲場的程式（11-15 為實體加的地圖防撞曾讓遊戲場導航整個卡住）。
這個檔案只從 skills.py 借兩個共用的小工具（_plain_label、_save_frame）；skills.py 不依賴這裡。

service.py 的動作處理器直接呼叫這裡的實作。
"""

import asyncio
import time
import json
import os
import math
import re
from typing import Optional, Tuple

from robot_agent import perception
from robot_agent.skills import _plain_label, _save_frame

# ---- 從高處頂面抓取（2026-11-13，使用者：「拿箱子上面的小瓶子」）----
# 舊 pick 靠底盤高度的前向 ToF 把東西送進指間；箱頂的瓶子 ToF 只看得到箱子正面，所以改用「夾爪上的相機 + 手臂幾何」：
# 現行做法（pick_from_top）：腕關節 90° 的觀察姿勢 → 視覺模型指出物件 → 相對「同一張畫面裡的夾爪」對準 → 伸過去合爪
# （NexArm 夾爪伺服卡住 = 夾到）→ 抬起、晃一下確認 → 水平搬運姿勢。相機在上臂、角度會跑，所以不用固定的像素/俯仰校正。
# 幾何：NexArm 底座平面離地 0.162 m、在車身正中。
NEX_BASE_H_M = 0.162
# 夾爪水平（pitch 0 / −20 查的是同一張表）時「前伸 r mm 最低到得了的高度 z mm」，取自 nexarm_workspace.json（REACH.z_span）。
# 越低要伸越遠：前伸 150 最低 200，伸到 325 才到得了 30。2026-11-14 使用者把箱子放倒（站著 0.35 m 抓的時候相機被短排線扯掉），
# 矮箱子的夾取高度常低於 150，要用這張表挑得到的前伸距離。
_NEX_HORIZ_MIN_Z = ((150, 200), (175, 190), (200, 180), (225, 160), (250, 140), (275, 110), (300, 70), (325, 30))


# EP 相機掛在夾爪下面：相機殼最低點在夾爪中心（arm_to 的高度）下方多少。2026-11-14 實測推回來的估計：夾爪中心在
# 箱頂上方 6–7 cm 往前伸時，相機正對箱子側面的條碼、把 0.16 m 的紙箱往前推（runs/_skill_frames/20261114_132839_*），
# 0.35 m 那次「抓取時相機掉下來」也是相機殼撞到箱子前緣。夾取/放置高度一律讓相機下緣離箱頂 ≥2 cm。量到實際值用 env 改。
# 2026-11-14 下午：相機改裝到手臂上節，夾爪下面只剩夾爪本體（約 3 cm）。相機又裝回夾爪下面的話設 0.075。
NEX_CAM_BOTTOM_M = float(os.environ.get("ROBOT_NEX_CAM_BOTTOM_M", "0.03"))


def _face_angle(samples):
    """前方感測器在幾個朝向量到的距離 → 箱面歪幾度。samples = [(相對朝向 度, 左為正, 距離 mm), ...]（至少 3 筆）。
    回 (phi 度, corner)：phi > 0 = 箱面左邊比較遠（要往左繞、車頭往右轉才會正對）；corner = 中間那點比兩邊連線近 2.5 cm 以上（對著的是箱角）。"""
    pts = [((r / 1000.0 + 0.16) * math.cos(math.radians(a)), (r / 1000.0 + 0.16) * math.sin(math.radians(a))) for a, r in samples]
    mx, my = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    den = sum((p[1] - my) ** 2 for p in pts)
    m = sum((p[1] - my) * (p[0] - mx) for p in pts) / den if den > 1e-9 else 0.0
    by_y = sorted(pts, key=lambda p: p[1])
    mid, lo, hi = by_y[len(by_y) // 2], by_y[0], by_y[-1]
    t = (mid[1] - lo[1]) / (hi[1] - lo[1]) if abs(hi[1] - lo[1]) > 1e-6 else 0.5
    chord_x = lo[0] + t * (hi[0] - lo[0])
    return math.degrees(math.atan(m)), (chord_x - mid[0]) > 0.025


async def _square_to_box(robot, front_tof, notes):
    """把車擺到「正對箱子的一個平面」（2026-11-19 使用者：「他不能直角面對箱子嗎？然後正正的放」）。
    今天三趟放置都掉：車是照相機看到的箱子方向直直開過去的，從夾取的位置過去剛好對著箱角或斜對著箱面，
    之後「感測器距離 + 7 cm」「往旁邊找空位」全部以為箱面是正的，算出來的位置就在邊上或外面。
    做法：原地左右各轉 7°，前方感測器量三個距離 → 三個點連成的線就是箱面的方向，知道它歪幾度（phi）。
    然後繞著感測器打到的那一點走到箱面的正前方：橫移 D·sin(phi)、前進 D·(1−cos(phi))、車頭轉 phi（麥輪可以橫著走）。
    走完再量一次，最多三輪，歪 7° 以內就算正。對著的是箱角（中間最近、兩邊都遠）就往比較正的那一面橫移 12 cm 再量。
    回 True = 已經正對；False = 量不出來（照舊繼續，後面還有用碰的確認）。"""
    def hd(st_):
        return st_.get("real_world_heading_deg", st_.get("world_heading_deg"))

    async def read():
        await asyncio.sleep(0.7)
        vs = []
        for _ in range(3):
            v = front_tof(await robot.state())
            if isinstance(v, (int, float)) and 60 < v < 1200:
                vs.append(float(v))
            await asyncio.sleep(0.15)
        return (sorted(vs)[len(vs) // 2] if vs else None), hd(await robot.state())

    async def turn(deg):
        await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(float(deg), 1), "push": False, "near_ok": True})

    def diff(a, b):
        return (((a - b) + 180.0) % 360.0) - 180.0

    log = []
    for it in range(3):
        r0, h0 = await read()
        if r0 is None or r0 > 900 or not isinstance(h0, (int, float)):
            notes.append(f"could not square up to the box (front sensor reads {r0} mm): going on as it is")
            return False
        samples, missed = [(0.0, r0)], []
        for step, name in ((9.0, "left"), (-18.0, "right")):
            await turn(step)
            r, h = await read()
            a = diff(h, h0) if isinstance(h, (int, float)) else (9.0 if name == "left" else -9.0)
            if r is not None and r < r0 + 250.0:
                samples.append((a, r))
            else:
                missed.append(name)
        _, h = await read()
        back = diff(h0, h) if isinstance(h, (int, float)) else 9.0
        if abs(back) > 1.5:
            await turn(back)
        if missed:
            # 轉 7–9° 就打不到箱子 = 那一邊已經超出箱面：往還打得到的那一邊橫移 10 cm 再量；兩邊都打不到 = 箱面很窄（窄的那一頭），不量了
            if len(missed) == 2 or it == 2:
                log.append("the face in front is too narrow to measure its angle")
                break
            await _move_exact(robot, 0.0, 0.10 if missed[0] == "left" else -0.10, tol=0.03)
            log.append(f"the box ends on the {missed[0]}: slid 10 cm to the {'right' if missed[0] == 'left' else 'left'}")
            continue
        phi, corner = _face_angle(samples)
        log.append("face " + " ".join(f"{a:+.0f}deg:{r:.0f}mm" for a, r in sorted(samples)) + f" -> off by {phi:+.0f} deg" + (" (a corner)" if corner else ""))
        if corner and it < 2:
            left_r = max(samples, key=lambda q: q[0])[1]
            right_r = min(samples, key=lambda q: q[0])[1]
            await _move_exact(robot, 0.0, -0.12 if left_r < right_r else 0.12, tol=0.03)      # 往讀值比較近（比較正）的那一面移
            continue
        if abs(phi) <= 7.0:
            notes.append("squared up to the box: " + "; ".join(log))
            return True
        phi = max(-40.0, min(40.0, phi))
        d = r0 / 1000.0 + 0.16
        left = max(-0.35, min(0.35, d * math.sin(math.radians(phi))))
        await _move_exact(robot, d * (1.0 - math.cos(math.radians(phi))), -left, tol=0.03)
        await turn(-phi)
    notes.append("tried to square up to the box: " + "; ".join(log))
    return True


async def _creep_to_surface(robot, front_tof, notes, what, target_mm=110, max_step=0.20):
    """底盤用前向 ToF 貼到箱子前 ~11 cm。只貼一次、最多 20 cm；開完 ToF 沒跟著變小 = 箱子被推走了（或感測器沒在看它），
    回 (tof, 錯誤訊息)。2026-11-14：舊版每次呼叫都 push=True 開 (tof-110)，箱子被相機推走後 ToF 變大，三次技能一路把箱子推了 40 cm。"""
    tof = front_tof(await robot.state())
    # 15:36 agent 實跑：approach 回報的 tof_front_mm 是 min(模擬, 實體)=214，實體其實 370，舊門檻 350 直接拒絕了四次。放寬到 600，分兩段貼近
    if tof is None or tof == 9999 or tof > 700:     # 700：_drive_to_visible 的「到了」門檻是 600，讀值抖一下（o_h2 讀到 614）不該整個拒絕
        return tof, (f"{what}: the REAL front distance sensor reads {tof} mm, so the box is not right in front of the robot "
                     f"(tof_front_mm in the telemetry can be the simulated value: look at real_tof_front_mm). "
                     + ("Call place_on_top again: it finds the box and drives to it by itself. Do NOT recenter_arm, face or approach while "
                        "carrying: in those arm poses the held object hides the distance sensor." if what == "place_on_top" else
                        "Turn to face the box (face('<the box>')) and call this again; do not use pick() instead."))
    for _ in range(6):      # 11-17：一步只走 7–10 cm，從 640 mm 開始三步只到 339，手臂搆不到箱子、東西放在箱子前面的地上
        if tof <= target_mm + 40:      # 底盤走不出 4 cm 以下的小步（實測送 0.02 m 完全沒動）
            return tof, None
        step = min(max_step, (tof - target_mm) / 1000.0)
        # 17:30 使用者：「你直接撞到箱子了，你不是有防撞的 sensor 嗎？」——舊版這裡送 push=True（關掉伺服器的 ToF 防撞）一次衝 0.30 m
        # 到離箱子 6 cm，煞不住就撞上。現在一律不送 push：伺服器會自己停在障礙前 0.15 m，剩下的距離由手臂伸過去。
        r = await robot.act("move_chassis", {"forward_m": round(step, 2), "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
        # 2026-11-18：0.4 s 太早。手臂舉高時送 0.10 m，走完那一刻讀 202 mm，之後兩秒內滑到 170（里程計也多走了 2 cm）；
        # 用 202 算出來的前伸量多了 3 cm，手臂伸到 432 mm 下垂、被判成「伸不到」。
        await asyncio.sleep(1.0)
        new = front_tof(await robot.state())
        notes.append(f"crept {step:.2f} m closer ({'ok' if r.get('ok') else 'failed'}), front sensor {tof} -> {new} mm")
        if step < 0.08 and isinstance(new, (int, float)) and new != 9999 and new <= tof + 15:
            # 小步（< 8 cm）底盤常常動不了或只動一點（11-17：送 0.06 讀值 222→202；送 0.04 讀值 207→205，兩次都被誤判成「箱子被推走」）。
            # 讀值沒變大就不是被推走：已經在目標附近幾公分內了，就停在這裡，剩下的交給手臂。
            if new > tof - 0.5 * step * 1000:
                return new, None
            tof = new
            continue
        if new is None or new == 9999 or new > tof - 0.4 * step * 1000:
            return new, (f"{what}: drove {step:.2f} m toward the surface but the front distance only went {tof} -> {new} mm: "
                         f"the box is probably sliding away (being pushed) or the sensor is not looking at it. Stopped; "
                         f"check the scene before trying again.")
        tof = new
    return tof, None



async def _free_lane(llm, jpg, ext, tof):
    """箱頂上已經有東西時，手臂要從哪個方向伸進去（2026-11-17 o_h2，使用者：「要放的地方前面有一個球，擋到了手臂」）。
    之前只對箱子的水平中心，中心那條線上剛好有球：手臂伸過去壓在球上，東西放歪倒下。
    做法：請模型把箱頂上每樣東西點出來 → 每樣東西左右各留 6 cm（夾爪張開的半寬 + 餘裕）→ 箱子寬度內剩下的空檔裡，
    選離箱子中心最近的方位。從車上看是一條「射線」，所以擋在前面或後面的都避得開。
    回 (相對夾著的東西的方位角, 說明) 或 None（箱頂是空的 / 沒有空檔 / 模型沒回）。"""
    # 問法（11-17 實測各 5 次）：「點出箱子上的東西」常漏掉白管（被夾著的棒子擋到一半）；模型原廠的通用問法「點出畫面裡的東西」
    # 5 次都同時找到白管、毛球、箱子、夾著的棒子 → 用通用問法，哪些算「箱頂上的障礙」自己用位置篩。同時問三次取聯集。
    prompt = ('Point to no more than 10 items in the image. The answer should follow the json format: '
              '[{"point": <point>, "label": <label>}, ...]. The points are in [y, x] format normalized to 0-1000.')
    res = await asyncio.gather(*[perception._points_all(llm, jpg, prompt) for _ in range(3)], return_exceptions=True)
    pts, boxes = [], []
    for r in res:
        for x, y, lb in ([] if isinstance(r, Exception) else r[0]):
            if "box" in lb.lower() or "carton" in lb.lower():
                boxes.append((x, y))
            elif not any(abs(x - q[0]) < 22 and abs(y - q[1]) < 45 for q in pts):
                pts.append((x, y, lb))
    if ext is None:
        for bx_, by_ in boxes:                                   # 沒給箱子的框：用模型點到的箱子自己圈
            ext = perception.surface_extent(jpg, bx_, by_)
            if ext is not None:
                break
    B = _bearing
    gap0 = B(NEX_GAP_REF[0])
    d = 0.16 + (tof / 1000.0 if isinstance(tof, (int, float)) and tof < 2000 else 0.6) + 0.07
    c = math.degrees(math.atan2(0.06, d))
    far = ("tree", "chair", "stool", "rug", "skirt", "suitcase", "desk", "table", "floor", "wall", "door", "window", "light", "gripper", "robot")
    pts = [q for q in pts if not any(w in q[2].lower() for w in far)
           and not (abs(q[0] - NEX_GAP_REF[0]) < 45 and q[1] > 600)]      # 背景的東西、夾著的東西（夾爪正上方、畫面下半）不算
    if ext is None:
        obs = [q for q in pts if 250 <= q[1] <= 800]
        lo, hi, centre = -14.0, 14.0, 0.0
    else:
        # 箱頂上的東西：左右在箱子範圍內，點的位置不低於箱子框的中段（更低的是箱子正面）
        obs = [q for q in pts if ext[0] - 30 <= q[0] <= ext[2] + 30 and q[1] <= ext[1] + 0.5 * (ext[3] - ext[1])]
        lo, hi = max(-14.0, B(ext[0]) - gap0 + 0.6 * c), min(14.0, B(ext[2]) - gap0 - 0.6 * c)
        centre = max(lo, min(hi, 0.0)) if hi > lo else 0.0       # 車頭已經對著箱子：優先放正前方，被擋才往旁邊找
    if not obs:
        return None
    blocks = []
    for x, y, lb in obs:
        bx = perception.box_around(jpg, x, y)
        half = max(B(bx[2]) - B(bx[0]), 2.0) / 2.0 if (bx[2] - bx[0]) < 200 else 2.0
        blocks.append((B(x) - gap0 - half - c, B(x) - gap0 + half + c))
    gaps, cur = [], lo
    for b0, b1 in sorted(blocks):
        if b0 > cur:
            gaps.append((cur, min(b0, hi)))
        cur = max(cur, b1)
    if cur < hi:
        gaps.append((cur, hi))
    gaps = [g for g in gaps if g[1] - g[0] >= 0.5]
    if not gaps:
        return None, ", ".join(sorted({lb or "object" for _, _, lb in obs}))      # 有東西、但沒有空檔
    best = min((max(g[0], min(g[1], centre)) for g in gaps), key=lambda t: abs(t - centre))
    return best, ", ".join(sorted({lb or "object" for _, _, lb in obs}))


async def _drive_to_visible(robot, llm, label, front_tof, notes, low_view=False, carrying=False):
    """看著目標開過去（不靠地圖）：相機找到目標 → 轉到正對 → 往前一段 → 重複，直到實體 ToF ≤ 600 mm 或目標已在畫面最下緣。
    2026-11-14：箱子在 0.8 m 以外時 ToF 讀值在 0.8–1.4 m/9999 之間亂跳，地圖上的方塊放了又拆，approach 拿到錯的位置（1.74 m）；
    相機反而一直看得到棒子。回 None = 到了；回字串 = 失敗原因。"""
    if carrying:      # 夾著東西：不能回看路姿勢（夾爪會前傾 26°，東西歪掉），用夾爪水平、相機朝前下方的姿勢邊看邊開
        # 2026-11-17 夜：原本用 (380,120)，那個姿勢 ToF 被夾爪擋住、永遠讀 280–340 mm → 下面「ToF ≤ 600 = 到了」永遠成立，
        # 車頭一對正就回報到了（其實還有 0.8–2.6 m）。改用量過的 NEX_CARRY_LOOK。
        await robot.act("arm_to", {"x_mm": NEX_CARRY_LOOK[0], "y_mm": NEX_CARRY_LOOK[1], "pitch_deg": 0.0, "yaw_deg": 0.0})
    elif low_view:    # 地上的東西：看路姿勢（平視）在 1 m 外只拍得到它的頂端或完全拍不到，改用往下看約 25° 的姿勢找
        await robot.act("arm_to", {"x_mm": 250, "y_mm": 150, "pitch_deg": 0.0, "yaw_deg": 0.0})
    else:
        await robot.act("recenter_arm", {})
    backed = False
    turns_in_row = 0
    for i in range(16):
        await asyncio.sleep(0.6)
        await robot.frame()
        jpg = await robot.frame()
        _save_frame(jpg, f"drive_to_{i+1}")
        f = await perception.locate(llm, jpg, label)
        if not f.found and i == 0:
            # 起跑時看不到（使用者把箱子換了位置）：原地每次轉 45° 找一圈
            for k in range(7):
                await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": 45.0, "push": False, "near_ok": True})
                await asyncio.sleep(0.6)
                await robot.frame()
                jpg = await robot.frame()
                _save_frame(jpg, f"search_{k+1}")
                f = await perception.locate(llm, jpg, label)
                if f.found:
                    notes.append(f"not in view at first: found it after turning {45 * (k + 1)} deg")
                    break
        if not f.found and i > 0:
            # 走到一半模型偶爾漏看（11-15 13:56：橘色棒子明明在畫面中間，第三張回「沒看到」）：換個說法、重拍再問兩次
            for alt in (f"{label} standing on the floor" if low_view else f"{label} on the box", label):
                await asyncio.sleep(0.5)
                jpg = await robot.frame()
                f = await perception.locate(llm, jpg, alt)
                if f.found:
                    break
        if not f.found and i > 0 and carrying and not backed:
            # 重問兩次還是看不到：多半是離箱子太近（箱子在畫面下緣外）或轉過頭。先直直退 0.4 m 拉開距離再看（只做一次）。
            # 11-17：原本一漏看就先退，箱子明明在 2 m 外的畫面正中間，白退 0.4 m。
            backed = True
            await robot.act("move_chassis", {"forward_m": -0.4, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
            await asyncio.sleep(0.6)
            await robot.frame()
            jpg = await robot.frame()
            f = await perception.locate(llm, jpg, label)
        if not f.found:
            return (f"cannot see '{label}' with the front camera from here (after {i} moves toward it"
                    f"{', and a full turn in place' if i == 0 else ''}). Drive somewhere it can be seen from, then call this again.")
        # 2026-11-16：夾著東西時原本容許 ±3° 而且每轉一次就重看，模型給的方位本身會抖幾度，11-15 最後一趟十圈全花在轉頭、一步沒前進。
        # 改成 ±5°，而且最多連轉兩次就一定往前走一步（走近之後方位誤差對應的橫向偏差也會變小）。
        # 夾著東西時一律以「夾爪所在的畫面位置」為基準（x≈553，不是畫面正中央）：以前圈得出整個箱子時用 553、圈不出來時用 500，
        # 兩種基準差 4°，剛好在 ±5° 的門檻兩邊跳。
        bearing = _bearing(f.x) - (_bearing(NEX_GAP_REF[0]) if carrying else 0.0)
        if carrying:
            # 2026-11-16 o_h2（使用者：「會一直撞到箱子」「放太邊了所以掉下去」）：模型只回箱子上的「一個點」，箱子在 0.5 m 外就佔 26° 寬，
            # 點落在左半邊還是右半邊每次不同 → 車頭左轉 20°、右轉 16° 來回擺，最後歪著貼上去（存下來的畫面：箱子中心在夾著的棒子右邊 12°）。
            # 改用影像把「整個箱子」找出來（同色區域，中間被夾著的東西擋住也接得起來），對它的水平中心；
            # 而且要對到「夾著的東西」所在的畫面位置（夾爪在畫面 x≈553，不是正中央 500）。
            ext = perception.surface_extent(jpg, f.x, f.y)
            if ext is not None:
                bearing = _bearing((ext[0] + ext[2]) / 2.0) - _bearing(NEX_GAP_REF[0])
                if perception.LOCK_HOOK is not None:
                    perception.LOCK_HOOK(ext, label)
            _drive_to_visible.last_bearing = (round(bearing, 1), ext is not None)
        if abs(bearing) > (5.0 if carrying else 4.0):
            await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": _turn_cmd(bearing), "push": False, "near_ok": True})
            await asyncio.sleep(0.4)
            turns_in_row += 1
            _drive_to_visible.last_bearing = None      # 轉過了：上面量到的方位作廢
            if carrying and turns_in_row < 2:
                continue                       # 夾著東西對箱子：轉完重看一次確認置中；連轉兩次還沒中就先往前
        turns_in_row = 0
        tof = front_tof(await robot.state())
        if isinstance(tof, (int, float)) and tof != 9999 and tof <= 600:
            lb = getattr(_drive_to_visible, "last_bearing", None)
            _drive_to_visible.last = {"bearing": lb[0], "tof": tof} if (carrying and lb and lb[1]) else None
            notes.append(f"drove toward the {label} by camera in {i} moves, front sensor now {tof} mm"
                         + (f", box centre {lb[0]:+.1f} deg from the held object" if (carrying and lb and lb[1]) else ""))
            return None
        # 不再用「目標在畫面多低」判斷太近：模型常把物件指在底部（y≥960），從 1 m 外就誤判，18:29 / 18:37 兩趟各白白放棄 2–3 次。
        # 只看 ToF：每次都先把車頭對正目標，箱子進到 0.8 m 內 ToF 就讀得到；真的太近目標會掉出畫面，下一圈回報看不到。
        near = isinstance(tof, (int, float)) and tof != 9999 and tof < 1000
        # 還遠（感測器讀不到箱子，或讀到 1.5 m 以上）：一步 0.35 m。11-17 從 2.3 m 外出發，每步 0.20 m、十圈走不到。
        far = isinstance(tof, (int, float)) and (tof == 9999 or tof >= 1500)
        step = min(0.40, (tof - 350) / 1000.0) if near else (0.35 if far else 0.20)
        await robot.act("move_chassis", {"forward_m": round(max(0.08, step), 2), "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
    return f"still not close to the {label} after 16 moves toward it."


def _nex_depth_from_base_y(reach_mm, z_mm, y1000, surface_m):
    """觀察姿勢（腕 90°、夾爪水平）下，物件「底部」在畫面的 y（0–1000）→ 物件中心離手臂底座軸的水平距離 m。
    相機在上臂：位置用 FK 算，俯仰 = 上臂仰角 − 74°（2026-11-14 17:26 成功那趟校正：底部 y=940 ↔ 距離 0.365 m、箱頂 0.25 m）。
    畫面每 40/1000 約 2 cm；y ≥ 990（貼著下緣）或視線沒往下就回 None。"""
    if y1000 is None or y1000 >= 990:
        return None
    k = _NEX_FK
    wx, wz = reach_mm - k["ax"] - k["L3"], z_mm - k["az"]
    d = min(math.hypot(wx, wz), k["L1"] + k["L2"] - 1e-6)
    p1 = math.atan2(wz, wx) + math.acos(max(-1.0, min(1.0, (k["L1"] ** 2 + d * d - k["L2"] ** 2) / (2 * k["L1"] * d))))
    cx = (k["ax"] + k["L1"] * math.cos(p1)) / 1000.0 + 0.03 * math.cos(p1)
    ch = NEX_BASE_H_M + (k["az"] + k["L1"] * math.sin(p1)) / 1000.0 + 0.03 * math.sin(p1)
    ty = math.tan(math.radians(perception.CAMERA_HFOV_DEG) / 2.0) * 9.0 / 16.0
    dep = -(math.degrees(p1) + NEX_LOOK_TILT_DEG) + math.degrees(math.atan((y1000 / 1000.0 - 0.5) * 2.0 * ty))
    if dep < 8.0 or ch <= surface_m:
        return None
    return cx + (ch - surface_m) / math.tan(math.radians(dep))


NEX_LOOK_TILT_DEG = float(os.environ.get("ROBOT_NEX_LOOK_TILT_DEG", "-76.5"))
# 相機固定座會鬆動（11-15 一小時內同一姿勢下夾爪在畫面的 y 從 429 → 512 → 790，等於俯仰變了 24°），所以畫面上的對位目標
# 一律用「相對於夾爪開口」的偏移，每次現量夾爪開口在哪。夾爪 (380,120) 與 0.73 m 外的地面點從相機看過去角度只差 3°，
# 所以這個偏移幾乎不受相機俯仰影響。基準（11-15 校正）：夾爪開口 (553,512)，白點 (471,415) = 車中心前方 0.73 m、左 0.10 m。
NEX_GAP_REF = (553.0, 512.0)
NEX_FLOOR_AHEAD_M = 0.87              # 目標對到「夾爪開口正上方 97 單位」時它在車中心正前方這麼遠（11-15 實測兩次：0.73 差 6 cm、0.79 還差 7–8 cm）
NEX_FLOOR_DY = -97.0


def _gripper_gap_cv(jpg, centre=False):
    """影像備援：俯視姿勢下夾爪是畫面中下方最大的幾塊黑色區域裡「最上面」那塊（手臂本體在它下面），指尖上緣往下 25 單位當開口。
    centre=True（靠近夾取用，2026-11-17）：x 改用夾爪「橫桿」的水平中心。使用者：「他還是沒有太對準去夾他」「會把東西撞倒」——
    相機重新固定後畫面裡右指尖比左指尖高，「最上面那幾列」只剩右指尖（寬 20 px），量到的 x 是右指尖（625–690）不是開口中心（≈566）：
    今天每一趟都把棒子對到右邊那根手指上，合爪時手指把棒子推倒。橫桿是整塊最寬的那幾列，不受指尖高低、棒子陰影影響（16 張實拍都在 563–567）。
    回家對白點那邊的校正值是用舊量法存的，所以預設不變。"""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    m = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 50).astype(np.uint8)
    m[: int(h * 0.30)] = 0
    m[:, : int(w * 0.25)] = 0
    m[:, int(w * 0.85):] = 0
    n, lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for k in range(1, n):
        bottom = st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT]
        cx = st[k, cv2.CC_STAT_LEFT] + st[k, cv2.CC_STAT_WIDTH] / 2.0
        if (st[k, cv2.CC_STAT_AREA] >= 1200 and bottom >= 0.80 * h and 0.40 * w <= cx <= 0.72 * w
                and (best is None or st[k, cv2.CC_STAT_TOP] < st[best, cv2.CC_STAT_TOP])):
            best = k
    if best is None:
        return None
    top = st[best, cv2.CC_STAT_TOP]
    ys, xs = np.nonzero(lab[top: top + max(8, h // 25)] == best)
    if len(xs) == 0:
        return None
    if centre:
        blob = lab == best
        rows = np.nonzero(blob.any(axis=1))[0]
        ext = [(np.nonzero(blob[r])[0].min(), np.nonzero(blob[r])[0].max()) for r in rows]
        wmax = max(b - a for a, b in ext)
        bar = [(a + b) / 2.0 for a, b in ext if b - a >= 0.85 * wmax][:8]
        return float(sum(bar) / len(bar) / w * 1000.0), float(top / h * 1000.0 - 45.0)
    return float((xs.min() + xs.max()) / 2.0 / w * 1000.0) - 15.0, float(top / h * 1000.0 - 45.0)     # 偏移對齊模型指的「開口」位置（實測差 +70/+15）


async def _gripper_gap(robot, llm):
    """俯視姿勢 (380,120) 下夾爪開口在畫面的 (x, y)/1000；量不到回 None。先問模型，問不到用影像找黑色夾爪。"""
    jpg = None
    for _ in range(2):
        await asyncio.sleep(0.5)
        await robot.frame()
        jpg = await robot.frame()
        g = await perception.locate(llm, jpg, "the gap between the two black gripper fingers")
        if g.found and g.y > 150:
            return float(g.x), float(g.y)
    return _gripper_gap_cv(jpg) if jpg else None


def _hue_ref(jpg, x1000, y1000):
    """模型指到的那一點附近的顏色（HSV 中位數）。太灰（飽和度 < 90）回 None：灰色的東西沒辦法靠顏色跟。"""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    cx, cy = int(x1000 / 1000.0 * w), int(y1000 / 1000.0 * h)
    patch = cv2.cvtColor(img[max(0, cy - 6): cy + 7, max(0, cx - 6): cx + 7], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    if len(patch) == 0:
        return None
    hh, ss, vv = (float(np.median(patch[:, k])) for k in range(3))
    return (hh, ss, vv) if ss >= 90 and vv >= 60 else None


def _color_base(jpg, ref, near=None):
    """用顏色找物件：跟 ref 同色相（±9）、夠鮮豔的最大一塊，回它「最下緣中點」(x, y)/1000 = 立在地上的物件的底部。
    視覺模型指同一個位置可以差 ±100 單位（11-15 邊看邊往前因此左右擺、漏看一次就開過頭），顏色分割每張都一樣而且不用等模型。"""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    dh = np.abs(hsv[:, :, 0] - int(ref[0]))
    dh = np.minimum(dh, 180 - dh)
    # 飽和度門檻要緊：橘色棒子 (H21,S247) 跟紙箱 (H23,S185) 色相幾乎一樣，只差在鮮豔程度。舊門檻 ref×0.55 把紙箱也算進來，
    # 11-15 18:0x 整趟追到箱子上去（讀值一直 −300~−500、+600~+700 不變）。改成比基準低不超過 35。
    m = ((dh <= 9) & (hsv[:, :, 1] >= max(80, ref[1] - 35)) & (hsv[:, :, 2] >= max(70, ref[2] - 75))).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    best, best_key = None, None
    for k in range(1, n):
        a = st[k, cv2.CC_STAT_AREA]
        if a < 120:
            continue
        key = -a
        if near is not None:
            key = math.hypot(cen[k][0] / w * 1000 - near[0], (st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT]) / h * 1000 - near[1]) - a * 0.02
        if best is None or key < best_key:
            best, best_key = k, key
    if best is None:
        return None
    bot = st[best, cv2.CC_STAT_TOP] + st[best, cv2.CC_STAT_HEIGHT] - 1
    ys, xs = np.nonzero(lab[max(0, bot - 5): bot + 1] == best)
    _color_base.last_top = float(st[best, cv2.CC_STAT_TOP] / h * 1000.0)       # 這一塊的最上緣（物件頂端）
    _color_base.last_wh = (int(st[best, cv2.CC_STAT_WIDTH]), int(st[best, cv2.CC_STAT_HEIGHT]))   # 像素寬高，給「是不是直立的細長物」檢查用
    # 每一列的左右中點（給 _color_x_at 用）：{列(0–1000): x(0–1000)}
    blob = lab == best
    prof = {}
    for yy in range(int(st[best, cv2.CC_STAT_TOP]), bot + 1):
        cols = np.nonzero(blob[yy])[0]
        if len(cols) >= 6:
            prof[int(round(yy / h * 1000.0))] = float((cols.min() + cols.max()) / 2.0 / w * 1000.0)
    _color_base.last_profile = prof
    return float(xs.mean() / w * 1000.0), float(bot / h * 1000.0)


def _color_x_at(y_lo, y_hi):
    """上一次 _color_base 找到的那一塊，在畫面 y_lo–y_hi（0–1000）這幾列的左右中點（中位數）；那幾列沒有它回 None。
    2026-11-19（使用者在旁邊看：「那個抓夾中間的空間根本沒有對準棒子」）：左右對準原本看的是色塊「最下緣」的中點。棒子靠近之後底部被
    手指擋住，最下緣只剩手指上方露出來的一小角，加上從上往下看直立的棒子在畫面上是斜的，量到的 x 比夾爪高度那一段偏了 60–100 單位：
    錄影裡棒子整根在左手指的左邊，讀值卻是 −13（算對準了），連合六次爪都是手指頂著棒子。
    要對的是「棒子在指尖這個高度的位置」= 色塊在指尖上緣上面那幾列的中點，那一段不會被擋。"""
    prof = getattr(_color_base, "last_profile", None) or {}
    v = sorted(x for y, x in prof.items() if y_lo <= y <= y_hi)
    return v[len(v) // 2] if len(v) >= 5 else None


def _real_arm_rz(res):
    """arm_to 回覆裡手臂實際停的位置：「arm now at x=416 y=-124 z=274 mm」→ (徑向前伸 mm, 高度 mm)。鏡射時看實體那一份。讀不到回 None。"""
    msg = str((res.get("real") or {}).get("message", "")) or str(res.get("message", ""))
    m = re.search(r"x=(-?\d+) y=(-?\d+) z=(-?\d+)", msg)
    return (math.hypot(float(m.group(1)), float(m.group(2))), float(m.group(3))) if m else None


# 手臂抬在 (r0, 150)、夾爪張開時，前方距離感測器看得到立在地上的物件（夾爪不擋）：讀值 + 這個數 ≈ 手臂第一次夾得到它的前伸量。
# 2026-11-18 實機一筆：讀值 427–432 mm，手臂伸 413 合爪沒夾到、428 夾到 → 約 −1 到 −16，取中間。資料多了再修。
def _real_travel_m(res):
    """move_chassis 回覆裡底盤實際走的距離（里程計，m）。鏡射時看實體那一份。讀不到回 None。"""
    rep = res.get("real") or res
    d = (rep.get("measured") or {}).get("distance_m")
    if d is None:
        m = re.search(r"travelled (\d+\.\d+)m", str(rep.get("message", "")))
        d = m.group(1) if m else None
    return None if d is None else float(d)


NEX_TOF_CONTACT_MM = float(os.environ.get("ROBOT_NEX_TOF_CONTACT_MM", "-8"))
# 實體底盤橫移送不到約 4.5 cm 就完全不動（11-18/19 實測：送 3.2–4.2 cm 共 24 次，里程計都是 0.00 m；4.7、5.9 cm 各走 3 cm；
# 12 cm 走 9 cm）。要橫移就至少送這麼多。
NEX_SIDE_MIN_M = float(os.environ.get("ROBOT_NEX_SIDE_MIN_M", "0.055"))


async def _tof_close_step(robot, arm, r0, yaw, gp, z_mm, expect_contact_mm, log):
    """地上夾取的最後一段底盤補位，改用前方距離感測器直接量（2026-11-18）。
    昨天的做法是看畫面估「還差幾公分」然後底盤盲走一步：今天實機送 0.058 m 只走了 21 mm（感測器 450→429、畫面 dy 125→44 都這麼說），
    棒子還在 7 cm 外，手臂伸到 428–443 mm（極限）才夾到 → 那個前伸量手臂抬不起來（指令抬 40 mm 實際只抬 16 mm），棒子底還拖在地上：
    晃動測試手臂轉 ±12° 時反而是底盤被帶著轉 ∓12°（陀螺儀讀值），手臂縮回時底盤又被往前拖了 21 cm（沒有任何底盤指令）。
    現在：手臂抬到 150（這個姿勢感測器看得到物件，今天一路 785→551→450→429 跟著底盤變），量一次、走一小步、再量，
    補到「手臂再伸約 4–5 cm 就夾得到」為止；走過頭就退。手臂全程抬著，底盤怎麼動都碰不到東西。最後分兩段把手臂降回夾取高度。
    感測器讀到的跟相機估的差太多（不是在看這個物件）→ 手臂降回去、回 False，呼叫端照舊用畫面估。"""
    async def read():
        vals = []
        for _ in range(3):
            st = await robot.state()
            t = st.get("real_tof_front_mm")
            if not isinstance(t, (int, float)):
                t = st.get("tof_front_mm")
            if isinstance(t, (int, float)) and 100 < t < 1500:
                vals.append(float(t))
            await asyncio.sleep(0.2)
        return sorted(vals)[len(vals) // 2] if len(vals) >= 2 else None

    async def lower():
        if z_mm < 60:
            await arm(r0, 60, gp, yaw)             # 分兩段降：一次從 150 降到夾取高度，夾爪會往前甩一段弧
        await arm(r0, z_mm, gp, yaw)

    await arm(r0, 150, 0.0, yaw)
    await asyncio.sleep(0.6)
    t = await read()
    if t is None or abs(t + NEX_TOF_CONTACT_MM - expect_contact_mm) > 40.0:
        log.append(f"[front sensor reads {t} mm with the arm lifted but the object should be about "
                   f"{expect_contact_mm - NEX_TOF_CONTACT_MM:.0f} mm away: not using the sensor for the last step]")
        await lower()
        return False
    # 目標：再伸 47 mm 夾到（容許 ±11）。換算常數有 ±8 mm 的不確定，所以實際落在 28–66 mm：
    # 下限 ≥ 25 mm 夾爪降下去時指尖還在物件前面；上限讓第一次夾到的前伸量 ≤ 約 423 mm。
    goal, steps = r0 + 47.0 - NEX_TOF_CONTACT_MM, []
    for _ in range(6):
        err = t - goal
        if abs(err) <= 11.0:
            break
        # 實體底盤的小步不是線性的：送 0.09–0.10 m 會走 100–150 mm，送 0.058 只走 21 mm，送 0.04 幾乎不動。
        # 還差 8 cm 以上照距離走；之內用 5–6 cm 的指令當「微調」，每步都重量，走多走少下一步修。
        mm = min(100.0, err) if err >= 80.0 else math.copysign(60.0 if abs(err) >= 25.0 else 50.0, err)
        await robot.act("move_chassis", {"forward_m": round(mm / 1000.0, 3), "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
        await asyncio.sleep(1.0)                   # 底盤回報走完之後還會再滑 1–3 cm，等停穩再量
        t2 = await read()
        if t2 is None or abs((t - t2) - mm) > 0.9 * abs(mm) + 25.0:
            steps.append(f"{mm:+.0f} mm: sensor {t:.0f} -> {t2} (no longer on the object)")
            t = None
            break
        steps.append(f"{mm:+.0f} mm: sensor {t:.0f} -> {t2:.0f}")
        t = t2
    if t is not None and t - goal < -11.0:         # 六步還是太近：再退一步，寧可手臂多伸一點
        await robot.act("move_chassis", {"forward_m": -0.06, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
        await asyncio.sleep(1.0)
        t2 = await read()
        steps.append(f"-60 mm: sensor {t:.0f} -> {t2}")
        t = t2 if t2 is not None else t
    await lower()
    log.append("last chassis steps measured by the front sensor with the arm lifted ("
               + ("; ".join(steps) if steps else "already in place")
               + (f"): the fingers should reach it about {t + NEX_TOF_CONTACT_MM - r0:.0f} mm further out" if t is not None else ")")
               + "; now only the arm moves")
    return True


async def _close_in_by_sight(robot, llm, label, arm, r0, z_mm, gp, yaw, notes, arm_only=False):
    """夾取高度下「邊看邊往前」（使用者 11-15：「他可以對準之後再往前吧」「相機沒有能夠非常固定的位置」）。
    手臂停在夾取姿勢 (r0, z)，同一張畫面裡量夾爪指尖那條線和物件底部：底部還在指尖線上方 = 還沒到，車往前一小段再看；
    左右偏了用底盤橫移修（不轉底座：轉了畫面整個變，11-15 轉完模型讀值亂跳把棒子弄丟）。底部降到指尖線（或被夾爪擋住）就停。
    模型指的位置會亂跳 → 每一步連看兩張，兩次差太多就再看第三張取中間值。只比同一張畫面裡的兩樣東西，相機歪了也不影響。
    回 (yaw, None) = 到指尖前了；(yaw, 原因字串) = 失敗。"""
    what = f"the bottom end of the {label}, where it touches the floor"

    # arm_only（2026-11-17）：物件已經在夾爪前幾公分時的「重新對一下左右」。底盤一步都不動、手臂不伸，只轉底座；看不到就當作在指間。
    # 用第一次靠近時記下的顏色，不重新問模型：這麼近的色塊又大又寬，會被「不像直立物件」的檢查退回 → 變成「看不到」→ 往前衝 12 cm
    # （19:53 那趟就是這樣把已經在夾爪裡的棒子又推了一次）。
    color = {"ref": getattr(_close_in_by_sight, "last_ref", None) if arm_only else None, "last": None}

    async def look():
        """回 (gap, (x, y) 或 None)。有顏色基準就用顏色找（穩、快）；還沒有就問模型，問到的同時取它的顏色當基準。"""
        pts, gap = [], None
        if color["ref"] is not None:
            await asyncio.sleep(0.5)
            await robot.frame()
            jpg = await robot.frame()
            _save_frame(jpg, "closein")
            gap = _gripper_gap_cv(jpg, centre=True)
            cb = _color_base(jpg, color["ref"], color["last"])
            if cb is not None:
                color["last"] = cb
            return gap, cb
        for k in range(3):
            await asyncio.sleep(0.5)
            await robot.frame()
            jpg = await robot.frame()
            if k == 0:
                _save_frame(jpg, "closein")
            g = _gripper_gap_cv(jpg, centre=True)
            if g is not None:
                gap = g
            if color["ref"] is None:
                fo = await perception.locate(llm, jpg, label)          # 指物件本體（中間），取那裡的顏色
                if fo.found:
                    ref = _hue_ref(jpg, fo.x, fo.y)
                    cb = _color_base(jpg, ref, (fo.x, fo.y)) if ref else None
                    if cb is not None:
                        # 模型第一眼可能指到別的同色東西（紙箱跟橘色棒子色相一樣）。立在地上的東西影像是直立的：高要明顯大於寬，
                        # 而且不能是一大片；不像就不採用這個顏色基準，下一張再問。
                        bw, bh = getattr(_color_base, "last_wh", (0, 0))
                        if not (bh >= 1.25 * bw and bw <= 140):
                            notes.append(f"colour blob at the model's point is {bw}x{bh} px, not an upright object: not tracking that")
                            continue
                        color["ref"], color["last"] = ref, cb
                        _close_in_by_sight.last_ref = ref
                        notes.append(f"tracking the {label} by colour (hue {ref[0]:.0f})")
                        return gap, cb
            f = await perception.locate(llm, jpg, what)
            if not f.found:
                f = await perception.locate(llm, jpg, f"{label} standing on the floor")
            if f.found:
                pts.append((float(f.x), float(f.y)))
            if len(pts) == 2 and abs(pts[0][0] - pts[1][0]) < 60 and abs(pts[0][1] - pts[1][1]) < 60:
                break
            if k == 1 and not pts:
                break
        if not pts:
            return gap, None
        xs, ys = sorted(p[0] for p in pts), sorted(p[1] for p in pts)
        return gap, (xs[len(xs) // 2], ys[len(ys) // 2])

    frac_stop = 0.45        # 指尖線到物件影像高度的 45% 就停（碰到時約 0.8）：差的那一小段交給後面每次 1.5 cm 的試夾

    async def chassis(fwd, right):
        """使用者 11-15：「你還是會不小心把他推倒」「你撞到箱子了」——手臂降在地面附近時底盤一動（尤其小步修正的來回晃）夾爪就掃到東西，
        而且那個姿勢 ToF 被夾爪擋住、防撞是關的。規則：手臂降下去之後底盤不准動。要動底盤就先把手臂抬到 150（ToF 看得到前面、防撞開著），
        動完再降回夾取高度。"""
        # 2026-11-17：這裡原本用 _move_exact（看里程計補到 2 cm 內）。實體小步本來就差 ±3 cm（同樣送 0.10，ToF 量到走了 96–112 mm；
        # 送 0.05–0.06 走了 20–67 mm），補正只會前後來回晃、把對好的左右晃掉。這個迴圈每一步都會重看畫面，所以改成「看一次、動一次」。
        await arm(r0, 150, 0.0, yaw)
        res = await robot.act("move_chassis", {"forward_m": round(float(fwd), 3), "right_m": round(float(right), 3), "turn_left_deg": 0,
                                               "push": False, "near_ok": True})
        await asyncio.sleep(0.5)
        if z_mm < 60:
            await arm(r0, 60, gp, yaw)             # 分兩段降：一次從 150 降到夾取高度，夾爪會往前甩一段弧
        await arm(r0, z_mm, gp, yaw)
        return _real_travel_m(res)

    seen_close, log, blind = False, [], 0
    nudged, arm_fix = False, 0
    min_dy = 1e9
    was_near, yaw0, side_fix = False, float(yaw), 0
    near_stuck = False
    last_dy, jumps = None, 0
    side_stall, side_off = 0, False
    last_far, same_far = None, 0
    r_cur = float(r0)
    NEAR = 150.0        # 11-17：230 時底盤太早停，手臂要伸到 441 mm（極限）才碰到棒子，只夾到指尖、沒辦法再伸進去重夾，一轉就掉。原值 230； 底部離指尖線不到這麼多（畫面單位）= 很接近了：底盤不准再動，只動手臂
    # 2026-11-19（使用者：「那個抓夾中間的空間根本沒有對準棒子」）：真正歪掉的是「夾爪在畫面上的位置」。棒子靠到手指旁邊時，
    # 它的影子/暗部跟黑色夾爪連成一塊，量到的夾爪中心從 566 跳到 471（另一張 661），指尖線也差了 125 單位；
    # 拿棒子去對一個錯的中心，讀值寫 −13「對準了」，實際棒子整根在左手指外面，連合六次爪都夾不到。
    # 夾爪是跟著手臂的：同一個前伸量，它在畫面上的位置不會變。第一次（棒子還遠）量到的當基準，之後跳掉的讀值不用，用基準推。
    gref = getattr(_close_in_by_sight, "gref", None) if arm_only else None
    if gref is None:
        gref = {"x": None, "y": None, "r": None}
    _close_in_by_sight.gref = gref

    def fix_gap(g):
        if g is None:
            return (gref["x"], gref["y"] - 3.8 * (r_cur - gref["r"])) if gref["x"] is not None else None
        gx, gy = g
        if gref["x"] is None:
            if not 520.0 <= gx <= 615.0:
                return (566.0, gy)
            gref.update(x=gx, y=gy, r=r_cur)
            return g
        py = gref["y"] - 3.8 * (r_cur - gref["r"])
        if abs(gx - gref["x"]) > 30.0 or abs(gy - py) > 60.0:
            log.append(f"[gripper misread at {gx:.0f},{gy:.0f}: using {gref['x']:.0f},{py:.0f}]")
            return (gref["x"], py)
        gref.update(x=0.7 * gref["x"] + 0.3 * gx, y=gy, r=r_cur)
        return g

    for i in range(26):
        gap, pt = await look()
        gap = fix_gap(gap)
        if gap is None:
            return yaw, f"cannot see the robot's own gripper at the grasp pose (camera pointing away): {' '.join(log)}", r_cur
        tip_y = gap[1] + 45.0                      # 影像法回的是指尖上緣 − 45
        if pt is None:
            # 曾經很接近過（上一個可信讀值在 NEAR 內）之後看不到 = 被夾爪擋住了，當作到了。絕對不能再讓底盤往前
            # （11-15：到了 +57 之後一次漏看就觸發「看不到往前 12 cm」，整台車開過頭把東西撞開）
            if arm_only or seen_close or (last_dy is not None and last_dy <= NEAR):
                log.append("hidden behind the gripper")
                break
            if blind < 6:                          # 這個姿勢只看得到車前約半公尺：還太遠就每次往前 12 cm 再看，最多 6 次
                blind += 1
                await chassis(0.12, 0.0)
                r_cur = float(r0)
                log.append("not seen, +0.12")
                continue
            return yaw, f"cannot see the {label} from the grasp pose ({' '.join(log)})", r_cur
        ex, dy = pt[0] - gap[0], tip_y - pt[1]     # dy > 0：底部在指尖線上方（還在前面）
        if color["ref"] is not None:
            x_tip = _color_x_at(tip_y - 70.0, tip_y - 8.0)      # 指尖上緣上方那一段（見 _color_x_at）；棒子還遠、那幾列沒有它時維持用底部
            if x_tip is not None:
                ex = x_tip - gap[0]
        if last_dy is not None and last_dy <= NEAR and dy > last_dy + 120 and jumps < 2:
            jumps += 1                             # 已經很近了卻突然「變遠」一大截 = 模型指錯地方，這張不算，重看
            log.append(f"[ignored jump {dy:+.0f}]")
            continue
        if was_near and dy > NEAR and min_dy <= 60.0:
            # 2026-11-16 o_h2（使用者：「明明就已經碰到他了怎麼沒有抓，然後還換位置把棍子弄到了」）：棒子已經進到兩指之間（存下來的畫面看得很清楚），
            # 它的下半截在夾爪的陰影裡、顏色變暗，顏色追蹤的「底部」一下跳到上面去 → 讀值從 +31 變成 +236 → 被當成「還很遠」→ 底盤往前開，把棒子撞倒。
            # 曾經很近過之後又「變遠」= 被夾爪擋住/遮暗了，東西就在指間：停，交給合爪。接近之後底盤絕對不再動。
            log.append(f"({ex:+.0f},{dy:+.0f}) reads far again after being close: it is between the fingers (hidden/shaded by the gripper)")
            break
        # 2026-11-17 21:0x：讀值 +144 → +150（門檻 150 上下抖 6 個單位）就被上面那條當成「近過又變遠」而結束，棒子其實還在 10 cm 外，
        # 後面合了 13 次爪、伸到 447 mm 才用指尖夾到。要真的近過（曾經 ≤ 60）才算。
        min_dy = min(min_dy, dy)
        last_dy = dy
        log.append(f"({ex:+.0f},{dy:+.0f})")
        if arm_only:
            new_yaw = max(yaw0 - 8.0, min(yaw0 + 8.0, yaw - max(-5.0, min(5.0, ex * 0.04))))
            if abs(ex) > 25 and arm_fix < 3 and abs(new_yaw - yaw) >= 0.5:
                arm_fix += 1
                yaw = new_yaw
                await arm(r_cur, z_mm, gp, yaw)
                continue
            break
        near = dy <= NEAR
        was_near = was_near or near
        # 2026-11-19 保險：還遠的時候連續 5 次讀值一模一樣 = 底盤根本沒在動，馬上結束，不要把 26 圈耗完
        # （今天那趟 (-92,+403) 重複 23 次、五分鐘，使用者：「為什麼他一直重複同一個動作」）。
        same_far = same_far + 1 if (not near and last_far is not None and abs(ex - last_far[0]) <= 6 and abs(dy - last_far[1]) <= 6) else 0
        last_far = None if near else (ex, dy)
        if same_far >= 4:
            return yaw, (f"the view did not change over 5 looks although the chassis was told to move: the chassis is not making "
                         f"these small steps ({' '.join(log)})"), r_cur
        # 「到了沒」改用比例判斷（11-15：只看底部會提早/太晚，因為底部在地上、指尖在離地 12 cm，兩者的畫面關係隨相機俯仰變）：
        # 指尖線落在物件影像「從底往上幾成」的位置 = frac。指尖真的碰到物件時 frac ≈ 夾點高度 ÷ 物件高度（跟相機角度無關）。
        frac = None
        if color["ref"] is not None and getattr(_color_base, "last_top", None) is not None:
            top_y = _color_base.last_top
            # 物件頂端頂到畫面上緣（相機離得很近、或相機角度變了）時「頂端」是被畫面切掉的假值，比例會被壓得很小、永遠到不了門檻：
            # 這時不用比例，退回「底部到指尖線就停」。11-16：頂端 y=0，比例算出 −0.05，棒子明明在指間還繼續伸。
            if pt[1] - top_y > 40 and top_y > 8:
                frac = (pt[1] - tip_y) / (pt[1] - top_y)
                log[-1] = log[-1][:-1] + f",f={frac:+.2f})"
        if frac is not None and abs(ex) <= 40 and frac >= frac_stop:
            break
        if abs(ex) > 25:
            if near:
                # 使用者 11-15：「很接近的時候不能讓機器人往前，這樣會把東西弄到，需要讓機械手臂往前」——夾爪已經降在車頭前面，
                # 底盤一動（前進或橫移）夾爪就掃到東西，兩次都把棍子掃倒。接近之後左右用手臂底座小角度轉。
                # 接近之後底座最多再轉 ±8°：11-16 棒子已在指間、追蹤點亂跳時連轉三次共 15°，整隻手臂橫掃把棒子掃倒
                new_yaw = max(yaw0 - 8.0, min(yaw0 + 8.0, yaw - max(-5.0, min(5.0, ex * 0.04))))
                if abs(new_yaw - yaw) < 0.5 and abs(ex) > 60 and side_fix < 2:
                    # 底座已經轉到上限、還偏很多（11-16 第三次夾：(-133,-108) 重複 12 圈）：手臂抬起來，底盤橫移，再降下來重新靠近
                    side_fix += 1
                    yaw = yaw0
                    await chassis(0.0, max(-0.08, min(0.08, 0.6 * ex * 0.0006)))
                    r_cur, was_near, last_dy = float(r0), False, None
                    log.append("sideways off with the base at its limit: lifted the arm, slid the chassis, closing in again")
                    continue
                if abs(new_yaw - yaw) < 0.5:
                    # 18:4x：底座到上限、還偏 45 單位（不到 60，不橫移）→ 同一個讀值 (-45,+136) 看了 18 次什麼都沒做。
                    # 偏 60 以內棒子還在兩指開口內（半個開口約 80 單位）：不再修左右，直接往下走「手臂往前伸」。
                    near_stuck = True
                else:
                    near_stuck = False
                    yaw = new_yaw
                    await arm(r_cur, z_mm, gp, yaw)
            else:                                  # 還遠：底盤橫移（物件在夾爪右邊 → 車往右）
                # 11-15：舊增益 (0.5+dy×0.0012) 讓左右來回擺（−208→+194→…），18 次都耗在這。減半，而且偏不多時同一圈接著往前
                mm_per_unit = 0.3 + max(0.0, dy) * 0.0005
                lat = max(-0.10, min(0.10, 0.7 * ex * mm_per_unit / 1000.0))
                # 2026-11-16 o_h1：偏 33 單位算出來只要橫移 1.3 cm，比 _move_exact 的容許誤差 (2 cm) 還小 → 底盤完全沒動，
                # 卻照樣 continue 重看，18 圈讀值一模一樣（-33,+561）一步都沒前進。橫移不到 3 cm 就不橫移，同一圈直接往前；
                # 剩下的一點左右偏差等靠近之後用手臂底座轉。
                # 2026-11-19（使用者：「為什麼他一直重複同一個動作」）：偏 92 單位算出來要橫移 3.2 cm，實體底盤這麼小的橫移完全不動
                # （里程計 0.00 m），讀值不變 → 同樣的 3.2 cm 連送 23 次，五分鐘什麼都沒做，最後回報夾不到。昨天沒發生只是因為那趟
                # 算出來是 5.9、4.7 cm，剛好走得動（中間 4.2、4.1 cm 那兩次一樣沒動）。
                # 改成：至少送 NEX_SIDE_MIN_M；看里程計，沒動就下一次加 2 cm；連三次沒動就不再橫移，直接往前，
                # 剩下的左右偏差等靠近之後用手臂底座轉（±8° ≈ ±200 單位，夠用）。
                if abs(lat) >= 0.03 and not side_off:
                    cmd = math.copysign(min(0.10, max(abs(lat), NEX_SIDE_MIN_M) + 0.02 * side_stall), lat)
                    moved = await chassis(0.0, cmd)
                    r_cur = float(r0)
                    if moved is not None and moved < 0.35 * abs(cmd):
                        side_stall += 1
                        log.append(f"[slide {cmd * 100:+.0f} cm: the chassis moved {moved * 100:.0f} cm]")
                        if side_stall >= 3:
                            side_off = True
                            log.append("[the chassis will not make this sideways step: going on, the arm base turns the rest when close]")
                    continue
            if near and not near_stuck:
                continue
        if near and not nudged and r_cur <= r0 + 1.0:
            # 2026-11-17 最後一段的距離問題（使用者：「你還是把東西會弄倒阿，我真的不知道為什麼」）。今天四趟的數字：
            # 底部對到指尖線（dy≈+5）時，手臂還要再伸 60–75 mm 以上夾爪才第一次碰到它（368→443、381→441、380→440、381→441 還碰不到）——
            # 因為底部在地上、指尖在離地 12 cm，從上面的相機看過去「對齊」其實還隔一段；再近底部就被夾爪擋住，看不到了。
            # 手臂從這裡只剩 6–8 cm 可伸，所以每趟都是伸到極限用指尖夾、或伸到底還沒碰到 → 車子「盲開」8 cm 再把手臂降下去，
            # 夾爪直接降在棒子上（19:53 存下來的畫面：棒子被壓歪卡在橫桿前），棒子就是這樣被推倒的。
            # 改成：趁棒子還看得到、還隔著一段時，手臂抬高，底盤一次補到「再伸約 4 cm 就碰到」的位置，之後底盤不再動、只伸手臂。
            # 每多 1 mm 距離 dy 約多 3.8 單位（手臂 357→381 時 dy +94→+3）。底盤小步誤差 ±3 cm，所以留 4 cm 而不是貼上去。
            nudged = True
            # 2026-11-18：先用前方感測器量著補（見 _tof_close_step）。畫面估的接觸前伸量 = r0 + dy/3.8 + 52
            # （今天兩筆：dy 125 → 感測器換算 442、公式 442；dy 44 → 421、公式 421；原本的 65 多估了 13 mm）。
            if await _tof_close_step(robot, arm, r0, yaw, gp, z_mm, r0 + 52.0 + dy / 3.8, log):
                continue
            adv = min(0.10, (65.0 + dy / 3.8 - 40.0) / 1000.0)
            if adv >= 0.03:        # 不到 3 cm 就不補：手臂自己伸得到，而底盤最小步的誤差（±2.5 cm）比要補的還大
                # 2026-11-19：最少送 6 cm（原 4 cm）。那趟送 4 cm 底盤完全沒動（里程計 0.00 m），結果手臂伸到 427 mm 才夾到；
                # 送 5–6 cm 實際走 3–4 cm。
                moved = await chassis(max(0.06, adv), 0.0)
                log.append(f"final chassis step {max(0.06, adv) * 100:.0f} cm with the arm lifted"
                           + (f" (the chassis really moved {moved * 100:.0f} cm)" if moved is not None else "") + ": now only the arm moves")
                continue
        if dy <= 35:      # 以前還要求 frac 是 None：實測棒子完全在指間時 frac 也只有 0.2（底部被橫桿擋住），等 0.45 等不到，手臂就一直往前伸、把棒子在地上推著走
            break
        seen_close = dy <= 160
        if near:
            if r_cur >= 438.0:                     # 手臂伸到底了還沒到線：交給後面的「縮回手臂、車往前 8 cm 再伸」
                log.append("arm at full reach")
                break
            r_cur = min(440.0, r_cur + (max(10.0, min(30.0, dy * 0.25)) if frac is None else (25.0 if frac < 0.0 else 12.0)))
            await arm(r_cur, z_mm, gp, yaw)
        else:                                      # 還遠才動底盤，而且保守：每步最多 10 cm，寧可多看幾次
            await chassis(max(0.05, min(0.10, dy * 0.0003)), 0.0)
            r_cur = float(r0)
    else:
        return yaw, f"could not get the {label} to the fingertips in 26 looks ({' '.join(log)})", r_cur
    notes.append(f"closed in by sight at grasp height {' '.join(log)}; arm now at reach {r_cur:.0f} mm")
    return yaw, None, r_cur


async def _servo_to_floor_object(robot, llm, label, notes):
    """地上的細東西 ToF 掃不到 → 跟回家看白點同一招：相機轉到俯視姿勢 (380,120)，把物件底部對到「夾爪開口正上方」的固定偏移
    （= 車中心正前方 0.73 m），再用里程計往前走到物件在手臂底座前 0.42 m。回 None = 到位；字串 = 失敗原因。"""
    # 先找到它、把車頭轉正；看不到就原地每 45° 找一圈。用俯視姿勢 (380,120) 找：相機固定不住常常朝上，
    # 看路姿勢會整個看天花板（11-15 兩次「轉一圈都找不到」就是這樣），這個姿勢朝下最多、看得到前方 0.4–2 m 的地板
    await robot.act("arm_to", {"x_mm": 380, "y_mm": 120, "pitch_deg": 0.0, "yaw_deg": 0.0})
    seen = False
    for k in range(9):
        await asyncio.sleep(0.7)
        await robot.frame()
        jpg = await robot.frame()
        _save_frame(jpg, f"floor_face_{k+1}")
        f = await perception.locate(llm, jpg, f"{label} standing on the floor")
        if f.found:
            seen = True
            if abs(_bearing(f.x)) <= 5.0:
                break
            await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": _turn_cmd(_bearing(f.x)), "push": False, "near_ok": True})
        elif not seen and k == 0:
            # 平視第一眼就看不到：多半是太近（15 cm 高的東西在 0.5 m 內掉在畫面下緣外）→ 先直直退再看，不要馬上轉走。
            # 2026-11-19（使用者：「他撞到後面的牆壁了」）：車後面沒有感測器，這一步每呼叫一次就盲退 0.5 m；棒子倒在地上夾不起來，
            # agent 連叫三次 pick_from_top，就一路退到牆上。改成退 0.3 m，而且三分鐘內只退一次。
            if time.time() - getattr(_servo_to_floor_object, "backed_t", 0.0) > 180.0:
                _servo_to_floor_object.backed_t = time.time()
                await robot.act("move_chassis", {"forward_m": -0.3, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
        elif not seen:
            await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": 45.0, "push": False, "near_ok": True})
        else:
            break
    if not seen:
        return f"cannot see the {label} on the floor anywhere around (turned a full circle)."
    await robot.act("arm_to", {"x_mm": 380, "y_mm": 120, "pitch_deg": 0.0, "yaw_deg": 0.0})
    gap = await _gripper_gap(robot, llm)
    if gap is None or gap[1] > 800:
        # 這個姿勢看不到（或只看到一點）夾爪：粗對位做不了也沒關係，車頭已經對著它，剩下交給夾取高度的「邊看邊往前」
        # 改成：一邊把它保持在畫面正中、一邊往前，直到它降到畫面下緣附近（= 已經在車前半公尺左右）
        steps = 0
        for _ in range(9):
            await asyncio.sleep(0.7)
            await robot.frame()
            jpg = await robot.frame()
            _save_frame(jpg, "floor_drive")
            f = await perception.locate(llm, jpg, f"the bottom end of the {label}, where it touches the floor")
            if not f.found:
                f = await perception.locate(llm, jpg, f"{label} standing on the floor")
            if not f.found or f.y >= 820:
                break
            if abs(_bearing(f.x)) > 5.0:
                await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": _turn_cmd(_bearing(f.x)), "push": False, "near_ok": True})
            await _move_exact(robot, 0.25 if f.y < 650 else 0.12, 0.0, tol=0.03)
            steps += 1
        notes.append(f"gripper not in view at the look pose: drove toward it by bearing in {steps} steps")
        return None
    tx, ty = gap[0], gap[1] + NEX_FLOOR_DY
    what = f"the bottom end of the {label}, where it touches the floor"
    log = []
    for i in range(12):
        await asyncio.sleep(0.8)
        await robot.frame()
        jpg = await robot.frame()
        _save_frame(jpg, f"floor_servo_{i+1}")
        f = await perception.locate(llm, jpg, what)
        if not f.found:
            f = await perception.locate(llm, jpg, f"{label} standing on the floor")
        if not f.found:
            # 俯視看不到：車頭已經對著它，不是太遠（在畫面上緣外）就是太近（被夾爪擋住）。前幾次當成太遠往前 0.3 m，之後退 0.4 m
            for step in ((0.3, 0.3, -0.4) if i == 0 else (-0.25,)):
                await robot.act("move_chassis", {"forward_m": step, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
                await asyncio.sleep(0.8)
                await robot.frame()
                jpg = await robot.frame()
                f = await perception.locate(llm, jpg, what)
                if not f.found:
                    f = await perception.locate(llm, jpg, f"{label} standing on the floor")
                if f.found:
                    break
        if not f.found:
            return f"cannot see the {label} on the floor with the camera looking down (after {i} corrections: {' '.join(log)})."
        ex, ey = f.x - tx, f.y - ty
        log.append(f"({ex:+.0f},{ey:+.0f})")
        if abs(ex) <= 30 and abs(ey) <= 30:
            notes.append(f"lined up on the {label} on the floor by camera (gripper gap at {gap[0]:.0f},{gap[1]:.0f}) in {i} corrections {' '.join(log)}")
            await robot.act("arm_to", {"x_mm": 250, "y_mm": 200, "pitch_deg": 0.0, "yaw_deg": 0.0})
            await _move_exact(robot, max(0.0, NEX_FLOOR_AHEAD_M - 0.62), 0.0)     # 只先靠到約 0.6 m，剩下的在夾取高度邊看邊走
            return None
        if abs(ex) > 250:                          # 偏很多：先轉向它（畫面 1000 單位 ≈ 100°）
            await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(-ex * 0.09, 1), "push": False, "near_ok": True})
            continue
        fwd = max(-0.20, min(0.30, -0.9 * ey * 0.0015))        # 物件在畫面上方（遠）→ 往前
        right = max(-0.20, min(0.20, 0.9 * ex * 0.0015))
        for key, v in (("forward_m", fwd), ("right_m", right)):
            if abs(v) < 0.012:
                continue
            if abs(v) < 0.09:
                back = -0.12 if v > 0 else 0.12
                await robot.act("move_chassis", {"forward_m": back if key == "forward_m" else 0, "right_m": back if key == "right_m" else 0,
                                                 "turn_left_deg": 0, "push": False, "near_ok": True})
                v -= back
            await robot.act("move_chassis", {"forward_m": round(v, 3) if key == "forward_m" else 0,
                                             "right_m": round(v, 3) if key == "right_m" else 0, "turn_left_deg": 0, "push": False, "near_ok": True})
    return f"could not line up on the {label} on the floor after 12 corrections ({' '.join(log)})."


_NEX_FK = {"ax": -0.12, "az": 110.35, "L1": 227.45, "L2": 145.31, "L3": 129.79}      # Test_Robot/nexarm_workspace.py 擬合的平面 FK（mm）


def _nex_r_wrist90(z_mm):
    """夾爪水平、前臂垂直往下（腕關節 90°，使用者 2026-11-14 貼的尺寸圖右邊那個姿勢）時的前伸距離 mm。
    肘在腕的正上方：(r − L3)² + (z + L2 − az)² = L1²。"""
    k = _NEX_FK
    h = z_mm + k["L2"] - k["az"]
    return k["ax"] + k["L3"] + math.sqrt(max(0.0, k["L1"] ** 2 - h * h))


_TOP_GRIP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "_last_top_grip.json")
# pick_from_top 夾到東西時記下「夾爪中心在物件底部上方多高」，place_on_top 用它算放下時瓶底離箱頂多少（2026-11-14）
_LAST_TOP_GRIP = {}


def _nex_min_z(reach_mm):
    """水平夾爪在這個前伸距離最低到得了多高（mm，多留 10 mm 餘裕）。"""
    t = _NEX_HORIZ_MIN_Z
    if reach_mm <= t[0][0]:
        return t[0][1] + 10.0
    for (r0, z0), (r1, z1) in zip(t, t[1:]):
        if reach_mm <= r1:
            return z0 + (z1 - z0) * (reach_mm - r0) / float(r1 - r0) + 10.0
    return t[-1][1] + 10.0


NEX_BOX_TOP_M = float(os.environ.get("ROBOT_BOX_TOP_M", "0.25"))     # 放倒的 Hiwonder 紙箱箱頂高度（使用者的固定道具）
# 2026-11-16 使用者：「箱子的尺寸又不是固定的，他不能夠自己適應嗎？」——箱頂高度不再當成固定值：place_on_top 放下時用手臂「碰到面」
# 量出實際高度，記在這個檔案；之後 30 分鐘內 pick_from_top / place_on_top 沒指定高度時就用量到的值，沒量過才退回上面的預設。
_BOX_TOP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "_box_top.json")
# 夾著東西「邊看邊開」的手臂姿勢（前伸, 高度）。2026-11-17 夜實測（夾著橘色棒子、前方實際 2.6 m）：
#   (152,98) 原位 → ToF 讀 105 mm；(380,120) → 280–337 mm：感測器照到的是夾爪和手上的東西，不是前面。
#   (380,160)、(380,200)、(300,150)、(250,200 以上) → 讀得到 2.5–2.7 m。鏡頭方面 (380,160) 看得到地面和箱子，(250,250)/(250,295) 只看得到天花板。
NEX_CARRY_LOOK = (380, 160)
# 手臂相機的水平視角。2026-11-17 夜在實機量的：原地轉 7 個角度（實體朝向 21.9°–57.7°），用影像比對追同一個箱子，
# 朝向每變 10.7° 箱子在畫面橫移 106/1000 → tan(HFOV/2)=0.90 → 約 84°。perception 的預設值是 100°（遊戲場模擬相機用的），
# 拿來算這顆相機的方位角會大三成；再加上實體底盤轉小角度會多轉三成（送 8° 轉了 10–10.7°，送 16° 轉 15–16°），
# 「看到偏幾度就轉幾度」等於每次修 1.7 倍：車頭在目標兩側來回擺（使用者：「他現在會一直轉」；11-16 的「左轉 20°、右轉 16° 來回擺」也是）。
NEX_CAM_HFOV_DEG = float(os.environ.get("ROBOT_NEX_CAM_HFOV", "84"))


def _bearing(x1000):
    """手臂相機畫面 x（0–1000）→ 方位角（右為正），用量過的視角。"""
    return perception.bearing_from_x(x1000, NEX_CAM_HFOV_DEG)


def _turn_cmd(bearing_right_deg):
    """把偏 bearing 度的目標轉到正前方，move_chassis 的 turn_left_deg 要送多少：12° 以內的小角度實體會多轉三成，先打 75 折。"""
    t = -float(bearing_right_deg)
    return round(t * 0.75 if abs(t) < 12.0 else t, 1)
NEX_CARRY_Z_MM = 295.0            # 夾著東西靠近箱子時手臂舉到的高度（相機排線的上限 300 mm）：夾點離物件底部 12 cm 時底部離地 0.34 m


def _box_top_default():
    try:
        with open(_BOX_TOP_FILE, encoding="utf-8") as fh:
            d = json.load(fh)
        if time.time() - float(d.get("t", 0)) < 1800 and 0.08 <= float(d["top_m"]) <= 0.40:
            return float(d["top_m"])
    except Exception:
        pass
    return NEX_BOX_TOP_M


def _save_box_top(top_m):
    try:
        with open(_BOX_TOP_FILE, "w", encoding="utf-8") as fh:
            json.dump({"top_m": round(float(top_m), 3), "t": time.time()}, fh)
    except Exception:
        pass
NEX_GRASP_PITCH_DEG = float(os.environ.get("ROBOT_NEX_GRASP_PITCH_DEG", "0"))
NEX_AIM_LEFT_DEG = float(os.environ.get("ROBOT_NEX_AIM_LEFT_DEG", "3"))
NEX_DEPTH_EXTRA_MM = float(os.environ.get("ROBOT_NEX_DEPTH_EXTRA_MM", "30"))
NEX_GRIP_X = 547.0          # 觀察姿勢下夾爪開口中心在畫面的 x（0–1000）；模型指不到夾爪時的備援


async def pick_from_top(robot, llm, obj: str, surface_m: Optional[float] = None, object_h_m: Optional[float] = None,
                        depth_m: Optional[float] = None, _side_try: bool = False) -> Tuple[bool, str]:
    """抓放在箱頂/矮櫃上的小物件，水平夾法。2026-11-14 下午改版（相機在上臂；使用者：「你不能自己量嗎？東西是隨機刷新的」）：
    底盤用 ToF 貼到箱子前 → 手臂到 (250,150,水平)，相機俯 30° 看箱頂 → 轉底座把物件對到正中 → 自己量：讓模型指
    「箱子前上緣」（配 ToF 的箱面距離 → 箱頂高度）、「物件底部」（視線射到箱頂平面 → 離前緣多遠）、「物件頂端」（→ 物件高度）→
    夾爪在物件 7 成高度水平伸過去 → 合爪（伺服卡住 = 夾到）→ 抬、收回。surface_m/object_h_m/depth_m 有給就用給的。"""
    label = _plain_label(obj)
    given_s = None if surface_m is None else max(0.0, min(0.45, float(surface_m)))      # 0 = 站在地板上的東西
    given_h = None if not object_h_m else max(0.03, min(0.25, float(object_h_m)))
    r_look, z_look = 320.0, 0.13      # 相機俯 ~24°；箱面在 0.45 m 時箱緣、物件底、物件頂都在畫面裡（試算過 0.16 與 0.35 m 的箱子）
    notes = []

    async def fresh(tag="look"):
        await asyncio.sleep(0.8)
        await robot.frame()
        jpg = await robot.frame()
        _save_frame(jpg, "pick_from_top_" + tag)
        return jpg

    async def arm(reach_mm, z_mm, pitch, yaw=0.0):
        r = await robot.act("arm_to", {"x_mm": int(round(reach_mm)), "y_mm": int(round(z_mm)), "pitch_deg": float(pitch),
                                       "yaw_deg": round(float(yaw), 1)})
        real = r.get("real") or {}
        if real:                       # 鏡射時以實體手臂為準（模擬手臂到不了桌面以下的姿勢，實體可以）
            if not real.get("ok"):
                r["message"] = str(r.get("message", "")) + " | real arm: " + str(real.get("message", ""))[:120]
            r["ok"] = bool(real.get("ok"))
        return r

    async def arm_pose():
        ap = (await robot.state()).get("arm_pose") or {}
        return (float(ap.get("reach_mm", 0)) / 1000.0, float(ap.get("z_mm", 0)) / 1000.0,
                float(ap.get("yaw_deg", 0.0)), float(ap.get("pitch_deg", 0.0)))

    async def home():
        await robot.act("recenter_arm", {})

    # 0. 貼近：前向 ToF 讀箱子正面（感測器在底座前 0.16）。夾爪水平最遠伸 450，瓶子在箱面後 10–20 cm 的話車得幾乎貼著箱子。
    def front_tof(s):
        # 鏡射模式下 tof_front_mm 是 min(模擬, 實體)；箱子只有實體看得準（模擬那邊是 ToF 打到點放的方塊），有實體讀值就用實體的
        rt = s.get("real_tof_front_mm")
        return rt if isinstance(rt, (int, float)) else s.get("tof_front_mm")

    st = await robot.state()
    # ===== 2026-11-14 15:20 改版：不靠相機幾何量距離（近距離誤差太大，兩次實跑量出箱高 0.29 / 0.08，夾爪壓到箱頂）=====
    # 使用者：「為什麼不拿棍子中間，這樣就不會碰到箱子」。做法：夾爪固定在「箱頂上方 7.5 cm」（棍子中段）的高度，
    # 相機從上臂俯視，用畫面裡的夾爪當基準把底座轉到對正，然後水平一小步一小步往前伸，每步合一次爪，
    # 伺服被擋住 = 夾到了。高度只跟箱頂高度有關（放倒的紙箱 0.16，ROBOT_BOX_TOP_M 可改，任務有給就用給的）。
    surface_m = given_s if given_s is not None else _box_top_default()
    object_h_m = given_h if given_h is not None else 0.15
    # 15:20 實跑：夾爪俯 25° 時指尖比夾點低、實體手臂也到不了 (309,73,-25)，使用者：「爪子不是平的所以一直撞到箱子」→ 夾爪保持水平，
    # 改用「夾高一點」換空隙：箱頂上方 9 cm（15 cm 的棍子還有 6 cm 在夾點上面）
    # 15:21 直接試：實體在前伸 300 只降得到 100 mm（表上說 70），(300,88) 到不了 → 夾點再高：箱頂上 11 cm（棍子頂下方 4 cm），從前伸 320 起
    # 15:30 用合起的夾爪往下探：z≈135 mm 被箱頂頂住 → 箱頂 ≈ 0.25 m（之前一直當 0.16，所以夾爪每次都放在箱頂的高度、把箱子推走）。
    # 夾棍子正中間：箱頂上方 7.5 cm；觀察姿勢也用同一個高度、前伸 280（夾爪還沒過箱緣）
    grip_above = max(NEX_CAM_BOTTOM_M + 0.04, min(0.075, 0.5 * object_h_m))
    on_floor = surface_m < 0.03
    if on_floor:
        # 地板模式（2026-11-15 使用者：「他應該可以拿地上的東西」；實測手臂在車頭前方、前伸 355 可降到 z=−70 = 夾爪離地 9 cm，−90 頂到地）：
        # 夾點離地 10 cm，所以東西至少要 12 cm 高；車要停遠一點（腕 90° 姿勢前伸 355，東西要在更前面）
        if object_h_m < 0.12:
            return False, (f"pick_from_top({obj}): it stands on the floor and is only {object_h_m*100:.0f} cm tall; the gripper cannot "
                           f"get lower than 9 cm above the floor, so floor objects must be at least 12 cm tall. Nothing was moved.")
        # 使用者 11-15：「不用在最底下抓，可以抓棍子的中間或上半段，這樣比較好看到棍子」→ 夾在離頂端 3 cm 處（15 cm 的棍子 = 離地 12 cm）
        grip_above = max(0.10, min(0.20, object_h_m - 0.03))
    z_grasp = max(-0.070 if on_floor else 0.03, surface_m + grip_above - NEX_BASE_H_M)
    # 使用者 15:25：「機械手臂是一體的，手肘會撞到箱子」。夾爪水平時腕關節跟夾爪同高（箱頂上 7.5 cm），腕部伺服殼更低，
    # 伸到箱子上方就磨到箱頂。夾爪改成前低後高（俯 25°）：腕關節比夾點高 5.5 cm，整支前臂從上方斜斜伸進去。
    gp = NEX_GRASP_PITCH_DEG
    notes.append(f"gripping {grip_above*100:.1f} cm above the box top (surface {surface_m:.2f} m), gripper pitch {gp:.0f} deg")

    # 使用者 15:40：「從爪子往後數的那節要 90 度，不然會撞到箱子」→ 起始姿勢 = 前臂垂直、夾爪水平（前伸只有 ~24 cm），
    # 靠底盤貼到箱子前 6 cm 把夾爪送到箱緣上方；之後只在需要時才一步 2 cm 往前伸（前臂才慢慢變斜）
    r_look, z_look = _nex_r_wrist90(z_grasp * 1000), z_grasp
    notes.append(f"wrist-90 pose: reach {r_look:.0f} mm")
    if on_floor:
        why = await _servo_to_floor_object(robot, llm, label, notes)
        if why:
            await robot.act("recenter_arm", {})
            return False, f"pick_from_top({obj}) [{'; '.join(notes)}]: {why}"
    t_start = front_tof(await robot.state())
    if on_floor:
        pass
    elif not isinstance(t_start, (int, float)) or t_start == 9999 or t_start > 600:
        why = await _drive_to_visible(robot, llm, label, front_tof, notes, low_view=on_floor)       # 還遠：先看著目標開過去
        if why:
            return False, f"pick_from_top({obj}) [{'; '.join(notes)}]: {why}"
    tof, err = await _creep_to_surface(robot, front_tof, notes, f"pick_from_top({obj})", target_mm=150, max_step=0.25) if not on_floor else (9999, None)
    if err:
        return False, err
    tof0 = tof
    await robot.act("gripper", {"state": "open"})

    async def claw_now():
        return (await robot.state()).get("real_gripper_claw_deg")

    async def aim(r_mm, z_mm, yaw, tag, gain=1.0, tries=2, pitch=0.0):
        """用畫面裡的夾爪當基準，把底座轉到物件跟夾爪開口同一條直線上。回 (yaw, 有沒有看到物件)。"""
        seen = False
        for i in range(tries):
            jpg = await fresh(f"{tag}_{i+1}")
            # 指物件的「底部」：棍子在俯視畫面裡是斜的，指整根會指到上半截（偏很遠），15:18 實跑底座因此來回轉 36°/6°/30°
            f = await perception.locate(llm, jpg, f"the bottom end of the {label}, where it stands on the box")
            if not f.found:
                f = await perception.locate(llm, jpg, label)
            if not f.found:
                break
            seen = True
            g = await perception.locate(llm, jpg, "the gap between the two black gripper fingers at the bottom of the image")
            ref = g.bearing_right_deg if g.found else perception.bearing_from_x(NEX_GRIP_X)
            e = f.bearing_right_deg - ref
            if abs(e) <= 2.0:
                break
            # 相機在底座軸前方 20 cm、物件離相機 ~25 cm：畫面上的角度差大約是底座該轉角度的兩倍 → 乘 0.5，單步最多 12°
            step = max(-12.0, min(12.0, 0.5 * gain * e))
            yaw = max(-60.0, min(60.0, yaw - step))            # 物件在夾爪右邊 → 底座往右轉（yaw 負）
            r = await arm(r_mm, z_mm, pitch, yaw)
            if not r.get("ok"):
                break
        return yaw, seen

    # 1. 俯視對正（夾爪在箱緣前、比箱頂高）。地板模式先在高處伸出去再垂直降，手臂不會掃過車頭
    if on_floor:
        await arm(r_look, 150, 0.0, 0.0)
    r = await arm(r_look, z_look * 1000, 0.0, 0.0)
    if not r.get("ok"):
        await home()
        return False, f"pick_from_top({obj}): could not set the look pose (reach {r_look:.0f}, height {z_look*1000:.0f} mm): {str(r.get('message',''))[:120]}"
    if on_floor:
        yaw, seen = 0.0, True          # 地板模式：左右對正交給後面夾取高度的「邊看邊往前」，這裡的箱頂對正用詞與距離都不適用
    else:
        yaw, seen = await aim(r_look, z_look * 1000, 0.0, "aim", tries=4)
        for y0 in (20.0, -20.0, 40.0, -40.0):          # 正前方看不到（17:12 實跑：棍子在畫面左下角只露一點）→ 底座左右掃
            if seen:
                break
            await arm(r_look, z_look * 1000, 0.0, y0)
            yaw, seen = await aim(r_look, z_look * 1000, y0, f"aim_scan{y0:+.0f}", tries=4)
        if not seen:
            await home()
            return False, (f"pick_from_top({obj}) [{'; '.join(notes)}]: the camera does not see '{label}' on the surface in front "
                           f"of the gripper. Arm back home.")
    notes.append(f"aimed with the gripper in view: base yaw {yaw:+.1f} deg")

    # 2. 到夾取高度（水平夾爪降到這個高度至少要伸 r0），再對一次
    r0 = r_look
    r = await arm(r0, z_grasp * 1000, gp, yaw)
    if not r.get("ok"):
        await home()
        return False, (f"pick_from_top({obj}) [{'; '.join(notes)}]: the arm cannot get to reach {r0:.0f} mm at {z_grasp*1000:.0f} mm "
                       f"height: {str(r.get('message',''))[:140]}")
    if not on_floor:
        yaw, _ = await aim(r0, z_grasp * 1000, yaw, "level", gain=0.7, tries=2, pitch=gp)
        # 使用者 17:50 在旁邊看：「需要再往左邊一點點，不然拿不好」→ 對正完再固定往左補 NEX_AIM_LEFT_DEG（正 = 左）
        yaw = max(-60.0, min(60.0, yaw + NEX_AIM_LEFT_DEG))
        notes.append(f"plus {NEX_AIM_LEFT_DEG:+.1f} deg left bias -> base yaw {yaw:+.1f}")

    floor_ready = False
    if on_floor:
        yaw, why, r_seen = await _close_in_by_sight(robot, llm, label, arm, r0, z_grasp * 1000, gp, yaw, notes)
        if why:
            await robot.act("gripper", {"state": "open"})
            await arm(r0, 150, 0.0, 0.0)
            # 2026-11-19：失敗時東西還立在車頭前 40 cm，手臂一回原位相機就看不到它（在畫面下緣外）。那趟 agent 找不到它，
            # 就 go_to("table") 直直往前開了 1.2 m，把棒子撞倒推走。先退 35 cm 讓它回到畫面裡，並且明講它在哪、不准往前開。
            await robot.act("move_chassis", {"forward_m": -0.35, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
            await home()
            return False, (f"pick_from_top({obj}) [{'; '.join(notes)}]: {why}. The {label} is STILL STANDING on the floor straight ahead, "
                           f"about 1 m in front of the robot (I backed up 0.35 m so the camera sees it again). Do NOT drive forward, "
                           f"go_to or navigate from here: the robot would run it over. Call pick_from_top again from here.")
        floor_ready = True
    # 3. 使用者 18:00：「不用一直夾，不然快碰到瓶子的時候會移動到瓶子」→ 先用相機估距離（物件底部在畫面的高低），
    # 直接伸到定位只合一次爪；沒夾到才再往前 1.5 cm 補夾，最多補 3 次。估不出來才退回「從頭一步步試夾」。
    jpg = await fresh("depth")
    pb = await perception.locate(llm, jpg, f"the bottom end of the {label}, where it stands on the box")
    d_est = _nex_depth_from_base_y(r0, z_grasp * 1000, pb.y if pb.found else None, surface_m)
    if floor_ready:
        # 底部到指尖線時，指尖離物件還有約「夾爪離地高度 ÷ tan(視線俯角)」≈ 6–10 cm：直接再伸 7 cm 合爪，沒夾到最多補 2 次各 2 cm
        # 11-15 使用者：「你幹嘛一直推倒他」——一口氣伸 7 cm 時，只要左右差一點指頭就頂到棍子把它推倒
        # （夾爪開口 51 mm、棍子 50 mm，幾乎進不去兩指之間）。改成先伸 3 cm 合爪，沒夾到每次只多伸 1.5 cm。
        rr = min(440.0, r_seen)
        rr_max = min(450.0, rr + 75.0)
        d_est = None
        notes.append(f"trying the grip from {rr:.0f} mm in 1.5 cm steps")
    elif d_est is not None and r0 / 1000.0 <= d_est <= 0.45:
        # 18:15 使用者：「你還是移動到他了，你夾的太淺了」——伸到估計位置時瓶子只到指尖，合爪把它往外撥（兩趟都是第 2 次才夾到）。
        # 多伸 NEX_DEPTH_EXTRA_MM 讓瓶子進到夾爪裡面再合爪
        rr = max(r0, min(450.0, d_est * 1000 + NEX_DEPTH_EXTRA_MM))
        rr_max = min(450.0, rr + 45.0)
        notes.append(f"camera depth estimate {d_est:.3f} m (base at image y={pb.y}) -> reaching straight to {rr:.0f} mm")
    else:
        rr, rr_max = r0, 450.0
        notes.append(f"no usable depth estimate (base y={pb.y if pb.found else None}, est {d_est}), probing step by step")
    holding, real_msg, g, reach_mm = False, "", {}, r0
    n_close = 0
    while rr <= rr_max:
        r = await arm(rr, z_grasp * 1000, gp, yaw)
        if not r.get("ok"):
            notes.append(f"arm stopped at reach {rr:.0f} mm: {str(r.get('message',''))[:80]}")
            break
        t_now = front_tof(await robot.state())
        if isinstance(t_now, (int, float)) and t_now != 9999 and t_now > tof0 + 30:
            await robot.act("gripper", {"state": "open"})
            await arm(rr, min(z_grasp * 1000 + 80, 250), 0.0, yaw)
            await home()
            return False, (f"pick_from_top({obj}) [{'; '.join(notes)}]: the box moved away (front sensor {tof0} -> {t_now} mm) while "
                           f"reaching at {rr:.0f} mm: the gripper is pushing something. Stopped, arm back home.")
        g = await robot.act("gripper", {"state": "close"})
        n_close += 1
        real_msg = str((g.get("real") or {}).get("message", ""))
        holding = ("closed on something" in real_msg) if real_msg else bool(g.get("holding"))
        reach_mm = rr
        if holding:
            # 不做「鬆開再伸進去重夾」：NexArm 爪口全開 51 mm（原廠尺寸圖），棍子直徑 50 mm，進不到兩指之間，
            # 17:28 實測再伸 3 cm 只會把棍子推倒。只能用指尖夾住它的近側。
            # 細的物件（合爪後夾爪角度 > −5°，約 3.5 cm 以下）進得去兩指之間：鬆開、再伸 2.5 cm、重夾，讓它坐進夾爪裡；
            # 重夾沒夾到就退回原位再夾。粗的（5 cm 棍子合到 −22°）不做。
            # 2026-11-17（使用者：「他只有抓到一點點…抓到了之後再把爪子打開然後往前一點點去真的抓穩」）：今天對準改用開口中心後，
            # 棒子是正對兩指中間的（存下來的畫面），只被指尖夾住底部、舉起來是歪的。之前「再伸會推倒」是因為當時其實對在右指上。
            # 粗的（合到 −25° 以內 = 進得了開口）也重夾，但只多伸 15 mm；重夾沒夾到一樣退回原位再夾。
            first = await claw_now()
            # 2026-11-19（使用者在旁邊看：「他剛剛夾到了阿 幹嘛放掉」）：這裡原本「伸超過 420 mm 才夾到就整個放開、手臂抬起、
            # 車往前、重新對、從頭再伸一次」。那趟 427 mm 明明夾住了卻放掉，多花 50 秒、多合了 4 次爪才又夾到。拿掉：夾到就留著，
            # 伸太遠抬不動的問題改在下面「先水平縮回再抬」處理。
            deeper = 25 if (isinstance(first, (int, float)) and first > -5) else (15 if (isinstance(first, (int, float)) and first > -25) else 0)
            if deeper and rr + deeper <= 450:
                await robot.act("gripper", {"state": "open"})
                await asyncio.sleep(0.8)
                r2 = await arm(rr + deeper, z_grasp * 1000, gp, yaw)
                g2 = await robot.act("gripper", {"state": "close"})
                m2 = str((g2.get("real") or {}).get("message", ""))
                if r2.get("ok") and "closed on something" in m2:
                    g, real_msg, reach_mm = g2, m2, rr + deeper
                    notes.append(f"opened and re-seated {deeper} mm deeper for a firm grip (claw {first} -> {await claw_now()} deg)")
                else:
                    await robot.act("gripper", {"state": "open"})
                    await asyncio.sleep(0.6)
                    await arm(rr, z_grasp * 1000, gp, yaw)
                    g = await robot.act("gripper", {"state": "close"})
                    real_msg = str((g.get("real") or {}).get("message", ""))
                    holding = ("closed on something" in real_msg) if real_msg else bool(g.get("holding"))
                    notes.append("deeper re-seat missed, gripped at the first contact again")
            else:
                notes.append(f"claw stopped at {first} deg")
            break
        await robot.act("gripper", {"state": "open"})
        await asyncio.sleep(0.6)
        rr += 15.0
    # 手臂伸到底還沒夾到（使用者 11-15：「你伸長不了的時候可以往前開一點」）：夾爪張開、手臂縮回起始前伸，車往前 8 cm，再從頭伸一次；最多 3 次。
    # 只在前面沒有箱子擋著時做（地板模式）；箱子模式車已經貼在防撞距離上，不能再往前。
    advances = 0
    while not holding and on_floor and advances < 3:
        advances += 1
        await robot.act("gripper", {"state": "open"})
        await arm(r0, z_grasp * 1000, gp, yaw)
        await arm(r0, 150, 0.0, yaw)               # 手臂先抬起來才動底盤（降著時底盤不准動）
        # 伸到 441 都沒碰到 = 它至少還在指尖前 1–2 cm：只往前 5 cm（一次、不補正），手臂分兩段降，降下去時它還在指尖前面
        await robot.act("move_chassis", {"forward_m": 0.05, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
        await asyncio.sleep(0.5)
        await arm(r0, 60, gp, yaw)
        await arm(r0, z_grasp * 1000, gp, yaw)
        rr = r0
        # 底盤動過，左右一定跟剛才不一樣：只轉手臂底座重新對一次（底盤不動、看不到也不往前）
        y2, why2, r2 = await _close_in_by_sight(robot, llm, label, arm, r0, z_grasp * 1000, gp, yaw, notes, arm_only=True)
        if not why2:
            yaw, rr = y2, max(r0, min(450.0, r2))
        while rr <= 450.0:
            r = await arm(rr, z_grasp * 1000, gp, yaw)
            if not r.get("ok"):
                break
            g = await robot.act("gripper", {"state": "close"})
            n_close += 1
            real_msg = str((g.get("real") or {}).get("message", ""))
            holding = ("closed on something" in real_msg) if real_msg else bool(g.get("holding"))
            reach_mm = rr
            if holding:
                # 這裡夾到的也一樣先鬆開、再伸 1.5 cm 重夾（原本只有第一輪會做，補位之後夾到的都是指尖淺夾）
                first = await claw_now()
                if isinstance(first, (int, float)) and first > -25 and rr + 15 <= 450:
                    await robot.act("gripper", {"state": "open"})
                    await asyncio.sleep(0.8)
                    r2 = await arm(rr + 15, z_grasp * 1000, gp, yaw)
                    g2 = await robot.act("gripper", {"state": "close"})
                    m2 = str((g2.get("real") or {}).get("message", ""))
                    if r2.get("ok") and "closed on something" in m2:
                        g, real_msg, reach_mm = g2, m2, rr + 15
                        notes.append(f"opened and re-seated 15 mm deeper for a firm grip (claw {first} -> {await claw_now()} deg)")
                    else:
                        await robot.act("gripper", {"state": "open"})
                        await asyncio.sleep(0.6)
                        await arm(rr, z_grasp * 1000, gp, yaw)
                        g = await robot.act("gripper", {"state": "close"})
                        real_msg = str((g.get("real") or {}).get("message", ""))
                        holding = ("closed on something" in real_msg) if real_msg else bool(g.get("holding"))
                        notes.append("deeper re-seat missed, gripped at the first contact again")
                break
            await robot.act("gripper", {"state": "open"})
            await asyncio.sleep(0.6)
            rr += 15.0
    if advances:
        notes.append(f"arm at full reach without a grip: drove 5 cm closer {advances} time(s) and reached again")
    notes.append(f"{'closed on it at' if holding else 'nothing up to'} {reach_mm:.0f} mm after {n_close} close(s)")
    _save_frame(await robot.frame(), "pick_from_top_closed")

    if not holding:
        await robot.act("gripper", {"state": "open"})
        await arm(min(reach_mm, 400), min(z_grasp * 1000 + 80, 250), 0.0, yaw)
        await home()
        if _side_try:
            return False, (f"pick_from_top({obj}) [{'; '.join(notes)}]: nothing between the fingers from this side either. "
                           f"The object must stand within about 0.17 m of an edge.")
        # 從這一邊搆不到（長方形箱子的短邊、東西在中間）→ 用地圖繞到側面再試一次
        st = await robot.state()
        if "real_tof_front_mm" in st:        # 實體：貼著箱子繞行撞過箱子（17:30），不自動繞，回報給上層決定
            return False, (f"pick_from_top({obj}) [{'; '.join(notes)}]: nothing between the fingers within reach from this side. "
                           f"Not driving around the box automatically (collision risk); back up and approach from another side.")
        xy, hd = st.get("world_xy"), st.get("world_heading_deg")
        if xy and hd is not None:
            a2 = math.radians(hd + yaw)
            ox, oy = xy[0] + 0.50 * math.cos(a2), xy[1] + 0.50 * math.sin(a2)
            for side in (+1, -1):
                b2 = math.radians(hd + 90 * side)
                gx, gy = ox + 0.60 * math.cos(b2), oy + 0.60 * math.sin(b2)
                r = await robot.act("navigate_to", {"x": round(gx, 2), "y": round(gy, 2), "tolerance_m": 0.08, "face": False})
                if not r.get("ok"):
                    continue
                st = await robot.state()
                xy2, hd2 = st.get("world_xy") or (gx, gy), st.get("world_heading_deg") or hd
                turn = (math.degrees(math.atan2(oy - xy2[1], ox - xy2[0])) - hd2 + 180) % 360 - 180
                await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(turn, 1), "push": False})
                ok2, msg2 = await pick_from_top(robot, llm, obj, surface_m if given_s else None, object_h_m if given_h else None,
                                                depth_m, _side_try=True)
                return ok2, f"[nothing within reach from the first side; went around to the {'left' if side > 0 else 'right'} side] " + msg2
        return False, f"pick_from_top({obj}) [{'; '.join(notes)}]: nothing between the fingers within reach, and could not drive around to another side."

    # 4. 抬起確認、收回
    await asyncio.sleep(0.5)
    z_hold = min(z_grasp * 1000 + (60 if reach_mm <= 420 else 40), 295.0)     # 俯 25° 伸到 450 時最高只到 120
    r_hold = reach_mm
    if on_floor and reach_mm > 370:
        # 2026-11-18：伸到 443 時原地抬 40 mm 只抬得起 16 mm（工作範圍邊緣），東西的底還拖在地上，接著晃動測試、縮回手臂都是在拖它，
        # 底盤被帶著轉、被往前拉。
        # 2026-11-19：改成「邊抬邊縮回 405」也不行——417 mm 夾到，送 (405, +60) 只到 (393, +20)、夾爪被壓低 8°，
        # 晃動測試時棒子的底還黏在地上不動，反而是底盤跟著左右轉了 12–16°、被往前拉 7 cm。這根棒子重、底又止滑，手臂伸這麼遠抬不動。
        # 先在夾取高度水平縮回 365（手臂有力的範圍；棒子直立著被帶過來，或底盤被拉過去，兩種都只是把距離縮短），再原地抬。
        await arm(365.0, z_grasp * 1000, gp, yaw)
        r_hold, z_hold = 365.0, min(z_grasp * 1000 + 60, 295.0)
        notes.append(f"gripped at {reach_mm:.0f} mm: pulled back level to 365 mm before lifting (the arm cannot lift it further out)")
        if reach_mm > 425.0:
            # 2026-11-19 19:33：442 mm 才夾到 = 只有指尖夾著（伸到底了，沒辦法再伸進去夾深）。照樣抬起來，手腕一轉成橫的棒子就滑掉了。
            # 現在已經縮回 365，前面有空間了：在這裡開爪、往前伸 2 cm 把它吃進夾爪裡再合（使用者 11-17：「抓到了之後再把爪子打開然後往前一點點去真的抓穩」）。
            await robot.act("gripper", {"state": "open"})
            await asyncio.sleep(0.7)
            deep = None
            for r_in in (385.0, 400.0, 415.0):
                await arm(r_in, z_grasp * 1000, gp, yaw)
                g3 = await robot.act("gripper", {"state": "close"})
                m3 = str((g3.get("real") or {}).get("message", ""))
                if ("closed on something" in m3) if m3 else bool(g3.get("holding")):
                    deep = r_in
                    break
                await robot.act("gripper", {"state": "open"})
                await asyncio.sleep(0.5)
            notes.append(f"it was only held by the fingertips: opened and took it deeper into the gripper at {deep:.0f} mm" if deep
                         else "it was only held by the fingertips: opened to take it deeper but did not get it again")
            if deep:
                await arm(365.0, z_grasp * 1000, gp, yaw)
    r_l = await arm(r_hold, z_hold, gp, yaw)
    p_l = _real_arm_rz(r_l)
    if on_floor and p_l is not None and p_l[1] < z_grasp * 1000 + 35.0 and r_hold > 330.0:
        # 還是沒離地（實際高度沒跟上）：再縮回一點再抬一次
        await arm(325.0, z_grasp * 1000 + 15.0, gp, yaw)
        r_hold = 325.0
        r_l = await arm(r_hold, z_hold, gp, yaw)
        p_l = _real_arm_rz(r_l)
        notes.append(f"it did not come off the floor at 365 mm: lifted again from 325 mm (arm now at height {p_l[1] if p_l else '?'} mm)")
    await asyncio.sleep(0.5)
    st = await robot.state()
    still = bool(st.get("real_gripper_holding")) if "real_gripper_holding" in st else holding
    _save_frame(await robot.frame(), "pick_from_top_lifted")
    if still:
        # 晃動測試：底座左右各轉 12° 再回來，還夾著才算穩
        for dy in (12.0, -12.0, 0.0):
            await arm(r_hold, z_hold, gp, yaw + dy)
        await asyncio.sleep(0.4)
        st = await robot.state()
        still = bool(st.get("real_gripper_holding")) if "real_gripper_holding" in st else still
        notes.append("shake test (base +-12 deg): " + ("still holding" if still else "DROPPED"))
    # 搬運姿勢：夾爪保持水平（看路姿勢會讓夾爪前傾 26°，棍子跟著歪、容易滑）。沒夾到才回看路姿勢
    if still:
        if on_floor:
            await arm(300, 150, 0.0, 0.0)
        await arm(_nex_r_wrist90(150.0 if on_floor else z_hold), 150.0 if on_floor else z_hold, 0.0, 0.0)
    else:
        await home()
    where = f"[{'; '.join(notes)}]"
    if still:
        _LAST_TOP_GRIP.update({"object": label, "grip_above_bottom_m": round(grip_above, 3), "object_h_m": object_h_m,
                               "surface_m": surface_m, "reach_mm": reach_mm, "yaw_deg": yaw, "z_mm": z_grasp * 1000,
                               "tof_mm": tof0, "t": time.time()})
        try:                                                    # 也寫到檔案：agent 重開或分兩次執行時 place_on_top 還找得到
            with open(_TOP_GRIP_FILE, "w", encoding="utf-8") as fh:
                json.dump(_LAST_TOP_GRIP, fh)
        except Exception:
            pass
        return True, (f"pick_from_top({obj}) {where}: GRASPED — the gripper servo stopped short of fully closed (something between the "
                      f"fingers) and it is still there after lifting and a shake test. Keep the gripper closed; the arm holds it level (do not recenter_arm while carrying, that tilts it).")
    await robot.act("gripper", {"state": "open"})
    return False, (f"pick_from_top({obj}) {where}: closed on it but it slipped out when lifting (real arm said: {real_msg[:100]}). "
                   f"Frames are in runs/_skill_frames/.")


async def place_on_top(robot, llm, surface_m: Optional[float] = None, object_h_m: Optional[float] = None,
                       depth_m: Optional[float] = None) -> Tuple[bool, str]:
    """把夾著的東西立著放回箱頂。2026-11-14 15:35 實機驗證過的路徑（pick_from_top 反過來，腕關節 90° 姿勢）：
    底盤貼到箱子前 → 手臂抬高 6 cm 移到放置點上方 → 垂直放下到比夾取高 5 mm → 爪子全開 → 水平退 2.5 cm 讓指尖離開 →
    抬高、歸位 → 用退出後的相機畫面確認它立著。剛用 pick_from_top 夾的而且車沒動過，就放回原本夾起來的位置。"""
    last = dict(_LAST_TOP_GRIP)
    crowded = False
    rolled = False
    held_hue = None
    aim = None                 # 夾著東西對箱子時最後量到的「箱子中心偏幾度」，放的時候用手臂底座轉過去補
    if not last:
        try:
            with open(_TOP_GRIP_FILE, encoding="utf-8") as fh:
                d = json.load(fh)
            if time.time() - float(d.get("t", 0)) < 900:
                last = d
        except Exception:
            pass
    surface = float(surface_m) if surface_m is not None else _box_top_default()      # surface_m=0 → 放到地上
    on_floor = surface < 0.03
    grip = float(last.get("grip_above_bottom_m") or 0.075)
    # 鬆手高度：比夾起來時再低一點，讓底部確實壓到面上才放。使用者 11-15：「你現在沒有放到底，所以棍子倒了」——
    # 之前用「夾取高度 + 4~5 mm」，東西在夾爪裡只要滑一點、或手臂伸長時下垂量不同，底部就懸空，一鬆手就倒。
    z_mm = (surface + grip - (0.010 if on_floor else 0.004) - NEX_BASE_H_M) * 1000
    if on_floor:
        z_mm = max(-70.0, z_mm)
    notes = [f"{'floor' if on_floor else f'surface {surface:.2f} m'}, lowering until its bottom presses on it before letting go"]

    async def arm(reach_mm, zz, yaw=0.0):
        r = await robot.act("arm_to", {"x_mm": int(round(reach_mm)), "y_mm": int(round(zz)), "pitch_deg": 0.0, "yaw_deg": round(float(yaw), 1)})
        real = r.get("real") or {}
        if real:                       # 鏡射時以實體手臂為準（模擬手臂到不了桌面以下的姿勢，實體可以）
            if not real.get("ok"):
                r["message"] = str(r.get("message", "")) + " | real arm: " + str(real.get("message", ""))[:120]
            r["ok"] = bool(real.get("ok"))
        return r

    def front_tof(s):
        rt = s.get("real_tof_front_mm")
        return rt if isinstance(rt, (int, float)) else s.get("tof_front_mm")

    def held(s):
        return bool(s.get("real_gripper_holding")) if "real_gripper_holding" in s else bool(s.get("gripper_holding"))

    if not held(await robot.state()):
        return False, "place_on_top: the gripper is not holding anything, nothing to put down."
    same_spot_hint = bool(last.get("reach_mm")) and float(last.get("surface_m") or 0.0) > 0.03      # 上一次是從箱頂夾的
    if on_floor:
        tof, err = 9999, None                      # 地上沒有箱面可以貼，就放在車前
    else:
        # 2026-11-17 夜（使用者看著實機：「抬高就看不到前面的路了，他現在會一直轉跟動手臂，因為他看不到前面」）。量了才知道更糟：
        # 夾著東西時，前方距離感測器在原位和舊的俯視姿勢都被夾爪/手上的東西擋住（讀 105 / 280–337 mm，實際前方 2.6 m）。於是
        #   ·「離箱子不到 0.75 m → 先退 0.5 m」每次呼叫都成立：那一趟四次呼叫，車子一路退到離箱子 2.6 m；
        #   · 開過去時「ToF ≤ 600 = 到了」也永遠成立：車頭一對正就說到了，手臂舉高才讀到真正的 0.8–2.6 m，然後失敗。
        # 現在：先縮到感測器看得到前面的姿勢量一次；橫著/直著都用 NEX_CARRY_LOOK 邊看邊開（感測器、鏡頭都看得到前面）；
        # 一路開到離箱子約 33 cm 才把手臂舉高（使用者：「快到了要放置的地方之後再舉高…再轉過來放下來」），最後 15 cm 只靠感測器貼近。
        await robot.act("arm_to", {"x_mm": 250, "y_mm": 200, "pitch_deg": 0.0, "yaw_deg": 0.0})
        await asyncio.sleep(0.5)
        t0 = front_tof(await robot.state())
        if not (same_spot_hint and isinstance(t0, (int, float)) and t0 <= 250):
            # 使用者 11-15：「你沒有對準箱子」——以前只在 ToF > 0.6 m 時才用相機對箱子；ToF 剛好掃到箱角或別的東西就直接往前貼，結果歪著。
            # 現在除了「剛從這個箱子上夾起來、車沒動」以外，一律先用相機把車頭轉正對著箱子（夾著東西、看著箱子開過去）
            # 11-15 17:54 放歪的原因：離箱子不到 0.6 m 時畫面只看得到箱角（還被夾著的東西擋住），沒轉正就直直貼上去，ToF 打到箱角，
            # 東西放到箱緣外、手臂把箱子頂歪。太近就先直直退 0.5 m，讓整個箱子進畫面再對正、靠近。
            if isinstance(t0, (int, float)) and t0 <= 750:
                await robot.act("move_chassis", {"forward_m": -0.5, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
                notes.append(f"front sensor {t0} mm, too close to see the whole box: backed up 0.5 m first")
            # 2026-11-17 使用者的做法：「夾到了之後把東西轉 90 度變成橫的，找到要放的地方以後再舉高，然後轉回來，再下放，
            # 這樣就不會擋到相機、也不用一直舉高」——直立夾著時棒子從夾爪往下垂 12 cm，正好擋在鏡頭中間、又容易撞到箱子。
            # 手腕轉 90° 讓它橫著（之後的 arm_to 不帶 roll，實體手臂會保持這個角度），伸到放置點上方之後才轉回直立。
            # 轉之前先記下夾著的東西的顏色（直立夾著時它就在夾爪開口正上方）：放完用來確認箱頂上真的有它
            await robot.act("arm_to", {"x_mm": NEX_CARRY_LOOK[0], "y_mm": NEX_CARRY_LOOK[1], "pitch_deg": 0.0, "yaw_deg": 0.0})
            await asyncio.sleep(0.6)
            await robot.frame()
            jpg_h = await robot.frame()
            held_hue = _hue_ref(jpg_h, NEX_GAP_REF[0], 700.0)
            if held_hue is None:
                # 上一次呼叫已經把它轉成橫的（這次是重試）：橫著時它在夾爪兩側、畫面最下面那一條
                held_hue = _hue_ref(jpg_h, 430.0, 920.0) or _hue_ref(jpg_h, 660.0, 920.0)
            # 使用者 11-17：「如果有物件拿起來會擋到鏡頭的話就可以轉 90 度」——不是每樣東西都轉：夾點以下垂著 10 cm 以上的長東西，
            # 或夾著時它的顏色在畫面裡佔了三成以上的高度，才算擋鏡頭；短的（例如夾在 7.5 cm 處的白管）照原本直立帶著走。
            blocks_view = grip >= 0.10
            if held_hue is not None and not blocks_view:
                if _color_base(jpg_h, held_hue, None) is not None:
                    blocks_view = getattr(_color_base, "last_wh", (0, 0))[1] >= 0.35 * 360
            if blocks_view:
                await robot.act("arm_to", {"x_mm": NEX_CARRY_LOOK[0], "y_mm": NEX_CARRY_LOOK[1], "pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 90.0})
                rolled = True
                notes.append("turned the held object sideways (wrist roll 90) so it does not block the camera on the way")
                # 「夾著」是合爪那一刻記下來的，東西後來掉了也還是 True（11-17：轉成橫的時掉在地上，後面整段對著空夾爪做完還回報成功）。
                # 轉完再合一次爪：夾爪還是合不到底 = 還在；合到底了 = 掉了。
                await asyncio.sleep(0.8)
                g_chk = await robot.act("gripper", {"state": "close"})
                m_chk = str((g_chk.get("real") or {}).get("message", ""))
                if m_chk and "closed on something" not in m_chk:
                    await robot.act("arm_to", {"x_mm": 250, "y_mm": 250, "pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0})
                    return False, (f"place_on_top [{'; '.join(notes)}]: the object slipped out of the gripper when it was turned sideways "
                                   f"(the fingers now close all the way). It is on the floor near the robot: find it (survey_scene) and pick it up again.")
            _drive_to_visible.last = None
            why = await _drive_to_visible(robot, llm, "cardboard box", front_tof, notes, carrying=True)
            if why:
                return False, f"place_on_top [{'; '.join(notes)}]: {why} (still holding it; call place_on_top again, do not recenter_arm)"
            aim = getattr(_drive_to_visible, "last", None)
            if await _square_to_box(robot, front_tof, notes) and isinstance(aim, dict):
                aim = dict(aim, bearing=0.0)        # 位置換過了，剛才量的「箱子中心偏幾度」不能再用；正對之後就放正前方
            # 箱頂空檔要在這裡看：手臂還在邊看邊開的姿勢、離箱子 35–60 cm，箱頂看得最清楚。11-17 自己實測：
            # 貼近之後手臂舉到 295，相機整個朝天花板（存下來的 place_lane 只有聖誕樹、柱子、電視），模型把「柱子、電視」當成箱頂的東西。
            t_look = front_tof(await robot.state())
            await robot.frame()
            jpg_l = await robot.frame()
            _save_frame(jpg_l, "place_lane")
            lane = await _free_lane(llm, jpg_l, None, t_look)
            if lane is not None and lane[0] is None:
                # 正前方左右都有東西（箱子窄的那一頭朝著車時常這樣，東西其實在箱子後段）：不轉手臂，改放靠近箱緣一點（5 cm），離後面的東西遠一點
                crowded = True
                notes.append(f"box top is crowded ({lane[1]}): no free lane, putting it nearer the front edge")
            elif lane is not None and isinstance(t_look, (int, float)) and t_look < 2000:
                aim = {"bearing": (aim or {}).get("bearing", 0.0), "lane": round(lane[0], 1), "tof": t_look}
                notes.append(f"things already on the box ({lane[1]}): free lane {lane[0]:+.0f} deg from straight ahead, the arm will turn to it")
            # 第一段貼近：手臂還在邊看邊開的姿勢（鏡頭看得到前面、夾爪離箱子還有 12 cm 以上），貼到離箱子約 33 cm
            tof, err = await _creep_to_surface(robot, front_tof, notes, "place_on_top", target_mm=300, max_step=0.10)
            if err:
                return False, err + " (still holding it; call place_on_top again, do not recenter_arm)"
        # 快到了才舉高（使用者 11-15：「你撞到箱子了，你要舉高一點點」；11-16：「他如果舉的不夠高會撞到箱子」「箱子的尺寸又不是固定的」→ 舉到最高；
        # 11-17：「橫著抓著的時候不需要抬到這麼高，抬高就看不到前面的路了…快到了之後再舉高」→ 所以只在最後這 15 cm 才舉）。
        # 真正的箱頂高度等一下放下時用手臂碰出來。
        await robot.act("arm_to", {"x_mm": 250, "y_mm": int(NEX_CARRY_Z_MM), "pitch_deg": 0.0, "yaw_deg": 0.0})
        # 2026-11-16：一步最多 10 cm（原 25 cm）、停在 19 cm（原 15 cm）。使用者：「會一直撞到箱子」——歪著貼近時 ToF 沒打在箱面上，
        # 一步 20–25 cm 走完才發現讀值沒變，箱子已經被推走了。小步走，第一步讀值沒跟著變就停。
        tof, err = await _creep_to_surface(robot, front_tof, notes, "place_on_top", target_mm=165, max_step=0.10)   # 165：停在 ≤205 mm，手臂只要伸 ≤435（伸到 450 會下垂、量不準）
    if err:
        return False, err + " (still holding it)"
    if not on_floor and depth_m is None and isinstance(tof, (int, float)) and tof > 215:
        # 手臂最多伸 435 mm = 箱緣後 7 cm 只在 tof ≤ 205 時搆得到。搆不到就不要放（11-17 兩次都在箱子前面/箱緣鬆手，東西掉到地上）
        return False, (f"place_on_top [{'; '.join(notes)}]: still {tof} mm from the box after creeping, too far for the arm to reach 7 cm "
                       f"behind its edge. Nothing was released (still holding it). Call place_on_top again.")
    if rolled:
        # 轉回直立要在這裡做：手臂還縮著、舉在最高、夾爪在箱子前面約 10 cm 的空中，四周沒有東西。
        # 2026-11-17 21:26 實機：我原本等伸到放置點正上方才轉回來，棒子下半截（12 cm）從左邊水平位置掃下來，
        # 放完之後原本站在放置點左邊約 7 cm 的白瓶子不見了（放之前的畫面還在），應該就是被這一掃打下箱子的。
        # 使用者說的順序本來就是「舉高 → 轉回來 → 再下放」。
        await robot.act("arm_to", {"x_mm": 250, "y_mm": int(NEX_CARRY_Z_MM), "pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0})
        await asyncio.sleep(1.0)
        rolled = False
        notes.append("in front of the box with the arm raised: turned it upright again before reaching over the box")
    if not on_floor and isinstance(tof, (int, float)) and tof < 700:
        # 手臂要伸多遠全看這個距離：底盤停穩之後取三次的中間值（走完那一刻的讀值會多 2–3 cm，見 _creep_to_surface）
        await asyncio.sleep(0.5)
        ts = []
        for _ in range(3):
            v = front_tof(await robot.state())
            if isinstance(v, (int, float)) and v < 700:
                ts.append(v)
            await asyncio.sleep(0.2)
        if ts:
            t_now = sorted(ts)[len(ts) // 2]
            if abs(t_now - tof) >= 8:
                notes.append(f"front sensor settled at {t_now} mm (read {tof} mm right after the last move)")
            tof = t_now
    r90 = _nex_r_wrist90(z_mm)
    same_spot = last.get("reach_mm") and isinstance(last.get("tof_mm"), (int, float)) and abs(tof - last["tof_mm"]) <= 25
    if on_floor:
        reach, yaw = min(440.0, max(r90 + 30.0, float(last.get("reach_mm") or 0.0))), float(last.get("yaw_deg") or 0.0)
    elif depth_m is not None:
        reach, yaw = (0.16 + tof / 1000.0 + float(depth_m)) * 1000, 0.0
    elif same_spot:
        reach, yaw = float(last["reach_mm"]), float(last.get("yaw_deg") or 0.0)
        notes.append("same standing spot as the pick: putting it back where it was")
    else:
        reach, yaw = (0.16 + tof / 1000.0 + (0.05 if crowded else 0.07)) * 1000, 0.0      # 不知道原位：放在箱緣後 7 cm（16:03 實測放在箱緣後 1 cm，太險）
        if aim and isinstance(aim.get("tof"), (int, float)):
            # 左右：最後一眼箱子中心偏 b 度（在離手臂底座 0.16+tof+0.07 m 處）→ 橫向偏 e 公尺；直直貼近後 e 不變，
            # 用手臂底座轉 atan(e/reach) 把東西送到箱子中線上（底座 + = 左，方位 + = 右）。最多 ±20°。
            tgt = aim["lane"] if isinstance(aim.get("lane"), (int, float)) else aim["bearing"]     # 箱頂有東西：對空檔；沒有：對箱子中心
            e = (0.16 + aim["tof"] / 1000.0 + 0.07) * math.tan(math.radians(tgt))
            e = max(-0.06, min(0.06, e))       # 11-19：+9° 的空檔在箱子外面。最多偏 6 cm，真的有沒有箱面等一下用碰的確認
            yaw = max(-20.0, min(20.0, -math.degrees(math.atan2(e, reach / 1000.0))))
            notes.append(f"{'free lane' if tgt != aim['bearing'] else 'box centre'} was {tgt:+.1f} deg off ({e * 100:+.0f} cm sideways): arm base yaw {yaw:+.1f} deg to put it there")
            reach = reach / max(0.9, math.cos(math.radians(yaw)))      # 斜著伸：同樣的前伸量往前只到 cos 倍，補回來才一樣在箱緣後 7 cm
    reach = max(r90, min(450.0 if on_floor else 435.0, reach))
    z_hi = 150.0 if on_floor else NEX_CARRY_Z_MM
    if not on_floor:
        await arm(160, z_hi)
    r = await arm(reach, z_hi, yaw)

    def over_spot(res):
        # 2026-11-18 實機：伸 432 mm、夾著棒子，手臂比指令低 21 mm（指令 295 實際 274；降到 265 實際 252），server 回「沒到位」，
        # 這裡就放棄了——可是存下來的畫面裡棒子直立在箱頂上方、毛球旁邊，位置完全是對的。徑向前伸到了（差 15 mm 內）、
        # 高度還在假設箱頂上方 3.5 cm 以上，就是「到了，只是下垂」：照常往下放（下面的碰觸判斷本來就把下垂算進去）。
        p = _real_arm_rz(res)
        return p is not None and abs(p[0] - reach) <= 15.0 and p[1] >= z_mm + 35.0

    if not r.get("ok") and not on_floor and over_spot(r):
        notes.append(f"arm sags at this reach (commanded z {z_hi:.0f}, really {_real_arm_rz(r)[1]:.0f} mm) but it is over the spot: going on")
        r["ok"] = True
    if not r.get("ok") and not on_floor:
        # 最高處伸不到這麼遠（手臂工作範圍）：每次降 3 cm 再試，降到「假設箱頂 + 6 cm」為止
        zz = z_hi
        while not r.get("ok") and zz - 30.0 >= z_mm + 35.0:      # 底部至少還在（假設的）箱頂上方 3 cm
            zz -= 30.0
            r = await arm(reach, zz, yaw)
            if not r.get("ok") and over_spot(r):
                r["ok"] = True
        if r.get("ok"):
            notes.append(f"cannot reach {reach:.0f} mm at full height: went over the box at z {zz:.0f} mm")
            z_hi = zz
    if r.get("ok") and rolled:
        # 已經在放置點正上方、最高處：轉回直立（底部離箱頂還有 9 cm 以上），停穩再往下放
        await robot.act("arm_to", {"x_mm": int(round(reach)), "y_mm": int(round(z_hi)), "pitch_deg": 0.0, "yaw_deg": round(float(yaw), 1), "roll_deg": 0.0})
        await asyncio.sleep(1.0)
        rolled = False
        notes.append("over the spot: turned it upright again")
    if r.get("ok") and on_floor:
        r = await arm(reach, z_mm, yaw)
    elif r.get("ok"):
        # 2026-11-16 o_h2：對準、貼近都對了，鬆手後棒子還是兩次都掉下去。箱頂高度寫死 0.25 m，但箱子換一面放就不是這個高度
        # （用旁邊 15 cm 的棒子當尺量畫面，今天的箱子大約 19 cm 高）→ 在箱頂上方 6 cm 就鬆手，棒子摔下去彈走。
        # 改成真的「放到底」：從上方每次降 8 mm，看實體手臂回報的高度；降不下去（實際高度比指令高 7 mm 以上、或回報沒到位）
        # = 底部頂到箱面了，就在那裡鬆手。最低降到假設箱頂只有 12 cm 的高度為止。
        def real_z(res):
            m = re.search(r"z=(-?\d+)", str((res.get("real") or {}).get("message", "")) or str(res.get("message", "")))
            return float(m.group(1)) if m else None
        z_low = max(_nex_min_z(reach) if callable(globals().get("_nex_min_z")) else -70.0, (0.12 + grip - NEX_BASE_H_M) * 1000)

        async def feel(reach_, yaw_, z_from, z_stop):
            """從 z_from 往下每次 10 mm，看實體手臂回報的高度；頂到東西（實際高度比指令高）回那個高度，降到 z_stop 都沒頂到回 None。
            2026-11-17：195 就頂到了卻等「落後 7 mm」才停，多壓三步把棒子壓歪 → 跟空中那幾步的落後量比，多 4 mm 就算碰到，退回不壓著。
            落後量自己會抖 ±3 mm → 再往下送 8 mm 確認一次：有跟著降 4 mm 以上只是雜訊。手臂伸遠時自己下垂（實際比指令低）不算碰到。"""
            z_cmd, air = z_from, []
            while z_cmd - 10.0 >= z_stop:
                z_cmd -= 10.0
                r_ = await arm(reach_, z_cmd, yaw_)
                zr = real_z(r_)
                lag = None if zr is None else zr - z_cmd
                base = sorted(air)[len(air) // 2] if len(air) >= 2 else None
                if (zr is None and not r_.get("ok")) or (lag is not None and (lag >= 7.0 or (base is not None and lag - base >= 4.0))):
                    if zr is not None and z_cmd - 8.0 >= z_stop:
                        r2 = await arm(reach_, z_cmd - 8.0, yaw_)
                        zr2 = real_z(r2)
                        if zr2 is not None and zr2 <= zr - 4.0:
                            z_cmd -= 8.0
                            continue
                    t_ = zr if zr is not None else z_cmd + 10.0
                    await arm(reach_, t_ + 3.0, yaw_)             # 放鬆：停在剛碰到的高度，不往下壓
                    return t_
                if lag is not None:
                    air.append(lag)
                await asyncio.sleep(0.15)
            return None

        # 2026-11-19（使用者：「他會放到箱子但是他沒對準箱子，東西掉下去了，他每次都這樣」）。今天三趟：
        #   18:50 斜對著箱角，棒子只壓在箱子右緣上，一鬆手翻下去；19:16 箱子窄的那一頭朝著車，手臂伸到箱子右邊外面，
        #   一路降到最低都沒碰到東西，卻照樣鬆手，棒子直接掉在地上。
        # 相機幾乎貼著箱頂的高度看過去，箱頂在畫面上只有十幾列，看不出哪裡是箱面、哪裡是邊緣；會碰的手臂才量得準：
        #   (1) 降到底都沒碰到 = 底下不是箱面 → 絕對不鬆手，改回正前方再碰一次，還是沒有就夾著退出來回報。
        #   (2) 碰到了，再往左右各 3.5 cm 碰一次（抬高越過箱頂的東西再移）：有一邊碰不到或低很多 = 那邊是邊緣 → 往另一邊挪 2.5 cm。
        #       已知有東西的那一邊不碰（免得壓到球或瓶子）。
        touched = await feel(reach, yaw, z_hi, z_low)
        if touched is None and abs(yaw) > 1.0:
            notes.append(f"nothing under the object at base yaw {yaw:+.1f} deg (lowered all the way): that spot is beside the box. Trying straight ahead")
            await arm(reach, z_hi, yaw)
            yaw = 0.0
            await arm(reach, z_hi, yaw)
            touched = await feel(reach, yaw, z_hi, z_low)
        if touched is None:
            await arm(reach, z_hi, yaw)
            await arm(160, z_hi)
            await arm(250, 200)
            return False, (f"place_on_top [{'; '.join(notes)}]: lowered the object all the way at reach {reach:.0f} mm and it never touched a surface: "
                           f"the box top is NOT under it (the robot is facing a corner or the narrow end of the box, or is not lined up with it). "
                           f"Nothing was released (still holding it; do not recenter_arm). Back up 0.4 m, drive to a spot where a long flat side of "
                           f"the box faces the robot squarely, then call place_on_top again.")
        # 2026-11-19 19:44（使用者在旁邊看：「這樣放到了不需要一直下壓」）：這裡原本碰到之後還往左右各 3.5 cm 再碰、回中間再碰一次，
        # 同一個東西在箱頂上壓了三次。現在車會先擺正對箱面（_square_to_box），放的位置在箱面裡面，碰到一次就放；
        # 「降到底都沒碰到就不鬆手」留著。
        top = touched / 1000.0 + NEX_BASE_H_M - grip
        notes.append(f"lowered until the object pressed on the surface: arm stopped at z {touched:.0f} mm -> the top is about {top:.2f} m high")
        if 0.08 <= top <= 0.40:
            _save_box_top(top)             # 記下量到的箱頂高度，之後從這個箱子上夾東西用得到
        z_mm = touched
        r = {"ok": True}
    if not r.get("ok"):
        # 2026-11-18：這裡原本 recenter_arm——東西還夾著，原位夾爪前傾 40°，棒子跟著歪（下一次呼叫重新合爪時夾爪角度從 −34° 變 −14°，它在指間滑過）。
        # 縮回來停在下一次 place_on_top 開頭用的姿勢就好。
        await arm(160, z_hi)
        await arm(250, 200)
        return False, (f"place_on_top [{'; '.join(notes)}]: could not get over the surface: {str(r.get('message',''))[:160]} "
                       f"(still holding it; call place_on_top again, do not recenter_arm)")
    await asyncio.sleep(0.5)
    await robot.act("gripper", {"state": "open"})
    await asyncio.sleep(1.0)
    if on_floor:
        await arm(max(r90 - 5, reach - 30), z_mm, yaw)             # 水平退出，指尖離開物件
    else:
        # 2026-11-19 19:44 錄影：位置對了、鬆手後棒子直直站在箱頂上，手臂「水平往後退 3 cm」時把它帶倒了——夾爪全開只比棒子寬一點，
        # 棒子是軟的止滑材質，兩根手指貼著它往後抽，夾點又在離底 12 cm 的高處，一拖就翻。
        # 改成直直往上抽離（摩擦力順著棒子的方向，它自己的重量壓著不會倒），高過頂端之後再往後退，退到它前面再降下來看。
        await arm(reach, z_mm + 40.0, yaw)
        await arm(reach, z_hi, yaw)
        r_back = max(r90 - 5, reach - 80)
        await arm(r_back, z_hi, yaw)
        await arm(r_back, z_mm, yaw)
    await asyncio.sleep(0.8)
    await robot.frame()
    jpg = await robot.frame()
    _save_frame(jpg, "place_on_top_check")
    name = str(last.get("object") or "object that was just put down")
    f = await perception.locate(llm, jpg, f"the {name} standing right in front of the camera")
    if not f.found:                                             # 退出後的畫面太近只看到頂端：再從看路姿勢看一次
        await arm(200, z_hi)
        await robot.act("recenter_arm", {})
        await asyncio.sleep(0.8)
        await robot.frame()
        jpg = await robot.frame()
        _save_frame(jpg, "place_on_top_check2")
        f = await perception.locate(llm, jpg, f"the {name} standing upright")
    tipped = _looks_tipped(jpg, f) if f.found else False
    missing = False
    if held_hue is not None and not on_floor:
        # 11-17：兩次都放在箱子前面的地上，模型卻回「看到它站在箱子上」（指到的是白瓶子）。有顏色的東西直接看畫面裡有沒有那個顏色的一大塊
        cb = _color_base(jpg, held_hue, None)
        bw, bh = getattr(_color_base, "last_wh", (0, 0)) if cb is not None else (0, 0)
        missing = cb is None or bw * bh < 2500 or bh < 60
    await arm(200, z_hi)
    await robot.act("recenter_arm", {})
    still = held(await robot.state())
    _LAST_TOP_GRIP.clear()
    notes.append(f"released at reach {reach:.0f} mm, base yaw {yaw:+.1f} deg")
    if still:
        return False, f"place_on_top [{'; '.join(notes)}]: opened the gripper but it still reports holding something."
    if missing:
        return False, (f"place_on_top [{'; '.join(notes)}]: released, but the camera does NOT see the {name} (its colour) on the box afterwards: "
                       f"it probably fell off or landed on the floor. Not a success; look for it (survey_scene) and try again.")
    if tipped:
        return False, (f"place_on_top [{'; '.join(notes)}]: released, but the {name} is NOT standing: in the camera it is as wide as it is "
                       f"tall (lying down or leaning on the gripper). Not a success; frame in runs/_skill_frames/.")
    if f.found:
        return True, (f"place_on_top [{'; '.join(notes)}]: released, the gripper is empty and the camera sees the {name} standing "
                      f"upright {'on the floor' if on_floor else 'on the box'}. The arm is back in the driving pose.")
    return False, (f"place_on_top [{'; '.join(notes)}]: released and the gripper is empty, but the camera could NOT confirm the {name} "
                   f"is standing upright (tipped over or fell off?). Not a success; frame in runs/_skill_frames/.")


def _looks_tipped(jpg, found) -> bool:
    """鬆手後的檢查（2026-11-17）：模型說「看到它站著」，實際是倒向夾爪、鏡頭只看到圓圓的頂端。有顏色的東西看色塊形狀：
    又大、又圓、又實心（寬高差不多、填滿外框 65% 以上）= 倒下來對著鏡頭。只抓這一種倒法；紙箱跟橘色同色相，
    色塊會連到箱子（形狀不圓、不實心）就不判；灰白色的東西圈不出來也不判（回 False）。"""
    import cv2
    import numpy as np
    ref = _hue_ref(jpg, found.x, found.y)
    if ref is None:
        return False
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dh = np.abs(hsv[:, :, 0].astype(np.int16) - int(ref[0]))
    mask = ((np.minimum(dh, 180 - dh) <= 8) & (hsv[:, :, 1] >= 150) & (hsv[:, :, 2] >= 60)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h, w = mask.shape
    k = lab[min(h - 1, int(found.y / 1000.0 * h)), min(w - 1, int(found.x / 1000.0 * w))]
    if k == 0:
        return False
    bw, bh, area = (int(stats[k, c]) for c in (cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
    return bw >= 0.10 * w and 0.75 <= bh / float(bw) <= 1.35 and area >= 0.65 * bw * bh


async def survey_scene(robot, llm) -> Tuple[bool, str]:
    """實體場地的「先看清楚再動手」（2026-11-14 陷阱題失敗後加的）：相機轉成俯 20–30° 拍箱子一帶，列出每樣小物件
    和它站在哪（箱子上 / 地上），手臂收回。不移動底盤。模擬裡 scan_overhead 做的事，這裡用前鏡頭做。"""
    st = await robot.state()
    tof = st.get("real_tof_front_mm", st.get("tof_front_mm"))
    await robot.act("arm_to", {"x_mm": 250, "y_mm": 150, "pitch_deg": 0.0, "yaw_deg": 0.0})
    await asyncio.sleep(1.0)
    await robot.frame()
    jpg = await robot.frame()
    _save_frame(jpg, "survey_scene")
    items, raw = await perception.survey_front(llm, jpg)
    await robot.act("recenter_arm", {})
    if items is None:
        return False, f"survey_scene: could not read the scene from the camera ({raw[:120]}). Try again from 0.8-1.2 m in front of the box."
    if not items:
        return True, ("survey_scene: no small objects seen on or around the box from here. If you are closer than 0.6 m the "
                      "box may be below the view: back up to about 1 m and survey again.")
    on_box = [i["label"] for i in items if "box" in i["where"]]
    floor = [i["label"] for i in items if "floor" in i["where"]]
    other = [i["label"] for i in items if "box" not in i["where"] and "floor" not in i["where"]]
    msg = (f"survey_scene (front camera looking down): ON THE BOX: {', '.join(on_box) if on_box else 'nothing'}. "
           f"ON THE FLOOR (not on the box): {', '.join(floor) if floor else 'nothing'}."
           + (f" Elsewhere: {', '.join(other)}." if other else "")
           + " Things ON THE BOX are grasped with pick_from_top('<label>'). Things standing ON THE FLOOR can be grasped too if they are at least 12 cm tall: pick_from_top('<label>', surface_m=0)."
           + (" NOTE: you are closer than 0.6 m, part of the box may be out of view; survey from about 1 m for a full list."
              if isinstance(tof, (int, float)) and tof < 600 else ""))
    return True, msg


_HOME_DOT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "_home_dot.json")
_HOME_DOT_POSE = {"x_mm": 380, "y_mm": 120, "pitch_deg": 0.0, "yaw_deg": 0.0}


def _find_park_row(jpg):
    """停車格（2026-11-16 使用者：「我貼了多個白色的點讓他能夠知道停車的位置，要停在點點的裡面」）：俯視姿勢下地上的白點排成一個框，
    取「最遠那一排」（車頭那一邊，畫面上 y 最小的一排）最左與最右兩個點的中點當定位基準。比單一白點穩：不會認錯是哪一顆，
    踢腳板、地毯的白色碎塊也湊不成一排。回 (mid_x, mid_y, 斜率, 點數)（0–1000）或 None。"""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 185) & (hsv[:, :, 1] < 60)).astype(np.uint8)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    dots = []
    for i in range(1, n):
        a, bw, bh = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        x, y = cent[i][0] / w * 1000.0, cent[i][1] / h * 1000.0
        # 地板上的貼紙：小、接近圓（俯視會壓扁成橫的橢圓）、在畫面下半部的中間地帶（上面是遠處的牆與踢腳板，兩側是地毯）
        # y > 500（原本 430）：新的停車格最遠那排只有 97 單位寬，下面的寬度下限要跟著放寬到 60，
        # 畫面左上角白地毯的碎塊（y 437–459、彼此相距 18–74）就可能湊成一排，所以把它們所在的那一帶排除。
        if 8 <= a <= 320 and 0.8 <= bw / max(1, bh) <= 3.0 and a / float(bw * bh) > 0.5 and y > 500 and 150 < x < 900:
            dots.append((x, y))
    if len(dots) < 2:
        return None
    dots.sort(key=lambda d: d[1])
    # 2026-11-19 停車格重貼之後：最遠那排（y 638–666）跟它後面左邊那兩顆（y 698–701）剛好差 60，同一個位置拍兩張，一張算 3 顆、一張算 5 顆，
    # 中點就在 (592,640) 和 (552,670) 之間跳（差 40 單位，比對準的容許 18 還大）。同一排的點 y 差不到 30，門檻收到 40。
    for k in range(len(dots) - 1):                      # 從最遠的點開始找「一排」：y 相近（40 內）的 2 顆以上、左右跨 60–450
        row = [d for d in dots if abs(d[1] - dots[k][1]) <= 40]
        xs = [d[0] for d in row]
        if len(row) >= 2 and 60 <= max(xs) - min(xs) <= 450:
            L, R = min(row, key=lambda d: d[0]), max(row, key=lambda d: d[0])
            return ((L[0] + R[0]) / 2.0, (L[1] + R[1]) / 2.0, (R[1] - L[1]) / max(1.0, R[0] - L[0]), len(row))
    return None



async def _move_exact(robot, fwd, right, tol=0.02, arm_low=False):
    """用實體里程計閉迴路走一段（車頭方向不變）：底盤指令的實際距離會差 ±10 cm（2026-11-15：送 −0.55 走了 0.595、送 0.642 只走 0.54），
    走完看里程計差多少再補，最多補 3 次；小於 9 cm 的補正用「反向退 12 cm 再走」。"""
    # arm_low=True：手臂降在車頭前方時夾爪擋在 ToF 前面，ToF 讀到的是自己的夾爪（0.14 m），防撞會把每一步砍成 0
    # （11-15：邊看邊往前 12 步畫面完全沒變）。這時 ToF 本來就是瞎的，只能關掉它；最前面的是夾爪，每步很短（≤15 cm）而且每步都看畫面。
    st = await robot.state()
    p0, hd = st.get("real_world_xy") or st.get("world_xy"), st.get("real_world_heading_deg", st.get("world_heading_deg", 0.0))
    h = math.radians(hd or 0.0)
    for _ in range(4):
        st = await robot.state()
        p = st.get("real_world_xy") or st.get("world_xy")
        dx, dy = p[0] - p0[0], p[1] - p0[1]
        done_f, done_r = dx * math.cos(h) + dy * math.sin(h), dx * math.sin(h) - dy * math.cos(h)
        rf, rr = fwd - done_f, right - done_r
        if abs(rf) <= tol and abs(rr) <= tol:
            break
        for key, v in (("forward_m", rf), ("right_m", rr)):
            if abs(v) <= tol:
                continue
            if abs(v) < 0.09:
                # 2026-11-17（使用者：「原本對準了…移動底盤之後又偏移了」）：舊做法是「反向退 12 cm 再走 20 cm」，夾取前為了往前 8 cm
                # 連做四五組來回，麥輪每次換向都滑一點，對好的左右就跑掉了。小步其實走得動（送 0.05–0.06，ToF 量到走了 2–7 cm），
                # 不退了，直接送要走的距離，最少 5 cm。（我一度改成「÷0.72」：那是照動作訊息裡的 travelled 算的，它少報；
                # 用 ToF 對照，送 0.10 實際走 96–112 mm，除了反而多走 3 cm 又退回來。）
                v = math.copysign(max(0.05, abs(v)), v)
            await robot.act("move_chassis", {"forward_m": round(v, 3) if key == "forward_m" else 0,
                                             "right_m": round(v, 3) if key == "right_m" else 0, "turn_left_deg": 0, "push": bool(arm_low), "near_ok": True})
        await asyncio.sleep(0.5)
    return rf, rr


async def return_home(robot, llm=None) -> Tuple[bool, str]:
    """回停車格：里程計回原點 → 退到停車格後方 → 相機俯視，把地上那一排白點（停車格車頭那一邊）的中點對到基準畫面位置 →
    往前開進格子。基準 runs/_home_dot.json 用 tools/park_calib.py 校正（車由人擺在格子正中間時執行），2026-11-16 實機驗證、
    使用者確認「有在框框裡面」。沒有基準檔、基準是舊的單點格式、不是實體場地、或看不到那排白點時，只做里程計回原點並照實說。"""
    r = await robot.act("return_to_origin", {})
    base = str(r.get("message", ""))[:200]
    st = await robot.state()
    if "real_world_xy" not in st or not os.path.exists(_HOME_DOT_FILE):
        return bool(r.get("ok")), base
    with open(_HOME_DOT_FILE, encoding="utf-8") as fh:
        ref = json.load(fh)
    if not ref.get("row"):
        return bool(r.get("ok")), ("return_to_origin: back at the start by odometry only (the parking reference is in the old single-dot "
                                   "format: recalibrate with tools/park_calib.py). " + base)
    await _move_exact(robot, -ref["forward_m"], -ref["right_m"], tol=0.04)
    await robot.act("arm_to", _HOME_DOT_POSE)
    ok, log, last_e = False, [], None
    travelled, moved_last, fwd_seek = 0.0, False, 0
    if ref.get("gap_x") is not None:
        # 2026-11-16：停車格基準的 gap 是影像法（_gripper_gap_cv）量的；這裡原本用 LLM 指的開口（定義不同：同一張畫面 y 差約 100、x 差 13），
        # 目標位置被平移錯，使用者：「沒有回到原位，他偏右邊」。用同一個方法量，差太多（> 60）就當作沒量到、直接用基準。
        await asyncio.sleep(0.8)
        await robot.frame()
        g = _gripper_gap_cv(await robot.frame())
        if g is not None and abs(g[0] - ref["gap_x"]) <= 60 and abs(g[1] - ref["gap_y"]) <= 60:
            ref = dict(ref, x=ref["x"] + g[0] - ref["gap_x"], y=ref["y"] + g[1] - ref["gap_y"])
            log.append(f"[gripper {g[0]:.0f},{g[1]:.0f} -> target {ref['x']:.0f},{ref['y']:.0f}]")
    for _ in range(12):
        await asyncio.sleep(1.0)
        await robot.frame()
        jpg = await robot.frame()
        _save_frame(jpg, "home_dot")
        pr = _find_park_row(jpg) if jpg else None      # 停車格最遠那一排白點的中點
        dot = (pr[0], pr[1]) if pr else None
        if dot is None:
            # 里程計漂掉時退到的位置會離白點太遠，白點在畫面上緣外（11-15：連兩次「dot not seen」，其實它就在上緣）→ 往前 0.3 m 再看，最多 3 次
            if last_e is None and fwd_seek < 3:
                fwd_seek += 1
                await robot.act("move_chassis", {"forward_m": 0.3, "right_m": 0, "turn_left_deg": 0, "push": False, "near_ok": True})
                log.append("not seen, moved 0.3 m forward to look")
                continue
            log.append("dot not seen")
            break
        ex, ey = dot[0] - ref["x"], dot[1] - ref["y"]
        log.append(f"({ex:+.0f},{ey:+.0f})")
        if (last_e is None and (abs(ex) > 330 or abs(ey) > 330)) or travelled > 0.9 or \
                (last_e is not None and moved_last and abs(ex - last_e[0]) < 8 and abs(ey - last_e[1]) < 8):
            if last_e is not None and abs(ex) <= 80 and abs(ey) <= 80:
                # 2026-11-16：誤差已經在 80 單位（約 12 cm）內卻「走了沒變」，是底盤走不出那幾公分的小步，不是追錯東西。
                # 以前這裡一律當成「不是白點」放棄，連最後補殘差那一步都跳過（兩題都停在離家 4–6 cm 還回報沒對到）。
                # 留著這個誤差，交給下面用里程計閉迴路的最後一段一起走掉。
                log.append("chassis cannot make this small correction: carrying the remainder into the final move")
                last_e = (ex, ey)
                break
            log.append("NOT the home marker (too far off / not responding to moves): stopped following it")
            last_e = None
            break
        last_e = (ex, ey)
        moved_last = False
        if abs(ex) <= 18 and abs(ey) <= 18:
            ok = True
            break
        # 畫面 1 單位 ≈ 1.5 mm；白點偏右 → 車往右，白點偏下（太近）→ 車後退。底盤走不出 5 cm 以下的小步：先反向退 8 cm 再走
        for key, v in (("forward_m", max(-0.25, min(0.25, -0.8 * ey * 0.0015))), ("right_m", max(-0.25, min(0.25, 0.8 * ex * 0.0015)))):
            if abs(v) < 0.008:
                continue
            if abs(v) < 0.09:        # 5–8 cm 的步子底盤也常常不動（19:00 回家前後方向卡在 −45 單位收斂不了）
                back = -0.12 if v > 0 else 0.12
                await robot.act("move_chassis", {"forward_m": back if key == "forward_m" else 0, "right_m": back if key == "right_m" else 0,
                                                 "turn_left_deg": 0, "push": False, "near_ok": True})
                v -= back
            await robot.act("move_chassis", {"forward_m": round(v, 3) if key == "forward_m" else 0,
                                             "right_m": round(v, 3) if key == "right_m" else 0, "turn_left_deg": 0, "push": False, "near_ok": True})
            travelled += abs(v)
            moved_last = True
    await robot.act("arm_to", {"x_mm": 250, "y_mm": 200, "pitch_deg": 0.0, "yaw_deg": 0.0})
    # 剩下沒對掉的幾公分（底盤小步走不準，前後方向常卡在 30–40 單位）直接加進最後這段用里程計閉迴路走的距離裡
    add_f = add_r = 0.0
    if last_e is not None and abs(last_e[0]) <= 80 and abs(last_e[1]) <= 80:
        add_f, add_r = -last_e[1] * 0.0015, last_e[0] * 0.0015
        ok = True
    if ok:
        await _move_exact(robot, ref["forward_m"] + add_f, ref["right_m"] + add_r)
    else:
        # 2026-11-19：白點從頭到尾沒對到時，上面「往前 0.3 m 再看」最多三次的 0.9 m 以前沒退回來，照樣再往前走 forward_m，
        # 車停在原點前面 0.9–1.05 m、離箱子 0.42 m（停車格那天被重貼、舊基準對不到）。沒對到就老實用里程計回原點。
        await robot.act("return_to_origin", {})
    await robot.act("recenter_arm", {})
    if ok:
        return True, f"return_to_origin: back home, aligned on the white floor marker (pixel error steps {' '.join(log)}). {base}"
    return bool(r.get("ok")), (f"return_to_origin: back at the start by odometry only; the white floor marker could not be aligned "
                               f"({' '.join(log)}), so the position may be several cm off. {base}")


# ---- 模擬裡的 NexArm：遊戲區的 pick / place（2026-11-19）----
# 使用者：「模擬的時候也要用最新的機器人」。遊戲區原本的 pick/place（skills.py）整套照 EP 原廠手臂校的：抓取姿勢、
# 前方 ToF 送進指間的距離、相機夾爪窗口與手腕力感測器的驗證。換成 NexArm 後手臂停在不對的地方，驗證還會誤判
# （h01：積木一公釐沒動，畫面裡的藍色溜滑梯被當成夾在指間的藍積木，回報 GRASPED）。
# 這裡不沿用那一套，改成照 NexArm 自己的幾何：物件的世界座標（俯視相機）＋ 機器人的位姿 → 物件在手臂底座座標裡的
# 距離與方位 → 手臂直接伸到那一點。夾到沒只看手臂自己回報的 gripper_holding（實體是夾爪伺服停在目標前，模擬是物件掛接上）。
# 實體場地不走這裡（沒有俯視相機與世界座標，用上面的 pick_from_top / place_on_top）。
NEX_SIM_BASE_H_M = 0.140     # 模擬模型：arm_to 的高度 0 在離地這麼高（指令 z=−77 時量到夾點離地 0.063 m）。實體是 0.162。
NEX_SIM_STAND_M = 0.37       # 夾取時車中心到物件的距離：手臂伸 370 mm，車頭（中心前 0.166 m）離物件 0.20 m
NEX_SIM_GRIP_UP_M = 0.085    # 直立的積木／垃圾（15–20 cm 高）夾在離底部這麼高的地方
NEX_SIM_BALL_UP_M = 0.136    # 球坐在球座上：球心離球座所在的面這麼高（球座 11.4 cm + 球半徑）
NEX_SIM_CARRY = (250, 150)   # 搬運姿勢：夾點離地 0.29 m，20 cm 的瓶子底部離地 0.20 m，高過桶壁（0.10 m）與前方 ToF 的光束（0.10 m）
_SIM_HELD = {}               # 手上那件東西：{"below_m": 夾點到物件底部的距離}，place 用它算放下的高度


def is_sim_nexarm(state: dict) -> bool:
    """純模擬、而且場景裡裝的是 NexArm。鏡射到實體時（遙測裡有 real_*）不算：實體有自己的技能。"""
    return bool(state) and "arm_pose" in state and state.get("mode") == "sim" and "real_tof_front_mm" not in state


def _sim_rel(state: dict, xy):
    """目標點相對機器人：(距離 m, 方位角 度，左為正)。"""
    rxy, hd = state.get("world_xy"), state.get("world_heading_deg")
    if not rxy or hd is None:
        return None, None
    d = math.hypot(xy[0] - rxy[0], xy[1] - rxy[1])
    b = (math.degrees(math.atan2(xy[1] - rxy[1], xy[0] - rxy[0])) - hd + 180) % 360 - 180
    return d, b


async def _sim_stand(robot, xy, want_m: float, tol_m: float = 0.04, push: bool = False):
    """面向 xy、站到離它 want_m 的地方（底盤小步修正，最多四輪）。回 (距離, 方位角)；位姿讀不到回 (None, None)。
    push=True 關掉前方 ToF 的防撞（放進桶子時車頭要貼到桶壁 3–4 cm，防撞會停在 15 cm 外）。"""
    d = b = None
    for _ in range(4):
        d, b = _sim_rel(await robot.state(), xy)
        if d is None:
            return None, None
        if abs(b) > 3.0:
            await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": round(b, 1)})
            continue
        if abs(d - want_m) <= tol_m:
            break
        step = max(-0.4, min(0.4, d - want_m))
        if abs(step) < 0.02:
            break
        await robot.act("move_chassis", {"forward_m": round(step, 2), "right_m": 0, "turn_left_deg": 0, "push": bool(push and step > 0),
                                         "near_ok": True})
    d, b = _sim_rel(await robot.state(), xy)
    return d, b


async def _sim_arm(robot, reach_m: float, h_m: float, yaw_deg: float):
    """夾點移到「離底座 reach_m、離地 h_m、底座轉 yaw_deg」；夾爪水平。"""
    return await robot.act("arm_to", {"x_mm": round(reach_m * 1000.0), "y_mm": round((h_m - NEX_SIM_BASE_H_M) * 1000.0),
                                      "pitch_deg": 0.0, "yaw_deg": round(yaw_deg, 1)})


async def _sim_holding(robot) -> bool:
    await asyncio.sleep(0.2)
    return bool((await robot.state()).get("gripper_holding"))


async def pick_sim(robot, llm, obj: str) -> Tuple[bool, str]:
    """模擬裡用 NexArm 夾起一個物件：導航到附近 → 面向它站到 0.37 m → 夾爪張開、從物件正上方降到夾取高度 → 合爪 →
    手臂回報夾住了才抬起收到搬運姿勢，再確認一次還夾著。"""
    from robot_agent import skills
    xy, kind, note, surface = await skills.resolve_target(robot, llm, obj)
    if kind == "front":
        # 「in front of me」：沒有座標（平台底下俯視相機看不到的東西），用前方 ToF 的讀值當距離，物件在正前方
        st = await robot.state()
        tof, rxy, hd = st.get("tof_front_mm"), st.get("world_xy"), st.get("world_heading_deg")
        if not isinstance(tof, (int, float)) or tof >= 700 or not rxy or hd is None:
            return False, (f"pick({obj}): nothing within 0.7 m on the front distance sensor, so there is no object straight ahead to grasp. "
                           f"face('{_plain_label(obj)}') and drive closer first.")
        d0 = tof / 1000.0 + 0.166 + 0.022
        xy = (rxy[0] + d0 * math.cos(math.radians(hd)), rxy[1] + d0 * math.sin(math.radians(hd)))
        surface = None
    if xy is None:
        return False, f"pick({obj}): {note}"
    where = f"[{kind} at world ({xy[0]:.2f}, {xy[1]:.2f}){note}]"
    if await _sim_holding(robot):
        return False, f"pick({obj}) {where}: the gripper is already holding something; place it first."
    d, _b = _sim_rel(await robot.state(), xy)
    if d is None or d > 0.6:
        res = await robot.act("navigate_to", await skills._nav_params(robot, xy, 0.30, surface, label=obj), timeout=skills.NAV_TIMEOUT_S)
        if not res.get("ok"):
            return False, f"pick({obj}) {where}: could not reach it — {res.get('message')}"
    is_ball = "ball" in obj.lower()
    base = float(surface["top_m"]) if surface else 0.0
    # 球先當它還在球座上；沒夾到再試一次貼著那個面的高度（掉下球座的球）。其他東西夾在離底部 8.5 cm 的地方。
    heights = [(NEX_SIM_BALL_UP_M, "on its post"), (0.03, "lying on the surface")] if is_ball else [(NEX_SIM_GRIP_UP_M, "upright")]
    attempts = []
    for up, how in heights:
        d, b = await _sim_stand(robot, xy, NEX_SIM_STAND_M)
        if d is None:
            return False, f"pick({obj}) {where}: lost the robot's world pose."
        if not (0.28 <= d <= 0.45):
            await robot.act("recenter_arm", {})
            return False, (f"pick({obj}) {where}: could not stand at grasping distance (now {d:.2f} m from it; the arm reaches 0.28–0.45 m). "
                           f"Something is in the way: approach it from another side and retry.")
        h = base + up
        await robot.act("gripper", {"state": "open"})
        await skills._snap(robot, "pick: gripper opened")
        r = await _sim_arm(robot, d, h + 0.09, b)
        if r.get("ok"):
            r = await _sim_arm(robot, d, h, b)
        if not r.get("ok"):
            attempts.append(f"{how}: the arm could not get to {d * 1000:.0f} mm out / {h * 100:.0f} cm above the floor ({str(r.get('message'))[:90]})")
            continue
        g = await robot.act("gripper", {"state": "close"})
        await skills._snap(robot, "pick: gripper closed on the object")
        if not (g.get("holding") or await _sim_holding(robot)):
            attempts.append(f"{how}: closed the gripper at {d * 1000:.0f} mm out, {h * 100:.0f} cm above the floor, and nothing was between the fingers")
            await robot.act("gripper", {"state": "open"})
            await _sim_arm(robot, d, h + 0.09, b)
            continue
        await _sim_arm(robot, d, h + 0.10, b)
        await skills._snap(robot, "pick: lifted")
        await robot.act("arm_to", {"x_mm": NEX_SIM_CARRY[0], "y_mm": NEX_SIM_CARRY[1], "pitch_deg": 0.0, "yaw_deg": 0.0})
        if not await _sim_holding(robot):
            attempts.append(f"{how}: the gripper closed on it but it was gone after lifting")
            continue
        _SIM_HELD.clear()
        _SIM_HELD["below_m"] = 0.0225 if is_ball else up
        await robot.act("set_held", {"tof_mm": None})
        return True, (f"pick({obj}) {where}: GRASPED — the gripper stopped on the object ({how}, {d * 1000:.0f} mm out, {h * 100:.0f} cm above "
                      f"the floor) and still reports holding it after lifting to the carry pose. Keep the gripper closed and use "
                      f"place(target) to put it down.")
    await robot.act("gripper", {"state": "close"})
    await robot.act("recenter_arm", {})
    return False, (f"pick({obj}) {where}: NOT grasped ({'; '.join(attempts)}). The object is probably not exactly at those coordinates "
                   f"(pushed, rolled away, or a different object): find it again with locate_overhead / scan_overhead and retry with the "
                   f"new coordinates, or report the failure honestly.")


async def place_sim(robot, llm, target: str) -> Tuple[bool, str]:
    """模擬裡用 NexArm 把手上的東西放到目標：桶子 → 夾點伸過桶緣上方再張開；桌子 → 放到桌面上方 2 cm；
    地墊／地面／座標 → 物件底部離地約 1.5 cm 再張開。放開後確認手臂回報已經沒夾著。"""
    from robot_agent import skills
    if not await _sim_holding(robot):
        return False, f"place({target}): the gripper is not holding anything (pick first)."
    xy, kind, note, surface = await skills.resolve_target(robot, llm, target)
    if kind == "front":
        return False, f"place({target}): 'in front of me' only makes sense for pick; give a landmark, an object description or coordinates"
    if xy is None:
        return False, f"place({target}): {note}"
    lname = note.split("'")[1] if kind == "landmark" and "'" in note else None
    is_bin = bool(lname and lname.startswith("bin_"))
    on_surface = surface if (kind == "landmark" and surface) else None
    where = f"[{kind} at world ({xy[0]:.2f}, {xy[1]:.2f}){note}]"
    below = float(_SIM_HELD.get("below_m", NEX_SIM_GRIP_UP_M))
    nav_target = on_surface or ({"name": lname} if is_bin else None)
    res = await robot.act("navigate_to", await skills._nav_params(robot, xy, 0.15 if is_bin else 0.25, nav_target, label=lname or target),
                          timeout=skills.NAV_TIMEOUT_S)
    if not res.get("ok"):
        return False, f"place({target}) {where}: could not reach it — {res.get('message')}; still holding the object"
    if is_bin or on_surface:
        lm = ((await robot.state()).get("static_landmarks") or {}).get(lname or on_surface["name"]) or {}
        bb = lm.get("bbox")
        half = min(bb[1] - bb[0], bb[3] - bb[2]) / 2 if bb else 0.245
        # 車頭貼到邊緣 4 cm：桶子 0.49 m 寬時車中心離桶心 0.45 m，手臂伸 0.43 m 剛好到桶心附近
        d, b = await _sim_stand(robot, xy, half + 0.166 + 0.04, tol_m=0.03, push=True)
        h = (0.10 if is_bin else float(on_surface["top_m"])) + below + (0.06 if is_bin else 0.02)
        how = "over the bin, above its rim" if is_bin else f"on top of the {on_surface['name']}"
        inside_min = (d or 9.9) - half + 0.07      # 至少伸過邊緣 7 cm，不然會掉在外面
    else:
        d, b = await _sim_stand(robot, xy, NEX_SIM_STAND_M)
        h = below + 0.015
        how = "on the floor / mat"
        inside_min = 0.28
    if d is None:
        return False, f"place({target}) {where}: lost the robot's world pose; still holding the object"
    r = {"ok": False, "message": "no reach tried"}
    for reach in (min(d, 0.43), 0.40, 0.36, 0.32, 0.28):
        if reach + 1e-6 < inside_min or reach > d + 0.02:
            continue
        r = await _sim_arm(robot, reach, h, b)
        if r.get("ok"):
            break
    if not r.get("ok"):
        await robot.act("arm_to", {"x_mm": NEX_SIM_CARRY[0], "y_mm": NEX_SIM_CARRY[1], "pitch_deg": 0.0, "yaw_deg": 0.0})
        return False, (f"place({target}) {where}: the arm could not get {how} from here ({d:.2f} m from its centre; "
                       f"{str(r.get('message'))[:100]}); still holding the object. Approach from another side and retry.")
    await skills._snap(robot, "place: positioned over the target")
    await robot.act("gripper", {"state": "open"})
    await robot.act("set_held", {"tof_mm": None})
    await skills._snap(robot, "place: gripper opened to release")
    await asyncio.sleep(1.0)
    still = await _sim_holding(robot)
    await robot.act("move_arm", {"forward_mm": 0, "up_mm": 80})
    await robot.act("move_chassis", {"forward_m": -0.25, "right_m": 0, "turn_left_deg": 0})
    await robot.act("recenter_arm", {})
    if still:
        return False, f"place({target}) {where}: opened the gripper {how} but the arm still reports holding the object."
    _SIM_HELD.clear()
    return True, (f"place({target}) {where}: released {how} (gripper {h * 100:.0f} cm above the floor, {reach * 1000:.0f} mm out); the arm "
                  f"reports the gripper empty, backed up 0.25 m, arm recentred. Check with scan_overhead / the camera if the task needs "
                  f"confirmation of where it ended up.")
