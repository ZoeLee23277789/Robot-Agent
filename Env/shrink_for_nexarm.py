#!/usr/bin/env python
"""把 NexArm 版遊戲區（Capstone_Playground_nexarm.ttt）裡要夾的東西縮到實體夾爪夾得住的寬度。

為什麼（2026-11-19）：實體 NexArm 夾爪開到底只有 5 cm，場景裡的球 6 cm、積木與垃圾 6.5–7 cm，實體夾不起來，
模擬照實體的開口做之後也夾不起來。使用者決定：只在 NexArm 版場景把物件縮小，原廠手臂的 Capstone_Playground.ttt 不動。

做法：球、小積木、垃圾的寬度一律縮到 4.5 cm（高度不變，俯視掃描靠「顏色 + 高度」分種類，高度不能動）；
球是整顆等比例縮，球心降到剛好坐在球座（tee，頂面不動）上。會先備份場景再存檔。

用法（無視窗 CoppeliaSim 先載 NexArm 版場景、模擬停止、robot_server 不要連著——它啟動時會在場景裡加感測器，不能一起存進去）：
    COPPELIASIM_SCENE=scenes/Capstone_Playground_nexarm.ttt ./start_sim_headless.sh
    .venv/bin/python Env/shrink_for_nexarm.py            # 修改並存檔
    .venv/bin/python Env/shrink_for_nexarm.py --dry-run  # 只列出會改什麼
重複執行沒關係：已經是 4.5 cm 的不會再縮。
"""
import argparse
import os
import shutil
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

WIDTH_M = 0.045
PREFIXES = ("ball_", "small_", "trash_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sim = RemoteAPIClient(port=a.port).require("sim")
    scene = sim.getStringParam(sim.stringparam_scene_path_and_name)
    if not os.path.isabs(scene):   # 用相對路徑啟動（start_sim_headless.sh）時回的也是相對路徑，相對於 CoppeliaSim 的資料夾
        scene = os.path.join(os.environ.get("COPPELIASIM_DIR", os.path.expanduser("~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04")), scene)
    if not os.path.isfile(scene):
        raise SystemExit("找不到場景檔 %s（設 COPPELIASIM_DIR）" % scene)
    if "nexarm" not in os.path.basename(scene).lower():
        raise SystemExit("載入的是 %s，不是 NexArm 版場景：不改" % scene)
    if sim.getSimulationState() != sim.simulation_stopped:
        raise SystemExit("模擬還在跑：先停止（存檔會把跑到一半的位置存進去）")
    changed = 0
    for sh in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0):
        name = sim.getObjectAlias(sh)
        if not name.startswith(PREFIXES) or sim.getObjectInt32Param(sh, sim.shapeintparam_static):
            continue
        size = sim.getShapeBB(sh)[0]
        w = max(size[0], size[1])
        if w <= WIDTH_M + 1e-4:
            continue
        k = WIDTH_M / w
        ball = name.startswith("ball_")
        pos = sim.getObjectPosition(sh, sim.handle_world)
        new_z = pos[2] - (size[2] - size[2] * k) / 2.0 if ball else pos[2]   # 球：底部留在原來的高度（球座頂面）
        print("%-16s %.1f x %.1f x %.1f cm -> 寬 %.1f cm%s" % (name, size[0] * 100, size[1] * 100, size[2] * 100, WIDTH_M * 100,
                                                            "，球心 z %.3f -> %.3f" % (pos[2], new_z) if ball else ""))
        if not a.dry_run:
            sim.scaleObject(sh, k, k, k if ball else 1.0, 0)
            sim.setObjectPosition(sh, [pos[0], pos[1], new_z], sim.handle_world)
        changed += 1
    if a.dry_run or not changed:
        print("（%s）" % ("dry-run，沒有改" if a.dry_run else "沒有需要縮的物件"))
        return
    backup = scene.replace(".ttt", time.strftime("_backup_%Y%m%d_%H%M%S.ttt"))
    shutil.copy2(scene, backup)
    sim.saveScene(scene)
    print("縮了 %d 個物件，已存檔 %s（備份 %s）" % (changed, scene, backup))


if __name__ == "__main__":
    main()
