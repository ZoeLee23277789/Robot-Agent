#!/usr/bin/env python
"""
用程式幫 CoppeliaSim 的場景放東西，不用在介面裡手動點。

它會在機器人周圍放三個 5 公分的彩色方塊 (紅、藍、綠)，設定成可以被推動和夾取，
並且把那塊很大的 ConcretBlock 搬到機器人後方遠處，免得擋住視野。
重複執行沒關係，舊的方塊會先被刪掉再重放。

使用方式 (在 Agent 的 .venv 裡，CoppeliaSim 要開著、模擬要先按停止)：
    pip install coppeliasim-zmqremoteapi-client
    python tools/scene_setup.py
    python tools/scene_setup.py --distance 1.5        # 方塊離機器人 1.5 公尺
    python tools/scene_setup.py --only red            # 只放紅色
    python tools/scene_setup.py --clear               # 把方塊全部移除
"""

import argparse
import sys

BOXES = {
    #  名稱        顏色 RGB         相對機器人的方向 (世界座標的 x, y 單位向量)
    "red":   ("RedBox",   [1.0, 0.05, 0.05], (1.0, 0.0)),
    "blue":  ("BlueBox",  [0.05, 0.2, 1.0],  (0.0, 1.0)),
    "green": ("GreenBox", [0.05, 0.8, 0.1],  (0.0, -1.0)),
}
SIZE = 0.05  # 公尺。EP 的夾爪張開大約 10 公分，5 公分的方塊最好夾


def set_pos(sim, handle, pos):
    # 新舊版 CoppeliaSim 的參數順序不同，兩種都試
    try:
        sim.setObjectPosition(handle, pos, sim.handle_world)
    except Exception:
        sim.setObjectPosition(handle, sim.handle_world, pos)


def get_pos(sim, handle):
    try:
        return sim.getObjectPosition(handle, sim.handle_world)
    except Exception:
        return sim.getObjectPosition(handle, -1)


def find(sim, path):
    try:
        return sim.getObject(path, {"noError": True})
    except Exception:
        try:
            return sim.getObject(path)
        except Exception:
            return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=float, default=1.2, help="方塊離機器人的距離 (公尺)")
    ap.add_argument("--only", choices=list(BOXES), action="append", help="只放指定顏色，可重複給")
    ap.add_argument("--clear", action="store_true", help="只移除方塊，不放新的")
    ap.add_argument("--keep-block", action="store_true", help="不要搬動 ConcretBlock")
    ap.add_argument("--robot", default="/RoboMaster", help="機器人在場景樹裡的名稱")
    ap.add_argument("--port", type=int, default=23001,
                    help="ZMQ remote API 的 port。CoppeliaSim 啟動時的訊息會印出 rpcPort=多少，照那個數字填")
    args = ap.parse_args()

    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError:
        sys.exit("請先安裝：pip install coppeliasim-zmqremoteapi-client")

    print(f"連線到 CoppeliaSim (127.0.0.1:{args.port}) ... 如果十秒內沒反應，代表 port 不對，按 Ctrl+C 後用 --port 換一個")
    sim = RemoteAPIClient(port=args.port).require("sim")

    if sim.getSimulationState() != sim.simulation_stopped:
        sys.exit("⚠️  模擬正在執行或暫停中。請先在 CoppeliaSim 按停止 (工具列的方形按鈕，不是暫停)，再執行這支程式。\n"
                 "    模擬進行中加入的物件，在停止模擬時會全部消失。")

    # 1) 移除上一次放的方塊
    for alias, _, _ in BOXES.values():
        h = find(sim, "/" + alias)
        if h != -1:
            sim.removeObjects([h])
            print(f"   移除舊的 {alias}")
    if args.clear:
        print("✅ 已清除")
        return

    # 2) 找機器人
    robot = find(sim, args.robot)
    if robot == -1:
        sys.exit(f"❌ 場景裡找不到 {args.robot}。請用 --robot 指定場景樹裡顯示的名稱 (前面加 /)。")
    rx, ry, _ = get_pos(sim, robot)
    print(f"   機器人在 x={rx:.2f} y={ry:.2f}")

    # 3) 把大水泥塊搬到機器人後方 3 公尺
    if not args.keep_block:
        block = find(sim, "/ConcretBlock")
        if block != -1:
            bz = get_pos(sim, block)[2]
            set_pos(sim, block, [rx - 3.0, ry, bz])
            print("   ConcretBlock 已搬到機器人後方 3 公尺 (不想搬的話加 --keep-block)")

    # 4) 放方塊
    for key in (args.only or list(BOXES)):
        alias, rgb, (dx, dy) = BOXES[key]
        h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [SIZE, SIZE, SIZE], 0)
        sim.setObjectAlias(h, alias)
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, rgb)
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 0)       # dynamic：會受重力、可以被推和夾
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)  # respondable：會跟夾爪和地板碰撞
        try:
            sim.setShapeMass(h, 0.05)  # 50 公克，太重夾爪會夾不起來
        except Exception:
            pass
        pos = [rx + dx * args.distance, ry + dy * args.distance, SIZE / 2 + 0.002]
        set_pos(sim, h, pos)
        try:
            sim.resetDynamicObject(h)
        except Exception:
            pass
        print(f"   ✅ {alias} 放在 x={pos[0]:.2f} y={pos[1]:.2f}")

    print("\n完成。請到 CoppeliaSim 確認畫面上出現了方塊，然後 File > Save scene 存檔，再按開始模擬。")


if __name__ == "__main__":
    main()
