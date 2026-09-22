#!/usr/bin/env python
"""
量相機的水平視角 (HFOV)，順便驗證「左轉」在模擬器裡是不是真的往左。

做法：先請 Gemini Robotics ER 指出目標在畫面的位置，命令機器人左轉一個小角度，再指一次。
左轉之後物體在畫面上應該往右移；移動了多少，配上轉了幾度，就能反推視角。

使用前：server 要開著，而且目標 (預設 red box) 要在畫面裡、大致靠中間。
    python tools/calibrate_fov.py
    python tools/calibrate_fov.py --object "blue box" --turn 12
量到的數字填進 .env：ROBOT_CAMERA_HFOV=量到的值
"""

import argparse
import asyncio
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

from robot_agent import perception  # noqa: E402
from robot_agent.client import RobotClient, check_link  # noqa: E402
from robot_agent.llm_factory import build_llm  # noqa: E402


def solve_hfov(x1: float, x2: float, delta_deg: float) -> float:
    u1, u2 = (x1 - 500) / 500, (x2 - 500) / 500
    best, best_err = 0.0, 1e9
    for h10 in range(300, 1700):  # 30.0 到 170.0 度
        t = math.tan(math.radians(h10 / 20))
        err = abs(math.degrees(math.atan(u2 * t) - math.atan(u1 * t)) - delta_deg)
        if err < best_err:
            best, best_err = h10 / 10, err
    return best


async def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="red box")
    ap.add_argument("--turn", type=float, default=15.0, help="測試用的左轉角度")
    ap.add_argument("--provider", default="robotics-er")
    ap.add_argument("--robot-url", default=os.getenv("ROBOT_SERVER_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--repeat", type=int, default=3, help="每個位置指幾次取中位數，降低 pointing 的雜訊")
    args = ap.parse_args()

    llm = build_llm(args.provider)
    robot = RobotClient(args.robot_url, os.getenv("ROBOT_SERVER_TOKEN", ""))

    check_link(await robot.health())

    async def median_x(tag: str):
        xs = []
        for _ in range(args.repeat):
            r = await perception.locate(llm, await robot.frame(), args.object)
            if r.found:
                xs.append(r.x)
        print(f"   {tag}: x = {xs}")
        return sorted(xs)[len(xs) // 2] if xs else None

    x1 = await median_x("轉之前")
    if x1 is None:
        sys.exit(f"畫面裡找不到 {args.object}。先讓機器人面向它再執行 (例如用 RobotAgent 下 face 的任務)。")
    if not 250 <= x1 <= 650:
        print("   提醒：目標不在畫面中間偏左的區域，左轉後可能會跑出畫面。")

    res = await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": args.turn})
    print("   ", res.get("message"))
    measured = (res.get("measured") or {}).get("dyaw_deg")
    x2 = await median_x("轉之後")
    await robot.act("move_chassis", {"forward_m": 0, "right_m": 0, "turn_left_deg": -args.turn})  # 轉回原位
    if x2 is None:
        sys.exit("轉完之後目標不見了。把 --turn 調小一點再試。")

    print()
    if x2 > x1 + 15:
        print("✅ 方向正確：命令左轉之後，物體在畫面上往右移。")
    elif x2 < x1 - 15:
        print("❌ 方向相反：命令左轉，物體卻往左移，代表機器人實際上是往右轉。請把這個結果告訴 Claude，要改 robot_server.py 的正負號。")
    else:
        sys.exit("物體幾乎沒移動，無法判斷。確認模擬有在跑，或把 --turn 調大。")

    delta = abs(measured) if measured and abs(abs(measured) - args.turn) < args.turn * 0.5 else args.turn
    if measured is not None:
        print(f"   里程計量到的 yaw 變化：{measured:+.1f} 度 (命令 {args.turn:+.1f}，左轉為正的慣例下"
              f"{'一致' if measured > 0 else '符號相反，SDK 的 yaw 以右轉為正，這是正常的'})")
    hfov = solve_hfov(min(x1, x2), max(x1, x2), delta)
    print(f"📐 估計的水平視角 HFOV ≈ {hfov:.0f} 度")
    print(f"   請在 .env 加上：ROBOT_CAMERA_HFOV={hfov:.0f}")


if __name__ == "__main__":
    asyncio.run(main())
