#!/usr/bin/env python
"""
相機裝在手臂上，手臂的姿勢決定相機的高度和俯仰角。這支程式把手臂掃過一組姿勢，
每個姿勢拍一張，拼成一張總覽圖，用來挑出「看遠」和「看近 (夾取)」兩個最好用的視角。
    .venv/bin/python tools/arm_view_sweep.py
輸出：runs/arm_views/contact_sheet.jpg
"""
import asyncio
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot_agent.client import RobotClient, check_link  # noqa: E402

XS = [80, 120, 160, 200]      # 手臂前伸 mm
YS = [130, 90, 50, 10]        # 手臂高度 mm，由高到低


async def main():
    robot = RobotClient(os.getenv("ROBOT_SERVER_URL", "http://127.0.0.1:8765"), os.getenv("ROBOT_SERVER_TOKEN", ""))
    check_link(await robot.health())
    out = "runs/arm_views"
    os.makedirs(out, exist_ok=True)
    start = (await robot.state()).get("arm_mm") or {"x": 108, "y": 67}
    rows = []
    for y in YS:
        row = []
        for x in XS:
            res = await robot.act("arm_to", {"x_mm": x, "y_mm": y})
            now = (await robot.state()).get("arm_mm") or {}
            jpg = await robot.frame()
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR) if jpg else np.zeros((360, 640, 3), np.uint8)
            img = cv2.resize(img, (480, 270))
            label = "cmd x=%d y=%d | actual x=%s y=%s%s" % (x, y, now.get("x"), now.get("y"), "" if res.get("ok") else " | LIMIT")
            cv2.rectangle(img, (0, 0), (480, 22), (255, 255, 255), -1)
            cv2.putText(img, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200) if not res.get("ok") else (0, 0, 0), 1)
            cv2.imwrite(f"{out}/arm_x{x}_y{y}.jpg", img)
            row.append(img)
            print("  ", label)
        rows.append(np.hstack(row))
    cv2.imwrite(f"{out}/contact_sheet.jpg", np.vstack(rows))
    await robot.act("arm_to", {"x_mm": start["x"], "y_mm": start["y"]})
    print(f"\n完成：{out}/contact_sheet.jpg  (手臂已回到原本的位置)")


if __name__ == "__main__":
    asyncio.run(main())
