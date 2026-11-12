"""
依序測試底盤、夾爪、手臂、相機，確認控制得動。

執行方式：
    source ~/rmenv/bin/activate
    cd ~/下載/Robot Test
    python test_movement.py
"""

import sys
import time
import cv2

from rm_connection import connect, start_camera

MODE = sys.argv[1] if len(sys.argv) > 1 else "sim"  # python test_movement.py real → 實體機器人


def main():
    print(f"[1/5] 連線中 (mode={MODE}) ...")
    ep = connect(MODE)
    print("連線成功，韌體版本：", ep.get_version())

    try:
        print("[2/5] 底盤測試：前進兩秒，停止，原地轉兩秒")
        ep.chassis.drive_speed(x=0.2, y=0, z=0, timeout=2)
        time.sleep(2)
        ep.chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(0.5)
        ep.chassis.drive_speed(x=0, y=0, z=30, timeout=2)
        time.sleep(2)
        ep.chassis.drive_speed(x=0, y=0, z=0)

        print("[3/5] 夾爪測試：開，關")
        ep.gripper.open()
        time.sleep(1.5)
        ep.gripper.close()
        time.sleep(1.5)

        print("[4/5] 手臂測試：小幅度前後移動")
        ep.robotic_arm.move(x=20, y=0).wait_for_completed()
        ep.robotic_arm.move(x=-20, y=0).wait_for_completed()

        print("[5/5] 相機測試：擷取一張畫面存成 snapshot.jpg")
        img = None
        if start_camera(ep):
            time.sleep(2)
            try:
                img = ep.camera.read_cv2_image(timeout=5)
            except Exception as e:
                print("串流開了但 5 秒內沒有收到畫面：", repr(e))
            ep.camera.stop_video_stream()
        else:
            print("相機串流開不起來（機器人沒有回應串流指令）")

        if img is not None:
            cv2.imwrite("snapshot.jpg", img)
            print(f"拍到影像，尺寸 {img.shape}，存成 snapshot.jpg，可以在 VSCode 檔案總管點開看")
        else:
            print("沒有拿到影像 (img is None)，相機這段之後要再檢查")

        print("全部測試跑完" + ("" if img is not None else "（相機除外）"))

    finally:
        # 不管中間有沒有出錯，離開前一定要讓底盤停下來再關閉連線
        ep.chassis.drive_speed(x=0, y=0, z=0)
        ep.close()
        print("已關閉連線")


if __name__ == "__main__":
    main()
