#!/usr/bin/env python
"""把 EP 設成路由器模式（STA）：產生 Wi-Fi QR code 給 EP 的相機掃，然後等它加入網路回報 IP。

    ~/rmenv/bin/python tools/ep_wifi_qr.py --ssid Arriba_5G --password '...'

流程（跟 RoboMaster App 的「路由器模式」一模一樣，只是 QR code 由這裡產生）：
  1. 中控上的連線開關撥到「路由器」那一側，EP 開機、相機要接著（中控只在開機時找相機）
  2. 這支程式把 QR code 存成 camera_shots/ep_wifi_qr.png，打開圖片放大，拿到 EP 相機前 20～30 cm
  3. EP 掃到之後會去連那個 Wi-Fi，連上後用 UDP 45678 廣播；這台電腦要在同一個網段（有線 eno1 也算）才收得到
  4. 收到就印出 EP 的 IP；之後 RM_CONN_TYPE=sta ./start_arm.sh --with-ep（rm_connection.pick_real_ep 會自己聽廣播找它）
EP 記得這組 Wi-Fi，之後開機只要開關還在路由器那側就會自己連，不用再掃。
"""
import argparse
import os
import socket
import sys

import qrcode
from robomaster import conn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssid", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--out", default="camera_shots/ep_wifi_qr.png")
    ap.add_argument("--no-wait", action="store_true", help="只產生圖片，不等機器人回報")
    a = ap.parse_args()
    helper = conn.ConnectionHelper()
    info = helper.build_qrcode_string(ssid=a.ssid, password=a.password)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    qrcode.make(info).resize((600, 600)).save(a.out)
    print("QR code 存在 %s（內容是 SSID %s 的連線資訊，不要外流）" % (os.path.abspath(a.out), a.ssid), flush=True)
    if a.no_wait:
        return 0
    print("等 EP 掃碼並連上 Wi-Fi（最多 6 分鐘）... 把圖片開大、拿到 EP 相機前 20～30 cm，中控燈變色就是掃到了", flush=True)
    if not helper.wait_for_connection():
        print("沒等到 EP 的回報：開關在路由器那側嗎？相機有接？Wi-Fi 密碼對嗎？電腦跟那個 Wi-Fi 是同一個網段嗎？")
        return 1
    print("EP 已加入 %s" % a.ssid)
    # 再聽一次廣播拿 IP（helper 本身沒回傳 IP）
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Test_Robot"))
    try:
        import rm_connection
        found = rm_connection.find_real_ep(timeout=15)
        for ip, sn in found:
            print("EP IP %s  SN %s" % (ip, sn))
        if not found:
            print("連上了但 15 秒內沒再聽到廣播；用 RM_ROBOT_IP=<ip> 指定或稍後再 ./start_arm.sh --with-ep")
    except Exception as e:
        print("找 IP 失敗：%s" % e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
