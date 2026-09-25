"""
RoboMaster 連線輔助模組
========================
把連線設定集中在這裡，模擬器和實體機器人共用同一組呼叫方式。
之後要換成接實體 EP，只需要改這個檔案，呼叫端的程式完全不用動。
"""

import time

import patch_ftp  # noqa: F401  模擬器沒有 FTP 服務，這個補丁讓 SDK 不會卡在等待上
import robomaster.config as config
from robomaster import robot


def connect(mode: str = "sim", retries: int = 4, retry_delay: float = 2.0) -> robot.Robot:
    """建立並回傳一個已初始化的 Robot 物件。

    mode="sim"  連到本機的 CoppeliaSim 模擬器 (127.0.0.1)
    mode="real" 連到實體 EP，走 router 模式，機器人和這台電腦要在同一個 Wi-Fi 上

    模擬器這條連線偶爾會在剛啟動模擬、或上一次連線沒乾淨關閉時失敗一次
    （"Robot: Can not connect to robot, check connection please."），
    但通常隔個一兩秒重試就會成功，所以這裡內建重試，不用手動重跑整支程式。
    """
    if mode == "sim":
        config.LOCAL_IP_STR = "127.0.0.1"
        config.ROBOT_IP_STR = "127.0.0.1"
    elif mode == "real":
        config.LOCAL_IP_STR = None
        config.ROBOT_IP_STR = None
    else:
        raise ValueError(f"未知的 mode: {mode!r}，只能是 'sim' 或 'real'")

    last_err = None
    for attempt in range(1, retries + 1):
        ep = robot.Robot()
        try:
            ep.initialize(conn_type="sta")
            return ep
        except Exception as e:
            last_err = e
            try:
                ep.close()
            except Exception:
                pass
            if attempt < retries:
                print(f"[rm_connection] 連線失敗（第 {attempt}/{retries} 次）：{e}，{retry_delay:.0f}s 後重試...")
                time.sleep(retry_delay)
    raise last_err
