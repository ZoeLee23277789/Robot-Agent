"""機器人版的 browser 套件。

原版這裡是 Chrome 的 session／事件匯流排／watchdog（一萬多行），機器人用不到，已整包移除。
Agent 迴圈真正需要的只剩三樣：
  1. BrowserProfile：Agent 會讀的幾個設定欄位（見 profile.py）
  2. views.py：BrowserStateSummary / BrowserStateHistory / TabInfo 這些狀態資料結構
  3. 一個叫 BrowserSession 的型別名稱給 type hint 與 registry 的型別檢查用；
     實際物件是 robot_agent.core_session.RobotSession，它繼承下面這個佔位類別。
"""

from .profile import DEFAULT_BROWSER_PROFILE, BrowserProfile


class BrowserSession:
	"""型別佔位：沒有任何行為，RobotSession 繼承它以通過 isinstance／型別註記。"""


Browser = BrowserSession  # 原版的別名，Agent(browser=...) 參數用

__all__ = ['BrowserSession', 'Browser', 'BrowserProfile', 'DEFAULT_BROWSER_PROFILE']
