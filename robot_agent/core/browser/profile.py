"""BrowserProfile 的機器人版。

原版 1,100 多行全是 Chrome 的啟動、連線、proxy、viewport 參數。Agent 迴圈（agent/service.py）
實際只讀下面七個欄位，其中對機器人真正有作用的只有 wait_between_actions（同一步裡多個動作之間的
等待秒數）；其他幾個是網址白名單、下載路徑之類，機器人永遠是 None，留著只是讓 Agent 讀得到。
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BrowserProfile(BaseModel):
	model_config = ConfigDict(extra='ignore', validate_assignment=True)

	allowed_domains: list[str] | set[str] | None = Field(default=None, description='原版的網址白名單；機器人沒有網址')
	wait_between_actions: float = Field(default=0.1, description='同一步裡多個動作之間的等待秒數')
	keep_alive: bool | None = None
	downloads_path: str | Path | None = None
	viewport: dict[str, Any] | None = None
	user_agent: str | None = None
	headless: bool | None = None


DEFAULT_BROWSER_PROFILE = BrowserProfile()
