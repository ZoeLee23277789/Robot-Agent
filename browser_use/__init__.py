"""
精簡版 browser_use：只保留 LLM 層 (browser_use/llm)。

原始套件裡跟瀏覽器有關的部分 (agent, browser, dom, tools 等) 已經移除，
因為 RoboMaster agent 只需要 Chat 類別、訊息型別和 structured output 的 schema 處理。
保留 browser_use 這個套件名稱，是為了讓 llm/ 底下的檔案一行都不用改，
之後想同步上游的修正時可以直接覆蓋。
"""

import os

from browser_use.logging_config import setup_logging

if os.environ.get('BROWSER_USE_SETUP_LOGGING', 'true').lower() != 'false':
	logger = setup_logging()
else:
	import logging

	logger = logging.getLogger('browser_use')

_LAZY = {
	'ChatOpenAI': 'browser_use.llm.openai.chat',
	'ChatAnthropic': 'browser_use.llm.anthropic.chat',
	'ChatGoogle': 'browser_use.llm.google.chat',
	'ChatOllama': 'browser_use.llm.ollama.chat',
}


def __getattr__(name: str):
	if name in _LAZY:
		from importlib import import_module

		return getattr(import_module(_LAZY[name]), name)
	raise AttributeError(f"module 'browser_use' has no attribute {name!r}")


__all__ = list(_LAZY)
