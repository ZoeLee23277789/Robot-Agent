"""
We have switched all of our code from langchain to openai.types.chat.chat_completion_message_param.

For easier transition we have
"""

from typing import TYPE_CHECKING

# Lightweight imports that are commonly used
from robot_agent.llm.base import BaseChatModel
from robot_agent.llm.messages import (
	AssistantMessage,
	BaseMessage,
	SystemMessage,
	UserMessage,
)
from robot_agent.llm.messages import (
	ContentPartImageParam as ContentImage,
)
from robot_agent.llm.messages import (
	ContentPartRefusalParam as ContentRefusal,
)
from robot_agent.llm.messages import (
	ContentPartTextParam as ContentText,
)

# 這份 vendor 進 RoboMaster agent 的版本只保留 robot_agent/llm_factory.py 會用到的四個 provider
# (openai / anthropic / google / ollama)。其他 provider (aws、azure、cerebras、deepseek、groq、
# openrouter、oci_raw、robot_agent cloud) 跟 models.py 的預設模型清單都已移除。

# Type stubs for lazy imports
if TYPE_CHECKING:
	from robot_agent.llm.anthropic.chat import ChatAnthropic
	from robot_agent.llm.google.chat import ChatGoogle
	from robot_agent.llm.ollama.chat import ChatOllama
	from robot_agent.llm.openai.chat import ChatOpenAI

# Lazy imports mapping for heavy chat models
_LAZY_IMPORTS = {
	'ChatAnthropic': ('robot_agent.llm.anthropic.chat', 'ChatAnthropic'),
	'ChatGoogle': ('robot_agent.llm.google.chat', 'ChatGoogle'),
	'ChatOllama': ('robot_agent.llm.ollama.chat', 'ChatOllama'),
	'ChatOpenAI': ('robot_agent.llm.openai.chat', 'ChatOpenAI'),
}


def __getattr__(name: str):
	"""Lazy import mechanism for heavy chat model imports."""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module

			module = import_module(module_path)
			return getattr(module, attr_name)
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	# Message types -> for easier transition from langchain
	'BaseMessage',
	'UserMessage',
	'SystemMessage',
	'AssistantMessage',
	# Content parts with better names
	'ContentText',
	'ContentRefusal',
	'ContentImage',
	# Chat models
	'BaseChatModel',
	'ChatOpenAI',
	'ChatGoogle',
	'ChatAnthropic',
	'ChatOllama',
]
