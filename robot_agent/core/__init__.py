import os
from typing import TYPE_CHECKING

from robot_agent.core.logging_config import setup_logging

# Only set up logging if not in MCP mode or if explicitly requested
if os.environ.get('ROBOT_AGENT_SETUP_LOGGING', 'true').lower() != 'false':
	from robot_agent.core.config import CONFIG

	# Get log file paths from config/environment
	debug_log_file = getattr(CONFIG, 'ROBOT_AGENT_DEBUG_LOG_FILE', None)
	info_log_file = getattr(CONFIG, 'ROBOT_AGENT_INFO_LOG_FILE', None)

	# Set up logging with file handlers if specified
	logger = setup_logging(debug_log_file=debug_log_file, info_log_file=info_log_file)
else:
	import logging

	logger = logging.getLogger('robot_agent.core')

# Monkeypatch BaseSubprocessTransport.__del__ to handle closed event loops gracefully
from asyncio import base_subprocess

_original_del = base_subprocess.BaseSubprocessTransport.__del__


def _patched_del(self):
	"""Patched __del__ that handles closed event loops without throwing noisy red-herring errors like RuntimeError: Event loop is closed"""
	try:
		# Check if the event loop is closed before calling the original
		if hasattr(self, '_loop') and self._loop and self._loop.is_closed():
			# Event loop is closed, skip cleanup that requires the loop
			return
		_original_del(self)
	except RuntimeError as e:
		if 'Event loop is closed' in str(e):
			# Silently ignore this specific error
			pass
		else:
			raise


base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


# Type stubs for lazy imports - fixes linter warnings
if TYPE_CHECKING:
	from robot_agent.core.agent.prompts import SystemPrompt
	from robot_agent.core.agent.service import Agent

	# from robot_agent.core.agent.service import Agent
	from robot_agent.core.agent.views import ActionModel, ActionResult, AgentHistoryList
	from robot_agent.core.browser import BrowserProfile, BrowserSession
	from robot_agent.core.browser import BrowserSession as Browser
	from robot_agent.llm import models
	from robot_agent.llm.anthropic.chat import ChatAnthropic
	from robot_agent.llm.azure.chat import ChatAzureOpenAI
	from robot_agent.llm.google.chat import ChatGoogle
	from robot_agent.llm.groq.chat import ChatGroq
	from robot_agent.llm.oci_raw.chat import ChatOCIRaw
	from robot_agent.llm.ollama.chat import ChatOllama
	from robot_agent.llm.openai.chat import ChatOpenAI
	from robot_agent.core.tools.service import Controller, Tools


# Lazy imports mapping - only import when actually accessed
_LAZY_IMPORTS = {
	# Agent service (heavy due to dependencies)
	'Agent': ('robot_agent.core.agent.service', 'Agent'),
	# System prompt (moderate weight due to agent.views imports)
	'SystemPrompt': ('robot_agent.core.agent.prompts', 'SystemPrompt'),
	# Agent views (very heavy - over 1 second!)
	'ActionModel': ('robot_agent.core.agent.views', 'ActionModel'),
	'ActionResult': ('robot_agent.core.agent.views', 'ActionResult'),
	'AgentHistoryList': ('robot_agent.core.agent.views', 'AgentHistoryList'),
	'BrowserSession': ('robot_agent.core.browser', 'BrowserSession'),
	'Browser': ('robot_agent.core.browser', 'BrowserSession'),  # Alias for BrowserSession
	'BrowserProfile': ('robot_agent.core.browser', 'BrowserProfile'),
	# Tools (moderate weight)
	'Tools': ('robot_agent.core.tools.service', 'Tools'),
	'Controller': ('robot_agent.core.tools.service', 'Controller'),  # alias
	# Chat models (very heavy imports)
	'ChatOpenAI': ('robot_agent.llm.openai.chat', 'ChatOpenAI'),
	'ChatGoogle': ('robot_agent.llm.google.chat', 'ChatGoogle'),
	'ChatAnthropic': ('robot_agent.llm.anthropic.chat', 'ChatAnthropic'),
	'ChatGroq': ('robot_agent.llm.groq.chat', 'ChatGroq'),
	'ChatAzureOpenAI': ('robot_agent.llm.azure.chat', 'ChatAzureOpenAI'),
	'ChatOCIRaw': ('robot_agent.llm.oci_raw.chat', 'ChatOCIRaw'),
	'ChatOllama': ('robot_agent.llm.ollama.chat', 'ChatOllama'),
	# LLM models module
	'models': ('robot_agent.llm.models', None),
	# Sandbox execution
}


def __getattr__(name: str):
	"""Lazy import mechanism - only import modules when they're actually accessed."""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module

			module = import_module(module_path)
			if attr_name is None:
				# For modules like 'models', return the module itself
				attr = module
			else:
				attr = getattr(module, attr_name)
			# Cache the imported attribute in the module's globals
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	'Agent',
	'BrowserSession',
	'Browser',  # Alias for BrowserSession
	'BrowserProfile',
	'Controller',
	'SystemPrompt',
	'ActionResult',
	'ActionModel',
	'AgentHistoryList',
	# Chat models
	'ChatOpenAI',
	'ChatGoogle',
	'ChatAnthropic',
	'ChatGroq',
	'ChatAzureOpenAI',
	'ChatOCIRaw',
	'ChatOllama',
	'Tools',
	'Controller',
	# LLM models module
	'models',
	# Sandbox execution
]
