# browser_use/llm/custom/chat.py
import os
import httpx
from typing import TypeVar, overload, Optional, Dict, Any, List

from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar("T")

class ChatBrowserUseLocal(BaseChatModel):
    """
    Drop-in replacement for ChatBrowserUse, but **NO cloud proxy**.
    - Calls OpenAI's API (or your own endpoint) directly.
    - Same interface expected by Agent.
    """

    def __init__(
        self,
        model: str = None,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        max_output_tokens: int = 400,
        request_timeout: float = 60.0,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        # model fallback
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for ChatBrowserUseLocal")

        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.request_timeout = float(request_timeout)
        self.extra_params = extra_params or {}

    # ---- identity for logs/telemetry (no cloud proxy) ----
    @property
    def provider(self) -> str:
        return "custom"  # 明確標示不是 browser-use 雲代理

    @property
    def name(self) -> str:
        return self.model

    # ---- core invoke ----
    @overload
    async def ainvoke(
        self, messages: List[BaseMessage], output_format: None = None, request_type: str = "browser_agent"
    ) -> ChatInvokeCompletion[str]: ...
    @overload
    async def ainvoke(
        self, messages: List[BaseMessage], output_format: type[T] = None, request_type: str = "browser_agent"
    ) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self, messages: List[BaseMessage], output_format: type[T] | None = None, request_type: str = "browser_agent"
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """
        Send request to OpenAI (or your endpoint) and return ChatInvokeCompletion.
        """
        # 1) 轉換訊息格式
        payload_msgs = [{"role": m.role, "content": m.content} for m in messages]

        # 2) 組 payload（採用 Chat Completions；你也可改成 Responses API）
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "messages": payload_msgs,
            # 為了速度與穩定：避免太花俏的輸出（必要時可加入 stop）
        }
        payload.update(self.extra_params)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 3) 呼叫 API（重試 1 次）
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            url = f"{self.base_url}/chat/completions"
            for attempt in (1, 2):
                try:
                    r = await client.post(url, json=payload, headers=headers)
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    # 第一次失敗：輕量退避
                    continue

        # 4) 解析結果 & 用量
        text = data["choices"][0]["message"]["content"]
        usage = None
        if "usage" in data:
            u = data["usage"]
            usage = ChatInvokeUsage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            )

        # 5) 結構化輸出（如果 Agent 要求 output_format）
        if output_format is not None:
            # 嘗試以 JSON 解析；不行就讓模型自己嘗試解析
            # 這裡保持簡單：交給 Pydantic 解析字串（模型提示可要求輸出 JSON）
            try:
                import json
                parsed = output_format.model_validate_json(text)
                return ChatInvokeCompletion(completion=parsed, usage=usage)
            except Exception:
                # 若解析失敗，回傳原文字（避免斷掉）
                return ChatInvokeCompletion(completion=text, usage=usage)

        return ChatInvokeCompletion(completion=text, usage=usage)
