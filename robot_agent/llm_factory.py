"""
LLM 工廠。直接重用 repo 裡 browser_use/llm 的 Chat 類別，
所以 provider 的切換方式、structured output 的處理都跟原本的 Adobe Express Agent 一樣。
"""

import logging
import os
from typing import Optional


class _TruncateFilter(logging.Filter):
    """LLM 偶爾會吐出幾千個字元的壞回應，函式庫會把它整段印出來洗版，這裡把每一行 log 截短。"""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if len(msg) > 400:
            record.msg, record.args = msg[:400] + f" ... [truncated {len(msg) - 400} chars]", ()
        return True


def _quiet_logs() -> None:
    for name in ("", "browser_use"):
        for h in logging.getLogger(name).handlers:
            if not any(isinstance(f, _TruncateFilter) for f in h.filters):
                h.addFilter(_TruncateFilter())

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
    "google": "gemini-2.5-flash",
    "robotics-er": "gemini-robotics-er-2-preview",
    "ollama": "qwen2.5vl:7b",
    "mlx": "default",
}


def _need(env: str) -> str:
    value = os.getenv(env)
    if not value:
        raise RuntimeError(f"Missing {env}. Put it in .env or export it.")
    return value


def build_llm(provider: str, model: Optional[str] = None):
    """機器人任務一定要用看得懂圖片的模型，否則 Agent 等於是閉著眼睛開車。"""
    provider = (provider or "openai").lower().strip()
    model = model or DEFAULT_MODELS.get(provider)
    _quiet_logs()

    if provider == "openai":
        from browser_use.llm.openai.chat import ChatOpenAI
        return ChatOpenAI(model=model, api_key=_need("OPENAI_API_KEY"))
    if provider == "anthropic":
        from browser_use.llm.anthropic.chat import ChatAnthropic
        return ChatAnthropic(model=model, api_key=_need("ANTHROPIC_API_KEY"))
    if provider in ("google", "gemini"):
        from browser_use.llm.google.chat import ChatGoogle
        return ChatGoogle(model=model, api_key=_need("GOOGLE_API_KEY"))
    if provider in ("robotics-er", "gemini-er", "er"):
        # Gemini Robotics ER：Google 專門為機器人做的 embodied reasoning 模型，走的是同一套 Gemini API。
        # 官方建議 thinking level 用 medium 兼顧延遲與準確度；這個模型是 Gemini 3 系列，temperature 維持預設的 1.0。
        from browser_use.llm.google.chat import ChatGoogle
        level = os.getenv("ROBOT_THINKING_LEVEL", "medium").lower()
        return ChatGoogle(
            model=model, api_key=_need("GOOGLE_API_KEY"), temperature=1.0,
            # 正常的一步輸出只有幾百個 token，但這個模型的 thinking_config 會把隱藏推理跟最終
            # JSON 輸出算進同一個額度：實測發現 4500 太緊，推理用量一多，JSON 常常寫到一半被
            # 硬截斷（"Expecting ',' delimiter"），反而比放寬上限更浪費步數。調到 7000，
            # 還是遠低於 SDK 預設的 8096，模型萬一陷入「120.02700000...」這種數字無限延伸的
            # 退化輸出時，一樣會在合理時間內被截斷並重試，不會真的卡到燒滿。
            max_output_tokens=int(os.getenv("ROBOT_MAX_OUTPUT_TOKENS", "7000")),
            config={"thinking_config": {"thinking_level": level}},
        )
    if provider == "ollama":
        from browser_use.llm.ollama.chat import ChatOllama
        return ChatOllama(model=model)
    if provider == "mlx":
        # 跟 Agent.py 相同：MLX 走 OpenAI 相容伺服器
        from browser_use.llm.openai.chat import ChatOpenAI
        return ChatOpenAI(base_url=os.getenv("MLX_BASE_URL", "http://localhost:8000/v1"), api_key="none", model=model)
    raise ValueError(f"Unsupported provider: {provider}")
