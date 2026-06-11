import os
from typing import Iterator, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .exceptions import LLMCallError


class MiniAgentLLM:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
    ):
        # 自动加载 .env 文件中的环境变量
        load_dotenv()

        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        if provider == "auto":
            self.provider = self._detect_provider(base_url)

        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL_ID", "deepseek-chat")

        if not self.api_key:
            raise LLMCallError(
                "未找到 API Key，请设置 LLM_API_KEY 环境变量或在构造时传入 api_key"
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _detect_provider(self, base_url: Optional[str]) -> str:
        url = base_url or os.getenv("LLM_BASE_URL", "")
        if "deepseek" in url:
            return "deepseek"
        if "modelscope" in url:
            return "modelscope"
        if "localhost:8000" in url or "localhost:8001" in url:
            return "vllm"
        if "localhost:11434" in url:
            return "ollama"
        if "zhipu" in url or "glm" in url:
            return "zhipu"
        return "openai"

    def chat(
        self,
        messages: list[dict],
        stream: bool = False,
        **kwargs,
    ) -> str | Iterator[str]:
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": stream,
        }
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens

        try:
            response = self._client.chat.completions.create(**params)
        except Exception as e:
            raise LLMCallError(f"LLM 调用失败: {e}") from e

        if stream:
            return self._stream_response(response)

        content = response.choices[0].message.content
        return content.strip() if content else ""

    def _stream_response(self, response) -> Iterator[str]:
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def think(self, messages: list[dict], **kwargs) -> str:
        return self.chat(messages, stream=False, **kwargs)

    def think_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        return self.chat(messages, stream=True, **kwargs)
