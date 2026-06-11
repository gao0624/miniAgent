from abc import ABC, abstractmethod
from typing import Optional

from .llm import MiniAgentLLM
from ..tools.registry import ToolRegistry


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        llm: MiniAgentLLM,
        system_prompt: str = "",
        tool_registry: Optional[ToolRegistry] = None,
        max_steps: int = 10,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_steps = max_steps
        self._history: list[dict] = []

    @abstractmethod
    def run(self, user_input: str) -> str:
        pass

    def _build_messages(self, user_input: str) -> list[dict]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def _add_to_history(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})

    def get_history(self) -> list[dict]:
        return self._history.copy()

    def clear_history(self) -> None:
        self._history.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
