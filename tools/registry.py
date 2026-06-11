from typing import Callable
from .base import BaseTool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._functions: dict[str, Callable] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        self._functions[tool.name] = tool.execute

    def register_function(
        self, name: str, func: Callable, description: str = ""
    ) -> None:
        self._functions[name] = func
        self._tools[name] = type(
            "DynamicTool",
            (BaseTool,),
            {"execute": lambda self, **kw: func(**kw)},
        )(name=name, description=description)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def execute(self, tool_name: str, **kwargs) -> str:
        func = self._functions.get(tool_name)
        if not func:
            raise ValueError(f"未找到工具: {tool_name}")
        return func(**kwargs)

    def list_tools(self) -> list[dict]:
        return [t.to_dict() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._functions

    def __len__(self) -> int:
        return len(self._tools)
