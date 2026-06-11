from abc import ABC, abstractmethod


class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> str:
        pass

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description}

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"
