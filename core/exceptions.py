class MiniAgentError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class LLMCallError(MiniAgentError):
    pass


class ToolExecutionError(MiniAgentError):
    pass


class ParseError(MiniAgentError):
    pass
