from ..base import BaseTool


class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算，支持加减乘除和基本数学表达式",
        )

    def execute(self, expression: str = "", **kwargs) -> str:
        if not expression:
            return "错误: 未提供数学表达式"
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"错误: 表达式包含不允许的字符"
        try:
            result = eval(expression)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"
