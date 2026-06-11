from ..core.agent import BaseAgent


class SimpleAgent(BaseAgent):
    """
    SimpleAgent 默认不保留对话历史，每次调用都是独立的单轮对话。
    这样可以直观地演示"没有记忆"时智能体会出现什么问题。

    如果需要多轮对话能力，可以取消下面"带记忆版本"的注释，
    并注释掉当前的"无记忆版本"。
    """

    # ========== 无记忆版本（当前启用）==========
    # 每次调用 run() 都只发送当前这一轮的消息，不带历史。
    # LLM 看不到之前的对话，所以无法记住用户之前说过的话。
    def run(self, user_input: str) -> str:
        messages = self._build_messages(user_input)
        response = self.llm.chat(messages)
        return response

    # ========== 带记忆版本（当前注释）==========
    # 取消下面这段代码的注释，同时注释掉上面的"无记忆版本"，
    # SimpleAgent 就能记住之前的对话历史。
    # _history 会在每轮对话后累积，下一轮调用时 LLM 能看到之前的内容。
    #
    # def run(self, user_input: str) -> str:
    #     messages = self._build_messages(user_input)
    #     response = self.llm.chat(messages)
    #     self._add_to_history("user", user_input)
    #     self._add_to_history("assistant", response)
    #     return response
