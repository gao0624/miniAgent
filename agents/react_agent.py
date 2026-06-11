import re
from typing import Optional

from ..core.agent import BaseAgent
from ..core.exceptions import ParseError, ToolExecutionError


class ReActAgent(BaseAgent):
    """ReAct 范式 Agent：思考 → 行动 → 观察 循环"""

    def __init__(self, name, llm, system_prompt="", tool_registry=None, max_steps=10):
        super().__init__(name, llm, system_prompt, tool_registry, max_steps)
        # 如果用户没有提供 system_prompt，就自动生成一个
        if not self.system_prompt:
            self.system_prompt = self._default_prompt()

    def _default_prompt(self) -> str:
        """
        根据已注册的工具，自动生成 system prompt。

        这个 prompt 是通用的——不管注册了什么工具，都能自动适配。
        工具列表和示例都是从 registry 动态生成的。
        """
        tools = self.tool_registry.list_tools()
        tools_desc = "\n".join(
            f"- {t['name']}: {t['description']}" for t in tools
        )
        # 用第一个注册工具生成示例（如果没有工具就不生成示例）
        first_tool = tools[0] if tools else None
        example = ""
        if first_tool:
            example = f'\nAction: {first_tool["name"]}(param="值")'
        return f"""你是一个智能助手。你需要使用可用工具来完成用户任务。

可用工具:
{tools_desc}

输出格式必须严格遵循:
Thought: 你的思考过程
Action: 工具名(param="value")
{example}
Action: Finish[最终答案]

规则:
1. 每次只能输出一对 Thought 和 Action
2. Action 必须在同一行
3. 当信息足够时必须使用 Finish"""

    def run(self, user_input: str) -> str:
        """
        ReAct 主循环

        整个流程就像这样：
        Step 1: 用户问 "郑州天气怎么样？"
                → LLM 思考：需要查天气 → 调用 get_weather → 得到结果
        Step 2: LLM 看到天气结果
                → 思考：信息够了 → 输出 Finish → 返回答案
        """
        # history 是传给 LLM 的上下文，包含了之前每一步的思考、行动和观察
        history = [f"用户问题: {user_input}"]

        for step in range(self.max_steps):
            print(f"\n{'='*20} Step {step + 1} {'='*20}")

            prompt = "\n".join(history)
            messages = self._build_messages(prompt)
            output = self.llm.chat(messages)

            if not output:
                history.append("Observation: 模型返回空内容")
                continue

            print(f"\n模型输出:\n{output}")
            history.append(output)
            self._add_to_history("assistant", output)

            # ============================================
            # 2. 解析 Action
            # ============================================
            tool_name, args = self._parse_action(output)

            if tool_name is None:
                history.append("Observation: 无法解析 Action 格式")
                continue

            # ============================================
            # 3. 如果是 Finish，返回最终答案
            # ============================================
            if tool_name == "Finish":
                # args 是 Finish[...] 中的答案文本（纯字符串）
                answer = args if isinstance(args, str) else args.get("answer", str(args))
                print(f"\n[任务完成]\n{answer}")
                self._add_to_history("user", f"最终答案: {answer}")
                return answer

            # ============================================
            # 4. 执行工具，得到真实的 Observation
            # ============================================
            if tool_name not in self.tool_registry:
                history.append(f"Observation: 未知工具 {tool_name}")
                continue

            try:
                # **args 会把字典拆成关键字参数，比如
                # args = {"city": "北京", "weather": "晴天"}
                # 等价于 registry.execute(tool_name, city="北京", weather="晴天")
                observation = self.tool_registry.execute(tool_name, **args)
            except Exception as e:
                observation = f"工具执行错误: {e}"

            print(f"\nObservation: {observation}")
            # 把真实的结果加入 history，下一轮 LLM 能看到这个结果
            history.append(f"Observation: {observation}")

        # 如果循环结束还没 Finish，说明任务没完成
        print("\n⚠️ 达到最大循环次数，任务结束")
        return "达到最大循环次数，任务未能完成"

    def _parse_action(self, text: str) -> tuple[Optional[str], dict | str]:
        action_match = re.search(r"Action:\s*(.*)", text)
        if not action_match:
            return None, {}
        action = action_match.group(1).strip()

        finish_match = re.search(r"Finish\[(.*)\]", action, re.DOTALL)
        if finish_match:
            return "Finish", finish_match.group(1)

        tool_match = re.search(r"(\w+)\((.*)\)", action)
        if not tool_match:
            return None, {}
        tool_name = tool_match.group(1)
        args_str = tool_match.group(2)

        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
        return tool_name, kwargs