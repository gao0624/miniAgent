import re
from typing import Optional

from ..core.agent import BaseAgent
from ..core.exceptions import ParseError, ToolExecutionError


class ReActAgent(BaseAgent):
    """ReAct 范式 Agent：思考 → 行动 → 观察 循环"""

    def __init__(self, name, llm, system_prompt="", tool_registry=None, max_steps=10):
        super().__init__(name, llm, system_prompt, tool_registry, max_steps)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """
        构建完整的 system prompt：默认的 ReAct 规范 + 用户自定义提示词

        用户只需要提供自定义部分（如任务描述、输出格式要求等），
        默认的 ReAct 规范会自动添加。
        """
        default_prompt = self._default_prompt()

        if self.system_prompt:
            return f"{default_prompt}\n\n{self.system_prompt}"
        else:
            return default_prompt

    def _default_prompt(self) -> str:
        """

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

输出格式必须严格遵循（每次只能输出一个 Thought 和一个 Action）:
Thought: 你的思考过程
Action: 工具名(param="value")
{example}
Action: Finish[最终答案]

【重要规则 - 必须严格遵守】:
1. 每次只能输出一个 Thought 和一个 Action，绝对不能输出多个
2. 禁止在一次输出中包含多个 Thought 或多个 Action
3. 禁止输出 Observation（这是系统自动生成的）
4. Action 必须在同一行
5. 当信息足够时必须使用 Action: Finish[最终答案]
6. 如果违反以上规则，系统会拒绝你的输出并要求重新生成

示例（正确格式）:
Thought: 我需要搜索相关信息
Action: web_search(query="Transformer架构")

示例（错误格式 - 禁止）:
Thought: 我需要搜索
Action: web_search(query="Transformer")
Thought: 搜索完成，现在分析
Action: analyze_text(text="...")"""

    def run(self, user_input: str) -> str:
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
            thought_count = len(re.findall(r'Thought:', output))
            action_count = len(re.findall(r'Action:', output))
            has_observation = 'Observation:' in output

            if thought_count > 1 or action_count > 1 or has_observation:
                print(f"⚠️ 检测到 {thought_count} 个 Thought、{action_count} 个 Action，包含 Observation: {has_observation}，违反 ReAct 规范，要求重新输出")
                history.append("Observation: 输出格式错误，一次只能输出一个 Thought 和一个 Action，且不能包含 Observation（Observation 由系统生成），请严格遵守 ReAct 规范重新输出")
                continue

            history.append(output)
            self._add_to_history("assistant", output)
            tool_name, args = self._parse_action(output)

            if tool_name is None:
                print("⚠️ 未检测到 Action 格式，LLM 必须使用 'Action: 工具名(param=\"值\")' 格式输出")
                history.append("Observation: 输出格式错误，必须使用 'Action: 工具名(param=\"值\")' 格式输出，不能直接输出内容。请严格遵守 ReAct 规范重新输出")
                continue

            if tool_name == "Finish":
                answer = args if isinstance(args, str) else args.get("answer", str(args))
                print(f"\n[任务完成]\n{answer}")
                self._add_to_history("user", f"最终答案: {answer}")
                return answer

            if tool_name not in self.tool_registry:
                history.append(f"Observation: 未知工具 {tool_name}")
                continue

            try:
                observation = self.tool_registry.execute(tool_name, **args)
            except Exception as e:
                observation = f"工具执行错误: {e}"

            print(f"\nObservation: {observation}")
            history.append(f"Observation: {observation}")

        print("\n达到最大循环次数，任务结束")
        return "达到最大循环次数，任务未能完成"

    def _parse_action(self, text: str) -> tuple[Optional[str], dict | str]:
        action_match = re.search(r"Action:\s*(.*)", text, re.DOTALL)
        if not action_match:
            return None, {}
        action = action_match.group(1).strip()

        finish_match = re.search(r"Finish\[(.*)\]", action, re.DOTALL)
        if finish_match:
            return "Finish", finish_match.group(1)

        tool_match = re.search(r"(\w+)\((.*)\)", action, re.DOTALL)
        if not tool_match:
            return None, {}
        tool_name = tool_match.group(1)
        args_str = tool_match.group(2)

        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str, re.DOTALL))
        return tool_name, kwargs