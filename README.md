# miniAgent

轻量级智能体框架，支持 ReAct（Reasoning + Acting）模式。

## 安装

### 方式一：开发模式安装（推荐）

```bash
# 进入 mini_agent 目录
cd mini_agent

# 以开发模式安装（修改代码后不需要重新安装）
pip install -e .
```

### 方式二：普通安装

```bash
# 进入 mini_agent 目录
cd mini_agent

# 普通安装
pip install .
```

### 方式三：直接运行（不安装）

```bash
# 直接运行 study_agent.py
cd mini_agent
python study_agent.py "机器学习中的过拟合是什么"
```

## 依赖

安装时会自动安装以下依赖：

- `openai>=1.0.0` - OpenAI API 客户端
- `python-dotenv>=1.0.0` - 环境变量管理

## 使用示例

```python
from mini_agent import MiniAgentLLM, ReActAgent
from mini_agent.tools import BaseTool, ToolRegistry

# 1. 初始化 LLM
llm = MiniAgentLLM(temperature=0.3)

# 2. 注册工具
registry = ToolRegistry()
registry.register(YourTool())

# 3. 创建 Agent
agent = ReActAgent(
    name="my_agent",
    llm=llm,
    tool_registry=registry,
    max_steps=10,
)

# 4. 运行
result = agent.run("你的问题")
print(result)
```

## 环境变量

需要设置以下环境变量：

```bash
# OpenAI API 密钥
OPENAI_API_KEY=your_api_key

# Tavily 搜索 API 密钥（可选，用于联网搜索）
TAVILY_API_KEY=your_tavily_key
```

## 目录结构

```
mini_agent/
├── __init__.py          # 包入口
├── setup.py             # 安装配置
├── README.md            # 说明文档
├── core/
│   ├── __init__.py
│   ├── agent.py         # 基础 Agent 类
│   ├── llm.py           # LLM 调用封装
│   └── exceptions.py    # 异常定义
├── agents/
│   ├── __init__.py
│   └── react_agent.py   # ReAct Agent 实现
└── tools/
    ├── __init__.py
    ├── base.py           # 工具基类
    └── registry.py       # 工具注册表
```

## 许可证

MIT License
