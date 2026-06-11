from setuptools import setup, find_packages

setup(
    name="mini-agent",
    version="0.1.0",
    description="MiniAgent - 轻量级智能体框架",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
    ],
)
