import subprocess
import sys

# Packages to install
packages = [
    "langchain",
    "langgraph",
    "langchain-google-genai",
    "langchain-mcp-adapters",
    "mcp",
    "nest_asyncio",
    "langgraph-checkpoint-sqlite",
    "pandas",
    "openpyxl",
    "pydantic"
]

# Install packages programmatically
subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
print("All packages installed successfully!")