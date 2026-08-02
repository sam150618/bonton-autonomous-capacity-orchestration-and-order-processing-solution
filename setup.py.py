# 1. Install required packages
!pip install -q langchain langgraph langchain-google-genai langchain-mcp-adapters mcp nest_asyncio pandas openpyxl pydantic

import os
import math
import asyncio
import nest_asyncio
import pandas as pd
from typing import Dict, List, Any
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Apply nest_asyncio for notebook compatibility
nest_asyncio.apply()

# Set Google Gemini API Key (replace with your key or set in environment)
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"