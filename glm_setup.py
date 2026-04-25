"""GLM/Zhipu API adapter for Agent4Rec.
Imported first by main.py to point openai SDK at Zhipu OpenAI-compatible endpoint.
"""
import os
import openai

GLM_KEY = os.environ.get(
    "GLM_API_KEY",
    "2572c468580744fc8bd4e6ca67b6c267.FpQXe5wsBg7F6wgW",
)
GLM_BASE = "https://open.bigmodel.cn/api/paas/v4"

openai.api_key = GLM_KEY
openai.api_base = GLM_BASE
os.environ["OPENAI_API_KEY"] = GLM_KEY
os.environ["OPENAI_API_BASE"] = GLM_BASE
