# -*- coding: utf-8 -*-
"""Constants and configuration values for OpenJudge Studio."""

from typing import Any

# ============================================================================
# Default Models
# ============================================================================

DEFAULT_MODELS: list[str] = [
    "qwen3-235b-a22b",
    "qwen3-32b",
    "qwen-max",
    "qwen-plus",
    "qwen-vl-max",  # Vision model
    "gpt-4o",
    "gpt-4o-mini",
    "deepseek-chat",
]

# Vision-capable models for multimodal graders
VISION_MODELS: list[str] = [
    "qwen-vl-max",
    "qwen-vl-plus",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-vision-preview",
]

# ============================================================================
# Grader Categories
# ============================================================================

GRADER_CATEGORIES: dict[str, dict[str, Any]] = {
    "common": {
        "name": "Common",
        "name_zh": "通用评测",
        "icon": "",
        "description": "General-purpose evaluation graders",
    },
    "text": {
        "name": "Text",
        "name_zh": "文本评测",
        "icon": "",
        "description": "Text similarity and matching graders",
    },
    "format": {
        "name": "Format",
        "name_zh": "格式评测",
        "icon": "",
        "description": "Format and structure validation graders",
    },
    "code": {
        "name": "Code",
        "name_zh": "代码评测",
        "icon": "",
        "description": "Code quality and style graders",
    },
    "math": {
        "name": "Math",
        "name_zh": "数学评测",
        "icon": "",
        "description": "Mathematical expression verification",
    },
    "multimodal": {
        "name": "Multimodal",
        "name_zh": "多模态评测",
        "icon": "",
        "description": "Image and vision-language graders",
    },
    "agent": {
        "name": "Agent",
        "name_zh": "智能体评测",
        "icon": "",
        "description": "Agent tool use and trajectory graders",
    },
}

# ============================================================================
# Example Data for Different Grader Types
# ============================================================================

EXAMPLE_DATA: dict[str, dict[str, Any]] = {
    "default": {
        "query": "What is the capital of France?",
        "response": (
            "The capital of France is Paris. It is the largest city in France "
            "and serves as the country's major cultural, economic, and political center."
        ),
        "reference_response": "Paris is the capital of France.",
        "context": "",
    },
    "text_similarity": {
        "query": "",
        "response": "The quick brown fox jumps over the lazy dog.",
        "reference_response": "A fast brown fox leaps over a sleepy dog.",
        "context": "",
    },
    "code_style": {
        "query": "Write a function to calculate factorial",
        "response": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
        "reference_response": "",
        "context": "",
    },
    "math_verify": {
        "query": "What is 2 + 2?",
        "response": "4",
        "reference_response": "2 + 2 = 4",
        "context": "",
    },
    "multimodal": {
        "query": "Describe this image",
        "response": "The image shows a beautiful sunset over the ocean.",
        "reference_response": "",
        "context": "Context text above the image.\n[IMAGE]\nContext text below the image.",
    },
    "agent_tool": {
        "query": "What's the weather in Beijing?",
        "response": "",
        "reference_response": "",
        "context": "",
        "tool_definitions": """[
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        }
    }
]""",
        "tool_calls": """[
    {
        "name": "get_weather",
        "arguments": {"location": "Beijing"}
    }
]""",
    },
}

# ============================================================================
# API Endpoints
# ============================================================================

DEFAULT_API_ENDPOINTS: dict[str, str] = {
    "DashScope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "OpenAI": "https://api.openai.com/v1",
    "DeepSeek": "https://api.deepseek.com/v1",
    "Custom": "",
}

# ============================================================================
# UI Settings
# ============================================================================

APP_VERSION = "0.3.0"
APP_NAME = "OpenJudge Studio"
