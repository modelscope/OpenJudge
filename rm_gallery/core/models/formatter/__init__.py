# -*- coding: utf-8 -*-
"""Model formatter module for processing and formatting model inputs and outputs.

This module provides formatters for different model providers like OpenAI and DashScope,
handling the conversion between internal representations and model-specific formats.
It includes base classes and implementations for various formatting strategies.
"""

from .base_formatter import BaseFormatter
from .dashscope_formatter import DashScopeChatFormatter, DashScopeMultiAgentFormatter
from .openai_formatter import OpenAIChatFormatter, OpenAIMultiAgentFormatter
from .truncated_formatter import TruncatedFormatterBase

__all__ = [
    "BaseFormatter",
    "TruncatedFormatterBase",
    "DashScopeChatFormatter",
    "DashScopeMultiAgentFormatter",
    "OpenAIChatFormatter",
    "OpenAIMultiAgentFormatter",
]
