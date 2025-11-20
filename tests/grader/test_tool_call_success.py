# -*- coding: utf-8 -*-
"""
Test Tool Call Success Grader

Tests for the ToolCallSuccessGrader class functionality.
"""

import asyncio

import pytest

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.gallery.grader.agent.tool_call_success import ToolCallSuccessGrader


def test_tool_call_success_grader_creation():
    """Test creating a ToolCallSuccessGrader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ToolCallSuccessGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "tool_call_success"


# @pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_tool_call_success_grader_execution():
    """Test executing the tool call success grader with actual model call"""
    # Initialize the grader
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ToolCallSuccessGrader(model=model)

    # Define tool definitions
    tool_definitions = [
        {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name",
                    },
                },
                "required": ["location"],
            },
        },
    ]

    # Define successful tool calls
    successful_tool_calls = [
        {
            "name": "get_weather",
            "arguments": {"location": "New York"},
            "result": {"temperature": 25, "condition": "sunny"},
        },
    ]

    # Evaluate successful tool calls
    result = await grader.aevaluate(
        tool_definitions=tool_definitions,
        tool_calls=successful_tool_calls,
    )
    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")

    # Define failed tool calls
    failed_tool_calls = [
        {
            "name": "get_weather",
            "arguments": {"location": "New York"},
            "result": {"error": "Connection timeout"},
        },
    ]

    # Evaluate failed tool calls
    result = await grader.aevaluate(
        tool_definitions=tool_definitions,
        tool_calls=failed_tool_calls,
    )
    print(result)
    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")


if __name__ == "__main__":
    asyncio.run(test_tool_call_success_grader_execution())
