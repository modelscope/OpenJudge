# -*- coding: utf-8 -*-
"""
Test Tool Call Accuracy Grader

Tests for the ToolCallAccuracyGrader class functionality.
"""

import pytest

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.gallery.grader.agent.tool_call_accuracy import ToolCallAccuracyGrader


def test_tool_call_accuracy_grader_creation():
    """Test creating a ToolCallAccuracyGrader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ToolCallAccuracyGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "tool_call_accuracy"


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_tool_call_accuracy_grader_execution():
    """Test executing the tool call accuracy grader with actual model call"""
    # Initialize the grader
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ToolCallAccuracyGrader(model=model)

    # Example conversation
    conversation = [
        {
            "role": "user",
            "content": "Can you check the weather in London and Paris?",
        },
    ]

    # Define the available tools
    tool_definitions = [
        {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name",
                    },
                },
                "required": ["location"],
            },
        },
    ]

    # Tool calls made by the agent
    tool_calls = [
        {
            "name": "get_weather",
            "arguments": {"location": "London"},
            "result": {"temperature": 18, "condition": "cloudy"},
        },
        {
            "name": "get_weather",
            "arguments": {"location": "Paris"},
            "result": {"temperature": 22, "condition": "sunny"},
        },
    ]

    # Execute the grader
    result = await grader.aevaluate(
        query=conversation,
        tool_definitions=tool_definitions,
        tool_calls=tool_calls,
    )

    # Verify the result
    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    assert hasattr(result, "metadata")
