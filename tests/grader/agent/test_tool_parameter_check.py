# -*- coding: utf-8 -*-
"""
Test Tool Parameter Check Grader

Tests for the ToolParameterCheckGrader class functionality.
"""

import asyncio
import pytest

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.template import LanguageEnum
from rm_gallery.gallery.grader.agent import ToolParameterCheckGrader


def test_tool_parameter_check_grader_creation():
    """Test creating a ToolParameterCheckGrader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ToolParameterCheckGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "tool_parameter_check"


def test_tool_parameter_check_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ToolParameterCheckGrader(
        model=model,
        language=LanguageEnum.ZH
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


#@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_tool_parameter_correct():
    """Test with correct parameter extraction"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ToolParameterCheckGrader(model=model)

    # Example conversation
    conversation = [
        {
            "role": "user",
            "content": "Search for all Python files in the src directory",
        },
    ]

    # Define the available tools
    tool_definitions = [
        {
            "name": "search_files",
            "description": "Search for files matching a pattern in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "File pattern to match (e.g., '*.py')",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in",
                    },
                },
                "required": ["pattern", "directory"],
            },
        },
    ]

    # Tool calls made by the agent
    tool_calls = [
        {
            "name": "search_files",
            "arguments": {"pattern": "*.py", "directory": "src"},
            "result": ["src/main.py", "src/utils.py"],
        },
    ]

    result = await grader.aevaluate(
        query=conversation,
        tool_definitions=tool_definitions,
        tool_calls=tool_calls,
    )

    print(result)
    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    assert hasattr(result, "metadata")


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_tool_parameter_hallucinated():
    """Test detecting hallucinated parameters"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ToolParameterCheckGrader(model=model)

    # Example conversation - note: user doesn't specify "recursive"
    conversation = [
        {
            "role": "user",
            "content": "Search for Python files",
        },
    ]

    # Define the available tools - no "recursive" parameter
    tool_definitions = [
        {
            "name": "search_files",
            "description": "Search for files matching a pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "File pattern to match",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in",
                    },
                },
                "required": ["pattern", "directory"],
            },
        },
    ]

    # Tool calls with hallucinated "recursive" parameter
    tool_calls = [
        {
            "name": "search_files",
            "arguments": {"pattern": "*.py", "directory": "src", "recursive": True},
            "result": ["src/main.py"],
        },
    ]

    result = await grader.aevaluate(
        query=conversation,
        tool_definitions=tool_definitions,
        tool_calls=tool_calls,
    )

    assert result is not None
    assert result.score == 0.0  # Should detect hallucinated parameter


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_tool_parameter_missing():
    """Test detecting missing required parameters"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ToolParameterCheckGrader(model=model)

    # Example conversation - user mentions both pattern and directory
    conversation = [
        {
            "role": "user",
            "content": "Search for Python files in src directory",
        },
    ]

    # Define the available tools
    tool_definitions = [
        {
            "name": "search_files",
            "description": "Search for files matching a pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "File pattern to match",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in",
                    },
                },
                "required": ["pattern", "directory"],
            },
        },
    ]

    # Tool calls missing 'directory' parameter
    tool_calls = [
        {
            "name": "search_files",
            "arguments": {"pattern": "*.py"},  # Missing 'directory'
            "result": {"error": "Missing required parameter: directory"},
        },
    ]

    result = await grader.aevaluate(
        query=conversation,
        tool_definitions=tool_definitions,
        tool_calls=tool_calls,
    )

    assert result is not None
    assert result.score == 0.0  # Should detect missing parameter


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_tool_parameter_with_conversation_history():
    """Test parameter check with conversation history"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ToolParameterCheckGrader(model=model)

    # Conversation with history
    conversation = [
        {
            "role": "user",
            "content": "I need to find Python files.",
        },
        {
            "role": "assistant",
            "content": "Which directory should I search in?",
        },
        {
            "role": "user",
            "content": "The config directory.",
        },
        {
            "role": "user",
            "content": "Search in that directory",
        },
    ]

    # Define the available tools
    tool_definitions = [
        {
            "name": "search_files",
            "description": "Search for files matching a pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "File pattern to match",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search in",
                    },
                },
                "required": ["pattern", "directory"],
            },
        },
    ]

    # Tool calls - parameters inferred from conversation history
    tool_calls = [
        {
            "name": "search_files",
            "arguments": {"pattern": "*.py", "directory": "config"},
            "result": ["config/settings.py", "config/database.py"],
        },
    ]

    result = await grader.aevaluate(
        query=conversation,
        tool_definitions=tool_definitions,
        tool_calls=tool_calls,
    )

    assert result is not None
    assert result.score == 1.0  # Parameters should be correct based on history

if __name__ == "__main__":
    asyncio.run(test_tool_parameter_correct())

