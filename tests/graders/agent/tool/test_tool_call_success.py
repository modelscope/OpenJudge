# -*- coding: utf-8 -*-
"""
Test Tool Call Success Grader

Tests for the ToolCallSuccessGrader class functionality.
"""

import pytest
import asyncio

from rm_gallery.core.graders.agent import ToolCallSuccessGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


def test_tool_call_success_grader_creation():
    """Test creating a ToolCallSuccessGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ToolCallSuccessGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "tool_call_success"


def test_tool_call_success_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="your-key", stream=False)
    grader = ToolCallSuccessGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
@pytest.mark.skip(reason="Model configuration issue with qwen3-32b enable_thinking parameter")
async def test_tool_call_success_with_successful_calls():
    """Test with successful tool calls"""
    model = OpenAIChatModel(model="qwen3-32b", api_key="your-key", stream=False)
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
        {
            "name": "search_files",
            "description": "Search for files by pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "File pattern"},
                    "directory": {
                        "type": "string",
                        "description": "Directory to search",
                    },
                },
                "required": ["pattern", "directory"],
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
        {
            "name": "search_files",
            "arguments": {"pattern": "*.py", "directory": "."},
            "result": ["main.py", "utils.py", "config.py"],
        },
    ]

    # Evaluate successful tool calls
    result = await grader.aevaluate(
        tool_definitions=tool_definitions,
        tool_calls=successful_tool_calls,
    )
    print(result)
    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    assert result.score > 0.5  # Should indicate success


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_tool_call_success_with_failed_calls():
    """Test with failed tool calls"""
    model = OpenAIChatModel(model="qwen3-32b", api_key="your-key", stream=False)
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
        {
            "name": "database_query",
            "description": "Query the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query"},
                },
                "required": ["query"],
            },
        },
    ]

    # Define failed tool calls
    failed_tool_calls = [
        {
            "name": "get_weather",
            "arguments": {"location": "New York"},
            "result": {"error": "Connection timeout"},
        },
        {
            "name": "database_query",
            "arguments": {"query": "SELECT * FROM users"},
            "result": {"error": "Authentication failed"},
        },
    ]

    # Evaluate failed tool calls
    result = await grader.aevaluate(
        tool_definitions=tool_definitions,
        tool_calls=failed_tool_calls,
    )

    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    assert result.score < 0.5  # Should indicate failure


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_tool_call_success_with_mixed_results():
    """Test with mixed successful and failed tool calls"""
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    grader = ToolCallSuccessGrader(model=model)

    # Define tool definitions
    tool_definitions = [
        {
            "name": "read_file",
            "description": "Read file contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "File path"},
                },
                "required": ["filepath"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["filepath", "content"],
            },
        },
        {
            "name": "delete_file",
            "description": "Delete a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "File path"},
                },
                "required": ["filepath"],
            },
        },
    ]

    # Define mixed tool calls - some succeed, some fail
    mixed_tool_calls = [
        {
            "name": "read_file",
            "arguments": {"filepath": "config.json"},
            "result": {"content": '{"setting": "value"}'},
        },
        {
            "name": "write_file",
            "arguments": {"filepath": "/readonly/file.txt", "content": "data"},
            "result": {"error": "Permission denied"},
        },
        {
            "name": "delete_file",
            "arguments": {"filepath": "temp.txt"},
            "result": {"success": True},
        },
    ]

    # Evaluate mixed tool calls
    result = await grader.aevaluate(
        tool_definitions=tool_definitions,
        tool_calls=mixed_tool_calls,
    )

    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    # Score should be moderate since some succeeded and some failed


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_tool_call_success_with_partial_results():
    """Test with tool calls that return partial or incomplete results"""
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    grader = ToolCallSuccessGrader(model=model)

    # Define tool definitions
    tool_definitions = [
        {
            "name": "api_request",
            "description": "Make an API request",
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string", "description": "API endpoint"},
                    "method": {"type": "string", "description": "HTTP method"},
                },
                "required": ["endpoint", "method"],
            },
        },
    ]

    # Define tool calls with partial results
    partial_tool_calls = [
        {
            "name": "api_request",
            "arguments": {"endpoint": "/api/data", "method": "GET"},
            "result": {
                "status": 200,
                "data": {"items": []},
                "warning": "Some data unavailable",
            },
        },
    ]

    # Evaluate partial results
    result = await grader.aevaluate(
        tool_definitions=tool_definitions,
        tool_calls=partial_tool_calls,
    )

    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")


if __name__ == "__main__":
    asyncio.run(test_tool_call_success_with_successful_calls())
