# -*- coding: utf-8 -*-
"""
Test Tool Selection Quality Grader

Tests for the ToolSelectionQualityGrader class functionality.
"""

import pytest

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.template import LanguageEnum
from rm_gallery.gallery.grader.agent import ToolSelectionQualityGrader


def test_tool_selection_quality_grader_creation():
    """Test creating a ToolSelectionQualityGrader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ToolSelectionQualityGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "tool_selection_quality"
    assert grader.threshold == 0.7


def test_tool_selection_quality_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ToolSelectionQualityGrader(
        model=model,
        threshold=0.8,
        language=LanguageEnum.ZH
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH
    assert grader.threshold == 0.8


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_tool_selection_quality_good():
    """Test with good tool selection"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ToolSelectionQualityGrader(model=model, threshold=0.7)

    # Example conversation
    conversation = [
        {
            "role": "user",
            "content": "Find all Python files that were modified in the last week",
        },
    ]

    # Define all available tools
    tool_definitions = [
        {
            "name": "search_files",
            "description": "Search for files by pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "File pattern"},
                    "directory": {"type": "string", "description": "Directory to search"},
                },
                "required": ["pattern", "directory"],
            },
        },
        {
            "name": "list_directory",
            "description": "List all files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "get_file_info",
            "description": "Get detailed file information",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "File path"},
                },
                "required": ["filepath"],
            },
        },
        {
            "name": "git_log",
            "description": "Get git history for a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "File path"},
                    "days": {"type": "integer", "description": "Number of days to look back"},
                },
                "required": ["filepath", "days"],
            },
        },
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
    ]

    # Good tool selection - agent correctly chose search_files and git_log
    tool_calls = [
        {
            "name": "search_files",
            "arguments": {"pattern": "*.py", "directory": "."},
            "result": ["main.py", "utils.py", "config.py"],
        },
        {
            "name": "git_log",
            "arguments": {"filepath": "main.py", "days": 7},
            "result": {"modified": True, "last_commit": "2024-11-20"},
        },
    ]

    result = await grader.aevaluate(
        query=conversation,
        tool_definitions=tool_definitions,
        tool_calls=tool_calls,
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score >= 0.7  # Should be good quality


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_tool_selection_quality_poor():
    """Test with poor tool selection"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ToolSelectionQualityGrader(model=model, threshold=0.7)

    # Example conversation
    conversation = [
        {
            "role": "user",
            "content": "Find all Python files modified in the last week",
        },
    ]

    # Define available tools
    tool_definitions = [
        {
            "name": "search_files",
            "description": "Search for files",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "directory": {"type": "string"},
                },
                "required": ["pattern", "directory"],
            },
        },
        {
            "name": "git_log",
            "description": "Get git history",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["filepath", "days"],
            },
        },
        {
            "name": "read_file",
            "description": "Read file contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                },
                "required": ["filepath"],
            },
        },
    ]

    # Poor tool selection - only reads README, doesn't search or check modification
    tool_calls = [
        {
            "name": "read_file",
            "arguments": {"filepath": "README.md"},
            "result": {"content": "# Project README..."},
        },
    ]

    result = await grader.aevaluate(
        query=conversation,
        tool_definitions=tool_definitions,
        tool_calls=tool_calls,
    )

    assert result is not None
    assert result.score < 0.5  # Should be poor quality


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_tool_selection_quality_with_history():
    """Test tool selection quality with conversation history"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ToolSelectionQualityGrader(model=model)

    # Conversation with history
    conversation = [
        {
            "role": "user",
            "content": "I need to find JSON configuration files.",
        },
        {
            "role": "assistant",
            "content": "Which directory should I search in?",
        },
        {
            "role": "user",
            "content": "The config folder.",
        },
        {
            "role": "user",
            "content": "Now search in the config folder",
        },
    ]

    # Define available tools
    tool_definitions = [
        {
            "name": "search_files",
            "description": "Search for files by pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "directory": {"type": "string"},
                },
                "required": ["pattern", "directory"],
            },
        },
        {
            "name": "list_directory",
            "description": "List all files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    ]

    # Good tool selection based on conversation context
    tool_calls = [
        {
            "name": "search_files",
            "arguments": {"pattern": "*.json", "directory": "config"},
            "result": ["config/app.json", "config/db.json"],
        },
    ]

    result = await grader.aevaluate(
        query=conversation,
        tool_definitions=tool_definitions,
        tool_calls=tool_calls,
    )

    assert result is not None
    assert result.score >= 0.7  # Good selection based on context

