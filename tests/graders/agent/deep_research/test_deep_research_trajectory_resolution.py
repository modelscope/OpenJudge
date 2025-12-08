# -*- coding: utf-8 -*-
"""
Test Trajectory Resolution Grader

Tests for the TrajectoryResolutionGrader class functionality.
"""

import pytest

from rm_gallery.core.graders.agent.deep_research.trajectory_resolution import (
    TrajectoryResolutionGrader,
)
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


def test_trajectory_resolution_grader_creation():
    """Test creating a TrajectoryResolutionGrader instance"""
    model = OpenAIChatModel(model="qwen3-max", api_key="your-key", stream=False)
    grader = TrajectoryResolutionGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "trajectory_resolution"


def test_trajectory_resolution_grader_creation_chinese():
    """Test creating a Chinese TrajectoryResolutionGrader instance"""
    model = OpenAIChatModel(model="qwen3-max", api_key="your-key", stream=False)
    grader = TrajectoryResolutionGrader(model=model, language=LanguageEnum.ZH)

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_trajectory_resolution_evaluation():
    """Test evaluating trajectory resolution with standard messages format"""
    model = OpenAIChatModel(model="qwen3-max", api_key="your-key", stream=False)
    grader = TrajectoryResolutionGrader(model=model)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please search and analyze the data"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "search", "arguments": '{"query": "test"}'}},
            ],
        },
        {"role": "tool", "name": "search", "content": "Found relevant information"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "analyze", "arguments": '{"data": "test"}'}},
            ],
        },
        {"role": "tool", "name": "analyze", "content": "Analysis complete"},
        {
            "role": "assistant",
            "content": "Based on the search and analysis, here are the results.",
        },
    ]

    result = await grader.aevaluate(messages=messages)

    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    assert hasattr(result, "metadata")
    assert 0.0 <= result.score <= 1.0

    # Check metadata structure
    assert "problem_solving" in result.metadata
    assert "trajectory_quality" in result.metadata
    assert "efficiency" in result.metadata
    assert "step_evaluations" in result.metadata
    assert "total_raw_score" in result.metadata
    assert "is_resolved" in result.metadata
    assert "resolution_threshold" in result.metadata
    assert "evaluation_type" in result.metadata

    # Check dimension scores structure
    problem_solving = result.metadata["problem_solving"]
    assert "raw_score" in problem_solving
    assert "max_score" in problem_solving
    assert "normalized_score" in problem_solving
    assert "reason" in problem_solving
    assert problem_solving["max_score"] == 40.0
    assert 0.0 <= problem_solving["raw_score"] <= 40.0

    trajectory_quality = result.metadata["trajectory_quality"]
    assert trajectory_quality["max_score"] == 35.0
    assert 0.0 <= trajectory_quality["raw_score"] <= 35.0

    efficiency = result.metadata["efficiency"]
    assert efficiency["max_score"] == 25.0
    assert 0.0 <= efficiency["raw_score"] <= 25.0

    # Check step_evaluations is a list
    assert isinstance(result.metadata["step_evaluations"], list)

    # Check is_resolved is boolean
    assert isinstance(result.metadata["is_resolved"], bool)
    assert result.metadata["evaluation_type"] == "trajectory_resolution"


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_trajectory_resolution_single_step():
    """Test with single step trajectory"""
    model = OpenAIChatModel(model="qwen3-max", api_key="your-key", stream=False)
    grader = TrajectoryResolutionGrader(model=model)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Search for information"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "search", "arguments": '{"query": "test"}'}},
            ],
        },
        {"role": "tool", "name": "search", "content": "Found information"},
        {"role": "assistant", "content": "Here is the information found."},
    ]

    result = await grader.aevaluate(messages=messages)

    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "metadata")
    assert 0.0 <= result.score <= 1.0

    # Verify metadata exists
    assert "is_resolved" in result.metadata
    assert "total_raw_score" in result.metadata
    assert "step_evaluations" in result.metadata


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_trajectory_resolution_custom_threshold():
    """Test with custom resolution threshold"""
    model = OpenAIChatModel(model="qwen3-max", api_key="your-key", stream=False)
    grader = TrajectoryResolutionGrader(model=model)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please search and analyze the data"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "search", "arguments": '{"query": "test"}'}},
            ],
        },
        {"role": "tool", "name": "search", "content": "Found relevant information"},
        {
            "role": "assistant",
            "content": "Based on the search, here are the results.",
        },
    ]

    result = await grader.aevaluate(messages=messages, resolution_threshold=0.8)

    assert result is not None
    assert hasattr(result, "metadata")
    assert "resolution_threshold" in result.metadata
    assert result.metadata["resolution_threshold"] == 0.8
    assert "is_resolved" in result.metadata
    # is_resolved should be determined by the custom threshold
    expected_resolved = result.score >= 0.8
    assert result.metadata["is_resolved"] == expected_resolved


@pytest.mark.asyncio
async def test_trajectory_resolution_empty_messages():
    """Test with empty messages"""
    model = OpenAIChatModel(model="qwen3-max", api_key="your-key", stream=False)
    grader = TrajectoryResolutionGrader(model=model)

    result = await grader.aevaluate(messages=[])

    assert result is not None
    assert result.score == 0.0
    assert "Empty" in result.reason
    assert hasattr(result, "metadata")
    assert "error" in result.metadata


def test_trajectory_resolution_metadata_structure():
    """Test the structure of metadata in trajectory resolution result"""
    # This is a unit test that doesn't require API calls
    # It validates that our understanding of the metadata structure is correct
    metadata = {
        "problem_solving": {
            "raw_score": 35.0,
            "max_score": 40.0,
            "normalized_score": 35.0 / 40.0,
            "reason": "Test reason",
        },
        "trajectory_quality": {
            "raw_score": 30.0,
            "max_score": 35.0,
            "normalized_score": 30.0 / 35.0,
            "reason": "Test reason",
        },
        "efficiency": {
            "raw_score": 20.0,
            "max_score": 25.0,
            "normalized_score": 20.0 / 25.0,
            "reason": "Test reason",
        },
        "step_evaluations": [
            {
                "step_index": 0,
                "step_description": "Search for information",
                "contribution_level": "critical",
                "reason": "Essential step",
            },
        ],
        "total_raw_score": 85.0,
        "is_resolved": False,
        "resolution_threshold": 0.9,
        "evaluation_type": "trajectory_resolution",
    }

    # Validate structure
    assert "problem_solving" in metadata
    assert "trajectory_quality" in metadata
    assert "efficiency" in metadata
    assert "step_evaluations" in metadata
    assert "total_raw_score" in metadata
    assert "is_resolved" in metadata
    assert "resolution_threshold" in metadata
    assert "evaluation_type" in metadata

    # Validate total score calculation
    total = (
        metadata["problem_solving"]["raw_score"]
        + metadata["trajectory_quality"]["raw_score"]
        + metadata["efficiency"]["raw_score"]
    )
    assert total == metadata["total_raw_score"]

    # Validate is_resolved determination
    normalized_score = total / 100.0
    expected_resolved = normalized_score >= metadata["resolution_threshold"]
    assert metadata["is_resolved"] == expected_resolved

    # Validate step_evaluations structure
    assert isinstance(metadata["step_evaluations"], list)
    if metadata["step_evaluations"]:
        step = metadata["step_evaluations"][0]
        assert "step_index" in step
        assert "step_description" in step
        assert "contribution_level" in step
        assert "reason" in step
