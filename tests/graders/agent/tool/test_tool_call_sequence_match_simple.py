# -*- coding: utf-8 -*-
"""
Test Tool Call Sequence Match Simple Grader

Tests for the ToolCallSequenceMatchSimpleGrader class functionality.
"""

import pytest

from openjudge.graders.agent.tool.tool_call_sequence_match_simple import (
    MetricType,
    ToolCallSequenceMatchSimpleGrader,
)


def test_tool_call_sequence_match_simple_grader_creation():
    """Test creating a ToolCallSequenceMatchSimpleGrader instance with default parameters."""
    grader = ToolCallSequenceMatchSimpleGrader()

    assert grader is not None
    assert grader.name == "tool_call_sequence_simple"
    assert grader.metric_type == MetricType.RECALL
    assert grader.match_arguments is False


def test_tool_call_sequence_match_simple_grader_precision_mode():
    """Test creating grader in precision mode."""
    grader = ToolCallSequenceMatchSimpleGrader(metric_type="precision")

    assert grader.metric_type == MetricType.PRECISION


def test_tool_call_sequence_match_simple_grader_recall_mode():
    """Test creating grader in recall mode."""
    grader = ToolCallSequenceMatchSimpleGrader(metric_type="recall")

    assert grader.metric_type == MetricType.RECALL


def test_tool_call_sequence_match_simple_grader_match_arguments():
    """Test creating grader with argument matching enabled."""
    grader = ToolCallSequenceMatchSimpleGrader(match_arguments=True)

    assert grader.match_arguments is True


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_both_empty():
    """Test with both empty tool call lists."""
    grader = ToolCallSequenceMatchSimpleGrader()

    result = await grader.aevaluate(tool_calls=[], reference_tool_calls=[])

    assert result is not None
    assert result.score == 1.0  # Both empty means perfect match


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_empty_predictions():
    """Test with empty predictions but non-empty reference."""
    grader = ToolCallSequenceMatchSimpleGrader(metric_type="recall")

    result = await grader.aevaluate(
        tool_calls=[],
        reference_tool_calls=[{"name": "search", "arguments": {}}],
    )

    assert result.score == 0.0  # Recall is 0 when nothing predicted


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_empty_reference():
    """Test with non-empty predictions but empty reference."""
    grader = ToolCallSequenceMatchSimpleGrader(metric_type="precision")

    result = await grader.aevaluate(
        tool_calls=[{"name": "search", "arguments": {}}],
        reference_tool_calls=[],
    )

    assert result.score == 0.0  # Precision is 0 when reference is empty


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_perfect_precision():
    """Test perfect precision - all predictions are correct."""
    grader = ToolCallSequenceMatchSimpleGrader(
        metric_type="precision",
        match_arguments=False,
    )

    result = await grader.aevaluate(
        tool_calls=[{"name": "search", "arguments": {}}],
        reference_tool_calls=[
            {"name": "search", "arguments": {}},
            {"name": "calculate", "arguments": {}},
        ],
    )

    # 1 correct out of 1 prediction = 1.0
    assert result.score == 1.0
    assert result.metadata["precision"] == 1.0


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_partial_precision():
    """Test partial precision - some predictions are wrong."""
    grader = ToolCallSequenceMatchSimpleGrader(
        metric_type="precision",
        match_arguments=False,
    )

    result = await grader.aevaluate(
        tool_calls=[
            {"name": "search", "arguments": {}},
            {"name": "wrong_tool", "arguments": {}},
        ],
        reference_tool_calls=[{"name": "search", "arguments": {}}],
    )

    # 1 correct out of 2 predictions = 0.5
    assert result.score == 0.5
    assert result.metadata["true_positives"] == 1
    assert result.metadata["total_predicted"] == 2


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_perfect_recall():
    """Test perfect recall - all references are found."""
    grader = ToolCallSequenceMatchSimpleGrader(
        metric_type="recall",
        match_arguments=False,
    )

    result = await grader.aevaluate(
        tool_calls=[
            {"name": "search", "arguments": {}},
            {"name": "calculate", "arguments": {}},
            {"name": "extra_tool", "arguments": {}},
        ],
        reference_tool_calls=[
            {"name": "search", "arguments": {}},
            {"name": "calculate", "arguments": {}},
        ],
    )

    # 2 correct out of 2 references = 1.0
    assert result.score == 1.0
    assert result.metadata["recall"] == 1.0


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_partial_recall():
    """Test partial recall - some references are missing."""
    grader = ToolCallSequenceMatchSimpleGrader(
        metric_type="recall",
        match_arguments=False,
    )

    result = await grader.aevaluate(
        tool_calls=[{"name": "search", "arguments": {}}],
        reference_tool_calls=[
            {"name": "search", "arguments": {}},
            {"name": "calculate", "arguments": {}},
        ],
    )

    # 1 correct out of 2 references = 0.5
    assert result.score == 0.5
    assert result.metadata["true_positives"] == 1
    assert result.metadata["total_reference"] == 2


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_loose_matching():
    """Test loose matching ignores argument differences."""
    grader = ToolCallSequenceMatchSimpleGrader(
        metric_type="precision",
        match_arguments=False,
    )

    result = await grader.aevaluate(
        tool_calls=[{"name": "search", "arguments": {"query": "different"}}],
        reference_tool_calls=[{"name": "search", "arguments": {"query": "original"}}],
    )

    # Names match, arguments ignored
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_strict_matching_mismatch():
    """Test strict matching requires same arguments."""
    grader = ToolCallSequenceMatchSimpleGrader(
        metric_type="precision",
        match_arguments=True,
    )

    result = await grader.aevaluate(
        tool_calls=[{"name": "search", "arguments": {"query": "different"}}],
        reference_tool_calls=[{"name": "search", "arguments": {"query": "original"}}],
    )

    # Names match but arguments differ
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_strict_matching_exact():
    """Test strict matching with same arguments."""
    grader = ToolCallSequenceMatchSimpleGrader(
        metric_type="precision",
        match_arguments=True,
    )

    result = await grader.aevaluate(
        tool_calls=[{"name": "search", "arguments": {"query": "test"}}],
        reference_tool_calls=[{"name": "search", "arguments": {"query": "test"}}],
    )

    # Both name and arguments match
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_function_wrapper_format():
    """Test function wrapper format: {"function": {"name": ..., "arguments": ...}}"""
    grader = ToolCallSequenceMatchSimpleGrader(match_arguments=False)

    result = await grader.aevaluate(
        tool_calls=[
            {"function": {"name": "search", "arguments": {"query": "test"}}},
        ],
        reference_tool_calls=[{"name": "search", "arguments": {}}],
    )

    assert result.score == 1.0


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_tool_call_wrapper_format():
    """Test tool_call wrapper format: {"tool_call": {"function": {...}}}"""
    grader = ToolCallSequenceMatchSimpleGrader(match_arguments=False)

    result = await grader.aevaluate(
        tool_calls=[
            {
                "tool_call": {
                    "function": {"name": "search", "arguments": {"query": "test"}},
                },
            },
        ],
        reference_tool_calls=[{"name": "search", "arguments": {}}],
    )

    assert result.score == 1.0


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_json_string_arguments():
    """Test arguments as JSON string."""
    grader = ToolCallSequenceMatchSimpleGrader(
        metric_type="precision",
        match_arguments=True,
    )

    result = await grader.aevaluate(
        tool_calls=[{"name": "search", "arguments": '{"query": "test"}'}],
        reference_tool_calls=[{"name": "search", "arguments": {"query": "test"}}],
    )

    assert result.score == 1.0


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_duplicate_predictions():
    """Test handling duplicate predictions."""
    grader = ToolCallSequenceMatchSimpleGrader(
        metric_type="precision",
        match_arguments=False,
    )

    result = await grader.aevaluate(
        tool_calls=[
            {"name": "search", "arguments": {}},
            {"name": "search", "arguments": {}},
        ],
        reference_tool_calls=[{"name": "search", "arguments": {}}],
    )

    # 1 match out of 2 predictions = 0.5
    assert result.score == 0.5
    assert result.metadata["true_positives"] == 1


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_duplicate_references():
    """Test handling duplicate references."""
    grader = ToolCallSequenceMatchSimpleGrader(
        metric_type="recall",
        match_arguments=False,
    )

    result = await grader.aevaluate(
        tool_calls=[{"name": "search", "arguments": {}}],
        reference_tool_calls=[
            {"name": "search", "arguments": {}},
            {"name": "search", "arguments": {}},
        ],
    )

    # 1 match out of 2 references = 0.5
    assert result.score == 0.5
    assert result.metadata["true_positives"] == 1


@pytest.mark.asyncio
async def test_tool_call_sequence_match_simple_grader_metadata_fields():
    """Test that all expected metadata fields are present."""
    grader = ToolCallSequenceMatchSimpleGrader(
        metric_type="precision",
        match_arguments=True,
    )

    result = await grader.aevaluate(
        tool_calls=[{"name": "search", "arguments": {"query": "test"}}],
        reference_tool_calls=[{"name": "search", "arguments": {"query": "test"}}],
    )

    assert "metric_type" in result.metadata
    assert "match_arguments" in result.metadata
    assert "precision" in result.metadata
    assert "recall" in result.metadata
    assert "true_positives" in result.metadata
    assert "total_predicted" in result.metadata
    assert "total_reference" in result.metadata


def test_tool_call_sequence_match_simple_grader_get_metadata():
    """Test that get_metadata returns a dictionary."""
    metadata = ToolCallSequenceMatchSimpleGrader.get_metadata()

    assert isinstance(metadata, dict)
    assert "description" in metadata
    assert "parameters" in metadata
    assert "score_range" in metadata
