# -*- coding: utf-8 -*-
"""
Test Financial Resolution Grader

Tests for the FinancialResolutionGrader class functionality.
"""

import pytest

from rm_gallery.core.graders.agent.deep_research.report_resolution import (
    ReportResolutionGrader,
)
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


def test_report_resolution_grader_creation():
    """Test creating a ReportResolutionGrader instance"""
    model = OpenAIChatModel(model="qwen3-max", api_key="your-key", stream=False)
    grader = ReportResolutionGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "report_resolution"


def test_report_resolution_grader_creation_chinese():
    """Test creating a Chinese ReportResolutionGrader instance"""
    model = OpenAIChatModel(model="qwen3-max", api_key="your-key", stream=False)
    grader = ReportResolutionGrader(model=model, language=LanguageEnum.ZH)

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_report_resolution_evaluation():
    """Test evaluating report resolution"""
    model = OpenAIChatModel(model="qwen3-max", api_key="your-key", stream=False)
    grader = ReportResolutionGrader(model=model)

    result = await grader.aevaluate(
        query="What are the best investment options for a conservative investor?",
        response="For conservative investors, I recommend government bonds, high-grade corporate bonds, and dividend-paying blue-chip stocks. These options provide stable returns with lower risk.",
        history="",
        chat_date="2024-01-15",
    )

    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    assert hasattr(result, "metadata")
    assert 0.0 <= result.score <= 1.0

    # Check metadata structure
    assert "precision_relevance" in result.metadata
    assert "completeness_depth" in result.metadata
    assert "data_support" in result.metadata
    assert "logical_rigor" in result.metadata
    assert "timeliness" in result.metadata
    assert "structure_readability" in result.metadata
    assert "total_raw_score" in result.metadata
    assert "is_resolved" in result.metadata
    assert "resolution_threshold" in result.metadata
    assert "evaluation_type" in result.metadata

    # Check dimension scores structure
    precision = result.metadata["precision_relevance"]
    assert "raw_score" in precision
    assert "max_score" in precision
    assert "normalized_score" in precision
    assert "reason" in precision
    assert precision["max_score"] == 35.0
    assert 0.0 <= precision["raw_score"] <= 35.0

    # Check is_resolved is boolean
    assert isinstance(result.metadata["is_resolved"], bool)
    assert result.metadata["evaluation_type"] == "report_resolution"


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_report_resolution_with_history():
    """Test with chat history"""
    model = OpenAIChatModel(model="qwen3-max", api_key="your-key", stream=False)
    grader = ReportResolutionGrader(model=model)

    chat_history = """
    User: What is my risk profile?
    Analyst: Based on your questionnaire, you have a conservative risk profile.
    """

    result = await grader.aevaluate(
        query="What should I invest in?",
        response="Given your conservative risk profile, I recommend bonds and stable dividend stocks.",
        history=chat_history,
        chat_date="2024-01-15",
    )

    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "metadata")
    assert 0.0 <= result.score <= 1.0

    # Verify metadata exists
    assert "is_resolved" in result.metadata
    assert "total_raw_score" in result.metadata


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_report_resolution_incomplete_answer():
    """Test with incomplete answer"""
    model = OpenAIChatModel(model="qwen3-max", api_key="your-key", stream=False)
    grader = ReportResolutionGrader(model=model)

    result = await grader.aevaluate(
        query="What are the risks and returns of investing in tech stocks?",
        response="Tech stocks can be good investments.",
        history="",
        chat_date="2024-01-15",
    )

    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "metadata")
    # Incomplete answer should have lower score
    assert 0.0 <= result.score <= 1.0

    # Incomplete answer likely won't be resolved
    assert "is_resolved" in result.metadata
    # Score should be less than default threshold (0.9)
    assert result.score < 0.9 or result.metadata["is_resolved"] is False


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_report_resolution_custom_threshold():
    """Test with custom resolution threshold"""
    model = OpenAIChatModel(model="qwen3-max", api_key="your-key", stream=False)
    grader = ReportResolutionGrader(model=model)

    result = await grader.aevaluate(
        query="What are the best investment options for a conservative investor?",
        response="For conservative investors, I recommend government bonds, high-grade corporate bonds, and dividend-paying blue-chip stocks. These options provide stable returns with lower risk.",
        history="",
        chat_date="2024-01-15",
        resolution_threshold=0.85,
    )

    assert result is not None
    assert hasattr(result, "metadata")
    assert "resolution_threshold" in result.metadata
    assert result.metadata["resolution_threshold"] == 0.85
    assert "is_resolved" in result.metadata
    # is_resolved should be determined by the custom threshold
    expected_resolved = result.score >= 0.85
    assert result.metadata["is_resolved"] == expected_resolved


def test_report_resolution_metadata_structure():
    """Test the structure of metadata in report resolution result"""
    # This is a unit test that doesn't require API calls
    # It validates that our understanding of the metadata structure is correct
    metadata = {
        "precision_relevance": {
            "raw_score": 30.0,
            "max_score": 35.0,
            "normalized_score": 30.0 / 35.0,
            "reason": "Test reason",
        },
        "completeness_depth": {
            "raw_score": 25.0,
            "max_score": 30.0,
            "normalized_score": 25.0 / 30.0,
            "reason": "Test reason",
        },
        "data_support": {
            "raw_score": 15.0,
            "max_score": 20.0,
            "normalized_score": 15.0 / 20.0,
            "reason": "Test reason",
        },
        "logical_rigor": {
            "raw_score": 8.0,
            "max_score": 10.0,
            "normalized_score": 8.0 / 10.0,
            "reason": "Test reason",
        },
        "timeliness": {
            "raw_score": 4.0,
            "max_score": 5.0,
            "normalized_score": 4.0 / 5.0,
            "reason": "Test reason",
        },
        "structure_readability": {
            "assessment": "Test assessment",
        },
        "total_raw_score": 82.0,
        "is_resolved": False,
        "resolution_threshold": 0.9,
        "evaluation_type": "report_resolution",
    }

    # Validate structure
    assert "precision_relevance" in metadata
    assert "completeness_depth" in metadata
    assert "data_support" in metadata
    assert "logical_rigor" in metadata
    assert "timeliness" in metadata
    assert "structure_readability" in metadata
    assert "total_raw_score" in metadata
    assert "is_resolved" in metadata
    assert "resolution_threshold" in metadata
    assert "evaluation_type" in metadata

    # Validate total score calculation
    total = (
        metadata["precision_relevance"]["raw_score"]
        + metadata["completeness_depth"]["raw_score"]
        + metadata["data_support"]["raw_score"]
        + metadata["logical_rigor"]["raw_score"]
        + metadata["timeliness"]["raw_score"]
    )
    assert total == metadata["total_raw_score"]

    # Validate is_resolved determination
    normalized_score = total / 100.0
    expected_resolved = normalized_score >= metadata["resolution_threshold"]
    assert metadata["is_resolved"] == expected_resolved
