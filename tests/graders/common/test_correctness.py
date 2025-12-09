# -*- coding: utf-8 -*-
"""
Tests for CorrectnessGrader
"""

import pytest

from rm_gallery.core.graders.common.correctness import CorrectnessGrader
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


@pytest.mark.asyncio
async def test_correctness_grader_with_ground_truth():
    """Test CorrectnessGrader with ground truth"""
    # Initialize grader with mock model
    grader = CorrectnessGrader(
        model={
            "api_key": "mock-key",
            "model": "qwen-max",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        threshold=0.7,
        language=LanguageEnum.EN,
    )

    # Test evaluation
    result = await grader.aevaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
        ground_truth="The capital of France is Paris, with a population of 2.2M.",
        context="Geography quiz question",
    )

    # Verify result structure
    assert result.name == "correctness"
    assert isinstance(result.score, (int, float))
    assert result.score >= 1 and result.score <= 5
    assert "Correctness score" in result.reason
    assert result.metadata["threshold"] == 0.7


@pytest.mark.asyncio
async def test_correctness_grader_without_ground_truth():
    """Test CorrectnessGrader without ground truth"""
    # Initialize grader with mock model
    grader = CorrectnessGrader(
        model={
            "api_key": "mock-key",
            "model": "qwen-max",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        threshold=0.7,
        language=LanguageEnum.EN,
    )

    # Test evaluation without ground_truth
    result = await grader.aevaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
    )

    # Verify result structure
    assert result.name == "correctness"
    assert isinstance(result.score, (int, float))
    assert result.metadata["threshold"] == 0.7

