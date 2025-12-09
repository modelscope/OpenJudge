# -*- coding: utf-8 -*-
"""
Tests for RelevanceEvaluator
"""

import pytest

from rm_gallery.core.graders.common.relevance import RelevanceEvaluator
from rm_gallery.core.models.schema.prompt_template import LanguageEnum


@pytest.mark.asyncio
async def test_relevance_evaluator_basic():
    """Test RelevanceEvaluator with basic query and response"""
    # Initialize evaluator with mock model
    evaluator = RelevanceEvaluator(
        model={
            "api_key": "mock-key",
            "model": "qwen-max",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        threshold=0.7,
        language=LanguageEnum.EN,
    )

    # Test evaluation
    result = await evaluator.aevaluate(
        query="What are Python decorators?",
        response="Decorators are functions that modify other functions. They use @syntax...",
    )

    # Verify result structure
    assert result.name == "relevance"
    assert isinstance(result.score, (int, float))
    assert result.score >= 1 and result.score <= 5
    assert "Relevance evaluation score" in result.reason
    assert result.metadata["threshold"] == 0.7


@pytest.mark.asyncio
async def test_relevance_evaluator_with_context():
    """Test RelevanceEvaluator with context"""
    # Initialize evaluator with mock model
    evaluator = RelevanceEvaluator(
        model={
            "api_key": "mock-key",
            "model": "qwen-max",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        threshold=0.7,
        language=LanguageEnum.EN,
    )

    # Test evaluation with context
    result = await evaluator.aevaluate(
        query="What's the weather like then?",
        response="July is summer in Europe with warm weather between 20-25°C.",
        context="Previous conversation about planning a July vacation to Europe",
    )

    # Verify result structure
    assert result.name == "relevance"
    assert isinstance(result.score, (int, float))
    assert result.score >= 1 and result.score <= 5
    assert result.metadata["threshold"] == 0.7


@pytest.mark.asyncio
async def test_relevance_evaluator_chinese():
    """Test RelevanceEvaluator with Chinese language"""
    # Initialize evaluator with Chinese language
    evaluator = RelevanceEvaluator(
        model={
            "api_key": "mock-key",
            "model": "qwen-max",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        threshold=0.7,
        language=LanguageEnum.ZH,
    )

    # Test evaluation
    result = await evaluator.aevaluate(
        query="什么是机器学习？",
        response="机器学习是人工智能的一个子集，它使系统能够从数据中学习。",
    )

    # Verify result structure
    assert result.name == "relevance"
    assert isinstance(result.score, (int, float))
    assert result.score >= 1 and result.score <= 5
    assert result.metadata["threshold"] == 0.7


@pytest.mark.asyncio
async def test_relevance_evaluator_with_ground_truth():
    """Test RelevanceEvaluator with ground truth for comparison"""
    # Initialize evaluator
    evaluator = RelevanceEvaluator(
        model={
            "api_key": "mock-key",
            "model": "qwen-max",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        threshold=0.7,
        language=LanguageEnum.EN,
    )

    # Test evaluation with ground truth
    result = await evaluator.aevaluate(
        query="What services does the premium membership include?",
        response="The premium membership includes priority customer support, exclusive content access, and early product releases.",
        ground_truth="Premium membership: priority support, exclusive content, early releases, and monthly webinars.",
    )

    # Verify result structure
    assert result.name == "relevance"
    assert isinstance(result.score, (int, float))
    assert result.score >= 1 and result.score <= 5
    assert result.metadata["threshold"] == 0.7

