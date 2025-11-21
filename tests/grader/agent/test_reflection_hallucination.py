# -*- coding: utf-8 -*-
"""
Test Reflection Hallucination Grader

Tests for the ReflectionHallucinationGrader class functionality.
"""

import pytest

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.template import LanguageEnum
from rm_gallery.gallery.grader.agent import ReflectionHallucinationGrader


def test_reflection_hallucination_grader_creation():
    """Test creating a ReflectionHallucinationGrader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ReflectionHallucinationGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "reflection_hallucination"


def test_reflection_hallucination_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ReflectionHallucinationGrader(
        model=model,
        language=LanguageEnum.ZH
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_hallucination_detection():
    """Test detecting hallucinated information in reflection"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ReflectionHallucinationGrader(model=model)

    # Test case with hallucinated details
    result = await grader.aevaluate(
        observation="You see a closed cabinet.",
        reflection="I observed a red vase on top of the cabinet with three flowers.",
        task_context="Task: Inventory room objects"
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Should detect hallucination


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_no_hallucination():
    """Test with correct reflection without hallucination"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ReflectionHallucinationGrader(model=model)

    result = await grader.aevaluate(
        observation="You see a closed cabinet and a table.",
        reflection="I observed a closed cabinet and a table in the room.",
        task_context="Task: Inventory room"
    )

    assert result is not None
    assert result.score == 1.0  # Should be correct


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_hallucination_with_history():
    """Test hallucination detection with history"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ReflectionHallucinationGrader(model=model)

    history = [
        {"observation": "Empty room", "reflection": "Room has no objects"},
    ]

    result = await grader.aevaluate(
        observation="You see an empty room.",
        reflection="I see a golden statue in the corner.",  # Hallucinated
        history_steps=history
    )

    assert result is not None
    assert result.score == 0.0

