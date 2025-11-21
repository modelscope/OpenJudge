# -*- coding: utf-8 -*-
"""
Test Reflection Progress Misjudge Grader

Tests for the ReflectionProgressMisjudgeGrader class functionality.
"""

import pytest

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.template import LanguageEnum
from rm_gallery.gallery.grader.agent import ReflectionProgressMisjudgeGrader


def test_reflection_progress_misjudge_grader_creation():
    """Test creating a ReflectionProgressMisjudgeGrader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ReflectionProgressMisjudgeGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "reflection_progress_misjudge"


def test_reflection_progress_misjudge_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ReflectionProgressMisjudgeGrader(
        model=model,
        language=LanguageEnum.ZH
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_progress_overestimation():
    """Test detecting progress overestimation"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ReflectionProgressMisjudgeGrader(model=model)

    # Test case with overestimated progress
    result = await grader.aevaluate(
        observation="Cabinet 1 is still empty. No items found.",
        reflection="Excellent progress! I'm making great headway toward finding the apples!",
        task_context="Task: Find apples in cabinets"
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Should detect error


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_progress_correct():
    """Test with correct progress assessment"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ReflectionProgressMisjudgeGrader(model=model)

    result = await grader.aevaluate(
        observation="Found 3 apples in cabinet 2.",
        reflection="Good progress! Found apples as required.",
        task_context="Task: Find apples"
    )

    assert result is not None
    assert result.score == 1.0


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_progress_with_history():
    """Test progress misjudge with history showing repeated failures"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ReflectionProgressMisjudgeGrader(model=model)

    history = [
        {"observation": "Cabinet 1 empty", "reflection": "No items"},
        {"observation": "Cabinet 1 still empty", "reflection": "Still nothing"},
        {"observation": "Cabinet 1 empty again", "reflection": "No progress"},
    ]

    result = await grader.aevaluate(
        observation="Cabinet 1 is empty.",
        reflection="Making excellent progress!",  # Overestimating
        history_steps=history,
        task_context="Task: Find items"
    )

    assert result is not None
    assert result.score == 0.0

