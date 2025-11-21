# -*- coding: utf-8 -*-
"""
Test Reflection Outcome Misinterpretation Grader

Tests for the ReflectionOutcomeMisinterpretationGrader class functionality.
"""

import pytest

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.template import LanguageEnum
from rm_gallery.gallery.grader.agent import ReflectionOutcomeMisinterpretationGrader


def test_reflection_outcome_misinterpretation_grader_creation():
    """Test creating a ReflectionOutcomeMisinterpretationGrader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ReflectionOutcomeMisinterpretationGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "reflection_outcome_misinterpretation"


def test_reflection_outcome_misinterpretation_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ReflectionOutcomeMisinterpretationGrader(
        model=model, 
        language=LanguageEnum.ZH
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_outcome_misinterpretation_detection():
    """Test detecting reflection outcome misinterpretation"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ReflectionOutcomeMisinterpretationGrader(model=model)

    # Test case with clear misinterpretation
    result = await grader.aevaluate(
        observation="The drawer is still closed. Action failed.",
        reflection="I successfully opened the drawer.",
        task_context="Task: Open the drawer"
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Should detect error


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_outcome_correct():
    """Test with correct reflection"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ReflectionOutcomeMisinterpretationGrader(model=model)

    # Test case with correct reflection
    result = await grader.aevaluate(
        observation="The drawer is now open.",
        reflection="I successfully opened the drawer.",
        task_context="Task: Open the drawer"
    )

    assert result is not None
    assert result.score == 1.0  # Should be correct


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_reflection_outcome_with_history():
    """Test with history steps"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ReflectionOutcomeMisinterpretationGrader(model=model)

    history = [
        {"observation": "Drawer is locked", "reflection": "Need to find key"},
    ]

    result = await grader.aevaluate(
        observation="Drawer is still locked.",
        reflection="The drawer is now open.",  # Wrong
        history_steps=history
    )

    assert result is not None
    assert result.score == 0.0

