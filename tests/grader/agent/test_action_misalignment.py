# -*- coding: utf-8 -*-
"""
Test Action Misalignment Grader

Tests for the ActionMisalignmentGrader class functionality.
"""

import pytest

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.template import LanguageEnum
from rm_gallery.gallery.grader.agent import ActionMisalignmentGrader


def test_action_misalignment_grader_creation():
    """Test creating an ActionMisalignmentGrader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ActionMisalignmentGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "action_misalignment"


def test_action_misalignment_grader_creation_chinese():
    """Test creating a Chinese ActionMisalignmentGrader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = ActionMisalignmentGrader(model=model, language=LanguageEnum.ZH)

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_action_misalignment_detection():
    """Test detecting action misalignment"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ActionMisalignmentGrader(model=model)

    # Test case with clear action misalignment
    result = await grader.aevaluate(
        plan="I will open drawer 1 to find the key.",
        action="close drawer 1",
        task_context="Task: Find the key in the room"
    )

    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    assert result.score == 0.0  # Should detect error
    assert "misalignment" in result.reason.lower() or "contradict" in result.reason.lower()


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_action_alignment_correct():
    """Test with correct action alignment"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ActionMisalignmentGrader(model=model)

    # Test case with correct alignment
    result = await grader.aevaluate(
        plan="I will open drawer 1 to find the key.",
        action="open drawer 1",
        task_context="Task: Find the key"
    )

    assert result is not None
    assert result.score == 1.0  # Should be correct


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_action_misalignment_with_history():
    """Test action misalignment with history steps"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = ActionMisalignmentGrader(model=model)

    history = [
        {"plan": "Check drawer 1", "action": "examine drawer 1"},
        {"plan": "Look inside", "action": "open drawer 1"},
    ]

    result = await grader.aevaluate(
        plan="I will close the drawer after searching.",
        action="open drawer 2",  # Wrong action
        history_steps=history,
        task_context="Task: Search for items"
    )

    assert result is not None
    assert hasattr(result, "score")

