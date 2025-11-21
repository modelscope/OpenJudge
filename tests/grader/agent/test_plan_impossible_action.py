# -*- coding: utf-8 -*-
"""
Test Plan Impossible Action Grader

Tests for the PlanImpossibleActionGrader class functionality.
"""

import pytest

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.template import LanguageEnum
from rm_gallery.gallery.grader.agent import PlanImpossibleActionGrader


def test_plan_impossible_action_grader_creation():
    """Test creating a PlanImpossibleActionGrader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = PlanImpossibleActionGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "plan_impossible_action"


def test_plan_impossible_action_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = PlanImpossibleActionGrader(
        model=model,
        language=LanguageEnum.ZH
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_plan_impossible_action_detection():
    """Test detecting impossible action in plan"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = PlanImpossibleActionGrader(model=model)

    # Test case with impossible action (using object before obtaining it)
    result = await grader.aevaluate(
        plan="I will use the key to unlock the door.",
        observation="The drawer is closed. You don't have any items.",
        memory="The key is inside the drawer, but the drawer is not opened yet.",
        task_context="Task: Unlock the door to exit"
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Should detect impossible action


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_plan_possible_action():
    """Test with feasible plan"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = PlanImpossibleActionGrader(model=model)

    result = await grader.aevaluate(
        plan="I will first open the drawer to get the key, then unlock the door.",
        observation="Drawer is closed. Key is inside.",
        memory="Key is in drawer 1.",
        task_context="Task: Unlock the door"
    )

    assert result is not None
    assert result.score == 1.0  # Should be correct


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_plan_impossible_action_with_history():
    """Test impossible action detection with history"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = PlanImpossibleActionGrader(model=model)

    history = [
        {"observation": "Door is locked", "plan": "Need to find key"},
        {"observation": "Drawer is closed", "plan": "Will check drawer"},
    ]

    result = await grader.aevaluate(
        plan="I will close the door.",  # Door is already locked, can't close
        observation="Door is locked, cannot be closed further.",
        memory="Door is locked.",
        history_steps=history
    )

    assert result is not None
    assert result.score == 0.0

