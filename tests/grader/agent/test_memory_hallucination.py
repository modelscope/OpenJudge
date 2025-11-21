# -*- coding: utf-8 -*-
"""
Test Memory Hallucination Grader

Tests for the MemoryHallucinationGrader class functionality.
"""

import pytest

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.template import LanguageEnum
from rm_gallery.gallery.grader.agent import MemoryHallucinationGrader


def test_memory_hallucination_grader_creation():
    """Test creating a MemoryHallucinationGrader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = MemoryHallucinationGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "memory_hallucination"


def test_memory_hallucination_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = MemoryHallucinationGrader(
        model=model,
        language=LanguageEnum.ZH
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_hallucination_detection():
    """Test detecting hallucinated information in memory"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = MemoryHallucinationGrader(model=model)

    # Test case with hallucinated memory
    result = await grader.aevaluate(
        observation="You see a closed cabinet.",
        memory="There is a red vase inside the cabinet with gold trim.",
        task_context="Task: Inventory room objects"
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Should detect hallucination


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_no_hallucination():
    """Test with correct memory without hallucination"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = MemoryHallucinationGrader(model=model)

    result = await grader.aevaluate(
        observation="Cabinet 1 contains 3 red apples.",
        memory="Cabinet 1 has 3 red apples.",
        task_context="Task: Inventory items"
    )

    assert result is not None
    assert result.score == 1.0  # Should be correct


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_hallucination_with_history():
    """Test memory hallucination with history"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = MemoryHallucinationGrader(model=model)

    history = [
        {"observation": "Cabinet is locked", "memory": "Cabinet 1 is locked"},
    ]

    result = await grader.aevaluate(
        observation="Cabinet is still locked. Cannot see inside.",
        memory="Cabinet contains 5 golden coins.",  # Cannot know this
        history_steps=history
    )

    assert result is not None
    assert result.score == 0.0

