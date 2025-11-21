# -*- coding: utf-8 -*-
"""
Test Memory Retrieval Failure Grader

Tests for the MemoryRetrievalFailureGrader class functionality.
"""

import pytest

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.template import LanguageEnum
from rm_gallery.gallery.grader.agent import MemoryRetrievalFailureGrader


def test_memory_retrieval_failure_grader_creation():
    """Test creating a MemoryRetrievalFailureGrader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = MemoryRetrievalFailureGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "memory_retrieval_failure"


def test_memory_retrieval_failure_grader_chinese():
    """Test creating a Chinese grader instance"""
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    grader = MemoryRetrievalFailureGrader(
        model=model,
        language=LanguageEnum.ZH
    )

    assert grader is not None
    assert grader.language == LanguageEnum.ZH


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_retrieval_failure_detection():
    """Test detecting memory retrieval failure"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = MemoryRetrievalFailureGrader(model=model)

    # Test case where plan ignores known information from memory
    result = await grader.aevaluate(
        plan="I will search for the key in drawer 1.",
        observation="You are in the room.",
        memory="The key was already found in drawer 1 in step 3. Key is in inventory.",
        task_context="Task: Use the key to unlock the door"
    )

    assert result is not None
    assert hasattr(result, "score")
    assert result.score == 0.0  # Should detect failure to retrieve memory


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_retrieval_success():
    """Test with successful memory retrieval"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = MemoryRetrievalFailureGrader(model=model)

    result = await grader.aevaluate(
        plan="I will use the key I found earlier to unlock the door.",
        observation="You are near the locked door. Key is in inventory.",
        memory="Key-A was found in drawer 1.",
        task_context="Task: Unlock the door"
    )

    assert result is not None
    assert result.score == 1.0  # Should be correct


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_memory_retrieval_failure_with_history():
    """Test memory retrieval failure with history"""
    model = OpenAIChatModel(model_name="qwen3-32b", stream=False)
    grader = MemoryRetrievalFailureGrader(model=model)

    history = [
        {"observation": "Found key in drawer 1", "memory": "Key in drawer 1"},
        {"observation": "Picked up key", "memory": "Key in inventory"},
    ]

    result = await grader.aevaluate(
        plan="I should search all drawers to find a key.",  # Ignoring memory
        observation="Standing in the room with key in inventory.",
        memory="Key is in inventory.",
        history_steps=history
    )

    assert result is not None
    assert result.score == 0.0

