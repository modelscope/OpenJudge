# -*- coding: utf-8 -*-
"""
Test HarmfulnessGrader Grader

Tests for the HarmfulnessGrader class functionality.
"""
from unittest.mock import AsyncMock

import pytest

from rm_gallery.core.graders.gallery.alignment.harmlessness import HarmfulnessGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel


def test_harmfulness_grader_creation():
    """Test creating a HarmfulnessGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", stream=False)
    grader = HarmfulnessGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "harmfulness"


@pytest.mark.asyncio
async def test_harmfulness_grader_execution():
    """Test executing the hallucination grader with actual model call"""
    # Initialize the grader
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    mock_parse_result = AsyncMock()
    mock_parse_result.metadata = {"score": 3.0, "reason": "perfect"}
    model.achat = AsyncMock(return_value=mock_parse_result)

    grader = HarmfulnessGrader(model=model)
    query = "How to stay healthy?"
    response = "Regular exercise and balanced diet are important."
    context = "Health advice question"

    # Execute the grader
    result = await grader.aevaluate(
        query=query,
        response=response,
        context=context,
    )
    print(result)

    # Verify the result
    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    assert hasattr(result, "metadata")

    assert "{query}" not in grader.to_dict().get("template")
    assert "{response}" not in grader.to_dict().get("template")
    assert "{context_section}" not in grader.to_dict().get("template")
    assert "{reference_section}" not in grader.to_dict().get("template")
