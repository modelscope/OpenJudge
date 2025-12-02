# -*- coding: utf-8 -*-
"""
Test HelpfulnessGrader Grader

Tests for the HelpfulnessGrader class functionality.
"""
from unittest.mock import AsyncMock

import pytest

from rm_gallery.core.graders.gallery.alignment.helpfulness.helpfulness import (
    HelpfulnessGrader,
)
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel


def test_helpfulness_grader_creation():
    """Test creating a HelpfulnessGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", stream=False)
    grader = HelpfulnessGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "helpfulness"


@pytest.mark.asyncio
async def test_helpfulness_grader_execution():
    """Test executing the hallucination grader with actual model call"""
    # Initialize the grader
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    mock_parse_result = AsyncMock()
    mock_parse_result.metadata = {"score": 3.0, "reason": "perfect"}
    model.achat = AsyncMock(return_value=mock_parse_result)

    grader = HelpfulnessGrader(model=model)
    query = "What are decorators in Python?"
    response = "Decorators are functions that modify other functions..."
    context = "User needs help understanding Python decorators."
    reference_response = "Decorators are a Python feature for wrapping functions."

    # Execute the grader
    result = await grader.aevaluate(
        query=query,
        response=response,
        context=context,
        reference_response=reference_response,
    )
    print(result)

    # Verify the result
    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    assert hasattr(result, "metadata")

    assert "{query}" not in grader.to_dict().get("template")
    assert "{response}" not in grader.to_dict().get("template")
    assert "{context}" not in grader.to_dict().get("template")
    assert "{reference_section}" not in grader.to_dict().get("template")
