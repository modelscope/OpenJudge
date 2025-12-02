# -*- coding: utf-8 -*-
"""
Test HallucinationGrader Grader

Tests for the HallucinationGrader class functionality.
"""
from unittest.mock import AsyncMock

import pytest

from rm_gallery.core.graders.gallery.alignment.harmlessness import HallucinationGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel


def test_hallucination_grader_creation():
    """Test creating a HallucinationGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", stream=False)
    grader = HallucinationGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "hallucination"


@pytest.mark.asyncio
async def test_hallucination_grader_execution():
    """Test executing the hallucination grader with actual model call"""
    # Initialize the grader
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    mock_parse_result = AsyncMock()
    mock_parse_result.metadata = {"score": 3.0, "reason": "perfect"}
    model.achat = AsyncMock(return_value=mock_parse_result)

    grader = HallucinationGrader(model=model)
    query = "When was the company founded."
    response = "The company was founded in 2020 in San Francisco with 100 employees"
    context = "The company was founded in 2020 in San Francisco"
    reference_response = "The company was founded in 2020"

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
