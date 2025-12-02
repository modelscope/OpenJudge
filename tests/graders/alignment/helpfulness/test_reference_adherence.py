# -*- coding: utf-8 -*-
"""
Test ReferenceAdherenceGrader Grader

Tests for the ReferenceAdherenceGrader class functionality.
"""
from unittest.mock import AsyncMock

import pytest

from rm_gallery.core.graders.gallery.alignment.harmlessness import (
    ReferenceAdherenceGrader,
)
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel


def test_reference_adherence_grader_creation():
    """Test creating a ReferenceAdherenceGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", stream=False)
    grader = ReferenceAdherenceGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "reference_adherence"


@pytest.mark.asyncio
async def test_reference_adherence_grader_execution():
    """Test executing the hallucination grader with actual model call"""
    # Initialize the grader
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    mock_parse_result = AsyncMock()
    mock_parse_result.metadata = {"score": 3.0, "reason": "perfect"}
    model.achat = AsyncMock(return_value=mock_parse_result)

    grader = ReferenceAdherenceGrader(model=model)
    query = "What is the capital of France?"
    response = "Paris is the capital of France."
    reference = "The capital of France is Paris, with a population of 2.2M."
    reference_type = "factual source"

    # Execute the grader
    result = await grader.aevaluate(
        query=query,
        response=response,
        reference=reference,
        reference_type=reference_type,
    )
    print(result)

    # Verify the result
    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    assert hasattr(result, "metadata")

    assert "{query}" not in grader.to_dict().get("template")
    assert "{response}" not in grader.to_dict().get("template")
    assert "{reference_section}" not in grader.to_dict().get("template")
    assert "{reference_type_section}" not in grader.to_dict().get("template")
