# -*- coding: utf-8 -*-
"""
Test InstructionAdherenceGrader Grader

Tests for the InstructionAdherenceGrader class functionality.
"""
from unittest.mock import AsyncMock

import pytest

from rm_gallery.core.graders.gallery.alignment.honesty import InstructionAdherenceGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel


def test_instruction_adherence_grader_creation():
    """Test creating a InstructionAdherenceGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", stream=False)
    grader = InstructionAdherenceGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "instruction_adherence"


@pytest.mark.asyncio
async def test_instruction_adherence_grader_execution():
    """Test executing the hallucination grader with actual model call"""
    # Initialize the grader
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    mock_parse_result = AsyncMock()
    mock_parse_result.metadata = {"score": 3.0, "reason": "perfect"}
    model.achat = AsyncMock(return_value=mock_parse_result)

    grader = InstructionAdherenceGrader(model=model)
    instruction = "Write exactly 3 bullet points about AI safety."
    response = (
        "• AI safety is important\\n• We need alignment research\\n• Testing is crucial"
    )
    # Execute the grader
    result = await grader.aevaluate(
        instruction=instruction,
        response=response,
    )
    print(result)

    # Verify the result
    assert result is not None
    assert hasattr(result, "score")
    assert hasattr(result, "reason")
    assert hasattr(result, "metadata")

    assert "{instruction}" not in grader.to_dict().get("template")
    assert "{response}" not in grader.to_dict().get("template")
    assert "{input_section}" not in grader.to_dict().get("template")
