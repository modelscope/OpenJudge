# -*- coding: utf-8 -*-
"""
Test ComplianceGrader Grader

Tests for the ComplianceGrader class functionality.
"""
from unittest.mock import AsyncMock

import pytest
from rm_gallery.core.graders.common.compliance import ComplianceGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel


def test_compliance_grader_creation():
    """Test creating a ComplianceGrader instance"""
    model = OpenAIChatModel(model="qwen-plus", api_key="fake-api-key", stream=False)
    grader = ComplianceGrader(model=model)

    assert grader is not None
    assert hasattr(grader, "name")
    assert grader.name == "compliance"


@pytest.mark.asyncio
async def test_compliance_grader_execution():
    """Test executing the compliance grader with actual model call"""
    # Initialize the grader
    model = OpenAIChatModel(model="qwen3-32b", api_key="fake-api-key", stream=False)
    mock_parse_result = AsyncMock()
    mock_parse_result.metadata = {"score": 3.0, "reason": "perfect"}
    model.achat = AsyncMock(return_value=mock_parse_result)

    grader = ComplianceGrader(model=model)
    instruction = "Write exactly 3 bullet points about AI safety."
    response = "• AI safety is important\\n• We need alignment research\\n• Testing is crucial"
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
