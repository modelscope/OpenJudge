# -*- coding: utf-8 -*-
"""Unit tests for QwenVLModel."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from rm_gallery.core.models.qwen_vl_model import QwenVLModel
from rm_gallery.core.models.schema.block import TextBlock
from rm_gallery.core.models.schema.message import ChatMessage
from rm_gallery.core.models.schema.response import ChatResponse


class PersonModelForTesting(BaseModel):
    """test model."""

    name: str
    age: int


@pytest.mark.unit
class TestQwenVLModel:
    """Test cases for QwenVLModel class."""

    def test_init(self):
        """Test initialization of QwenVLModel."""
        # Test basic initialization
        model = QwenVLModel(api_key="test-key", model="qwen-vl-plus")
        assert model.model == "qwen-vl-plus"
        assert not model.stream

        # Test initialization with custom parameters
        model = QwenVLModel(
            api_key="test-key",
            model="qwen-vl-max",
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
        )
        assert model.model == "qwen-vl-max"
        assert model.temperature == 0.7
        assert model.top_p == 0.9
        assert model.max_tokens == 1000

        # Test initialization without api_key should fail
        with pytest.raises(ValueError) as exc_info:
            QwenVLModel(model="qwen-vl-plus")
        assert "API key must be provided" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
