# -*- coding: utf-8 -*-
"""Unit tests for DashScopeChatModel."""
import pytest

from rm_gallery.core.models.dashscope_chat_model import DashScopeChatModel


@pytest.mark.unit
class TestDashScopeChatModel:
    """Test cases for DashScopeChatModel class."""

    def test_init(self):
        """Test initialization of DashScopeChatModel."""
        # Test basic initialization
        model = DashScopeChatModel(model="qwen-turbo", api_key="test-key")
        assert model.model == "qwen-turbo"
        assert model.stream

        # Test initialization with custom parameters
        model = DashScopeChatModel(
            model="qwen-plus",
            api_key="test-key",
            stream=False,
            enable_thinking=True,
            temperature=0.7,
        )
        assert model.model == "qwen-plus"
        # When enable_thinking is True, stream should be forced to True
        assert model.stream
        assert model.enable_thinking
        assert model.kwargs["temperature"] == 0.7
