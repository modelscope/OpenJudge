# -*- coding: utf-8 -*-
"""Unit tests for DashScopeFormatter."""
import pytest

from rm_gallery.core.models.formatter.dashscope_formatter import DashScopeChatFormatter
from rm_gallery.core.models.schema.block import ImageBlock, TextBlock
from rm_gallery.core.models.schema.message import ChatMessage


@pytest.mark.unit
class TestDashScopeFormatter:
    """Test cases for DashScopeFormatter class."""

    def test_init(self):
        """Test initialization of DashScopeFormatter."""
        formatter = DashScopeChatFormatter()
        assert isinstance(formatter, DashScopeChatFormatter)
        assert formatter.support_tools_api is True
        assert formatter.support_multiagent is False
        assert formatter.support_vision is True

    def test_assert_list_of_msgs_with_valid_input(self):
        """Test assert_list_of_msgs with valid input."""
        formatter = DashScopeChatFormatter()

        messages = [
            ChatMessage(role="user", content=[TextBlock(text="Hello")]),
            ChatMessage(role="assistant", content=[TextBlock(text="Hi")]),
        ]

        # Should not raise any exception
        formatter.assert_list_of_msgs(messages)

    def test_assert_list_of_msgs_with_invalid_input(self):
        """Test assert_list_of_msgs with invalid input."""
        formatter = DashScopeChatFormatter()

        # Test with non-list input
        with pytest.raises(TypeError, match="Input must be a list of ChatMessage objects"):
            formatter.assert_list_of_msgs("invalid input")

        # Test with list containing non-ChatMessage objects
        with pytest.raises(TypeError, match="Expected ChatMessage object"):
            formatter.assert_list_of_msgs(
                [
                    ChatMessage(role="user", content=[TextBlock(text="Hello")]),
                    "invalid message",
                ],
            )
