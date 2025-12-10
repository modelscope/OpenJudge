# -*- coding: utf-8 -*-
"""Unit tests for DashScopeMultiAgentFormatter."""
import pytest

from rm_gallery.core.models.formatter.dashscope_formatter import (
    DashScopeMultiAgentFormatter,
)
from rm_gallery.core.models.schema.block import TextBlock
from rm_gallery.core.models.schema.message import ChatMessage


@pytest.mark.unit
class TestDashScopeMultiAgentFormatter:
    """Test cases for DashScopeMultiAgentFormatter class."""

    def test_init(self):
        """Test initialization of DashScopeMultiAgentFormatter."""
        formatter = DashScopeMultiAgentFormatter()
        assert isinstance(formatter, DashScopeMultiAgentFormatter)
        assert formatter.support_tools_api is True
        assert formatter.support_multiagent is True
        assert formatter.support_vision is True

    @pytest.mark.asyncio
    async def test_format_agent_message(self):
        """Test _format_agent_message method."""
        formatter = DashScopeMultiAgentFormatter()

        messages = [
            ChatMessage(
                role="user",
                name="Alice",
                content=[TextBlock(text="Hello Bob")],
            ),
            ChatMessage(
                role="assistant",
                name="Bob",
                content=[TextBlock(text="Hi Alice")],
            ),
        ]

        formatted = await formatter._format_agent_message(messages)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"
        # Check that the content contains the expected text
        content_text = str(formatted[0]["content"])
        assert "Alice: Hello Bob" in content_text
        assert "Bob: Hi Alice" in content_text
        assert "<history>" in content_text
        assert "</history>" in content_text

    @pytest.mark.asyncio
    async def test_format_system_message(self):
        """Test _format_system_message method."""
        formatter = DashScopeMultiAgentFormatter()

        message = ChatMessage(
            role="system",
            content=[TextBlock(text="You are a helpful assistant.")],
        )

        formatted = await formatter._format_system_message(message)

        assert formatted["role"] == "system"
        assert formatted["content"] == "You are a helpful assistant."
