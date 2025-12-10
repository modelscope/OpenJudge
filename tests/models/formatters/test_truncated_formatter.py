# -*- coding: utf-8 -*-
"""Unit tests for TruncatedFormatterBase."""
from unittest.mock import MagicMock

import pytest

from rm_gallery.core.models.formatter.truncated_formatter import TruncatedFormatterBase
from rm_gallery.core.models.schema.block import TextBlock
from rm_gallery.core.models.schema.message import ChatMessage


@pytest.mark.unit
class TestTruncatedFormatterBase:
    """Test cases for TruncatedFormatterBase class."""

    def test_init(self):
        """Test initialization of TruncatedFormatterBase."""

        # Create a concrete implementation since TruncatedFormatterBase is abstract
        class ConcreteFormatter(TruncatedFormatterBase):
            async def _format(self, msgs):
                return []

        formatter = ConcreteFormatter()
        assert isinstance(formatter, TruncatedFormatterBase)
        assert formatter.token_counter is None
        assert formatter.max_tokens is None

    def test_init_with_params(self):
        """Test initialization of TruncatedFormatterBase with parameters."""

        # Create a concrete implementation since TruncatedFormatterBase is abstract
        class ConcreteFormatter(TruncatedFormatterBase):
            async def _format(self, msgs):
                return []

        mock_token_counter = MagicMock()
        formatter = ConcreteFormatter(
            token_counter=mock_token_counter,
            max_tokens=1000,
        )
        assert formatter.token_counter == mock_token_counter
        assert formatter.max_tokens == 1000

    def test_format_method_raises_not_implemented(self):
        """Test that instantiation does not raise TypeError for abstract class."""

        # Abstract classes can be instantiated in Python unless they have abstract methods
        # that are not implemented. But the error would occur when trying to call the abstract method.
        class IncompleteFormatter(TruncatedFormatterBase):
            pass

        # This should not raise an error during instantiation
        formatter = IncompleteFormatter()
        assert isinstance(formatter, TruncatedFormatterBase)

        # But calling the abstract method should raise NotImplementedError
        messages = [
            ChatMessage(role="user", content=[TextBlock(text="Hello")]),
        ]

        import asyncio

        with pytest.raises(NotImplementedError):
            asyncio.run(formatter.format(messages))

    def test_format_tool_sequence(self):
        """Test _format_tool_sequence method."""

        # Create a concrete implementation with all required methods
        class ConcreteFormatter(TruncatedFormatterBase):
            async def _format(self, msgs):
                # Simple mock implementation
                return [{"role": msg.role, "content": str(msg.content)} for msg in msgs]

            async def _format_tool_sequence(self, msgs):
                # Mock implementation
                return [{"role": msg.role, "content": str(msg.content)} for msg in msgs]

            async def _format_agent_message(self, msgs, is_first=True):
                # Mock implementation
                return [{"role": msg.role, "content": str(msg.content)} for msg in msgs]

        formatter = ConcreteFormatter()

        messages = [
            ChatMessage(role="user", content=[TextBlock(text="Hello")]),
            ChatMessage(role="assistant", content=[TextBlock(text="Hi")]),
        ]

        import asyncio

        result = asyncio.run(formatter._format_tool_sequence(messages))
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_format_agent_message(self):
        """Test _format_agent_message method."""

        # Create a concrete implementation with all required methods
        class ConcreteFormatter(TruncatedFormatterBase):
            async def _format(self, msgs):
                # Simple mock implementation
                return [{"role": msg.role, "content": str(msg.content)} for msg in msgs]

            async def _format_tool_sequence(self, msgs):
                # Mock implementation
                return [{"role": msg.role, "content": str(msg.content)} for msg in msgs]

            async def _format_agent_message(self, msgs, is_first=True):
                # Mock implementation
                return [{"role": msg.role, "content": str(msg.content)} for msg in msgs]

        formatter = ConcreteFormatter()

        messages = [
            ChatMessage(role="user", content=[TextBlock(text="Hello")]),
            ChatMessage(role="assistant", content=[TextBlock(text="Hi")]),
        ]

        import asyncio

        result = asyncio.run(formatter._format_agent_message(messages))
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
