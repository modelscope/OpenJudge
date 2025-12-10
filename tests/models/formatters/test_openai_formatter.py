# -*- coding: utf-8 -*-
"""Unit tests for OpenAIFormatter."""
from unittest.mock import mock_open, patch

import pytest

from rm_gallery.core.models.formatter.openai_formatter import (
    OpenAIChatFormatter,
    OpenAIMultiAgentFormatter,
    _to_openai_image_url,
)
from rm_gallery.core.models.schema.block import (
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    URLSource,
)
from rm_gallery.core.models.schema.message import ChatMessage


@pytest.mark.unit
class TestOpenAIFormatter:
    """Test cases for OpenAIFormatter class."""

    def test_init(self):
        """Test initialization of OpenAIFormatter."""
        formatter = OpenAIChatFormatter()
        assert isinstance(formatter, OpenAIChatFormatter)
        assert formatter.support_tools_api is True
        assert formatter.support_multiagent is True
        assert formatter.support_vision is True

    def test_assert_list_of_msgs_with_valid_input(self):
        """Test assert_list_of_msgs with valid input."""
        formatter = OpenAIChatFormatter()

        messages = [
            ChatMessage(role="user", content=[TextBlock(text="Hello")]),
            ChatMessage(role="assistant", content=[TextBlock(text="Hi")]),
        ]

        # Should not raise any exception
        formatter.assert_list_of_msgs(messages)

    def test_assert_list_of_msgs_with_invalid_input(self):
        """Test assert_list_of_msgs with invalid input."""
        formatter = OpenAIChatFormatter()

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

    @patch("os.path.exists")
    @patch("os.path.isfile")
    def test_to_openai_image_url_with_local_file(self, mock_isfile, mock_exists):
        """Test _to_openai_image_url with local file."""
        mock_exists.return_value = True
        mock_isfile.return_value = True

        # Test with valid local image file with correct extension
        with patch("builtins.open", mock_open(read_data=b"fake image data")):
            result = _to_openai_image_url("/path/to/image.png")
            assert result.startswith("data:image/png;base64,")

    @patch("os.path.exists")
    def test_to_openai_image_url_with_web_url(self, mock_exists):
        """Test _to_openai_image_url with web URL."""
        mock_exists.return_value = False

        # Test with valid web URL
        url = "https://example.com/image.png"
        result = _to_openai_image_url(url)
        assert result == url

    @patch("os.path.exists")
    @patch("os.path.isfile")
    def test_to_openai_image_url_with_invalid_extension(self, mock_isfile, mock_exists):
        """Test _to_openai_image_url with invalid file extension."""
        mock_exists.return_value = True
        mock_isfile.return_value = True

        # Test with invalid extension
        with pytest.raises(TypeError):
            _to_openai_image_url("/path/to/image.txt")

    def test_format_with_text_only_messages(self):
        """Test format method with text-only messages."""
        formatter = OpenAIChatFormatter()

        messages = [
            ChatMessage(role="user", content=[TextBlock(text="Hello")]),
            ChatMessage(role="assistant", content=[TextBlock(text="Hi there!")]),
        ]

        # Since format is an async method, we need to run it in an event loop
        import asyncio

        formatted = asyncio.run(formatter.format(messages))

        assert len(formatted) == 2
        assert formatted[0]["role"] == "user"
        # Access attributes properly instead of using dict notation
        assert formatted[0]["content"][0]["text"] == "Hello"
        assert formatted[1]["role"] == "assistant"
        assert formatted[1]["content"][0]["text"] == "Hi there!"

    def test_format_with_tool_use_block(self):
        """Test format method with tool use block."""
        formatter = OpenAIChatFormatter()

        messages = [
            ChatMessage(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id="call_123",
                        name="get_weather",
                        input={"location": "New York"},
                    ),
                ],
            ),
        ]

        import asyncio

        formatted = asyncio.run(formatter.format(messages))

        assert len(formatted) == 1
        assert formatted[0]["role"] == "assistant"
        assert "tool_calls" in formatted[0]
        assert len(formatted[0]["tool_calls"]) == 1
        assert formatted[0]["tool_calls"][0]["id"] == "call_123"
        assert formatted[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_format_with_tool_result_block(self):
        """Test format method with tool result block."""
        formatter = OpenAIChatFormatter()

        # Create a dummy class to allow 'tool' role for testing purposes
        class TestChatMessage(ChatMessage):
            role: str  # Override the literal restriction for testing

        messages = [
            TestChatMessage(
                role="tool",
                name="get_weather",
                content=[
                    ToolResultBlock(
                        id="call_123",
                        name="get_weather",
                        output="Sunny, 25°C",  # Using string output instead of list
                    ),
                ],
            ),
        ]

        import asyncio

        formatted = asyncio.run(formatter.format(messages))

        # Filter out empty messages
        filtered_formatted = [msg for msg in formatted if msg.get("content") or msg.get("tool_calls")]
        assert len(filtered_formatted) == 1
        assert filtered_formatted[0]["role"] == "tool"
        assert filtered_formatted[0]["tool_call_id"] == "call_123"
        assert filtered_formatted[0]["content"] == "Sunny, 25°C"
        assert filtered_formatted[0]["name"] == "get_weather"

    def test_format_with_image_block(self):
        """Test format method with image block."""
        formatter = OpenAIChatFormatter()

        messages = [
            ChatMessage(
                role="user",
                content=[
                    TextBlock(text="Here is an image:"),
                    ImageBlock(
                        type="image",
                        source=URLSource(
                            type="url",
                            url="https://example.com/image.jpg",
                        ),
                    ),
                ],
            ),
        ]

        import asyncio

        formatted = asyncio.run(formatter.format(messages))

        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"
        assert len(formatted[0]["content"]) == 2
        assert formatted[0]["content"][0]["text"] == "Here is an image:"
        assert formatted[0]["content"][1]["type"] == "image_url"
        assert formatted[0]["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"


@pytest.mark.unit
class TestOpenAIMultiAgentFormatter:
    """Test cases for OpenAIMultiAgentFormatter class."""

    def test_init(self):
        """Test initialization of OpenAIMultiAgentFormatter."""
        formatter = OpenAIMultiAgentFormatter()
        assert isinstance(formatter, OpenAIMultiAgentFormatter)
        assert formatter.support_tools_api is True
        assert formatter.support_multiagent is True
        assert formatter.support_vision is True

    def test_format_agent_message(self):
        """Test _format_agent_message method."""
        formatter = OpenAIMultiAgentFormatter()

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

        import asyncio

        formatted = asyncio.run(formatter._format_agent_message(messages))

        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"
        assert "<history>" in formatted[0]["content"][0]["text"]
        assert "Alice: Hello Bob" in formatted[0]["content"][0]["text"]
        assert "Bob: Hi Alice" in formatted[0]["content"][0]["text"]
        assert "</history>" in formatted[0]["content"][0]["text"]
