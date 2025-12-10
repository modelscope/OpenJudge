# -*- coding: utf-8 -*-
"""Unit tests for BaseFormatter."""
import base64
import os
import tempfile
from unittest.mock import mock_open, patch

import pytest

from rm_gallery.core.models.formatter.base_formatter import (
    BaseFormatter,
    _save_base64_data,
)
from rm_gallery.core.models.schema.block import (
    Base64Source,
    ImageBlock,
    TextBlock,
    URLSource,
)
from rm_gallery.core.models.schema.message import ChatMessage


class ConcreteFormatter(BaseFormatter):
    """Concrete implementation of BaseFormatter for testing purposes."""

    async def format(self, *args, **kwargs):
        """Concrete implementation of abstract method."""
        pass


@pytest.mark.unit
class TestBaseFormatter:
    """Test cases for BaseFormatter class."""

    def test_save_base64_data(self):
        """Test _save_base64_data function."""
        # Test saving base64 data to a temporary file
        media_type = "image/png"
        base64_data = base64.b64encode(b"fake image data").decode("utf-8")

        file_path = _save_base64_data(media_type, base64_data)

        # Check that file was created
        assert os.path.exists(file_path)

        # Check file content
        with open(file_path, "rb") as f:
            content = f.read()
            assert content == b"fake image data"

        # Clean up
        os.remove(file_path)

    def test_convert_tool_result_to_string_with_string(self):
        """Test convert_tool_result_to_string with string input."""
        result = BaseFormatter.convert_tool_result_to_string("simple text")
        assert result == "simple text"

    def test_convert_tool_result_to_string_with_text_blocks(self):
        """Test convert_tool_result_to_string with text blocks."""
        # This method is meant to work with raw dict data, not Pydantic models
        output = [
            {"type": "text", "text": "First line"},
            {"type": "text", "text": "Second line"},
        ]

        result = BaseFormatter.convert_tool_result_to_string(output)
        # When there are multiple text blocks, they should be joined with newlines and prefixed with "- "
        assert result == "- First line\n- Second line"

    def test_convert_tool_result_to_string_with_image_block_url(self):
        """Test convert_tool_result_to_string with image block from URL."""
        # This method is meant to work with raw dict data, not Pydantic models
        output = [
            {"type": "text", "text": "Image result:"},
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "http://example.com/image.png",
                },
            },
        ]

        result = BaseFormatter.convert_tool_result_to_string(output)
        assert "Image result:" in result
        assert "image can be found at: http://example.com/image.png" in result

    @patch("rm_gallery.core.models.formatter.base_formatter._save_base64_data")
    def test_convert_tool_result_to_string_with_image_block_base64(self, mock_save_base64):
        """Test convert_tool_result_to_string with image block from base64."""
        mock_save_base64.return_value = "/tmp/fake_image.png"

        # This method is meant to work with raw dict data, not Pydantic models
        output = [
            {"type": "text", "text": "Image result:"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "base64encodeddata",
                },
            },
        ]

        result = BaseFormatter.convert_tool_result_to_string(output)
        assert "Image result:" in result
        assert "image can be found at: /tmp/fake_image.png" in result

    def test_convert_tool_result_to_string_with_unsupported_block(self):
        """Test convert_tool_result_to_string with unsupported block type."""
        output = [
            {"type": "unsupported", "text": "Unsupported block"},
        ]

        with pytest.raises(ValueError) as exc_info:
            BaseFormatter.convert_tool_result_to_string(output)
        assert "Unsupported block type" in str(exc_info.value)

    def test_convert_tool_result_to_string_with_invalid_source(self):
        """Test convert_tool_result_to_string with invalid source type."""
        output = [
            {
                "type": "image",
                "source": {
                    "type": "invalid",
                    "url": "http://example.com/image.png",
                },
            },
        ]

        with pytest.raises(ValueError) as exc_info:
            BaseFormatter.convert_tool_result_to_string(output)
        assert "Invalid image source" in str(exc_info.value)

    def test_assert_list_of_msgs_with_valid_input(self):
        """Test assert_list_of_msgs with valid input."""
        msgs = [
            ChatMessage(role="user", content=[TextBlock(text="Hello")]),
            ChatMessage(role="assistant", content=[TextBlock(text="Hi")]),
        ]

        # Should not raise any exception
        BaseFormatter.assert_list_of_msgs(msgs)

    def test_assert_list_of_msgs_with_non_list_input(self):
        """Test assert_list_of_msgs with non-list input."""
        with pytest.raises(TypeError) as exc_info:
            BaseFormatter.assert_list_of_msgs("invalid input")
        assert "Input must be a list of ChatMessage objects" in str(exc_info.value)

    def test_assert_list_of_msgs_with_invalid_message_type(self):
        """Test assert_list_of_msgs with list containing non-ChatMessage objects."""
        msgs = [
            ChatMessage(role="user", content=[TextBlock(text="Hello")]),
            "invalid message",
        ]

        with pytest.raises(TypeError) as exc_info:
            BaseFormatter.assert_list_of_msgs(msgs)
        assert "Expected ChatMessage object" in str(exc_info.value)
