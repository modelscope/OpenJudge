# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    """The text block."""

    type: Literal["text"] = Field(
        default="text",
        description="The type of the block",
    )
    text: str = Field(default="", description="The text content")

    def format(self, **kwargs) -> "TextBlock":
        """Format the text block."""
        block = self.model_copy()
        block.text = block.text.format(**kwargs)
        return block


class ThinkingBlock(BaseModel):
    """The thinking block."""

    type: Literal["thinking"] = Field(
        default="thinking",
        description="The type of the block",
    )
    thinking: str = Field(default="", description="The thinking content")

    def format(self, **kwargs) -> "ThinkingBlock":
        """Format the thinking block."""
        block = self.model_copy()
        block.thinking = block.thinking.format(**kwargs)
        return block


class Base64Source(BaseModel):
    """The base64 source"""

    type: Literal["base64"] = Field(
        default="base64",
        description="The type of the src, must be `base64`",
    )
    media_type: str = Field(
        ...,
        description="The media type of the data, e.g. `image/jpeg` or `audio/mpeg`",
    )
    data: str = Field(
        ...,
        description="The base64 data, in format of RFC 2397",
    )

    def format(self, **kwargs) -> "Base64Source":
        """Format the base64 source."""
        block = self.model_copy()
        block.media_type = block.media_type.format(**kwargs)
        block.data = block.data.format(**kwargs)
        return block


class URLSource(BaseModel):
    """The URL source"""

    type: Literal["url"] = Field(
        default="url",
        description="The type of the src, must be `url`",
    )
    url: str = Field(..., description="The URL of the image or audio")

    def format(self, **kwargs) -> "URLSource":
        """Format the URL source."""
        block = self.model_copy()
        block.url = block.url.format(**kwargs)
        return block


class ImageBlock(BaseModel):
    """The image block"""

    type: Literal["image"] = Field(
        default="image",
        description="The type of the block, must be `image`",
    )
    source: Base64Source | URLSource = Field(
        ...,
        description="The source of the image",
    )

    def format(self, **kwargs) -> "ImageBlock":
        """Format the image block."""
        block = self.model_copy()
        block.source = block.source.format(**kwargs)
        return block


class AudioBlock(BaseModel):
    """The audio block"""

    type: Literal["audio"] = Field(
        default="audio",
        description="The type of the block, must be `audio`",
    )
    source: Base64Source | URLSource = Field(
        ...,
        description="The source of the audio",
    )

    def format(self, **kwargs) -> "AudioBlock":
        """Format the audio block."""
        block = self.model_copy()
        block.source = block.source.format(**kwargs)
        return block


class VideoBlock(BaseModel):
    """The video block"""

    type: Literal["video"] = Field(
        default="video",
        description="The type of the block, must be `video`",
    )
    source: Base64Source | URLSource = Field(
        ...,
        description="The source of the video",
    )

    def format(self, **kwargs) -> "VideoBlock":
        """Format the video block."""
        block = self.model_copy()
        block.source = block.source.format(**kwargs)
        return block


class ToolUseBlock(BaseModel):
    """The tool use block"""

    type: Literal["tool_use"] = Field(
        default="tool_use",
        description="The type of the block, must be `tool_use`",
    )
    id: str = Field(..., description="The identity of the tool call")
    name: str = Field(..., description="The name of the tool function")
    input: Dict[str, Any] = Field(
        default_factory=dict,
        description="The input arguments of the tool function",
    )

    def format(self, **kwargs) -> "ToolUseBlock":
        """Format the tool use block."""
        return self


class ToolResultBlock(BaseModel):
    """The tool result block"""

    type: Literal["tool_result"] = Field(
        default="tool_result",
        description="The type of the block, must be `tool_result`",
    )
    id: str = Field(..., description="The identity of the tool call result")
    output: str | List[TextBlock | ImageBlock | AudioBlock] = Field(
        ...,
        description="The output of the tool function",
    )
    name: str = Field(..., description="The name of the tool function")

    def format(self, **kwargs) -> "ToolResultBlock":
        """Format the tool result block."""
        block = self.model_copy()
        if isinstance(block.output, str):
            block.output = block.output.format(**kwargs)
        else:
            block.output = [
                output_block.format(**kwargs) for output_block in block.output
            ]
        return block


ContentBlock = (
    TextBlock
    | ThinkingBlock
    | ImageBlock
    | AudioBlock
    | VideoBlock
    | ToolUseBlock
    | ToolResultBlock
)
