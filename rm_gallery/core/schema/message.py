# -*- coding: utf-8 -*-
from datetime import datetime
from typing import Any, Dict, List, Literal, Sequence, overload

import shortuuid
from pydantic import BaseModel, Field

from rm_gallery.core.schema.block import (
    AudioBlock,
    ContentBlock,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    VideoBlock,
)


class ChatMessage(BaseModel):
    """A message in a chat conversation, compatible with AgentScope Msg class."""

    name: str = Field(default="", description="The name of the message sender")
    content: str | Sequence[ContentBlock] = Field(
        default="",
        description="The content of the message, either a string or a list of content blocks",
    )
    role: Literal["user", "assistant", "system"] = Field(
        default="user",
        description="The role of the message sender",
    )
    metadata: Dict[str, Any] | None = Field(
        default=None,
        description="The metadata of the message",
    )
    timestamp: str | None = Field(
        default=None,
        description="The created timestamp of the message",
    )
    invocation_id: str | None = Field(
        default=None,
        description="The related API invocation id",
    )
    id: str = Field(
        default_factory=lambda: shortuuid.uuid(),
        description="Unique identifier for the message",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize ChatMessage with current timestamp if not provided.

        Args:
            **data: Message data
        """
        if "timestamp" not in data or data.get("timestamp") is None:
            data["timestamp"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S.%f",
            )[:-3]
        super().__init__(**data)

    def to_dict(self) -> dict:
        """Convert the message into JSON dict data.

        Returns:
            Dictionary representation of the message
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, json_data: dict) -> "ChatMessage":
        """Load a message object from the given JSON data.

        Args:
            json_data: JSON data to load from

        Returns:
            ChatMessage instance
        """
        return cls(**json_data)

    def has_content_blocks(
        self,
        block_type: Literal[
            "text",
            "tool_use",
            "tool_result",
            "image",
            "audio",
            "video",
        ]
        | None = None,
    ) -> bool:
        """Check if the message has content blocks of the given type.

        Args:
            block_type (Literal["text", "tool_use", "tool_result", "image", \
            "audio", "video"] | None, defaults to None):
                The type of the block to be checked. If `None`, it will
                check if there are any content blocks.
        """
        return len(self.get_content_blocks(block_type)) > 0

    def get_text_content(self) -> str | None:
        """Get the pure text blocks from the message content."""
        if isinstance(self.content, str):
            return self.content

        gathered_text = None
        for block in self.content:
            if block.get("type") == "text":
                if gathered_text is None:
                    gathered_text = str(block.get("text"))
                else:
                    gathered_text += block.get("text")
        return gathered_text

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["text"],
    ) -> List[TextBlock]:
        ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["tool_use"],
    ) -> List[ToolUseBlock]:
        ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["tool_result"],
    ) -> List[ToolResultBlock]:
        ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["image"],
    ) -> List[ImageBlock]:
        ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["audio"],
    ) -> List[AudioBlock]:
        ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["video"],
    ) -> List[VideoBlock]:
        ...

    @overload
    def get_content_blocks(
        self,
        block_type: None = None,
    ) -> List[ContentBlock]:
        ...

    def get_content_blocks(
        self,
        block_type: Literal[
            "text",
            "thinking",
            "tool_use",
            "tool_result",
            "image",
            "audio",
            "video",
        ]
        | None = None,
    ) -> (
        List[ContentBlock]
        | List[TextBlock]
        | List[ThinkingBlock]
        | List[ToolUseBlock]
        | List[ToolResultBlock]
        | List[ImageBlock]
        | List[AudioBlock]
        | List[VideoBlock]
    ):
        """Get the content in block format. If the content is a string,
        it will be converted to a text block.

        Args:
            block_type (`Literal["text", "thinking", "tool_use", \
            "tool_result", "image", "audio", "video"] | None`, optional):
                The type of the block to be extracted. If `None`, all blocks
                will be returned.

        Returns:
            `List[ContentBlock]`:
                The content blocks.
        """
        blocks = []
        if isinstance(self.content, str):
            blocks.append(
                TextBlock(type="text", text=self.content),
            )
        else:
            blocks = self.content or []

        if block_type is not None:
            blocks = [_ for _ in blocks if _["type"] == block_type]

        return blocks

    # def __repr__(self) -> str:
    #     """Get the string representation of the message."""
    #     return (
    #         f"Msg(id='{self.id}', "
    #         f"name='{self.name}', "
    #         f"content={repr(self.content)}, "
    #         f"role='{self.role}', "
    #         f"metadata={repr(self.metadata)}, "
    #         f"timestamp='{self.timestamp}', "
    #         f"invocation_id='{self.invocation_id}')"
    #     )

    def format(self, **kwargs) -> "ChatMessage":
        message = self.model_copy()
        if isinstance(message.content, str):
            message.content = message.content.format(**kwargs)
        else:
            message.content = [block.format(**kwargs) for block in message.content]
        return message
