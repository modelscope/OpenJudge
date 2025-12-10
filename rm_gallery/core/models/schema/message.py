# -*- coding: utf-8 -*-
"""Chat message schema definition.

This module defines the ChatMessage class, which represents a message in a chat conversation.
It is compatible with AgentScope Msg class and supports both simple text messages and
rich content blocks including images, audio, video, and tool interactions.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Sequence, overload

import shortuuid
from pydantic import BaseModel, Field

from rm_gallery.core.models.schema.block import (
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
    """A message in a chat conversation, compatible with AgentScope Msg class.

    This class represents a single message in a chat conversation, supporting both
    simple text messages and rich content blocks. It includes metadata such as
    sender name, role, timestamp, and unique identifier.

    Attributes:
        name (str): The name of the message sender.
        content (Union[str, Sequence[ContentBlock]]): The content of the message,
            either a string or a list of content blocks.
        role (Literal["user", "assistant", "system"]): The role of the message sender.
        metadata (Dict[str, Any] | None): The metadata of the message.
        timestamp (str | None): The created timestamp of the message.
        invocation_id (str | None): The related API invocation id.
        id (str): Unique identifier for the message.

    Example:
        >>> # Create a simple text message
        >>> msg = ChatMessage(
        ...     name="Alice",
        ...     content="Hello, world!",
        ...     role="user"
        ... )
        >>> print(msg.name)
        Alice
        >>> print(msg.content)
        Hello, world!
        >>>
        >>> # Create a message with content blocks
        >>> msg = ChatMessage(
        ...     name="Assistant",
        ...     content=[TextBlock(type="text", text="Here's an image:")],
        ...     role="assistant"
        ... )
    """

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
        default_factory=lambda: shortuuid.uuid(),  # pylint: disable=unnecessary-lambda
        description="Unique identifier for the message",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize ChatMessage with current timestamp if not provided.

        Args:
            **data: Message data including name, content, role, metadata, etc.

        Example:
            >>> msg = ChatMessage(name="Bob", content="Hi there!", role="user")
            >>> print(isinstance(msg.timestamp, str))
            True
        """
        if "timestamp" not in data or data.get("timestamp") is None:
            data["timestamp"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S.%f",
            )[:-3]
        super().__init__(**data)

    def to_dict(self) -> dict:
        """Convert the message into JSON dict data.

        Returns:
            dict: Dictionary representation of the message.

        Example:
            >>> msg = ChatMessage(name="Test", content="Hello")
            >>> data = msg.to_dict()
            >>> print("name" in data and "content" in data)
            True
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, json_data: dict) -> "ChatMessage":
        """Load a message object from the given JSON data.

        Args:
            json_data: JSON data to load from.

        Returns:
            ChatMessage: Instance created from the JSON data.

        Example:
            >>> data = {"name": "Test", "content": "Hello", "role": "user"}
            >>> msg = ChatMessage.from_dict(data)
            >>> print(msg.name)
            Test
            >>> print(msg.content)
            Hello
        """
        return cls(**json_data)

    def has_content_blocks(
        self,
        block_type: (
            Literal[
                "text",
                "tool_use",
                "tool_result",
                "image",
                "audio",
                "video",
            ]
            | None
        ) = None,
    ) -> bool:
        """Check if the message has content blocks of the given type.

        Args:
            block_type: The type of the block to be checked. If `None`, it will
                check if there are any content blocks.

        Returns:
            bool: True if the message has content blocks of the specified type, False otherwise.

        Example:
            >>> msg = ChatMessage(content=[TextBlock(type="text", text="Hello")])
            >>> print(msg.has_content_blocks("text"))
            True
            >>> print(msg.has_content_blocks("image"))
            False
        """
        return len(self.get_content_blocks(block_type)) > 0

    def get_text_content(self) -> str | None:
        """Get the pure text blocks from the message content.

        Returns:
            str | None: The concatenated text content, or None if no text blocks exist.

        Example:
            >>> msg = ChatMessage(content="Hello world")
            >>> print(msg.get_text_content())
            Hello world
            >>>
            >>> msg = ChatMessage(content=[TextBlock(type="text", text="Hello"),
            ...                            TextBlock(type="text", text=" world")])
            >>> print(msg.get_text_content())
            Hello world
        """
        if isinstance(self.content, str):
            return self.content

        gathered_text = None
        for block in self.content:
            if block.type == "text":
                if gathered_text is None:
                    gathered_text = str(block.text)
                else:
                    gathered_text += block.text
        return gathered_text

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["text"],
    ) -> List[TextBlock]: ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["tool_use"],
    ) -> List[ToolUseBlock]: ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["tool_result"],
    ) -> List[ToolResultBlock]: ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["image"],
    ) -> List[ImageBlock]: ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["audio"],
    ) -> List[AudioBlock]: ...

    @overload
    def get_content_blocks(
        self,
        block_type: Literal["video"],
    ) -> List[VideoBlock]: ...

    @overload
    def get_content_blocks(
        self,
        block_type: None = None,
    ) -> List[ContentBlock]: ...

    def get_content_blocks(
        self,
        block_type: (
            Literal[
                "text",
                "thinking",
                "tool_use",
                "tool_result",
                "image",
                "audio",
                "video",
            ]
            | None
        ) = None,
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
            block_type: The type of the block to be extracted. If `None`, all blocks
                will be returned.

        Returns:
            List[ContentBlock]: The content blocks.

        Example:
            >>> # Get all content blocks
            >>> msg = ChatMessage(content=[TextBlock(type="text", text="Hello")])
            >>> blocks = msg.get_content_blocks()
            >>> print(len(blocks))
            1
            >>>
            >>> # Get specific type of blocks
            >>> text_blocks = msg.get_content_blocks("text")
            >>> print(len(text_blocks))
            1
        """
        blocks = []
        if isinstance(self.content, str):
            blocks.append(
                TextBlock(type="text", text=self.content),
            )
        else:
            blocks = self.content or []

        if block_type is not None:
            blocks = [_ for _ in blocks if _.type == block_type]

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
        """Format the message content with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to format the message content with.

        Returns:
            ChatMessage: A copy of the message with formatted content.

        Example:
            >>> msg = ChatMessage(content="Hello {name}!")
            >>> formatted_msg = msg.format(name="Alice")
            >>> print(formatted_msg.get_text_content())
            Hello Alice!
        """
        message = self.model_copy()
        if isinstance(message.content, str):
            message.content = message.content.format(**kwargs)
        else:
            message.content = [block.format(**kwargs) for block in message.content]
        return message
