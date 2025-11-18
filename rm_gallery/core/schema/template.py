# -*- coding: utf-8 -*-
import asyncio
import json
from abc import ABC
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Literal, Type

import yaml
from pydantic import BaseModel, Field

from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.response import ChatResponse
from rm_gallery.core.utils.instance import init_instance_by_config


class LanguageEnum(str, Enum):
    """Language enumeration for templates."""

    EN = "en"
    ZH = "zh"


class RequiredField(BaseModel):
    """Required fields for grading."""

    name: str = Field(default=..., description="name of the field")
    type: str = Field(default=..., description="type of the field")
    position: Literal["data", "sample", "grader"] = Field(
        default="data",
        description="position of the field",
    )
    description: str = Field(
        default=...,
        description="description of the field",
    )


class Template(BaseModel):
    """Template for generating chat messages."""

    messages: List[ChatMessage] | Dict[
        LanguageEnum,
        List[ChatMessage],
    ] = Field(
        default_factory=list,
        description="messages for generating chat",
    )

    # @model_validator(mode="before")
    # def validate_template(cls, values) -> dict:
    #     messages = values.get("messages", [])
    #     # Pattern to match placeholders like {word}, {word.word}, {word.word.word}, etc.
    #     placeholder_pattern = (
    #         r"\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\}"
    #     )

    #     # Handle both dict and RequiredField instances
    #     required_fields_raw = values.get("required_fields", [])
    #     required_fields = []
    #     for field in required_fields_raw:
    #         if isinstance(field, dict):
    #             required_fields.append(field.get("name"))
    #         elif isinstance(field, RequiredField):
    #             required_fields.append(field.name)
    #         elif hasattr(field, "name"):
    #             required_fields.append(field.name)
    #     required = []

    #     all_messages = []
    #     if isinstance(messages, list):
    #         messages = messages
    #     elif isinstance(messages, dict):
    #         for language, language_messages in messages.items():
    #             if not isinstance(language_messages, list):
    #                 raise ValueError("Invalid message format")
    #             all_messages.extend(language_messages)
    #     else:
    #         raise ValueError("Invalid message format")

    #     for message in all_messages:
    #         content = (
    #             message.get("content", "")
    #             if isinstance(message, dict)
    #             else getattr(message, "content", "")
    #         )
    #         # Find all placeholders in the content
    #         placeholders = re.findall(placeholder_pattern, content)
    #         for placeholder in placeholders:
    #             # Add to required if not already present
    #             if placeholder not in required:
    #                 required.append(placeholder)

    #     for name in required:
    #         if name not in required_fields:
    #             raise ValueError(f"Required field {name} is missing")
    #     return values

    def get(self, language: LanguageEnum | None = LanguageEnum.EN):
        if isinstance(self.messages, list):
            messages = self.messages
        elif isinstance(self.messages, dict):
            if language is None:
                language = LanguageEnum.EN
            assert language in self.messages
            messages = self.messages.get(language, [])
        else:
            raise ValueError("Invalid messages")

        return messages


class Chat(ABC):
    """Chat for generating response."""

    def __init__(self, template: Template | dict, model: dict | ChatModelBase):
        """
        Initialize a ChatTemplate.
        """
        self.template = (
            template
            if isinstance(template, Template)
            else Template(**template)
        )
        self.model = init_instance_by_config(model, accept_type=ChatModelBase)

    def format(
        self,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Format messages with provided keyword arguments.

        Args:
            **kwargs: Keyword arguments to format with

        Returns:
            List of formatted message dictionaries
        """
        messages = self.template.get(language)
        messages = [message.to_dict() for message in messages]

        for message in messages:
            message["content"] = message.get("content", "").format(**kwargs)
        return messages

    async def __call__(
        self,
        structured_model: Type[BaseModel] | None = None,
        language: LanguageEnum = LanguageEnum.EN,
        **kwargs,
    ) -> ChatResponse:
        """Generate chat response using the template.

        Args:
            structured_model: Optional structured model output
            **kwargs: Keyword arguments for formatting messages

        Returns:
            Chat response
        """
        messages = self.format(language=language, **kwargs)
        response = await self.model(
            messages=messages,
            structured_model=structured_model,
        )

        # Handle case where response might be an AsyncGenerator
        if isinstance(response, AsyncGenerator):
            # For streaming responses, collect all chunks
            content_parts = []
            metadata = {}
            usage = None

            async for chunk in response:
                content_parts.extend(chunk.content)
                if chunk.metadata:
                    metadata.update(chunk.metadata)
                if chunk.usage:
                    usage = chunk.usage

            # Create a consolidated response
            response = ChatResponse(
                content=content_parts,
                metadata=metadata or None,
                usage=usage,
            )

        return response

    @classmethod
    def load(cls, path: str):
        if path.endswith("json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif path.endswith("yaml"):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError("Invalid file format")
        return cls(**data)


if __name__ == "__main__":
    template = Template(
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="{question}"),
        ],
    )
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    chat = Chat(template=template, model=model)
    result = asyncio.run(chat(question="What is the capital of France?"))
    print(result)
