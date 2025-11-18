# -*- coding: utf-8 -*-
import asyncio
import json
from abc import ABC
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    TypedDict,
    Union,
)

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


class PromptDict(TypedDict, total=False):
    system: str | ChatMessage
    """"""
    user: str | ChatMessage
    """"""


Prompt = Union[str, PromptDict, List[ChatMessage]]


def _convert_prompt_to_messages(prompt: Prompt) -> List[ChatMessage]:
    """Convert various prompt formats to List[ChatMessage]."""
    if isinstance(prompt, str):
        return [ChatMessage(role="user", content=prompt)]
    elif isinstance(prompt, list):
        # Already in the correct format
        return prompt
    elif isinstance(prompt, dict):
        messages = []
        if "system" in prompt:
            system_content = prompt["system"]
            if isinstance(system_content, str):
                messages.append(
                    ChatMessage(role="system", content=system_content),
                )
            else:
                messages.append(system_content)
        if "user" in prompt:
            user_content = prompt["user"]
            if isinstance(user_content, str):
                messages.append(ChatMessage(role="user", content=user_content))
            else:
                messages.append(user_content)
        return messages
    else:
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")


class Template(BaseModel):
    """Template for generating chat messages."""

    messages: List[ChatMessage] | Dict[
        LanguageEnum,
        List[ChatMessage],
    ] = Field(
        default_factory=list,
        description="messages for generating chat",
    )

    def to_messages(self, language: LanguageEnum | None = LanguageEnum.EN):
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

    @classmethod
    def from_prompt(cls, prompt: Prompt) -> "Template":
        """Create a Template instance from a prompt."""
        messages = _convert_prompt_to_messages(prompt)
        return cls(messages=messages)

    @classmethod
    def from_multilingual(
        cls,
        prompt: Dict[LanguageEnum | str, Prompt],
    ) -> "Template":
        return cls(
            messages={
                LanguageEnum(lang): _convert_prompt_to_messages(prompt)
                for lang, prompt in prompt.items()
            },
        )


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
        language: LanguageEnum | None = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Format messages with provided keyword arguments.

        Args:
            language: Language code for multilingual templates
            **kwargs: Keyword arguments to format with

        Returns:
            List of formatted message dictionaries
        """
        messages = self.template.to_messages(language)
        messages = [message.to_dict() for message in messages]

        for message in messages:
            message["content"] = message.get("content", "").format(**kwargs)
        return messages

    async def __call__(
        self,
        callback: Type[BaseModel] | Callable | None = None,
        language: LanguageEnum | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Generate chat response using the template.

        Args:
            callback: Optional callback to output
            language: Language code for multilingual templates
            **kwargs: Keyword arguments for formatting messages

        Returns:
            Chat response
        """
        messages = self.format(language=language, **kwargs)
        # Check if callback is a Pydantic BaseModel class
        if (
            callback
            and isinstance(callback, type)
            and issubclass(callback, BaseModel)
        ):
            # If callback is a Pydantic class, pass it as structured_model
            response = await self.model(
                messages=messages,
                structured_model=callback,
            )
        else:
            # If callback is not a Pydantic class or is None, don't pass structured_model
            response = await self.model(
                messages=messages,
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

        # If callback is a function, call it with the response
        if callback and isinstance(callback, Callable):
            response.metadata = response.metadata or {}
            response.metadata.update(callback(response))

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
        messages={
            LanguageEnum.EN: [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "{question}"},
            ],
        },
    )
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    chat = Chat(template=template, model=model)
    messages = chat.format(
        language="en",
        question="What is the capital of France?",
    )
    print(messages)
    result = asyncio.run(chat(question="What is the capital of France?"))
    print(result)
