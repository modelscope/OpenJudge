# -*- coding: utf-8 -*-
import asyncio
import json
from abc import ABC
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Type,
    TypedDict,
    Union,
    Optional,
)

import yaml
from pydantic import BaseModel, Field, field_validator

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
                    ChatMessage(role="system", content=system_content)
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
    """Template for generating chat messages.

    The Template class supports different types of prompts:
    1. Simple prompts: str or List[ChatMessage]
    2. PromptDict: structured prompt with "system" and/or "user" keys
    3. LanguageDict: multilingual prompts mapping LanguageEnum to any of the above prompt types

    Attributes:
        prompt: Prompt for generating messages, can be a string, dict, or list of ChatMessage objects.
                Can also be a dictionary mapping language codes to prompts.

    Examples:
        >>> from rm_gallery.core.schema.template import Template, LanguageEnum
        >>> from rm_gallery.core.schema.message import ChatMessage
        >>>
        >>> # Simple string prompt (single user prompt)
        >>> template = Template(prompt="Hello, how are you?")
        >>>
        >>> # List of ChatMessage objects
        >>> messages = [
        ...     ChatMessage(role="system", content="You are a helpful assistant."),
        ...     ChatMessage(role="user", content="What is the weather like today?")
        ... ]
        >>> template = Template(prompt=messages)
        >>>
        >>> # PromptDict with system and user messages
        >>> prompt_dict = {
        ...     "system": "You are a helpful assistant.",
        ...     "user": "What is the weather like today?"
        ... }
        >>> template = Template(prompt=prompt_dict)
        >>>
        >>> # LanguageDict for multilingual support
        >>> multilingual_prompt = {
        ...     LanguageEnum.EN: "Hello, how are you?",
        ...     LanguageEnum.ZH: "你好，你怎么样？"
        ... }
        >>> template = Template(prompt=multilingual_prompt)
    """

    prompt: Prompt = Field(
        default=...,
        description="Prompt for generating messages, can be a string, dict, or list of ChatMessage objects.",
    )

    # 存储多语言模板的私有属性
    _multilingual_prompts: Optional[Dict[str, Prompt]] = None

    @field_validator("prompt", mode="before")
    @classmethod
    def validate_prompt(cls, v):
        """Validate prompt value."""
        if isinstance(v, dict):
            # Check if this looks like a LanguageDict (keys are language codes)
            if all(isinstance(k, (str, LanguageEnum)) for k in v.keys()):
                # If all keys are language codes, this should be handled via multilingual method
                # We'll allow it for backward compatibility but warn users
                pass
        return v

    def to_messages(self, language: Optional[str] = None) -> List[ChatMessage]:
        """Get messages for the specified language.

        This method handles different types of prompts:
        1. Single prompt (str or List[ChatMessage]) - directly converted to messages
        2. PromptDict - structured prompt with system/user keys
        3. Multilingual prompts - when language is specified and template was created with multilingual method

        Args:
            language: The language to retrieve messages for. Only used for multilingual templates.
                      If None or not found, will use the default prompt.

        Returns:
            List of ChatMessage objects for the specified language or prompt.

        Examples:
            >>> from rm_gallery.core.schema.template import Template
            >>> from rm_gallery.core.schema.message import ChatMessage
            >>>
            >>> # Single string prompt
            >>> template = Template(prompt="Hello, how are you?")
            >>> messages = template.to_messages()
            >>> len(messages)
            1
            >>> messages[0].role
            'user'
            >>>
            >>> # PromptDict with system and user messages
            >>> prompt_dict = {
            ...     "system": "You are a helpful assistant.",
            ...     "user": "What is the weather like today?"
            ... }
            >>> template = Template(prompt=prompt_dict)
            >>> messages = template.to_messages()
            >>> len(messages)
            2
            >>> messages[0].role
            'system'
            >>> messages[1].role
            'user'
            >>>
            >>> # Multilingual template
            >>> template = Template.multilingual({
            ...     "en": "Hello, how are you?",
            ...     "zh": "你好，你怎么样？"
            ... })
            >>> english_messages = template.to_messages("en")
            >>> len(english_messages)
            1
            >>> english_messages[0].content
            'Hello, how are you?'
            >>> chinese_messages = template.to_messages("zh")
            >>> chinese_messages[0].content
            '你好，你怎么样？'
        """
        # Handle multilingual prompts
        if self._multilingual_prompts is not None and language is not None:
            if language in self._multilingual_prompts:
                return _convert_prompt_to_messages(
                    self._multilingual_prompts[language]
                )
            elif "en" in self._multilingual_prompts:
                return _convert_prompt_to_messages(
                    self._multilingual_prompts["en"]
                )
            else:
                # Return first available language
                first_lang = next(iter(self._multilingual_prompts))
                return _convert_prompt_to_messages(
                    self._multilingual_prompts[first_lang]
                )

        # Handle regular prompt
        return _convert_prompt_to_messages(self.prompt)

    @classmethod
    def multilingual(cls, prompts: Dict[str, Prompt]) -> "Template":
        """Create a Template with multilingual support.

        Args:
            prompts: Dictionary mapping language codes to prompts

        Returns:
            Template instance with multilingual support

        Examples:
            >>> template = Template.multilingual({
            ...     "en": [{"role": "user", "content": "Hello!"}],
            ...     "zh": [{"role": "user", "content": "你好!"}]
            ... })
            >>> en_messages = template.to_messages("en")
            >>> zh_messages = template.to_messages("zh")
        """
        # Use English as default if available, otherwise use first language
        default_prompt = prompts.get("en", next(iter(prompts.values())))

        template = cls(prompt=default_prompt)
        template._multilingual_prompts = prompts
        return template


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
        language: Optional[str] = None,
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
        structured_model: Type[BaseModel] | None = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        """Generate chat response using the template.

        Args:
            structured_model: Optional structured model output
            language: Language code for multilingual templates
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
    template = Template.multilingual(
        {
            "en": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "{question}"},
            ],
        }
    )
    model = OpenAIChatModel(model_name="qwen-plus", stream=False)
    chat = Chat(template=template, model=model)
    result = asyncio.run(chat(question="What is the capital of France?"))
    print(result)
