# -*- coding: utf-8 -*-
"""
Test Template Module

Tests for the Template and Chat classes functionality.
"""

import asyncio

import pytest

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.template import Chat, LanguageEnum, Template


def test_template_main_example():
    """Test the example code from template.py __main__ section"""
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
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is the capital of France?"


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_chat_execution():
    """Test executing the chat with actual model call"""
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
    assert len(messages) == 2

    # This would actually call the model
    result = await chat(question="What is the capital of France?")
    assert result is not None
    assert hasattr(result, "content")
