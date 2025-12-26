# -*- coding: utf-8 -*-
"""Unit tests for PromptTemplate."""
import pytest

from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate


@pytest.mark.unit
class TestPromptTemplate:
    """Test cases for PromptTemplate class."""

    def _get_multi_language_prompt_template(self):
        return PromptTemplate(
            messages={
                LanguageEnum.EN: [
                    ChatMessage(
                        role="system",
                        content="system prompt text",
                    ),
                    ChatMessage(
                        role="user",
                        content="user prompt text",
                    ),
                ],
                LanguageEnum.ZH: [
                    ChatMessage(
                        role="system",
                        content="系统提示",
                    ),
                    ChatMessage(
                        role="user",
                        content="用户提示",
                    ),
                ],
            },
        )

    def _get_single_list_prompt_template(self):
        return PromptTemplate(
            messages=[
                ChatMessage(
                    role="system",
                    content="system prompt text",
                ),
                ChatMessage(
                    role="user",
                    content="user prompt text",
                ),
            ]
        )

    def test_init_with_multi_language_messages(self):
        """Test initialization of PromptTemplate."""
        pt = self._get_multi_language_prompt_template()

        assert len(pt.messages) == 2

        assert len(pt.messages[LanguageEnum.EN.value]) == 2
        assert pt.messages[LanguageEnum.EN.value][0].role == "system"
        assert pt.messages[LanguageEnum.EN.value][0].content == "system prompt text"
        assert pt.messages[LanguageEnum.EN.value][1].role == "user"
        assert pt.messages[LanguageEnum.EN.value][1].content == "user prompt text"

        assert len(pt.messages[LanguageEnum.ZH.value]) == 2
        assert pt.messages[LanguageEnum.ZH.value][0].role == "system"
        assert pt.messages[LanguageEnum.ZH.value][0].content == "系统提示"
        assert pt.messages[LanguageEnum.ZH.value][1].role == "user"
        assert pt.messages[LanguageEnum.ZH.value][1].content == "用户提示"

    def test_init_with_single_list_messages(self):
        """Test initialization of PromptTemplate."""
        pt = self._get_single_list_prompt_template()

        assert len(pt.messages) == 2
        assert pt.messages[0].role == "system"
        assert pt.messages[0].content == "system prompt text"
        assert pt.messages[1].role == "user"
        assert pt.messages[1].content == "user prompt text"

    def test_get_prompts_of_multi_languages(self):
        """Test initialization of PromptTemplate."""
        pt = self._get_multi_language_prompt_template()

        prompts = pt.get_prompt()
        assert len(prompts) == 2

        assert len(prompts[LanguageEnum.EN.value]) == 2

        d = prompts[LanguageEnum.EN.value][0]
        assert d["role"] == "system"
        assert d["content"] == "system prompt text"

        d = prompts[LanguageEnum.EN.value][1]
        assert d["role"] == "user"
        assert d["content"] == "user prompt text"

        assert len(prompts[LanguageEnum.ZH.value]) == 2

        d = prompts[LanguageEnum.ZH.value][0]
        assert d["role"] == "system"
        assert d["content"] == "系统提示"

        d = prompts[LanguageEnum.ZH.value][1]
        assert d["role"] == "user"
        assert d["content"] == "用户提示"

    def test_get_prompts_of_one_of_multi_languages(self):
        """Test initialization of PromptTemplate."""
        pt = self._get_multi_language_prompt_template()

        prompts = pt.get_prompt(LanguageEnum.ZH)
        assert len(prompts) == 1

        assert len(prompts[LanguageEnum.ZH.value]) == 2

        d = prompts[LanguageEnum.ZH.value][0]
        assert d["role"] == "system"
        assert d["content"] == "系统提示"

        d = prompts[LanguageEnum.ZH.value][1]
        assert d["role"] == "user"
        assert d["content"] == "用户提示"

    def test_get_prompts_of_single_list(self):
        """Test initialization of PromptTemplate."""
        pt = self._get_single_list_prompt_template()
        prompts = pt.get_prompt()

        assert len(prompts) == 1

        assert len(prompts[LanguageEnum.ANY.value]) == 2

        d = prompts[LanguageEnum.ANY.value][0]
        assert d["role"] == "system"
        assert d["content"] == "system prompt text"

        d = prompts[LanguageEnum.ANY.value][1]
        assert d["role"] == "user"
        assert d["content"] == "user prompt text"
