# -*- coding: utf-8 -*-
"""Tokenizer utilities for text processing.

This module provides a unified interface for different tokenization strategies
including tiktoken, jieba, and simple whitespace-based tokenizers. It supports
multilingual content processing with specialized handling for Chinese text.
"""

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List

from pydantic import BaseModel, Field


class TokenizerEnum(str, Enum):
    """
    Enum for tokenizer types.

    Supported tokenizer types: tiktoken, jieba, simple.

    Example:
        >>> tokenizer_type = TokenizerEnum.tiktoken
        >>> print(tokenizer_type.value)
        tiktoken
    """

    tiktoken = "tiktoken"
    jieba = "jieba"
    simple = "simple"


class BaseTokenizer(BaseModel, ABC):
    """
    Base tokenizer class providing unified tokenization interface.

    This abstract base class defines the interface for different tokenization
    strategies including tiktoken and jieba tokenizers.

    Attributes:
        name (str): Name of the tokenizer.

    Example:
        >>> class MyTokenizer(BaseTokenizer):
        ...     name: str = "my_tokenizer"
        ...     def tokenize(self, text: str) -> List[str]:
        ...         return text.split()
        >>> tokenizer = MyTokenizer(name="test")
        >>> print(tokenizer.name)
        test
    """

    name: str = Field(..., description="Name of the tokenizer")

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text into a list of tokens.

        Args:
            text: Input text to tokenize.

        Returns:
            List[str]: List of token strings.

        Example:
            >>> # This is an abstract method that must be implemented by subclasses
        """

    def preprocess_text(self, text: str, to_lower: bool = False) -> str:
        """
        Preprocess text before tokenization.

        Args:
            text: Input text.
            to_lower: Whether to convert to lowercase.

        Returns:
            str: Preprocessed text.

        Example:
            >>> tokenizer = BaseTokenizer(name="test")
            >>> result = tokenizer.preprocess_text("  Hello World  ", to_lower=True)
            >>> print(result)
            hello world
        """
        text = text.strip()
        if to_lower:
            text = text.lower()
        return text


class TiktokenTokenizer(BaseTokenizer):
    """
    Tiktoken-based tokenizer supporting multilingual content.

    Uses tiktoken encoding for robust tokenization of Chinese, English
    and other languages. Falls back to simple splitting if tiktoken fails.

    Attributes:
        name (str): Name of the tokenizer, defaults to "tiktoken".
        encoding_name (str): Tiktoken encoding name, defaults to "cl100k_base".

    Example:
        >>> tokenizer = TiktokenTokenizer()
        >>> tokens = tokenizer.tokenize("Hello, 世界!")
        >>> print(len(tokens) > 0)
        True
    """

    name: str = Field(default="tiktoken", description="Tiktoken tokenizer")
    encoding_name: str = Field(
        default="cl100k_base",
        description="Tiktoken encoding name",
    )

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using tiktoken encoder.

        Args:
            text: Input text to tokenize.

        Returns:
            List[str]: List of token strings.

        Example:
            >>> tokenizer = TiktokenTokenizer()
            >>> tokens = tokenizer.tokenize("Hello world")
            >>> print(isinstance(tokens, list))
            True
        """
        try:
            import tiktoken

            encoding = tiktoken.get_encoding(self.encoding_name)
            tokens = encoding.encode(text)
            # Convert token ids back to strings for comparison
            token_strings = [encoding.decode([token]) for token in tokens]
            return token_strings
        except Exception:
            # Fallback to simple splitting if tiktoken fails
            return text.split()


class JiebaTokenizer(BaseTokenizer):
    """
    Jieba-based tokenizer for Chinese text processing.

    Provides Chinese word segmentation using jieba library with optional
    Chinese character filtering and preprocessing capabilities.

    Attributes:
        name (str): Name of the tokenizer, defaults to "jieba".
        chinese_only (bool): Whether to keep only Chinese characters, defaults to False.

    Example:
        >>> tokenizer = JiebaTokenizer()
        >>> tokens = tokenizer.tokenize("你好世界")
        >>> print(isinstance(tokens, list))
        True
    """

    name: str = Field(default="jieba", description="Jieba Chinese tokenizer")
    chinese_only: bool = Field(
        default=False,
        description="Whether to keep only Chinese characters",
    )

    def _preserve_chinese(self, text: str) -> str:
        """
        Preserve only Chinese characters.

        Args:
            text: Input text.

        Returns:
            str: Text with only Chinese characters.

        Example:
            >>> tokenizer = JiebaTokenizer()
            >>> result = tokenizer._preserve_chinese("Hello 你好123")
            >>> print(result)
            你好
        """
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        return "".join(chinese_chars)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Chinese text using jieba.

        Args:
            text: Input text to tokenize.

        Returns:
            List[str]: List of token strings.

        Raises:
            ImportError: If jieba library is not installed.

        Example:
            >>> tokenizer = JiebaTokenizer()
            >>> tokens = tokenizer.tokenize("你好 世界")
            >>> print(isinstance(tokens, list))
            True
        """
        try:
            import jieba

            if self.chinese_only:
                text = self._preserve_chinese(text)
            return list(jieba.cut(text))
        except ImportError as e:
            raise ImportError("jieba library required for Chinese tokenization: pip install jieba") from e


class SimpleTokenizer(BaseTokenizer):
    """
    Simple whitespace-based tokenizer.

    Basic tokenizer that splits text on whitespace. Used as fallback
    when other tokenizers are not available or fail.

    Attributes:
        name (str): Name of the tokenizer, defaults to "simple".

    Example:
        >>> tokenizer = SimpleTokenizer()
        >>> tokens = tokenizer.tokenize("Hello world test")
        >>> print(tokens)
        ['Hello', 'world', 'test']
    """

    name: str = Field(default="simple", description="Simple whitespace tokenizer")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text by splitting on whitespace.

        Args:
            text: Input text to tokenize.

        Returns:
            List[str]: List of token strings.

        Example:
            >>> tokenizer = SimpleTokenizer()
            >>> tokens = tokenizer.tokenize("hello world")
            >>> print(tokens)
            ['hello', 'world']
        """
        return text.split()


def get_tokenizer(
    tokenizer_type: TokenizerEnum = TokenizerEnum.tiktoken,
    encoding_name: str = "cl100k_base",
    chinese_only: bool = False,
    **kwargs: Any,
) -> BaseTokenizer:
    """
    Factory function to create tokenizer instances.

    Args:
        tokenizer_type: Type of tokenizer ("tiktoken", "jieba", "simple").
        encoding_name: Tiktoken encoding name (for tiktoken tokenizer).
        chinese_only: Whether to keep only Chinese characters (for jieba tokenizer).
        **kwargs: Additional arguments for tokenizer initialization.

    Returns:
        BaseTokenizer: Tokenizer instance.

    Raises:
        ValueError: If tokenizer_type is not supported.

    Example:
        >>> tokenizer = get_tokenizer(TokenizerEnum.simple)
        >>> tokens = tokenizer.tokenize("hello world")
        >>> print(tokens)
        ['hello', 'world']
        >>>
        >>> tokenizer = get_tokenizer(TokenizerEnum.tiktoken)
        >>> tokens = tokenizer.tokenize("Hello, world!")
        >>> print(isinstance(tokens, list))
        True
    """
    tokenizer_type = TokenizerEnum(tokenizer_type)
    if tokenizer_type is TokenizerEnum.tiktoken:
        return TiktokenTokenizer(encoding_name=encoding_name, **kwargs)
    elif tokenizer_type is TokenizerEnum.jieba:
        return JiebaTokenizer(chinese_only=chinese_only, **kwargs)
    elif tokenizer_type is TokenizerEnum.simple:
        return SimpleTokenizer(**kwargs)
    else:
        raise ValueError(
            f"Unsupported tokenizer type: {tokenizer_type}. " f"Supported types: tiktoken, jieba, simple",
        )
