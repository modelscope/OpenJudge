# -*- coding: utf-8 -*-
"""Pointwise template for training."""
import re
from typing import Any, Dict

from pydantic import BaseModel, Field


class BasePromptTemplate(BaseModel):
    """Base template for prompt formatting and parsing"""

    @classmethod
    def _parse(cls, text: str) -> Dict[str, Any]:
        """Parse XML-like tags from text"""
        contents: Dict[str, Any] = {}
        for field_name, _ in cls.model_fields.items():
            if field_name == "think":
                pattern = r"<think>(.*?)</think>"
            else:
                pattern = rf"<{field_name}>(.*?)</{field_name}>"

            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                contents[field_name] = match.group(1).strip()
        return contents

    @classmethod
    def schema(  # pylint: disable=unused-argument
        cls,
        enable_thinking: bool = True,
        **kwargs: Any,
    ) -> str:
        """Generate schema description for output"""
        lines = []
        for field_name, field_info in cls.model_fields.items():
            description = field_info.description or ""
            if field_name == "think" and not enable_thinking:
                continue
            lines.append(f"<{field_name}>{description}</{field_name}>")
        return "\n".join(lines)


class PointwiseTrainTemplate(BasePromptTemplate):
    """Template for pointwise evaluation training."""

    score: int = Field(default=..., description="score of helpfulness from 0 to 4")

    @classmethod
    def parse(cls, text: str) -> "PointwiseTrainTemplate":
        """Parse text and create instance."""
        try:
            contents = cls._parse(text)
            if "score" in contents:
                contents["score"] = int(contents["score"])
            return cls(**contents)
        except Exception as e:
            raise ValueError(f"Failed to parse: {e}") from e

    @classmethod
    def format(
        cls,
        desc: str,
        rubrics: str,
        examples: str,
        query: str,
        context: str,
        answer: str,
        **kwargs: Any,
    ) -> str:
        """Format prompt for pointwise evaluation."""
        if examples:
            examples = f"# Examples\n{examples}\n"

        return f"""# Task Description
        {desc}
        # Rubrics
        {rubrics}

        {examples}

        # Query
        {query}

        # Context
        {context}

        # Answer
        {answer}

        # Output Requirement
        {cls.schema(**kwargs)}
        """
