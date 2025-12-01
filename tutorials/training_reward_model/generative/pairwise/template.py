# -*- coding: utf-8 -*-
"""Pairwise comparison template for training."""
import re
from typing import Any, Dict, Optional

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


class PairwiseComparisonTemplate(BasePromptTemplate):
    """Template for pairwise comparison tasks between two responses."""

    think: str = Field(default=..., description="your reasoning trace", alias="think")
    preference: Optional[str] = Field(
        default=None,
        description="which response is better: A, B, or tie",
    )

    @classmethod
    def parse(cls, text: str) -> "PairwiseComparisonTemplate":
        """Parse text and create instance."""
        try:
            contents = cls._parse(text)
            preference = contents.get("preference", "unknown").strip().upper()

            # Normalize preference values
            if preference in ["A", "RESPONSE A", "ANSWER A"]:
                contents["preference"] = "A"
            elif preference in ["B", "RESPONSE B", "ANSWER B"]:
                contents["preference"] = "B"
            elif preference in ["TIE", "EQUAL", "SAME"]:
                contents["preference"] = "tie"
            else:
                contents["preference"] = preference
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
        response_a: str,
        response_b: str,
        **kwargs: Any,
    ) -> str:
        """Format prompt for pairwise comparison."""
        if examples:
            examples = f"# Examples\n{examples}\n"

        return f"""# Task Description
{desc}

# Rubrics
{rubrics}

{examples}

# Query
{query}

# Response A
{response_a}

# Response B
{response_b}

# Output Requirement
{cls.schema(**kwargs)}
"""
