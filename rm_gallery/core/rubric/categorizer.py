"""
LLM-Based Rubric Categorizer

This module provides semantic categorization and aggregation of evaluation rubrics
using LLM-based classification. It merges similar rubrics into structured Theme-Tips
format to reduce redundancy and improve usability.

Features:
- LLM-powered semantic classification into Theme-Tips structure
- Configurable number of output categories
- Multi-language support (Chinese/English)
- Structured output with Pydantic models
- Integrated with AutoRubrics pipeline for automatic aggregation
- Compatible with both Single Mode and Batch Mode

"""

from typing import List

from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.model.template import LanguageEnum
from rm_gallery.core.rubric.prompts import RubricCategorizationTemplate


class RubricCategory(BaseModel):
    theme: str = Field(description="Theme statement for the category")
    tips: List[str] = Field(description="List of tips for the category")


class RubricCategorizationOutput(BaseModel):
    categories: List[RubricCategory] = Field(description="List of categorized rubrics")
    reason: str = Field(description="Reasoning for the categorization")


class LLMRubricCategorizer:
    """Enhanced LLM-based Rubric classifier compatible with main.py output"""

    def __init__(
        self,
        num_categories: int = 5,
        model_name: str = "qwen3-32b",
        language: str = "zh",
    ):
        self.num_categories = num_categories
        self.language = LanguageEnum(language)

        # Initialize LLM using new architecture
        self.llm = OpenAIChatModel(model_name=model_name)

        # Create model config for ChatTemplate
        self.model_config = {
            "model_name": model_name,
            "stream": False,
        }

        # Initialize categorization template
        self.categorization_template = RubricCategorizationTemplate.categorization(
            self.model_config
        )

    async def categorize_rubrics(self, rubrics: List[str]) -> tuple[List[str], dict]:
        """Main method: perform semantic classification of rubrics

        Returns:
            tuple: (categorized_rubrics, aggregation_info)
        """

        if len(rubrics) == 0:
            logger.error("Input rubrics list is empty")
            return [], {"categorization_successful": False, "error": "Empty input"}

        try:
            # Format rubrics text
            rubrics_text = "\n".join(
                [f"{i+1}. {rubric}" for i, rubric in enumerate(rubrics)]
            )

            # Call LLM using ChatTemplate with structured output
            response_obj = await self.categorization_template(
                chat_output=RubricCategorizationOutput,
                rubrics_text=rubrics_text,
                num_categories=self.num_categories,
                language=self.language,
            )

            # Get structured data from metadata
            if not response_obj.metadata or "categories" not in response_obj.metadata:
                raise ValueError("No categories in structured response")

            categories = response_obj.metadata["categories"]

            # Generate directly usable string list
            ready_to_use_list = []
            for category in categories:
                theme = category.get("theme", "")
                tips = category.get("tips", [])

                # Assemble into single string: Theme + Tips
                theme_str = f"Theme: {theme}"
                tips_str = "\n".join(
                    [f"- Tip{i}: {tip}" for i, tip in enumerate(tips, start=1)]
                )

                # Combine into complete evaluation rubric string
                complete_rubric = f"{theme_str}\n{tips_str}"
                ready_to_use_list.append(complete_rubric)

            logger.info(f"Generated {len(ready_to_use_list)} categorized rubrics")

            # Return both rubrics and aggregation info as expected
            aggregation_info = {
                "num_categories": len(ready_to_use_list),
                "original_rubrics_count": len(rubrics),
                "categorization_successful": True,
            }
            return ready_to_use_list, aggregation_info

        except Exception as e:
            logger.error(f"Rubric categorization failed: {e}")
            return [], {"categorization_successful": False, "error": str(e)}
