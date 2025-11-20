# -*- coding: utf-8 -*-
from typing import Any, List

from rm_gallery.core.grader.base import GraderMode, GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import BaseHelpfulnessGrader

RUBRICS = (
    "Accuracy in Translation:\n    "
    "Faithfully convey the original text's meaning, intent, and nuances without "
    "distortion, omission, or addition.\n"
    "Contextual Appropriateness:\n    "
    "Preserve the original context, tone, and purpose while adapting to target "
    "language conventions and specified formatting requirements.\n"
    "Fluency and Naturalness:\n    "
    "Produce translations that read naturally in the target language, avoiding "
    "awkward phrasing or literal translations that compromise readability.\n"
    "Cultural Sensitivity:\n    "
    "Appropriately handle culture-specific references, idioms, and concepts, "
    "ensuring they are correctly adapted or explained for the target audience."
)

# Translation Score System Prompt
TRANSLATION_POINTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Translation Score User Prompt
TRANSLATION_POINTWISE_USER_PROMPT = """# Task Description
Please act as an impartial judge and evaluate the quality of a translation.
You should assess the translation based on accuracy, fluency, and cultural sensitivity.
Be as objective as possible.

# Rubrics
{rubrics}

# Original Text
{query}

# Translation
{answer}

# Output Requirement
```json
{
    "score": "A numerical score from 0.0 to 1.0 representing the quality of the translation."
    "reason": "The reason for the score."
}
```
"""

# Translation Rank System Prompt
TRANSLATION_LISTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Translation Rank User Prompt
TRANSLATION_LISTWISE_USER_PROMPT = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate answers. Your task is to select the single best answer \
from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance.These rubrics can serve as supplementary knowledge for your judgment. If you find any of the rubrics helpful for the current problem, feel free to use them as supplements.If all answers meet all rubrics, you can judge and choose one answer by yourself.

# Rubrics
{rubrics}

# Query
{query}

# Answers
{answer}

# Output Requirement
```json
{
    "rank": ["The rank score of the answer in the list."],
    "reason": "The reason for the score."
}
```
"""

TRANSLATION_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=TRANSLATION_POINTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=TRANSLATION_POINTWISE_USER_PROMPT,
        ),
    ],
)

TRANSLATION_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=TRANSLATION_LISTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=TRANSLATION_LISTWISE_USER_PROMPT,
        ),
    ],
)


class TranslationGrader(BaseHelpfulnessGrader):
    """Translation: Accurately converts text between languages while preserving meaning, tone, and cultural context."""

    _point_template = TRANSLATION_POINTWISE_TEMPLATE
    _list_template = TRANSLATION_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Template | None = None,
        mode: GraderMode = GraderMode.LISTWISE,
        rubrics: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the TranslationGrader.

        Args:
            model: The language model used for evaluation. Can be either a ChatModelBase
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            mode: The grader mode. Defaults to LISTWISE.
            rubrics: Custom rubrics for evaluation. If None, default rubrics will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="translation",
            mode=mode,
            model=model,
            template=template,
            rubrics=rubrics,
            description="Accurately converts text between languages while preserving meaning, tone, and cultural context.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs: Any,
    ) -> GraderScore | GraderRank:
        """Evaluate the quality of the translation based on the query.

        Evaluates translation responses for their ability to accurately convert
        text between languages while preserving meaning, tone, and cultural context.
        The grader focuses on accuracy, fluency, and cultural appropriateness.

        Args:
            query (str): The original text to be translated.
            answer (str | List[str]): The translation(s) to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical translation quality score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of translations by quality
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise translation grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model_name="gpt-3.5-turbo")
            >>> grader = TranslationGrader(mode=GraderMode.POINTWISE, model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="Hello, how are you today?",
            ...     answer="Hola, ¿cómo estás hoy?"
            ... ))
            >>> print(result.score, result.reason)
            1.0 The translation accurately conveys the meaning and maintains appropriate tone.
        """
        return await super().aevaluate(query=query, answer=answer, **kwargs)
