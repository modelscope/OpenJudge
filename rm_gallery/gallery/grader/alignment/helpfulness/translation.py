# -*- coding: utf-8 -*-
from typing import List
from rm_gallery.core.grader.base import (
    GraderMode,
    LLMGrader,
    GraderScore,
    GraderRank,
)
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import (
    BaseHelpfulnessGrader,
)

RUBRICS = """Accuracy in Translation: Faithfully convey the original text's meaning, intent, and nuances without distortion, omission, or addition.
Contextual Appropriateness: Preserve the original context, tone, and purpose while adapting to target language conventions and specified formatting requirements.
Fluency and Naturalness: Produce translations that read naturally in the target language, avoiding awkward phrasing or literal translations that compromise readability.
Cultural Sensitivity: Appropriately handle culture-specific references, idioms, and concepts, ensuring they are correctly adapted or explained for the target audience.
"""

TRANSLATION_SCORE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
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
""",
        ),
    ],
)

TRANSLATION_RANK_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
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
    "rank": ["The rank score of the answer in the list."]
    "reason": "The reason for the score."
}
```
""",
        ),
    ],
)


class TranslationGrader(BaseHelpfulnessGrader):
    """Translation: Converts text from one language to another while preserving meaning, tone, and cultural nuances."""

    _point_template = TRANSLATION_SCORE_TEMPLATE
    _list_template = TRANSLATION_RANK_TEMPLATE
    _rubrics = RUBRICS

    def __init__(self, model: ChatModelBase | dict, template: Template | None = None, mode: GraderMode = GraderMode.LISTWISE, **kwargs):
        """Initialize the SafetyGrader."""
        super().__init__(
            name="Translation",
            mode=mode,
            model=model,
            template=template,
            description="Translation: Converts text from one language to another while preserving meaning, tone, and cultural nuances.",
            **kwargs,
        )
    
    async def a_evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
    ) -> GraderScore | GraderRank:
        """Evaluate the translation quality of the response based on the query.

        Evaluates translation responses for their ability to convert text from one
        language to another while preserving meaning, tone, and cultural nuances.
        The grader focuses on accuracy, fluency, and cultural sensitivity.

        Args:
            query (str): The original text to be translated.
            answer (str | List[str]): The translation(s) to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

                Each GraderScore contains:
                    - score: A numerical score assigned by the grader
                    - reason: Explanation of how the score was determined
                    - metadata: Optional additional information from the evaluation

        Example:
            >>> grader = TranslationGrader()
            >>> result = await grader.evaluate(
            ...     query="Translate to French: Hello, how are you today?",
            ...     answer="Bonjour, comment allez-vous aujourd'hui?"
            ... )
        """
        return await super().a_evaluate(query=query, answer=answer, **kwargs)
