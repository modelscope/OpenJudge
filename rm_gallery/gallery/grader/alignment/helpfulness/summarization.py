# -*- coding: utf-8 -*-
from typing import Any, List

from rm_gallery.core.grader.base import GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.grader import GraderMode
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import BaseHelpfulnessGrader

RUBRICS = (
    "Comprehensive Coverage of Core Content: A superior summary captures all critical "
    "elements, themes, and details central to the source material without omitting key "
    "information. Avoidance of Irrelevant or Tangential Information: Focuses exclusively "
    "on the primary subject, eliminating extraneous details that distract from the core "
    "narrative or argument. Logical Structure and Coherence: Information is organized in "
    "a clear, hierarchical, or chronological sequence to ensure readability and logical "
    "progression of ideas. Factual Accuracy and Neutral Objectivity: The summary must "
    "faithfully represent the source material without introducing distortions, opinions, "
    "or subjective interpretations, maintaining a neutral tone throughout."
)

# Summarization Score System Prompt
SUMMARIZATION_POINTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Summarization Score User Prompt
SUMMARIZATION_POINTWISE_USER_PROMPT = """# Task Description
Please act as an impartial judge and evaluate the quality of a summary.
You should assess the summary based on comprehensive coverage, avoidance of irrelevant information, logical structure, and factual accuracy.
Be as objective as possible.

# Rubrics
{rubrics}

# Original Text
{query}

# Summary
{answer}

# Output Requirement
```json
{
    "score": "A numerical score from 0.0 to 1.0 representing the quality of the summary."
    "reason": "The reason for the score."
}
```
"""

# Summarization Rank System Prompt
SUMMARIZATION_LISTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Summarization Rank User Prompt
SUMMARIZATION_LISTWISE_USER_PROMPT = """# Task Description

Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate answers. Your task is to select the single best answer \
from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. \
These rubrics are ordered from highest to lowest importance. You must check each \
candidate answer in turn to see if it violates any rubric, and provide reasons for \
any violations you find. These reasons should be used as references for ranking \
the answers.
You may organize your reasoning as you see fit, but keep your thought process as concise as possible.

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
"""

SUMMARIZATION_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=SUMMARIZATION_POINTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=SUMMARIZATION_POINTWISE_USER_PROMPT,
        ),
    ],
)

SUMMARIZATION_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=SUMMARIZATION_LISTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=SUMMARIZATION_LISTWISE_USER_PROMPT,
        ),
    ],
)


class SummarizationGrader(BaseHelpfulnessGrader):
    """Summarization: Condenses information while preserving key points and overall meaning."""

    _point_template = SUMMARIZATION_POINTWISE_TEMPLATE
    _list_template = SUMMARIZATION_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Template | None = None,
        mode: GraderMode = GraderMode.LISTWISE,
        rubrics: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the SummarizationGrader.

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
            name="summarization",
            mode=mode,
            model=model,
            template=template,
            rubrics=rubrics,
            description="Condenses information while preserving key points and overall meaning.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs: Any,
    ) -> GraderScore | GraderRank:
        """Evaluate the quality of the summary based on the query.

        Evaluates summarization responses for their ability to condense information
        while preserving key points and overall meaning. The grader focuses on
        key point preservation, conciseness, and clarity.

        Args:
            query (str): The original text to be summarized.
            answer (str | List[str]): The summary/summaries to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical summarization quality score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of summaries by quality
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise summarization grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model_name="gpt-3.5-turbo")
            >>> grader = SummarizationGrader(mode=GraderMode.POINTWISE, model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="Climate change is a significant global challenge that affects ecosystems, economies, and societies worldwide. It is caused primarily by human activities such as burning fossil fuels, deforestation, and industrial processes. The impacts include rising sea levels, extreme weather events, biodiversity loss, and food security issues. Addressing climate change requires international cooperation, policy changes, and technological innovations.",
            ...     answer="Climate change, caused by human activities, has global impacts on ecosystems and societies. It requires international cooperation and technological solutions."
            ... ))
            >>> print(result.score, result.reason)
            0.85 The summary captures the key points but could include more specific impacts.
        """
        return await super().aevaluate(query=query, answer=answer, **kwargs)
