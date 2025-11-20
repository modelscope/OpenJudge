# -*- coding: utf-8 -*-
from typing import Any, List

from rm_gallery.core.grader.base import GraderMode, GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import BaseHelpfulnessGrader

RUBRICS = (
    "Preservation of Meaning:\n    "
    "The rewritten text must accurately preserve the original meaning without "
    "introducing factual errors or distortions.\n"
    "Enhancement of Clarity:\n    "
    "The rewritten text should be clearer and easier to understand than the "
    "original, with improved logical flow.\n"
    "Style and Format Adaptation:\n    "
    "The rewritten text should appropriately adapt its style and format to better "
    "suit the target audience or purpose.\n"
    "Elimination of Redundancy:\n    "
    "Remove unnecessary repetition while retaining essential information.\n"
    "Improved Readability:\n    "
    "Enhance sentence structure, vocabulary, and overall readability appropriate "
    "to the target audience."
)

REWRITE_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in evaluating rewritten text quality. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Please act as an impartial judge and evaluate the quality of a rewritten text.
You should critically and accurately assess the rewritten text based on the provided rubrics.
Focus on how well the rewriting improves clarity, style, or format while preserving the original meaning.

# Rubrics
{rubrics}

# Original Text
{query}

# Rewritten Text
{answer}

# Output Requirement
```json
{
    "score": "Numerical rewrite quality score (0.0-1.0)."
    "reason": "Explanation of how the score was determined."
}
```
""",
        ),
    ],
)

REWRITE_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in evaluating rewritten text quality. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Your role is that of a professional evaluation expert. I will provide you with an original text and several rewritten versions. Your task is to select the single best rewritten version from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance. You must check each rewritten version in turn to see if it violates any rubric, and provide reasons for any violations you find. These reasons should be used as references for ranking the rewritten versions.
You may organize your reasoning as you see fit, but keep your thought process as concise as possible.

# Rubrics
{rubrics}

# Original Text
{query}

# Rewritten Versions
{answer}

# Output Requirement
```json
{
    "rank": ["The rank score of the rewritten versions in the list."]
    "reason": "The reason for the score."
}
```
""",
        ),
    ],
)


class RewriteGrader(BaseHelpfulnessGrader):
    """Rewrite: Improves text quality by correcting errors and enhancing clarity, fluency, and style."""

    _point_template = REWRITE_POINTWISE_TEMPLATE
    _list_template = REWRITE_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Template | None = None,
        mode: GraderMode = GraderMode.LISTWISE,
        rubrics: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the RewriteGrader.

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
            name="rewrite",
            mode=mode,
            model=model,
            template=template,
            rubrics=rubrics,
            description="Improves text quality by correcting errors and enhancing clarity, fluency, and style.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs: Any,
    ) -> GraderScore | GraderRank:
        """Evaluate the quality of the rewritten text based on the query.

        Evaluates rewrite responses for their ability to improve text quality by
        correcting errors and enhancing clarity, fluency, and style. The grader
        focuses on error correction, clarity improvement, and stylistic enhancement.

        Args:
            query (str): The original text to be rewritten.
            answer (str | List[str]): The rewritten text(s) to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical rewrite quality score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of rewritten texts by quality
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise rewrite grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model_name="gpt-3.5-turbo")
            >>> grader = RewriteGrader(mode=GraderMode.POINTWISE, model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="This is a bad writen sentance with alot of erors.",
            ...     answer="This is a poorly written sentence with many errors."
            ... ))
            >>> print(result.score, result.reason)
            1.0 The rewrite correctly fixes grammar and spelling errors while improving clarity.
        """
        return await super().aevaluate(query=query, answer=answer, **kwargs)
