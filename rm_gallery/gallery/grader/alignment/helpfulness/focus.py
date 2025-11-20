# -*- coding: utf-8 -*-
from typing import Any, List

from rm_gallery.core.grader.base import GraderMode, GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import BaseHelpfulnessGrader

RUBRICS = (
    "Direct Relevance to Core Query:\n    "
    "Prioritize completions that explicitly address the specific question, task, "
    "or scenario posed in the query without introducing tangential concepts, "
    "unnecessary details, or unrelated analysis.\n"
    "Maintaining Central Theme:\n    "
    "Ensure the response stays focused on the main topic throughout, avoiding "
    "digressions or shifts to peripheral subjects.\n"
    "Filtering Irrelevant Information:\n    "
    "Eliminate or avoid including information that does not directly contribute "
    "to answering the core query or completing the primary task.\n"
    "Adhering to Length Constraints:\n    "
    "Provide responses that are appropriately detailed without unnecessary "
    "elaboration, keeping the focus sharp and concise."
)


# Focus Score System Prompt
FOCUS_POINTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Focus Score User Prompt
FOCUS_POINTWISE_USER_PROMPT = """# Task Description
Please act as an impartial judge and evaluate the focus of a response.
You should assess how well the response maintains strict adherence to the main topic while filtering out irrelevant information.
Be as objective as possible.

# Rubrics
{rubrics}

# Query
{query}

# Answer
{answer}

# Output Requirement
```json
{
    "score": "A numerical score from 0.0 to 1.0 representing the focus quality of the response."
    "reason": "The reason for the score."
}
```
"""

# Focus Rank System Prompt
FOCUS_LISTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Focus Rank User Prompt
FOCUS_LISTWISE_USER_PROMPT = """# Task Description
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

FOCUS_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=FOCUS_POINTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=FOCUS_POINTWISE_USER_PROMPT,
        ),
    ],
)

FOCUS_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=FOCUS_LISTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=FOCUS_LISTWISE_USER_PROMPT,
        ),
    ],
)


class FocusGrader(BaseHelpfulnessGrader):
    """Focus: Maintains strict adherence to the main topic while filtering out irrelevant information."""

    _point_template = FOCUS_POINTWISE_TEMPLATE
    _list_template = FOCUS_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Template | None = None,
        mode: GraderMode = GraderMode.LISTWISE,
        rubrics: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the FocusGrader.

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
            name="focus",
            mode=mode,
            model=model,
            template=template,
            rubrics=rubrics,
            description="Maintains strict adherence to the main topic while filtering out irrelevant information.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs: Any,
    ) -> GraderScore | GraderRank:
        """Evaluate the focus quality of the response based on the query.

        Evaluates responses for their ability to maintain strict adherence to the
        main topic while filtering out irrelevant information. The grader focuses
        on direct relevance to the core query and elimination of tangential content.

        Args:
            query (str): The query to evaluate.
            answer (str | List[str]): The answer(s) to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical focus quality score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of responses by focus quality
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise focus grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model_name="gpt-3.5-turbo")
            >>> grader = FocusGrader(mode=GraderMode.POINTWISE, model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="Explain the process of photosynthesis",
            ...     answer="Photosynthesis is the process by which plants convert light energy into chemical energy..."
            ... ))
            >>> print(result.score, result.reason)
            0.95 The response stays focused on explaining photosynthesis without digressing.
        """
        return await super().aevaluate(query=query, answer=answer, **kwargs)
