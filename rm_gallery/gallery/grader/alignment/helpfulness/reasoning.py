# -*- coding: utf-8 -*-
from typing import Any, List

from rm_gallery.core.grader.base import GraderMode, GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import BaseHelpfulnessGrader

RUBRICS = (
    "Logical Soundness:\n    "
    "Ensure all reasoning steps follow logically from premises and established "
    "rules of inference, avoiding logical fallacies or invalid conclusions.\n"
    "Step-by-Step Clarity:\n    "
    "Present the reasoning process in a clear, sequential manner that allows for "
    "easy verification and understanding of how conclusions are reached.\n"
    "Evidence-Based Conclusions:\n    "
    "Base conclusions on solid evidence, clearly distinguishing between facts, "
    "assumptions, and inferences.\n"
    "Comprehensiveness:\n    "
    "Address all aspects of the problem or question, considering alternative "
    "approaches and potential edge cases where relevant."
)


# Reasoning Score System Prompt
REASONING_POINTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Reasoning Score User Prompt
REASONING_POINTWISE_USER_PROMPT = """# Task Description
Please act as an impartial judge and evaluate the quality of a reasoning response.
You should assess the response based on logical soundness, step-by-step clarity, and evidence-based conclusions.
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
    "score": "A numerical score from 0.0 to 1.0 representing the quality of the reasoning."
    "reason": "The reason for the score."
}
```
"""

# Reasoning Rank System Prompt
REASONING_LISTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Reasoning Rank User Prompt
REASONING_LISTWISE_USER_PROMPT = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate answers. Your task is to select the single best answer \
from the candidates.

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

REASONING_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=REASONING_POINTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=REASONING_POINTWISE_USER_PROMPT,
        ),
    ],
)

REASONING_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=REASONING_LISTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=REASONING_LISTWISE_USER_PROMPT,
        ),
    ],
)


class ReasoningGrader(BaseHelpfulnessGrader):
    """Reasoning: Applies logical thinking and systematic approaches to solve problems and answer questions."""

    _point_template = REASONING_POINTWISE_TEMPLATE
    _list_template = REASONING_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Template | None = None,
        mode: GraderMode = GraderMode.LISTWISE,
        rubrics: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the ReasoningGrader.

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
            name="reasoning",
            mode=mode,
            model=model,
            template=template,
            rubrics=rubrics,
            description="Applies logical thinking and systematic approaches to solve problems and answer questions.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs: Any,
    ) -> GraderScore | GraderRank:
        """Evaluate the reasoning quality of the response based on the query.

        Evaluates reasoning responses for their ability to apply logical thinking
        and systematic approaches to solve problems and answer questions. The
        grader focuses on logical soundness, step-by-step clarity, and
        evidence-based conclusions.

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
                    - score (float): Numerical reasoning quality score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of responses by reasoning quality
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise reasoning grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model_name="gpt-3.5-turbo")
            >>> grader = ReasoningGrader(mode=GraderMode.POINTWISE, model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="If all roses are flowers and some flowers are red, is it true that some roses are red?",
            ...     answer="This is undetermined because we don't know which flowers are red."
            ... ))
            >>> print(result.score, result.reason)
            0.9 The response demonstrates logical reasoning by correctly identifying the undetermined nature of the statement.
        """
        return await super().aevaluate(query=query, answer=answer, **kwargs)
