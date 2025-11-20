# -*- coding: utf-8 -*-
from typing import Any, List

from rm_gallery.core.grader.base import GraderMode, GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import BaseHelpfulnessGrader

RUBRICS = (
    "Factual Accuracy:\n    "
    "Prioritize completely accurate information without any factual errors or "
    "hallucinations. Every statement should be verifiable against authoritative "
    "sources.\n"
    "Precision and Conciseness:\n    "
    "Provide responses that directly and precisely answer the question without "
    "unnecessary elaboration or ambiguity.\n"
    "Comprehensiveness within Scope:\n    "
    "Include all relevant information required to fully answer the question, but "
    "avoid including tangential or excessive details.\n"
    "Logical Coherence:\n    "
    "Structure responses in a clear, logical manner that enhances understanding "
    "and maintains focus on the core question."
)


# Closed QA Score System Prompt
CLOSED_QA_POINTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Closed QA Score User Prompt
CLOSED_QA_POINTWISE_USER_PROMPT = """# Task Description
Please act as an impartial judge and evaluate the quality of a closed QA response.
You should assess the response based on factual accuracy, precision, and completeness.
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
    "score": "A numerical score from 0.0 to 1.0 representing the quality of the QA response."
    "reason": "The reason for the score."
}
```
"""

# Closed QA Rank System Prompt
CLOSED_QA_LISTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Closed QA Rank User Prompt
CLOSED_QA_LISTWISE_USER_PROMPT = """# Task Description
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

CLOSED_QA_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=CLOSED_QA_POINTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=CLOSED_QA_POINTWISE_USER_PROMPT,
        ),
    ],
)

CLOSED_QA_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=CLOSED_QA_LISTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=CLOSED_QA_LISTWISE_USER_PROMPT,
        ),
    ],
)


class ClosedQAGrader(BaseHelpfulnessGrader):
    """Closed QA: Provides precise, fact-based answers to questions with definitive correct responses."""

    _point_template = CLOSED_QA_POINTWISE_TEMPLATE
    _list_template = CLOSED_QA_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Template | None = None,
        mode: GraderMode = GraderMode.LISTWISE,
        rubrics: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the ClosedQAGrader.

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
            name="closed_qa",
            mode=mode,
            model=model,
            template=template,
            rubrics=rubrics,
            description="Provides precise, fact-based answers to questions with definitive correct responses.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs: Any,
    ) -> GraderScore | GraderRank:
        """Evaluate the closed QA response based on the query.

        Evaluates closed QA responses for their ability to provide precise,
        fact-based answers to questions with definitive correct responses.
        The grader emphasizes precision in fact-based responses, avoids
        hallucinations, and focuses on explicit requirements.

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
                    - score (float): Numerical helpfulness score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of responses by quality
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise closed QA grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model_name="gpt-3.5-turbo")
            >>> grader = ClosedQAGrader(mode=GraderMode.POINTWISE, model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="What is the capital of France?",
            ...     answer="The capital of France is Paris."
            ... ))
            >>> print(result.score, result.reason)
            1.0 The response correctly identifies Paris as the capital of France.
        """
        return await super().aevaluate(query=query, answer=answer, **kwargs)
