# -*- coding: utf-8 -*-
from typing import List
from rm_gallery.core.grader.base import (
    GraderMode,
    LLMGrader,
    GraderScore,
    GraderRank,
)
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import RequiredField, Template
from rm_gallery.gallery.grader.alignment.helpfulness import (
    BaseHelpfulnessGrader,
)

RUBRICS = """Factual Accuracy: Prioritize completely accurate information without any factual errors or hallucinations. Every statement should be verifiable against authoritative sources.
Precision and Conciseness: Provide responses that directly and precisely answer the question without unnecessary elaboration or ambiguity.
Comprehensiveness within Scope: Include all relevant information required to fully answer the question, but avoid including tangential or excessive details.
Logical Coherence: Structure responses in a clear, logical manner that enhances understanding and maintains focus on the core question.
"""


CLOSED_QA_SCORE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
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
""",
        ),
    ],
)

CLOSED_QA_RANK_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.

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


class ClosedQAGrader(BaseHelpfulnessGrader):
    """Closed QA: Provides precise, fact-based answers to questions with definitive correct responses."""

    _point_template = CLOSED_QA_SCORE_TEMPLATE
    _list_template = CLOSED_QA_RANK_TEMPLATE
    _rubrics = RUBRICS

    async def evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
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
                    - rank (List[int]): Ranking of answers by quality
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise closed QA grader
            >>> grader = ClosedQAGrader(mode=GraderMode.POINTWISE)
            >>> result = await grader.evaluate(
            ...     query="What is the capital of France?",
            ...     answer="The capital of France is Paris."
            ... )
            >>> print(result.score, result.reason)

            >>> # Example for listwise closed QA grader
            >>> ranking_grader = ClosedQAGrader(mode=GraderMode.LISTWISE)
            >>> result = await ranking_grader.evaluate(
            ...     query="What is the capital of France?",
            ...     answer=["The capital of France is Paris.", "The capital of France is London."]
            ... )
            >>> print(result.rank, result.reason)
        """
        return await super().evaluate(query=query, answer=answer, **kwargs)
