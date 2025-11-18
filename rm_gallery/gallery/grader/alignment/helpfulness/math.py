# -*- coding: utf-8 -*-
from typing import List
from rm_gallery.core.grader.base import GraderMode, GraderScore, GraderRank
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import (
    BaseHelpfulnessGrader,
)


RUBRICS = """Mathematical Accuracy: Ensure all calculations, formula applications, and logical steps are error-free, as even minor inaccuracies (e.g., arithmetic mistakes, misapplied rules) invalidate results despite otherwise correct methodologies.
Logical Coherence: Present solutions with clear, sequential reasoning where each step follows logically from the previous one, enabling easy verification and understanding of the problem-solving process.
Notational Clarity: Use appropriate mathematical notation, symbols, and formatting consistently throughout the solution to avoid ambiguity and enhance readability.
Problem-Specific Precision: Address all aspects of the given problem completely, providing exact solutions in the required format without omitting critical details or providing extraneous information."""


MATH_SCORE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Please act as an impartial judge and evaluate the mathematical correctness of a response.
You should assess the response based on mathematical accuracy, logical coherence, and notational clarity.
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
    "score": "A numerical score from 0.0 to 1.0 representing the mathematical correctness of the response."
    "reason": "The reason for the score."
}
```
""",
        ),
    ],
)

MATH_RANK_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance. You must check each candidate answer in turn to see if it violates any rubric, and provide reasons for any violations you find. These reasons should be used as references for ranking the answers.
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
""",
        ),
    ],
)


class MathGrader(BaseHelpfulnessGrader):
    """Math: Solves mathematical problems accurately with clear logical reasoning and proper notation."""

    _point_template = MATH_SCORE_TEMPLATE
    _list_template = MATH_RANK_TEMPLATE
    _rubrics = RUBRICS

    async def evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
    ) -> GraderScore | GraderRank:
        """Evaluate the mathematical correctness of the response based on the query.

        Evaluates math responses for their ability to solve mathematical problems
        accurately with clear logical reasoning and proper notation. The grader
        focuses on mathematical accuracy, logical coherence, and notational clarity.

        Args:
            query (str): The query to evaluate.
            answer (str | List[str]): The answer(s) to evaluate. For POINTWISE mode,
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
            >>> grader = MathGrader()
            >>> result = await grader.evaluate(
            ...     query="Solve for x: 2x + 5 = 15",
            ...     answer="2x + 5 = 15; 2x = 10; x = 5"
            ... )
        """
        return await super().evaluate(query=query, answer=answer, **kwargs)
