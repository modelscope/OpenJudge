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


RUBRICS = """Logical Soundness: Ensure all reasoning steps follow logically from premises and established rules of inference, avoiding logical fallacies or invalid conclusions.
Step-by-Step Clarity: Present the reasoning process in a clear, sequential manner that allows for easy verification and understanding of how conclusions are reached.
Evidence-Based Conclusions: Base conclusions on solid evidence, clearly distinguishing between facts, assumptions, and inferences.
Comprehensiveness: Address all aspects of the problem or question, considering alternative approaches and potential edge cases where relevant."""


REASONING_SCORE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
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
""",
        ),
    ],
)

REASONING_RANK_TEMPLATE = Template(
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


class ReasoningGrader(BaseHelpfulnessGrader):
    """Reasoning: Applies logical thinking and systematic approaches to solve problems and answer questions."""

    _point_template = REASONING_SCORE_TEMPLATE
    _list_template = REASONING_RANK_TEMPLATE
    _rubrics = RUBRICS

    async def evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
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

                Each GraderScore contains:
                    - score: A numerical score assigned by the grader
                    - reason: Explanation of how the score was determined
                    - metadata: Optional additional information from the evaluation

        Example:
            >>> grader = ReasoningGrader()
            >>> result = await grader.evaluate(
            ...     query="If all roses are flowers and some flowers are red, is it true that some roses are red?",
            ...     answer="This is undetermined because we don't know which flowers are red."
            ... )
        """
        return await super().evaluate(query=query, answer=answer, **kwargs)
