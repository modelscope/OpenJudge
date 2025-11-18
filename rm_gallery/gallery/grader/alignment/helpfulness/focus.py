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


RUBRICS = """Direct Relevance to Core Query: Prioritize completions that explicitly address the specific question, task, or scenario posed in the query without introducing tangential concepts, unnecessary details, or unrelated analysis.
Maintaining Central Theme: Ensure the response stays focused on the main topic throughout, avoiding digressions or shifts to peripheral subjects.
Filtering Irrelevant Information: Eliminate or avoid including information that does not directly contribute to answering the core query or completing the primary task.
Adhering to Length Constraints: Provide responses that are appropriately detailed without unnecessary elaboration, keeping the focus sharp and concise."""


FOCUS_SCORE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
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
""",
        ),
    ],
)

FOCUS_RANK_TEMPLATE = Template(
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


class FocusGrader(BaseHelpfulnessGrader):
    """Focus: Maintains strict adherence to the main topic while filtering out irrelevant information."""

    _point_template = FOCUS_SCORE_TEMPLATE
    _list_template = FOCUS_RANK_TEMPLATE
    _rubrics = RUBRICS

    async def evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
    ) -> GraderScore | GraderRank:
        """Evaluate the focus of the response based on the query.

        Evaluates responses for their ability to maintain strict adherence to
        the main topic while filtering out irrelevant information. The grader
        focuses on maintaining the central theme, filtering irrelevant
        information, and adhering to length constraints.

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
            >>> grader = FocusGrader()
            >>> result = await grader.evaluate(
            ...     query="Explain photosynthesis",
            ...     answer="Photosynthesis is the process by which plants convert light energy into chemical energy."
            ... )
        """
        return await super().evaluate(query=query, answer=answer, **kwargs)
