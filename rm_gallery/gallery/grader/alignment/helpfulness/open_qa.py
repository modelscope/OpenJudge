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


RUBRICS = """Comprehensive Coverage: Address all aspects of the question thoroughly, providing a well-rounded response that considers multiple angles and relevant factors without omitting significant points.
Source Reliability and Citation: Base information on credible, up-to-date sources and clearly attribute facts or claims to their origins when appropriate, particularly for contentious or specialized topics.
Nuanced Understanding: Demonstrate sophisticated comprehension of the topic by acknowledging complexities, uncertainties, or trade-offs rather than presenting overly simplified or one-sided views.
Clarity and Organization: Present information in a clear, logically structured format that enhances readability and facilitates understanding, using appropriate headings, paragraphs, and transitions."""


OPEN_QA_SCORE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Please act as an impartial judge and evaluate the quality of an open QA response.
You should assess the response based on comprehensive coverage, source reliability, nuanced understanding, and clarity.
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

OPEN_QA_RANK_TEMPLATE = Template(
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


class OpenQAGrader(BaseHelpfulnessGrader):
    """Open QA Grader for evaluating open-ended question answering performance."""

    _point_template = OPEN_QA_SCORE_TEMPLATE
    _list_template = OPEN_QA_RANK_TEMPLATE
    _rubrics = RUBRICS

    def __init__(self, model: ChatModelBase | dict, template: Template | None = None, mode: GraderMode = GraderMode.LISTWISE, **kwargs):
        """Initialize the SafetyGrader."""
        super().__init__(
            name="Open QA",
            mode=mode,
            model=model,
            template=template,
            description="Evaluate open QA responses for their ability to provide comprehensive, well-researched answers to open-ended questions.",
            **kwargs,
        )
    
    async def a_evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
    ) -> GraderScore | GraderRank:
        """Evaluate the open QA response based on the query.

        Evaluates open QA responses for their ability to provide comprehensive,
        well-researched answers to open-ended questions. The grader focuses on
        comprehensive coverage, source reliability, and nuanced understanding.

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
            >>> grader = OpenQAGrader()
            >>> result = await grader.evaluate(
            ...     query="What are the potential impacts of climate change on agriculture?",
            ...     answer="Climate change can affect agriculture through..."
            ... )
        """
        return await super().a_evaluate(query=query, answer=answer, **kwargs)
