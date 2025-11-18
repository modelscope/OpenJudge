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

RUBRICS = """Adherence to Instructional Specificity: Prioritize addressing all explicit requirements (e.g., format, content scope, tone) with precise alignment to ensure completeness and fidelity to the task's intent.
Depth and Originality in Content: Deliver nuanced, actionable insights or creative elements that exceed generic responses through specific examples, contextual relevance, and imaginative elaboration.
Structural Coherence and Logical Flow: Maintain organized progression (e.g., clear hierarchy, thematic sequencing) to enhance readability while avoiding contradictions or deviations from established frameworks."""


GENERATION_SCORE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Please act as an impartial judge and evaluate the quality of a generated content.
You should assess the content based on adherence to instructions, content depth, and structural coherence.
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
    "score": "A numerical score from 0.0 to 1.0 representing the quality of the generated content."
    "reason": "The reason for the score."
}
```
""",
        ),
    ],
)

GENERATION_RANK_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance.These rubrics can serve as supplementary knowledge for your judgment. If you find any of the rubrics helpful for the current problem, feel free to use them as supplements.

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


class GenerationGrader(BaseHelpfulnessGrader):
    """Generation: Produces rich, creative content tailored to specific formats and audiences."""

    _point_template = GENERATION_SCORE_TEMPLATE
    _list_template = GENERATION_RANK_TEMPLATE

    async def evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
    ) -> GraderScore | GraderRank:
        """Evaluate the generation quality of the response based on the query.

        Evaluates generation responses for their ability to produce rich,
        creative content tailored to specific formats and audiences. The grader
        focuses on content richness and depth, creative expression, and audience
        appropriateness.

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
            >>> grader = GenerationGrader()
            >>> result = await grader.evaluate(
            ...     query="Write a short story about a robot learning to paint",
            ...     answer="Once upon a time, there was a robot named ART-1 who discovered the joy of painting..."
            ... )
        """
        return await super().evaluate(query=query, answer=answer, **kwargs)
