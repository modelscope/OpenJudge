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

RUBRICS = """Creative Relevance and Contextual Alignment: Prioritize completions that balance novel ideas with direct ties to the scenario's core context, ensuring ideas are both imaginative and grounded in the specific problem or theme.
Practical Feasibility and Actionable Detail: Favor completions that offer concrete, implementable solutions or insights, avoiding abstract or overly speculative suggestions that lack real-world applicability.
Structural Coherence and Logical Organization: Prefer completions that present ideas in a clear, logically sequenced framework (e.g., categorized sections, step-by-step processes) to enhance readability and development potential.
"""


BRAINSTORMING_SCORE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Please act as an impartial judge and evaluate the creativity and usefulness of brainstorming ideas.
You should critically and accurately assess the ideas with the key rubrics that are presented from most important to least important.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
Do not allow the length of the responses to influence your evaluation.
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
    "score": "A numerical score from 0.0 to 1.0 representing the quality of ideas."
    "reason": "The reason for the score."
}
```
""",
        ),
    ],
)

BRAINSTORMING_RANK_TEMPLATE = Template(
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


class BrainstormingGrader(BaseHelpfulnessGrader):
    """Brainstorming: Generates creative ideas and suggestions to address user challenges."""

    _point_template = BRAINSTORMING_SCORE_TEMPLATE
    _list_template = BRAINSTORMING_RANK_TEMPLATE
    _rubrics = RUBRICS

    async def evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
    ) -> GraderScore | GraderRank:
        """Evaluate the brainstorming response based on the query.

        Evaluates brainstorming responses for their ability to generate creative
        ideas and suggestions that address user challenges. The grader focuses on
        idea diversity, relevance to the core question, and practical implementation
        guidance.

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
            >>> grader = BrainstormingGrader()
            >>> result = await grader.evaluate(
            ...     query="Give me ideas for a birthday gift for my 10-year-old",
            ...     answer="Here are some ideas: 1) Art supplies kit, 2) Science experiment set, 3) Board game, 4) Book series"
            ... )
        """
        return await super().evaluate(query=query, answer=answer, **kwargs)
