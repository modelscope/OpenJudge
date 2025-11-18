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


RUBRICS = """Character and Contextual Fidelity: Prioritize maintaining the assigned character's persona, motivations, and world-building consistency while strictly adhering to the scenario's established rules, terminology, and thematic boundaries to ensure immersive authenticity.
Consistent Character Voice: Ensure the character's speech patterns, vocabulary, and mannerisms are consistent throughout the interaction, reflecting their background, personality, and current emotional state.
Contextual Appropriateness: Responses should be appropriate to the scenario's setting, time period, and cultural context, avoiding anachronisms or out-of-place references that break immersion.
Engagement Quality: The interaction should be engaging and maintain the role-playing scenario's momentum, with the character responding naturally to prompts while adding depth to the narrative."""


ROLE_PLAYING_SCORE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Please act as an impartial judge and evaluate the quality of a role playing response.
You should assess the response based on character consistency, contextual appropriateness, and engagement quality.
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
    "score": "A numerical score from 0.0 to 1.0 representing the quality of the role playing response."
    "reason": "The reason for the score."
}
```
""",
        ),
    ],
)

ROLE_PLAYING_RANK_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content="You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words.",
        ),
        ChatMessage(
            role="user",
            content="""# Task Description
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. These rubrics are ordered from highest to lowest importance.These rubrics can serve as supplementary knowledge for your judgment, though not necessarily required. First, think independently. Use these rubrics only when unsure about certain answers, selecting specific ones based on the questions and answers.

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


class RolePlayingGrader(BaseHelpfulnessGrader):
    """Role Playing: Embodies distinct characters or personas to create immersive, contextually appropriate interactions."""

    _point_template = ROLE_PLAYING_SCORE_TEMPLATE
    _list_template = ROLE_PLAYING_RANK_TEMPLATE
    _rubrics = RUBRICS

    async def evaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs,
    ) -> GraderScore | GraderRank:
        """Evaluate the role playing quality of the response based on the query.

        Evaluates role playing responses for their ability to embody distinct
        characters or personas to create immersive, contextually appropriate
        interactions. The grader focuses on character consistency, contextual
        appropriateness, and engagement quality.

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
            >>> grader = RolePlayingGrader()
            >>> result = await grader.evaluate(
            ...     query="Greetings, wise wizard!",
            ...     answer="Ah, greetings young traveler! What brings you to my tower this eve?"
            ... )
        """
        return await super().evaluate(query=query, answer=answer, **kwargs)
