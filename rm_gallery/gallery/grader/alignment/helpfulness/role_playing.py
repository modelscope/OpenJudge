# -*- coding: utf-8 -*-
from typing import Any, List

from rm_gallery.core.grader.base import GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.grader import GraderMode
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import BaseHelpfulnessGrader

RUBRICS = (
    "Character and Contextual Fidelity:\n    "
    "Prioritize maintaining the assigned character's persona, motivations, and "
    "world-building consistency while strictly adhering to the scenario's "
    "established rules, terminology, and thematic boundaries to ensure immersive "
    "authenticity.\n"
    "Consistent Character Voice:\n    "
    "Ensure the character's speech patterns, vocabulary, and mannerisms are "
    "consistent throughout the interaction, reflecting their background, "
    "personality, and current emotional state.\n"
    "Contextual Appropriateness:\n    "
    "Responses should be appropriate to the scenario's setting, time period, and "
    "cultural context, avoiding anachronisms or out-of-place references that "
    "break immersion.\n"
    "Engagement Quality:\n    "
    "The interaction should be engaging and maintain the role-playing scenario's "
    "momentum, with the character responding naturally to prompts while adding "
    "depth to the narrative."
)


# Role Playing Score System Prompt
ROLE_PLAYING_POINTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Role Playing Score User Prompt
ROLE_PLAYING_POINTWISE_USER_PROMPT = """# Task Description
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
"""

ROLE_PLAYING_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=ROLE_PLAYING_POINTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=ROLE_PLAYING_POINTWISE_USER_PROMPT,
        ),
    ],
)

ROLE_PLAYING_LISTWISE_SYSTEM_PROMPT = """You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."""

ROLE_PLAYING_LISTWISE_USER_PROMPT = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate answers. Your task is to select the single best answer \
from the candidates.
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
"""

ROLE_PLAYING_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=ROLE_PLAYING_LISTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=ROLE_PLAYING_LISTWISE_USER_PROMPT,
        ),
    ],
)


class RolePlayingGrader(BaseHelpfulnessGrader):
    """Role Playing: Engages in immersive roleplay scenarios with consistent character portrayal and contextual awareness."""

    _point_template = ROLE_PLAYING_POINTWISE_TEMPLATE
    _list_template = ROLE_PLAYING_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Template | None = None,
        mode: GraderMode = GraderMode.POINTWISE,
        rubrics: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the RolePlayingGrader.

        Args:
            model: The language model used for evaluation. Can be either a ChatModelBase
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: The template for generating prompts. If None, a default template will be used.
            mode: The grader mode. Defaults to POINTWISE.
            rubrics: Custom rubrics for evaluation. If None, default rubrics will be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name="role_playing",
            mode=mode,
            model=model,
            template=template,
            rubrics=rubrics,
            description="Engages in immersive roleplay scenarios with consistent character portrayal and contextual awareness.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs: Any,
    ) -> GraderScore | GraderRank:
        """Evaluate the quality of the role playing response based on the query.

        Evaluates role playing responses for their ability to engage in immersive
        roleplay scenarios with consistent character portrayal and contextual awareness.
        The grader focuses on character consistency, contextual awareness, and
        immersive engagement.

        Args:
            query (str): The role playing scenario or prompt.
            answer (str | List[str]): The role playing response(s) to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical role playing quality score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of responses by quality
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise role playing grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model_name="gpt-3.5-turbo")
            >>> grader = RolePlayingGrader(mode=GraderMode.POINTWISE, model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="You are a medieval blacksmith. A knight approaches requesting a sword.",
            ...     answer="Ah, good sir knight! I shall forge you a blade worthy of your noble quest. What specifications would you desire in your weapon?"
            ... ))
            >>> print(result.score, result.reason)
            0.9 The response maintains character consistency and appropriately engages with the scenario.
        """
        return await super().aevaluate(query=query, answer=answer, **kwargs)
