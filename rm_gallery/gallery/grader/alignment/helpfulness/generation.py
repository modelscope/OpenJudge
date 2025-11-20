# -*- coding: utf-8 -*-
from typing import Any, List

from rm_gallery.core.grader.base import GraderMode, GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import BaseHelpfulnessGrader

RUBRICS = (
    "Adherence to Instructional Specificity:\n    "
    "Prioritize addressing all explicit requirements (e.g., format, content scope, "
    "tone) with precise alignment to ensure completeness and fidelity to the "
    "task's intent.\n"
    "Depth and Originality in Content:\n    "
    "Deliver nuanced, actionable insights or creative elements that exceed generic "
    "responses through specific examples, contextual relevance, and imaginative "
    "elaboration.\n"
    "Structural Coherence and Logical Flow:\n    "
    "Maintain organized progression (e.g., clear hierarchy, thematic sequencing) "
    "to enhance readability while avoiding contradictions or deviations from "
    "established frameworks."
)

# Generation Score System Prompt
GENERATION_POINTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Generation Score User Prompt
GENERATION_POINTWISE_USER_PROMPT = """# Task Description
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
"""

# Generation Rank System Prompt
GENERATION_LISTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Generation Rank User Prompt
GENERATION_LISTWISE_USER_PROMPT = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate answers. Your task is to select the single best answer \
from the candidates.
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
"""

GENERATION_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=GENERATION_POINTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=GENERATION_POINTWISE_USER_PROMPT,
        ),
    ],
)

GENERATION_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=GENERATION_LISTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=GENERATION_LISTWISE_USER_PROMPT,
        ),
    ],
)


class GenerationGrader(BaseHelpfulnessGrader):
    """Generation: Creates high-quality, instruction-following content across diverse formats and topics."""

    _point_template = GENERATION_POINTWISE_TEMPLATE
    _list_template = GENERATION_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Template | None = None,
        mode: GraderMode = GraderMode.LISTWISE,
        rubrics: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the GenerationGrader.

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
            name="generation",
            mode=mode,
            model=model,
            template=template,
            rubrics=rubrics,
            description="Creates high-quality, instruction-following content across diverse formats and topics.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs: Any,
    ) -> GraderScore | GraderRank:
        """Evaluate the quality of the generated content based on the query.

        Evaluates content generation for adherence to instructions, content depth,
        and structural coherence. The grader focuses on following specified formats,
        providing comprehensive content, and maintaining logical organization.

        Args:
            query (str): The content generation instruction or query.
            answer (str | List[str]): The generated content(s) to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical generation quality score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of content by quality
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise generation grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model_name="gpt-3.5-turbo")
            >>> grader = GenerationGrader(mode=GraderMode.POINTWISE, model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="Write a short story about a robot learning to paint",
            ...     answer="Once upon a time, there was a robot named ART-1 who discovered..."
            ... ))
            >>> print(result.score, result.reason)
            0.85 The story creatively addresses the prompt with good narrative structure.
        """
        return await super().aevaluate(query=query, answer=answer, **kwargs)
