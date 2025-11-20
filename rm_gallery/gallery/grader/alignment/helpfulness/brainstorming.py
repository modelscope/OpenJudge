# -*- coding: utf-8 -*-
from typing import Any, List

from rm_gallery.core.grader.base import GraderMode, GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import BaseHelpfulnessGrader

RUBRICS = (
    "Creative Relevance and Contextual Alignment:\n    "
    "Prioritize completions that balance novel ideas with direct ties to the "
    "scenario's core context, ensuring ideas are both imaginative and grounded "
    "in the specific problem or theme.\n"
    "Practical Feasibility and Actionable Detail:\n    "
    "Favor completions that offer concrete, implementable solutions or insights, "
    "avoiding abstract or overly speculative suggestions that lack real-world "
    "applicability.\n"
    "Structural Coherence and Logical Organization:\n    "
    "Prefer completions that present ideas in a clear, logically sequenced "
    "framework (e.g., categorized sections, step-by-step processes) to enhance "
    "readability and development potential."
)


# Brainstorming Score System Prompt
BRAINSTORMING_POINTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Brainstorming Score User Prompt
BRAINSTORMING_POINTWISE_USER_PROMPT = """# Task Description
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
"""

# Brainstorming Rank System Prompt
BRAINSTORMING_LISTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Brainstorming Rank User Prompt
BRAINSTORMING_LISTWISE_USER_PROMPT = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate answers. Your task is to select the single best answer \
from the candidates.

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

BRAINSTORMING_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=BRAINSTORMING_POINTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=BRAINSTORMING_POINTWISE_USER_PROMPT,
        ),
    ],
)

BRAINSTORMING_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=BRAINSTORMING_LISTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=BRAINSTORMING_LISTWISE_USER_PROMPT,
        ),
    ],
)


class BrainstormingGrader(BaseHelpfulnessGrader):
    """Brainstorming: Generates creative ideas and suggestions to address user challenges."""

    _point_template = BRAINSTORMING_POINTWISE_TEMPLATE
    _list_template = BRAINSTORMING_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Template | None = None,
        mode: GraderMode = GraderMode.LISTWISE,
        rubrics: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the BrainstormingGrader.

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
            name="brainstorming",
            mode=mode,
            model=model,
            template=template,
            rubrics=rubrics,
            description="Generates creative ideas and suggestions to address user challenges.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs: Any,
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
            >>> result = await grader.aevaluate(
            ...     query="Give me ideas for a birthday gift for my 10-year-old",
            ...     answer="Here are some ideas: 1) Art supplies kit, 2) Science experiment set, 3) Board game, 4) Book series"
            ... )
        """
        return await super().aevaluate(query=query, answer=answer, **kwargs)
