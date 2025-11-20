# -*- coding: utf-8 -*-
from typing import Any, List

from rm_gallery.core.grader.base import GraderMode, GraderRank, GraderScore
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template
from rm_gallery.gallery.grader.alignment.helpfulness import BaseHelpfulnessGrader

RUBRICS = (
    "Comprehensive Coverage:\n    "
    "Address all aspects of the question thoroughly, providing a well-rounded "
    "response that considers multiple angles and relevant factors without "
    "omitting significant points.\n"
    "Source Reliability and Citation:\n    "
    "Base information on credible, up-to-date sources and clearly attribute "
    "facts or claims to their origins when appropriate, particularly for "
    "contentious or specialized topics.\n"
    "Nuanced Understanding:\n    "
    "Demonstrate sophisticated comprehension of the topic by acknowledging "
    "complexities, uncertainties, or trade-offs rather than presenting overly "
    "simplified or one-sided views.\n"
    "Clarity and Organization:\n    "
    "Present information in a clear, logically structured format that enhances "
    "readability and facilitates understanding, using appropriate headings, "
    "paragraphs, and transitions."
)


# Open QA Score System Prompt
OPEN_QA_POINTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Open QA Score User Prompt
OPEN_QA_POINTWISE_USER_PROMPT = """# Task Description
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
"""

# Open QA Rank System Prompt
OPEN_QA_LISTWISE_SYSTEM_PROMPT = "You are a helpful assistant skilled in reward evaluation. Please make reward judgments based on the given prompt words."

# Open QA Rank User Prompt
OPEN_QA_LISTWISE_USER_PROMPT = """# Task Description
Your role is that of a professional evaluation expert. I will provide you with a \
question and several candidate answers. Your task is to select the single best answer \
from the candidates.
I will also provide you with a set of rubrics, listed under the heading #Rubrics. \
These rubrics are ordered from highest to lowest importance. You must check each \
candidate answer in turn to see if it violates any rubric, and provide reasons for \
any violations you find. These reasons should be used as references for ranking \
the answers.
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
"""

OPEN_QA_POINTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=OPEN_QA_POINTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=OPEN_QA_POINTWISE_USER_PROMPT,
        ),
    ],
)

OPEN_QA_LISTWISE_TEMPLATE = Template(
    messages=[
        ChatMessage(
            role="system",
            content=OPEN_QA_LISTWISE_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=OPEN_QA_LISTWISE_USER_PROMPT,
        ),
    ],
)


class OpenQAGrader(BaseHelpfulnessGrader):
    """Open QA: Provides comprehensive, nuanced answers to open-ended questions without definitive correct responses."""

    _point_template = OPEN_QA_POINTWISE_TEMPLATE
    _list_template = OPEN_QA_LISTWISE_TEMPLATE
    _rubrics = RUBRICS

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Template | None = None,
        mode: GraderMode = GraderMode.LISTWISE,
        rubrics: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the OpenQAGrader.

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
            name="open_qa",
            mode=mode,
            model=model,
            template=template,
            rubrics=rubrics,
            description="Provides comprehensive, nuanced answers to open-ended questions without definitive correct responses.",
            **kwargs,
        )

    async def aevaluate(
        self,
        query: str,
        answer: str | List[str],
        **kwargs: Any,
    ) -> GraderScore | GraderRank:
        """Evaluate the quality of the open QA response based on the query.

        Evaluates open QA responses for their comprehensiveness, source reliability,
        nuanced understanding, and clarity. The grader focuses on addressing all
        aspects of the question and providing well-rounded responses.

        Args:
            query (str): The open-ended question to evaluate.
            answer (str | List[str]): The response(s) to evaluate. For POINTWISE mode,
                this should be a single string. For LISTWISE mode, this should be
                a list of strings.
            **kwargs: Additional arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical open QA quality score (0.0-1.0)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of responses by quality
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise open QA grader
            >>> import asyncio
            >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
            >>> from rm_gallery.core.grader.base import GraderMode
            >>> model = OpenAIChatModel(model_name="gpt-3.5-turbo")
            >>> grader = OpenQAGrader(mode=GraderMode.POINTWISE, model=model)
            >>> result = asyncio.run(grader.aevaluate(
            ...     query="What are the potential impacts of climate change on global agriculture?",
            ...     answer="Climate change can impact global agriculture in several ways..."
            ... ))
            >>> print(result.score, result.reason)
            0.9 The response comprehensively addresses multiple aspects of climate change impacts on agriculture.
        """
        return await super().aevaluate(query=query, answer=answer, **kwargs)
