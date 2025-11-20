# -*- coding: utf-8 -*-
# https://arxiv.org/pdf/2410.21545


from typing import Any

from rm_gallery.core.grader.base import GraderMode, GraderRank, LLMGrader
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Chat, Template
from rm_gallery.core.utils.utils import _json_loads_with_repair

# Criteria Generation System Prompt
CRITERIA_GENERATION_SYSTEM_PROMPT = (
    "You are an impartial judge tasked with generating rubrics for evaluating "
    "responses provided by AI assistants to an instruction."
)

# Criteria Generation User Prompt
CRITERIA_GENERATION_USER_PROMPT = """# Task Description
- Your job is to identify important rubrics, along with detailed descriptions, \
that a human would use to objectively evaluate the quality of the response based \
on the given instruction.
- The rubrics should ensure that responses accurately fulfill the requirements \
of the instruction.
- The rubrics should be designed to ensure that responses are honest, helpful, \
and harmless (do not contain offensive content).
- The descriptions of the rubrics should be framed as chain-of-thought detailed \
questions that assess whether the response meets the user's instruction.
- The length of the response should only be considered a rubric if it is \
specified in the instruction.\n
# Input
## Instruction
{instruction}

# Output Requirements
```json
{
    "rubrics": [
        "rubric 1",
        ...
    ]
}
```"""

# Relative Evaluation System Prompt
RELATIVE_EVALUATION_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses "
    "provided by two AI assistants to the user instruction shown below."
)

# Relative Evaluation User Prompt
RELATIVE_EVALUATION_USER_PROMPT = """Task Description
- Please act as an impartial judge and evaluate the quality of the responses \
provided by two AI assistants to the user instruction shown below. You should \
choose the assistant that follows the user's instructions and answers the user's \
instruction better.
- Your evaluation should consider the provided rubrics.
- Provide detailed reasons assessing the quality of the responses based on each \
rubric individually. Clearly specify which assistant performed better for each rubric.
- After assessing all rubrics, provide a final verdict based on the overall \
performance of the assistants.
- Don't be influenced by the order in which the responses are presented. Do not \
favor certain names of the assistants. Be as objective as possible.\n
# Input
## Instruction
{instruction}

## Rubrics
{rubrics}

## Completion
{completion}

# Output Requirements
```json
{{
    "rank": "The rank of each completions",
    "reason": "The reason for the rank."
}}
```"""

CriteriaGenerationTemplate = Template(
    messages=[
        ChatMessage(
            role="system",
            content=CRITERIA_GENERATION_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=CRITERIA_GENERATION_USER_PROMPT,
        ),
    ],
)

RelativeEvaluationTemplate = Template(
    messages=[
        ChatMessage(
            role="system",
            content=RELATIVE_EVALUATION_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=RELATIVE_EVALUATION_USER_PROMPT,
        ),
    ],
)


class Cramo(LLMGrader):
    """Cramo grader for evaluation of AI assistant responses.

    Implements the CRAMO (Criteria-Aware Relative Evaluation of Model Outputs) approach
    for evaluating responses from AI assistants. This grader first generates
    evaluation criteria based on the instruction, then evaluates the response
    using those criteria.

    The grader uses two-stage evaluation:
    1. Criteria Generation: Generates rubrics for evaluating responses based on the instruction
    2. Response Evaluation: Evaluates the response based on the generated rubrics
    """

    def __init__(self, model: ChatModelBase | dict):
        """Initialize a Cramo grader."""
        super().__init__(
            name="cramo_eval",
            mode=GraderMode.LISTWISE,
            template=RelativeEvaluationTemplate,
            model=model,
        )

    async def aevaluate(
        self,
        instruction: str,
        completion: str,
        **kwargs: Any,
    ) -> GraderRank:
        """Evaluate and rank responses using criteria-aware relative evaluation.

        This method implements the CRAMO approach which first automatically generates
        evaluation criteria (rubrics) based on the instruction, then evaluates and ranks
        the responses based on these criteria.

        Args:
            instruction: The instruction to evaluate the response against.
            completion: The response to evaluate.
            **kwargs: Additional keyword arguments.

        Returns:
            GraderRank: A grader rank with ranking and reason.

        Example:
            >>> from rm_gallery.core.model.dashscope_llm import DashScopeLLM
            >>> model_config = {"model_name": "qwen-plus"}
            >>> grader = Cramo(name="cramo_eval", model=model_config)
            >>> result = await grader.aevaluate(
            ...     instruction="Explain the process of photosynthesis",
            ...     completion="Photosynthesis is a process used by plants..."
            ... )
            >>> print(result.rank)
            >>> print(result.reason)
        """
        # Generate rubrics using CriteriaGenerationTemplate
        chat = Chat(
            template=CriteriaGenerationTemplate,
            model=self.model,
        )
        rubrics = chat(
            char_output=_json_loads_with_repair,
            instruction=instruction,
        ).metadata.get("rubrics", [])

        # Evaluate the completion using the generated rubrics
        result = await super().aevaluate(
            instruction=instruction,
            completion=completion,
            rubrics="\n".join(rubrics),
            **kwargs,
        )
        return result
