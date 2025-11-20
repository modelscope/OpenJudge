# -*- coding: utf-8 -*-
"""
Instruction Adherence Grader

Evaluates whether model outputs correctly follow the given instructions, including
content requirements, format constraints, style guidelines, and other specified criteria.
"""

import textwrap
from typing import Any, Optional

from loguru import logger

from rm_gallery.core.grader.base import Grader
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.grader import GraderMode, GraderScore
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import LanguageEnum, RequiredField, Template

# pylint: disable=line-too-long

# English Prompt
INSTRUCTION_ADHERENCE_PROMPT_EN = """
You are a professional data annotator responsible for evaluating whether the model output follows the given instructions. Your task is to score according to the following criteria:

<Scoring Criteria>
A response that perfectly follows instructions should:
- Address all required topics, questions, or tasks mentioned in the instruction.
- Follow the specified format exactly (e.g., JSON, bullet points, numbered list, essay format).
- Adhere to all constraints specified (e.g., word/sentence count, tone, style, vocabulary level).
- Include all required elements (e.g., introduction, conclusion, specific sections).
- Avoid adding information not requested by the instruction.
- Meet quality requirements if specified (e.g., "detailed", "concise", "professional").

Points should be deducted for:
- Missing required content or topics.
- Incorrect format or structure.
- Violating specified constraints (e.g., too long/short, wrong tone).
- Omitting required elements.
- Adding excessive unrequested information.
- Misinterpreting the instruction's intent.
</Scoring Criteria>

<Guidance>
- Carefully parse the instruction to identify ALL requirements and constraints.
- Break down complex instructions into individual requirements.
- Check each requirement systematically against the output.
- Consider both explicit requirements (stated clearly) and implicit requirements (strongly implied).
- Evaluate format, structure, and style requirements separately from content requirements.
- Be strict: partial fulfillment should result in lower scores.
</Guidance>

<Reminder>
The goal is to evaluate instruction-following capability, not content quality per se. A response can be well-written but score low if it doesn't follow instructions. Conversely, a simple response that perfectly follows all instructions should score high.
</Reminder>

Evaluate the following:

<instruction>
{instruction}
</instruction>

{input_section}

<output>
{output}
</output>

# Output Instructions
Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 0 and 10, where 10 means perfect instruction adherence and 0 means complete failure to follow instructions>,
    "reasoning": "<brief explanation for the assigned score, specifically mentioning which instruction requirements were met or violated>"
}}

JSON:
"""

# Chinese Prompt
INSTRUCTION_ADHERENCE_PROMPT_ZH = """
你是一名专业的数据标注员，负责评估模型输出是否遵循给定的指令。你的任务是根据以下标准进行评分：

<评分标准>
完美遵循指令的回答应该：
- 涵盖指令中提到的所有必需主题、问题或任务。
- 完全遵循指定的格式（例如，JSON、项目符号、编号列表、论文格式）。
- 遵守所有指定的约束（例如，字数/句子数、语气、风格、词汇水平）。
- 包含所有必需的元素（例如，引言、结论、特定部分）。
- 避免添加指令未要求的信息。
- 满足指定的质量要求（例如，"详细"、"简洁"、"专业"）。

以下情况应扣分：
- 缺少必需的内容或主题。
- 格式或结构不正确。
- 违反指定的约束（例如，太长/太短、错误的语气）。
- 遗漏必需的元素。
- 添加过多未要求的信息。
- 误解指令的意图。
</评分标准>

<指导>
- 仔细解析指令以识别所有要求和约束。
- 将复杂的指令分解为单个要求。
- 系统地根据输出检查每个要求。
- 考虑明确的要求（清楚陈述的）和隐含的要求（强烈暗示的）。
- 将格式、结构和风格要求与内容要求分开评估。
- 严格要求：部分满足应导致较低的分数。
</指导>

<提醒>
目标是评估指令遵循能力，而不是内容质量本身。一个回答可能写得很好，但如果不遵循指令就会得分低。相反，一个简单但完美遵循所有指令的回答应该得到高分。
</提醒>

评估以下内容：

<instruction>
{instruction}
</instruction>

{input_section}

<output>
{output}
</output>

# 输出指令
请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0到10之间的整数，其中10表示完美遵循指令，0表示完全未能遵循指令>,
    "reasoning": "<对所给分数的简要解释，特别提到满足或违反了哪些指令要求>"
}}

JSON:
"""


class InstructionAdherenceGrader(Grader):
    """
    Instruction Adherence Grader

    Evaluates how well model outputs follow the given instructions across multiple dimensions
    including content, format, style, constraints, and completeness.

    Key evaluation dimensions:
    - Content Relevance: Does output address all required topics/questions?
    - Format Compliance: Does output follow specified format (e.g., JSON, bullet points, essay)?
    - Constraint Adherence: Are all constraints satisfied (e.g., length, tone, style)?
    - Completeness: Are all instruction requirements fulfilled?
    - Precision: Does output avoid adding unrequested information?

    Attributes:
        name: Grader name
        model: OpenAIChatModel instance for evaluation
        threshold: Success threshold [0, 1] (default: 0.7)

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",  # pragma: allowlist secret
        ...     model_name="gpt-4o",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>> grader = InstructionAdherenceGrader(model=api, threshold=0.7)
        >>>
        >>> result = await grader.aevaluate(
        ...     instruction="Write a 3-sentence summary in formal tone about climate change.",
        ...     output="Climate change is a big problem. It's getting hotter. We need to act now!",
        ...     input="Summarize the climate situation."
        ... )
        >>> print(f"Instruction adherence score: {result.score:.2f}")
    """

    def __init__(
        self,
        model: OpenAIChatModel,
        name: str = "instruction_adherence",
        threshold: float = 0.7,
        description: str = "Evaluate whether output follows the given instructions",
        template: Optional[Template] = None,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name=name,
            mode=GraderMode.POINTWISE,
            description=description,
            required_fields=[
                RequiredField(
                    name="instruction",
                    type="str",
                    position="data",
                    description="The instruction or prompt given to the model",
                ),
                RequiredField(
                    name="output",
                    type="str",
                    position="data",
                    description="Model output to evaluate",
                ),
                RequiredField(
                    name="input",
                    type="str",
                    position="data",
                    description="Original user input or question (optional)",
                    required=False,
                ),
            ],
        )
        self.model = model
        self.threshold = threshold
        self.template = (
            template if template is not None else DEFAULT_INSTRUCTION_ADHERENCE_TEMPLATE
        )
        self.language = language
        self.evaluation_cost = 0.0

    async def aevaluate(  # pylint: disable=redefined-builtin,unused-argument
        self,
        instruction: str,
        output: str,
        input: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate instruction adherence in output

        Args:
            instruction: The instruction or prompt given to the model
            output: Model output to evaluate
            input: Optional original user input or question
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized instruction adherence value [0, 1]
                        where 1.0 means perfect adherence, 0.0 means complete failure

        Example:
            >>> result = await grader.aevaluate(
            ...     instruction="Write exactly 3 bullet points about AI safety.",
            ...     output="• AI safety is important\\n• We need alignment research\\n• Testing is crucial",
            ... )
        """
        self.evaluation_cost = 0.0

        # Prepare input section
        input_section = ""
        if input:
            input_section = f"""<input>
{input}
</input>"""

        # Get template and format prompt
        messages = self.template.get(self.language)
        prompt = messages[0].content.format(
            instruction=instruction,
            output=output,
            input_section=input_section,
        )

        try:
            # Call LLM for evaluation
            response = await self.model(
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response from text content
            import json

            text_content = "".join(
                [block.text for block in response.content if hasattr(block, "text")],
            )

            # Parse JSON response
            result_data = json.loads(text_content.strip())
            score = float(result_data.get("score", 0))
            reasoning = result_data.get("reasoning", "No reasoning provided")

            # Normalize score from 0-10 to 0-1
            normalized_score = score / 10.0

        except Exception as e:
            logger.error(f"Error evaluating instruction adherence: {e}")
            score = 0.0
            normalized_score = 0.0
            reasoning = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "evaluation_cost": self.evaluation_cost,
            "threshold": self.threshold,
            "raw_score": score,
        }

        # Generate final reason
        reason = f"Instruction adherence score: {normalized_score:.4f}\n{reasoning}"

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


# Build default template from prompts
DEFAULT_INSTRUCTION_ADHERENCE_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(INSTRUCTION_ADHERENCE_PROMPT_EN),
                name="User",
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(INSTRUCTION_ADHERENCE_PROMPT_ZH),
                name="User",
            ),
        ],
    },
    required_fields=[
        RequiredField(
            name="instruction",
            type="str",
            position="data",
            description="The instruction given",
        ),
        RequiredField(
            name="output",
            type="str",
            position="data",
            description="Output to evaluate",
        ),
        RequiredField(
            name="input_section",
            type="str",
            position="data",
            description="Optional input section",
        ),
    ],
)


__all__ = ["InstructionAdherenceGrader", "DEFAULT_INSTRUCTION_ADHERENCE_TEMPLATE"]
