# -*- coding: utf-8 -*-
"""
Instruction Adherence Grader

Evaluates whether model response correctly follow the given instructions, including
content requirements, format constraints, style guidelines, and other specified criteria.
"""

import textwrap
from typing import Optional

from loguru import logger

from rm_gallery.core.grader.base import LLMGrader
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.grader import (
    GraderError,
    GraderMode,
    GraderRank,
    GraderScore,
)
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import LanguageEnum, Template

# pylint: disable=line-too-long

# English Prompt
INSTRUCTION_ADHERENCE_PROMPT_EN = """
You are a professional data annotator responsible for evaluating whether the model response follows the given instructions. Your task is to score according to the following criteria:

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
- Check each requirement systematically against the response.
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

<response>
{response}
</response>

# Output Instructions
Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 0 and 10, where 10 means perfect instruction adherence and 0 means complete failure to follow instructions>,
    "reason": "<brief explanation for the assigned score, specifically mentioning which instruction requirements were met or violated>"
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

<response>
{response}
</response>

# 输出指令
请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0到10之间的整数，其中10表示完美遵循指令，0表示完全未能遵循指令>,
    "reason": "<对所给分数的简要解释，特别提到满足或违反了哪些指令要求>"
}}

JSON:
"""


# Build default template from prompts
DEFAULT_INSTRUCTION_ADHERENCE_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(INSTRUCTION_ADHERENCE_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(INSTRUCTION_ADHERENCE_PROMPT_ZH),
            ),
        ],
    },
)


class InstructionAdherenceGrader(LLMGrader):
    """
    Instruction Adherence Grader
    
    Purpose:
        Evaluates how precisely model outputs follow given instructions across content,
        format, style, and constraints. Essential for ensuring AI systems execute tasks
        as specified without deviation.
    
    What it evaluates:
        - Content Relevance: Addresses all required topics and questions
        - Format Compliance: Follows specified format (JSON, bullet points, essay, etc.)
        - Constraint Adherence: Satisfies length, tone, style requirements
        - Completeness: Fulfills all instruction requirements
        - Precision: Avoids adding unrequested information
        - Structural Accuracy: Maintains requested organization
    
    When to use:
        - Structured output generation (JSON, XML, specific formats)
        - Task completion verification in agent systems
        - Evaluating instruction-tuned models
        - Quality control for templated content generation
        - Testing AI assistants' ability to follow complex instructions
    
    Scoring:
        - 10: Perfect adherence to all instruction aspects
        - 7-9: Follows most instructions with minor deviations
        - 4-6: Partial adherence, misses some requirements
        - 0-3: Significant instruction violations or misunderstanding
    
    Args:
        model: ChatModelBase instance or dict config for OpenAIChatModel
        threshold: Minimum score [0, 1] to pass (default: 0.7)
        template: Custom evaluation template (default: DEFAULT_INSTRUCTION_ADHERENCE_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)
    
    Returns:
        GraderScore object with:
            - score: Normalized score [0, 1] where 1.0 = perfect adherence
            - reason: Detailed analysis of adherence and violations
            - metadata: Raw score, threshold, and evaluation details
    
    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.gallery.grader.llm_judge import InstructionAdherenceGrader
        >>>
        >>> # Initialize grader
        >>> model = OpenAIChatModel(api_key="sk-...", model="gpt-4o")
        >>> grader = InstructionAdherenceGrader(model=model, threshold=0.7)
        >>>
        >>> # Good adherence
        >>> result = await grader.aevaluate(
        ...     instruction="Write exactly 3 sentences in formal academic tone.",
        ...     output="Climate change poses serious risks. Research shows rising temperatures. Action is urgently needed."
        ... )
        >>> print(result.score)  # 1.0 - follows all requirements
        >>>
        >>> # Poor adherence
        >>> result = await grader.aevaluate(
        ...     instruction="Write a 3-sentence summary in formal tone about climate change.",
        ...     response="Climate change is a big problem. It's getting hotter. We need to act now!",
        ...     query="Summarize the climate situation."
        ... )
        >>> print(result.score)  # 0.3 - informal tone, poor structure
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        threshold: float = 0.7,
        template: Optional[Template] = DEFAULT_INSTRUCTION_ADHERENCE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """
        Initialize InstructionAdherenceGrader

        Args:
            model: ChatModelBase instance or dict config for OpenAIChatModel
            threshold: Success threshold [0, 1] (default: 0.7)
            template: Template for evaluation prompts (default: DEFAULT_INSTRUCTION_ADHERENCE_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
        """
        super().__init__(
            name="instruction_adherence",
            mode=GraderMode.POINTWISE,
            description="Evaluate whether response follows the given instructions",
            model=model,
            template=template,
            language=language,
        )
        self.threshold = threshold

    async def aevaluate(
        self,
        instruction: str,
        response: str,
        query: Optional[str] = None,
    ) -> GraderScore | GraderRank | GraderError:
        """
        Evaluate instruction adherence in response

        Args:
            instruction: The instruction or prompt given to the model
            response: Model response to evaluate
            query: Optional original user query or question

        Returns:
            GraderScore: Score with normalized instruction adherence value [0, 1]
                        where 1.0 means perfect adherence, 0.0 means complete failure

        Example:
            >>> result = await grader.aevaluate(
            ...     instruction="Write exactly 3 bullet points about AI safety.",
            ...     response="• AI safety is important\\n• We need alignment research\\n• Testing is crucial",
            ... )
        """
        return await super().aevaluate(
            instruction=instruction,
            response=response,
            query=query,
        )

    async def _aevaluate(
        self,
        instruction: str,
        response: str,
        query: Optional[str] = None,
    ) -> GraderScore:
        # Prepare input section
        input_section = ""
        if query:
            input_section = f"""<query>
{query}
</query>"""

        try:
            result = await super()._aevaluate(
                instruction=instruction,
                response=response,
                input_section=input_section,
            )
            score = result.score
            reason = result.reason
            # Normalize score from 0-10 to 0-1
            normalized_score = score / 10.0

        except Exception as e:
            logger.error(f"Error evaluating instruction adherence: {e}")
            score = 0.0
            normalized_score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "threshold": self.threshold,
            "raw_score": score,
        }

        # Generate final reason
        reason = f"Instruction adherence score: {normalized_score:.4f}\n{reason}"

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = ["InstructionAdherenceGrader", "DEFAULT_INSTRUCTION_ADHERENCE_TEMPLATE"]
