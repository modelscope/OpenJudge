# -*- coding: utf-8 -*-
"""
Reference Adherence Grader

Evaluates whether model outputs align with and properly utilize reference materials,
including factual consistency, style matching, and appropriate citation/grounding.
"""

import textwrap
from typing import Any, Optional

from loguru import logger

from rm_gallery.core.grader.base import LLMGrader
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.grader import GraderMode, GraderScore
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import LanguageEnum, Template

# pylint: disable=line-too-long

# English Prompt
REFERENCE_ADHERENCE_PROMPT_EN = """
You are a professional data annotator responsible for evaluating whether the model output properly adheres to the provided reference material. Your task is to score according to the following criteria:

<Scoring Criteria>
A response that perfectly adheres to the reference should:
- Maintain factual consistency with all information in the reference.
- Include key points from the reference that are relevant to the input question.
- Match the style, tone, and format of the reference when appropriate.
- Not contradict, misrepresent, or distort information from the reference.
- Properly ground claims in the reference material without fabricating details.
- Use reference information accurately without taking it out of context.
- Balance reference adherence with responding appropriately to the specific input.

Points should be deducted for:
- Factual contradictions with the reference material.
- Omitting critical information present in the reference.
- Misrepresenting or distorting reference information.
- Adding claims not supported by the reference when grounding is required.
- Significantly departing from reference style/format when matching is expected.
- Taking reference information out of context.
- Over-relying on reference when original synthesis is needed.
</Scoring Criteria>

<Guidance>
- Carefully read the reference material to understand its key facts, style, and content.
- Compare each claim in the output against the reference material.
- Check if the output appropriately balances using reference info vs. answering the specific question.
- Evaluate whether the output matches the reference's level of detail and confidence.
- Consider if the output properly attributes or grounds information in the reference.
- Assess whether the output adds appropriate synthesis or only paraphrases.
</Guidance>

<Reminder>
The goal is to evaluate reference adherence, not general quality. A well-written response that contradicts the reference should score low. A simple response that accurately reflects and properly uses the reference should score high. Consider both accuracy and appropriate application of the reference material.
</Reminder>

{reference_type_section}

Evaluate the following:

<reference_material>
{reference}
</reference_material>

<input>
{input}
</input>

<output>
{output}
</output>

# Output Instructions
Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 0 and 10, where 10 means perfect reference adherence and 0 means complete failure to adhere to reference>,
    "reason": "<brief explanation for the assigned score, specifically mentioning how the output aligns with or deviates from the reference material>"
}}

JSON:
"""

# Chinese Prompt
REFERENCE_ADHERENCE_PROMPT_ZH = """
你是一名专业的数据标注员，负责评估模型输出是否正确遵循提供的参考材料。你的任务是根据以下标准进行评分：

<评分标准>
完美遵循参考的回答应该：
- 与参考中的所有信息保持事实一致性。
- 包含参考中与输入问题相关的关键点。
- 在适当时与参考的风格、语气和格式相匹配。
- 不与参考中的信息矛盾、歪曲或扭曲。
- 在参考材料中正确地支撑声明，而不捏造细节。
- 准确使用参考信息，不脱离上下文。
- 在遵循参考和适当回答特定输入之间取得平衡。

以下情况应扣分：
- 与参考材料的事实矛盾。
- 遗漏参考中存在的关键信息。
- 歪曲或扭曲参考信息。
- 在需要支撑时添加参考不支持的声明。
- 在预期匹配时明显偏离参考风格/格式。
- 脱离上下文使用参考信息。
- 在需要原创综合时过度依赖参考。
</评分标准>

<指导>
- 仔细阅读参考材料以理解其关键事实、风格和内容。
- 将输出中的每个声明与参考材料进行比较。
- 检查输出是否适当地平衡使用参考信息和回答特定问题。
- 评估输出是否与参考的细节水平和可信度相匹配。
- 考虑输出是否正确地在参考中归因或支撑信息。
- 评估输出是否添加了适当的综合还是仅仅转述。
</指导>

<提醒>
目标是评估参考遵循性，而不是一般质量。一个写得很好但与参考矛盾的回答应该得分低。一个简单但准确反映并正确使用参考的回答应该得分高。同时考虑准确性和参考材料的适当应用。
</提醒>

{reference_type_section}

评估以下内容：

<reference_material>
{reference}
</reference_material>

<input>
{input}
</input>

<output>
{output}
</output>

# 输出指令
请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0到10之间的整数，其中10表示完美遵循参考，0表示完全未能遵循参考>,
    "reason": "<对所给分数的简要解释，特别提到输出如何与参考材料一致或偏离>"
}}

JSON:
"""

# Build default template from prompts
DEFAULT_REFERENCE_ADHERENCE_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(REFERENCE_ADHERENCE_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(REFERENCE_ADHERENCE_PROMPT_ZH),
            ),
        ],
    },
)


class ReferenceAdherenceGrader(LLMGrader):
    """
    Reference Adherence Grader

    Evaluates how well model outputs adhere to provided reference materials, including
    factual alignment, style consistency, proper grounding, and appropriate use of
    reference information.

    Key evaluation dimensions:
    - Factual Consistency: Does output align with facts in reference?
    - Information Coverage: Are key points from reference appropriately included?
    - Style/Format Matching: Does output match reference style when required?
    - Proper Attribution: Are references used appropriately without misrepresentation?
    - Grounding Quality: Is output properly grounded in reference material?

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
        >>> grader = ReferenceAdherenceGrader(model=api, threshold=0.7)
        >>>
        >>> result = await grader.aevaluate(
        ...     reference="The product was launched in Q1 2023 in Europe.",
        ...     input="When and where was the product launched?",
        ...     output="The product was launched in early 2023 in European markets."
        ... )
        >>> print(f"Reference adherence score: {result.score:.2f}")
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        threshold: float = 0.7,
        template: Optional[Template] = DEFAULT_REFERENCE_ADHERENCE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="reference_adherence",
            mode=GraderMode.POINTWISE,
            description="Evaluate whether output adheres to provided reference materials",
            model=model,
            template=template,
            language=language,
        )
        self.threshold = threshold

    async def aevaluate(  # pylint: disable=redefined-builtin,unused-argument
        self,
        reference: str,
        input: str,
        output: str,
        reference_type: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate reference adherence in output

        Args:
            reference: Reference material to adhere to
            input: Original user input or question
            output: Model output to evaluate
            reference_type: Optional description of how reference should be used
                          (e.g., "style guide", "factual source", "example format")
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized reference adherence value [0, 1]
                        where 1.0 means perfect adherence, 0.0 means complete failure

        Example:
            >>> result = await grader.aevaluate(
            ...     reference="The capital of France is Paris, with a population of 2.2M.",
            ...     input="What is the capital of France?",
            ...     output="Paris is the capital of France.",
            ...     reference_type="factual source"
            ... )
        """

        # Prepare reference type section based on language
        reference_type_section = ""
        if reference_type:
            if self.language == LanguageEnum.ZH:
                reference_type_section = f"""<reference_type>
注意：参考材料应被视为：{reference_type}
请相应地评估遵循性。
</reference_type>
"""
            else:
                reference_type_section = f"""<reference_type>
Note: The reference material should be treated as: {reference_type}
Evaluate adherence accordingly.
</reference_type>
"""

        try:
            result = await super().aevaluate(
                reference=reference,
                input=input,
                output=output,
                reference_type_section=reference_type_section,
            )
            score = result.score
            reason = result.reason
            # Normalize score from 0-10 to 0-1
            normalized_score = score / 10.0

        except Exception as e:
            logger.error(f"Error evaluating reference adherence: {e}")
            score = 0.0
            normalized_score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "threshold": self.threshold,
            "raw_score": score,
            "reference_type": reference_type,
        }

        # Generate final reason
        reason = f"Reference adherence score: {normalized_score:.4f}\n{reason}"

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = ["ReferenceAdherenceGrader", "DEFAULT_REFERENCE_ADHERENCE_TEMPLATE"]
