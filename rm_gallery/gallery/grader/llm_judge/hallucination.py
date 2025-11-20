# -*- coding: utf-8 -*-
"""
Hallucination Grader

Evaluates whether model outputs contain hallucinations (fabricated information not
supported by the input context).
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
HALLUCINATION_PROMPT_EN = """
You are a professional data annotator responsible for evaluating whether the model output contains hallucinations. Your task is to score according to the following criteria:

<Scoring Criteria>
A hallucination-free answer should:
- Contain only verifiable facts directly supported by the input context.
- Not make unsupported claims or assumptions.
- Not add speculative or imagined details.
- Be completely accurate regarding dates, numbers, and specific details.
- Appropriately indicate uncertainty when information is incomplete.
</Scoring Criteria>

<Guidance>
- Thoroughly read the input context.
- Identify all claims made in the output.
- Cross-check each claim with the input context.
- Note any unsupported or contradictory information.
- Consider the severity and number of hallucinations.
</Guidance>

<Reminder>
Focus only on factual accuracy and support from the input context. Do not consider style, grammar, or presentation when scoring. A short but factual response should score higher than a longer response containing unsupported claims.
</Reminder>

Use the following context to help you evaluate whether there are hallucinations in the output:
<context>
{context}
</context>

<input>
{input}
</input>

<output>
{output}
</output>

{reference_section}

# Output Instructions
Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 0 and 10, where 10 means no hallucinations and 0 means severe hallucinations>,
    "reason": "<brief explanation for the assigned score, specifically mentioning any hallucinations found or confirming factual accuracy>"
}}

JSON:
"""

# Chinese Prompt
HALLUCINATION_PROMPT_ZH = """
你是一名专业的数据标注员，负责评估模型输出是否包含幻觉（虚构信息）。你的任务是根据以下标准进行评分：

<评分标准>
无幻觉的回答应该：
- 仅包含输入上下文直接支持的可验证事实。
- 不做出无依据的声明或假设。
- 不添加推测性或想象的细节。
- 在日期、数字和具体细节方面完全准确。
- 在信息不完整时适当地表示不确定性。
</评分标准>

<指导>
- 仔细阅读输入的上下文。
- 识别输出中的所有声明。
- 将每个声明与输入上下文进行交叉核对。
- 注意任何无依据或矛盾的信息。
- 考虑幻觉的严重程度和数量。
</指导>

<提醒>
仅关注事实准确性和输入上下文的支持。评分时不要考虑风格、语法或呈现方式。简短但真实的回答应该比包含无依据声明的较长回答得分更高。
</提醒>

使用以下上下文帮助你评估输出中是否存在幻觉：
<context>
{context}
</context>

<input>
{input}
</input>

<output>
{output}
</output>

{reference_section}

# 输出指令
请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0到10之间的整数，其中10表示无幻觉，0表示严重幻觉>,
    "reason": "<对所给分数的简要解释，特别提到发现的任何幻觉或确认事实准确性>"
}}

JSON:
"""

# Build default template from prompts
DEFAULT_HALLUCINATION_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(HALLUCINATION_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(HALLUCINATION_PROMPT_ZH),
            ),
        ],
    },
)


class HallucinationGrader(LLMGrader):
    """
    Hallucination Grader

    Evaluates whether model outputs contain hallucinations - information that is not
    supported by the provided context or makes unsupported claims.

    Key evaluation dimensions:
    - Factual grounding: Is information supported by input context?
    - Claim verification: Are all claims backed by provided evidence?
    - Speculation detection: Does output avoid adding imagined details?
    - Accuracy: Are dates, numbers, and specific details correct?

    Attributes:
        name: Grader name
        model: OpenAIChatModel instance for evaluation
        threshold: Success threshold [0, 1] (default: 0.7)
        template: Evaluation template (default: DEFAULT_HALLUCINATION_TEMPLATE)
        language: Language for evaluation prompts (default: LanguageEnum.EN)

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.core.schema.template import LanguageEnum
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",  # pragma: allowlist secret
        ...     model_name="gpt-4o",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>>
        >>> # English grader
        >>> grader_en = HallucinationGrader(model=api, threshold=0.7, language=LanguageEnum.EN)
        >>>
        >>> # Chinese grader
        >>> grader_zh = HallucinationGrader(model=api, threshold=0.7, language=LanguageEnum.ZH)
        >>>
        >>> result = await grader_en.aevaluate(
        ...     context="The company was founded in 2020 in San Francisco.",
        ...     input="When was the company founded?",
        ...     output="The company was founded in 2020 in San Francisco with 100 employees.",
        ...     reference_output="The company was founded in 2020."
        ... )
        >>> print(f"Hallucination score: {result.score:.2f}")
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        threshold: float = 0.7,
        template: Optional[Template] = DEFAULT_HALLUCINATION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="hallucination",
            mode=GraderMode.POINTWISE,
            description="Evaluate whether output contains hallucinations",
            model=model,
            template=template,
            language=language,
        )
        self.threshold = threshold
        self.template = (
            template if template is not None else DEFAULT_HALLUCINATION_TEMPLATE
        )

    async def aevaluate(  # pylint: disable=redefined-builtin,unused-argument
        self,
        context: str,
        input: str,
        output: str,
        reference_output: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate hallucination in output

        Args:
            context: Context information to verify against
            input: Input question or prompt
            output: Model output to evaluate
            reference_output: Optional reference output for comparison
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized hallucination value [0, 1]
                        where 1.0 means no hallucinations, 0.0 means severe hallucinations

        Example:
            >>> result = await grader.aevaluate(
            ...     context="The product launched in 2023.",
            ...     input="When did the product launch?",
            ...     output="The product launched in 2023 with great success.",
            ...     reference_output="The product launched in 2023."
            ... )
        """

        # Prepare reference section based on language
        reference_section = ""
        if reference_output:
            if self.language == LanguageEnum.ZH:
                reference_section = f"""如有需要，你也可以使用以下参考输出来帮助识别回答中的幻觉：
<reference_output>
{reference_output}
</reference_output>"""
            else:
                reference_section = f"""If available, you may also use the following reference outputs to help you identify hallucinations in the response:
<reference_output>
{reference_output}
</reference_output>"""

        try:
            result = await super().aevaluate(
                context=context,
                input=input,
                output=output,
                reference_section=reference_section,
            )
            score = result.score
            reason = result.reason
            # Normalize score from 0-10 to 0-1
            normalized_score = score / 10.0

        except Exception as e:
            logger.error(f"Error evaluating hallucination: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "threshold": self.threshold,
            "raw_score": score,
        }

        # Generate final reason
        reason = f"Hallucination evaluation score: {normalized_score:.4f}\n{reason}"

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = ["HallucinationGrader", "DEFAULT_HALLUCINATION_TEMPLATE"]
