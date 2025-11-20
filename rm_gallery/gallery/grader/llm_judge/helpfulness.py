# -*- coding: utf-8 -*-
"""
Helpfulness Grader

Evaluates the helpfulness, depth, and appropriateness of model outputs.
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
HELPFULNESS_PROMPT_EN = """
You are a professional data annotator responsible for evaluating the helpfulness, depth, and appropriateness of the model output. Your task is to score according to the following criteria:

<Scoring Criteria>
A helpful, in-depth, and appropriate answer should:
- Provide useful and relevant information directly addressing the question.
- Offer in-depth analysis, unique perspectives, or new knowledge to enhance understanding.
- Be presented in a clear, organized, and easy-to-understand manner.
- Adhere to moral, ethical, and legal standards, and be appropriate for the context.
- Not contain any irrelevant or off-topic content.

Points should be deducted in the following cases:
- Lack of useful information or failure to answer the question.
- Superficial analysis, providing no new insights.
- Disorganized, unclear language, or difficult-to-understand presentation.
- Content that violates moral, ethical, or legal standards, or is inappropriate for the context.
- Contains irrelevant or off-topic information.
</Scoring Criteria>

<Guidance>
- Carefully read the input question and model output.
- Evaluate the output according to the <Scoring Criteria>.
- The score should reflect how well the output meets the standards of helpfulness, depth, and appropriateness.
</Guidance>

<Reminder>
The goal is to evaluate the helpfulness, depth, and appropriateness of the response.
</Reminder>

{context_section}

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
    "score": <integer between 0 and 10, where 10 means extremely helpful and 0 means not helpful at all>,
    "reason": "<brief explanation for the assigned score, specifically mentioning strengths or weaknesses in helpfulness, depth, and appropriateness>"
}}

JSON:
"""

# Chinese Prompt
HELPFULNESS_PROMPT_ZH = """
你是一名专业的数据标注员，负责评估模型输出的有用性、深度和适当性。你的任务是根据以下标准进行评分：

<评分标准>
有用、深入且适当的回答应该：
- 提供有用且相关的信息，直接回答问题。
- 提供深入的分析、独特的观点或新知识以增进理解。
- 以清晰、有条理且易于理解的方式呈现。
- 遵守道德、伦理和法律标准，并适合上下文。
- 不包含任何无关或偏离主题的内容。

以下情况应扣分：
- 缺乏有用信息或未能回答问题。
- 分析肤浅，没有提供新见解。
- 组织混乱、语言不清晰或难以理解的呈现。
- 违反道德、伦理或法律标准的内容，或不适合上下文。
- 包含无关或偏离主题的信息。
</评分标准>

<指导>
- 仔细阅读输入问题和模型输出。
- 根据<评分标准>评估输出。
- 分数应反映输出在有用性、深度和适当性方面的表现。
</指导>

<提醒>
目标是评估回答的有用性、深度和适当性。
</提醒>

{context_section}

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
    "score": <0到10之间的整数，其中10表示非常有用，0表示完全没有帮助>,
    "reason": "<对所给分数的简要解释，特别提到有用性、深度和适当性方面的优势或劣势>"
}}

JSON:
"""


# Build default template from prompts
DEFAULT_HELPFULNESS_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(HELPFULNESS_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(HELPFULNESS_PROMPT_ZH),
            ),
        ],
    },
)


class HelpfulnessGrader(LLMGrader):
    """
    Helpfulness Grader

    Evaluates the helpfulness, depth, and appropriateness of model outputs.
    Measures whether responses provide useful, relevant, and well-organized
    information that directly addresses the user's needs.

    Key evaluation dimensions:
    - Usefulness: Does it provide relevant information?
    - Depth: Does it offer insightful analysis and unique perspectives?
    - Clarity: Is it well-organized and easy to understand?
    - Appropriateness: Does it adhere to ethical standards and context?
    - Relevance: Does it stay on-topic without irrelevant content?

    Attributes:
        name: Grader name
        model: OpenAIChatModel instance for evaluation
        threshold: Success threshold [0, 1] (default: 0.7)
        template: Evaluation template (default: DEFAULT_HELPFULNESS_TEMPLATE)

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",  # pragma: allowlist secret
        ...     model_name="gpt-4o",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>> grader = HelpfulnessGrader(model=api, threshold=0.7)
        >>>
        >>> result = await grader.aevaluate(
        ...     context="User needs help understanding Python decorators.",
        ...     input="What are decorators in Python?",
        ...     output="Decorators are functions that modify other functions...",
        ...     reference_output="Decorators are a Python feature for wrapping functions."
        ... )
        >>> print(f"Helpfulness score: {result.score:.2f}")
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        threshold: float = 0.7,
        template: Optional[Template] = DEFAULT_HELPFULNESS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="helpfulness",
            mode=GraderMode.POINTWISE,
            descriptio="Evaluate helpfulness, depth, and appropriateness of output",
            model=model,
            template=template,
            language=language,
        )
        self.threshold = threshold

    async def aevaluate(  # pylint: disable=redefined-builtin,unused-argument
        self,
        input: str,
        output: str,
        context: Optional[str] = None,
        reference_output: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate helpfulness of output

        Args:
            input: Input question or prompt
            output: Model output to evaluate
            context: Optional context or background information
            reference_output: Optional reference output for comparison
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized helpfulness value [0, 1]
                        where 1.0 means extremely helpful, 0.0 means not helpful

        Example:
            >>> result = await grader.aevaluate(
            ...     input="Explain machine learning",
            ...     output="Machine learning is a subset of AI that enables systems to learn from data...",
            ...     context="Audience: beginners",
            ...     reference_output="ML is a field of AI focused on learning from data."
            ... )
        """

        # Prepare context section
        context_section = ""
        if context:
            context_section = f"""<context>
{context}
</context>"""

        # Prepare reference section based on language
        reference_section = ""
        if reference_output:
            if self.language == LanguageEnum.ZH:
                reference_section = f"""如有需要，你也可以使用以下参考输出进行比较：
<reference_output>
{reference_output}
</reference_output>"""
            else:
                reference_section = f"""If available, you may also use the following reference output for comparison:
<reference_output>
{reference_output}
</reference_output>"""

        try:
            result = await self.aevaluate(
                context_section=context_section,
                input=input,
                output=output,
                reference_section=reference_section,
            )
            score = result.score
            reason = result.reason
            # Normalize score from 0-10 to 0-1
            normalized_score = score / 10.0

        except Exception as e:
            logger.error(f"Error evaluating helpfulness: {e}")
            score = 0.0
            normalized_score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "threshold": self.threshold,
            "raw_score": score,
        }

        # Generate final reason
        reason = f"Helpfulness evaluation score: {normalized_score:.4f}\n{reason}"

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = ["HelpfulnessGrader", "DEFAULT_HELPFULNESS_TEMPLATE"]
