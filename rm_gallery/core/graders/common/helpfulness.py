# -*- coding: utf-8 -*-
"""
Helpfulness Grader

Evaluates the helpfulness, depth, and appropriateness of model response.
"""

import textwrap
from typing import Optional

from loguru import logger

from rm_gallery.core.graders.base_grader import GraderMode, GraderScore
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
HELPFULNESS_PROMPT_EN = """
You are a professional data annotator responsible for evaluating the helpfulness, depth, and appropriateness of the model response. Your task is to score according to the following criteria:

<Scoring Criteria>
A helpful, in-depth, and appropriate response should:
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
- Carefully read the query and model response.
- Evaluate the response according to the <Scoring Criteria>.
- The score should reflect how well the response meets the standards of helpfulness, depth, and appropriateness.
</Guidance>

<Reminder>
The goal is to evaluate the helpfulness, depth, and appropriateness of the response.
</Reminder>

{context_section}

<query>
{query}
</query>

<response>
{response}
</response>

{reference_section}

# Output Instructions
Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 1 and 5, where 5 means extremely helpful and 1 means not helpful at all>,
    "reason": "<brief explanation for the assigned score, specifically mentioning strengths or weaknesses in helpfulness, depth, and appropriateness>"
}}

Scoring Scale:
- 5: Exceptionally helpful, comprehensive, and well-organized
- 4: Helpful and relevant with good depth
- 3: Provides some useful information but lacks depth or clarity
- 2: Limited helpfulness, mostly superficial or unclear
- 1: Not helpful, irrelevant, or confusing

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

<query>
{query}
</query>

<response>
{response}
</response>

{reference_section}

# 输出指令
请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <1到5之间的整数，其中5表示非常有用，1表示完全没有帮助>,
    "reason": "<对所给分数的简要解释，特别提到有用性、深度和适当性方面的优势或劣势>"
}}

评分标尺：
- 5: 极其有用、全面且组织良好
- 4: 有用且相关，有良好的深度
- 3: 提供了一些有用信息，但缺乏深度或清晰度
- 2: 帮助有限，大多肤浅或不清楚
- 1: 没有帮助、无关或令人困惑

JSON:
"""


# Build default template from prompts
DEFAULT_HELPFULNESS_TEMPLATE = PromptTemplate(
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

    Purpose:
        Evaluates how helpful, useful, and well-structured model outputs are in addressing
        user needs. Goes beyond correctness to assess practical value and user satisfaction.

    What it evaluates:
        - Usefulness: Provides relevant, actionable information
        - Depth: Offers insightful analysis beyond surface-level answers
        - Clarity: Well-organized, easy to understand structure
        - Completeness: Addresses all aspects of the query
        - Appropriateness: Matches context and user expertise level
        - Relevance: Stays on-topic without unnecessary tangents

    When to use:
        - Evaluating chatbot and assistant responses for user satisfaction
        - Ranking multiple candidate responses by utility
        - Training reward models for RLHF (Reinforcement Learning from Human Feedback)
        - Quality assurance for customer-facing AI systems
        - A/B testing different response generation strategies

    Scoring:
        - 5: Exceptionally helpful, comprehensive, and well-organized
        - 4: Helpful and relevant with good depth
        - 3: Provides some useful information but lacks depth or clarity
        - 2: Limited helpfulness, mostly superficial or unclear
        - 1: Unhelpful, irrelevant, or confusing

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [0, 1] to pass (default: 0.7)
        template: Custom evaluation template (default: DEFAULT_HELPFULNESS_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore object with:
            - score: Score [1, 5] where 5 = maximally helpful, 1 = not helpful
            - reason: Explanation of strengths and weaknesses
            - metadata: Threshold and evaluation details

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.core.alignment.helpfulness import HelpfulnessGrader
        >>>
        >>> # Initialize grader
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-max")
        >>> grader = HelpfulnessGrader(model=model, threshold=0.7)
        >>>
        >>> # Helpful output
        >>> result = await grader.aevaluate(
        ...     input="What are Python decorators?",
        ...     output="Decorators are functions that modify other functions. They use @syntax..."
        ... )
        >>> print(result.score)  # 5 - very helpful with good explanation
        >>>
        >>> # Unhelpful output
        >>> result = await grader.aevaluate(
        ...     query="What are decorators in Python?",
        ...     response="Decorators are functions that modify other functions...",
        ...     context="User needs help understanding Python decorators.",
        ...     reference_response="Decorators are a Python feature for wrapping functions."
        ... )
        >>> print(result.score)  # 2 - too vague and lacks depth
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 0.7,
        template: Optional[PromptTemplate] = DEFAULT_HELPFULNESS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """
        Initialize HelpfulnessGrader

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Success threshold [0, 1] (default: 0.7)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_HELPFULNESS_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
        """
        super().__init__(
            name="helpfulness",
            mode=GraderMode.POINTWISE,
            description="Evaluate helpfulness, depth, and appropriateness of response",
            model=model,
            template=template,
            language=language,
        )
        self.threshold = threshold

    async def aevaluate(
        self,
        query: str,
        response: str,
        context: str = "",
        reference_response: str = "",
    ) -> GraderScore:
        """
        Evaluate helpfulness of response

        Args:
            query: Input query or prompt
            response: Model response to evaluate
            context: Context or background information. Defaults to empty string.
            reference_response: Reference response for comparison. Defaults to empty string.

        Returns:
            GraderScore: Score with helpfulness value [1, 5]
                        where 5 means extremely helpful, 1 means not helpful

        Example:
            >>> result = await grader.aevaluate(
            ...     query="Explain machine learning",
            ...     answer="Machine learning is a subset of AI that enables systems to learn from data...",
            ...     context="Audience: beginners",
            ...     reference_response="ML is a field of AI focused on learning from data."
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
        if reference_response:
            if self.language == LanguageEnum.ZH:
                reference_section = f"""如有需要，你也可以使用以下参考输出进行比较：
<reference_response>
{reference_response}
</reference_response>"""
            else:
                reference_section = f"""If available, you may also use the following reference response for comparison:
<reference_response>
{reference_response}
</reference_response>"""

        try:
            result = await super().aevaluate(
                query=query,
                response=response,
                context_section=context_section,
                reference_section=reference_section,
            )
            score = result.score
            reason = result.reason

        except Exception as e:
            logger.error(f"Error evaluating helpfulness: {e}")
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "threshold": self.threshold,
        }

        # Generate final reason
        reason = f"Helpfulness evaluation score: {score}\n{reason}"

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata=metadata,
        )


__all__ = ["HelpfulnessGrader", "DEFAULT_HELPFULNESS_TEMPLATE"]
