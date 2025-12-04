# -*- coding: utf-8 -*-
"""
Hallucination Grader

Evaluates whether model response contain hallucinations (fabricated information not
supported by the context).
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
HALLUCINATION_PROMPT_EN = """
You are a professional data annotator responsible for evaluating whether the model response contains hallucinations. Your task is to score according to the following criteria:

<Scoring Criteria>
A hallucination-free response should:
- Contain only verifiable facts (supported by context if provided, or based on established facts/common knowledge).
- Not make unsupported claims or assumptions.
- Not add speculative or imagined details.
- Be completely accurate regarding dates, numbers, and specific details.
- Appropriately indicate uncertainty when information is incomplete.
</Scoring Criteria>

<Guidance>
- Thoroughly read the query and response.
- Identify all claims made in the response.
- If context is provided: Cross-check each claim with the context.
- If no context is provided: Verify claims against common knowledge and logical consistency.
- Note any unsupported, contradictory, or factually incorrect information.
- Consider the severity and number of hallucinations.
</Guidance>

<Reminder>
Focus only on factual accuracy. If context is provided, verify support from the context. If no context is provided, verify factual correctness based on common knowledge. Do not consider style, grammar, or presentation when scoring. A short but factual response should score higher than a longer response containing unsupported claims.
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
    "score": <integer between 1 and 5, where 5 means no hallucinations and 1 means severe hallucinations>,
    "reason": "<brief explanation for the assigned score, specifically mentioning any hallucinations found or confirming factual accuracy>"
}}

Scoring Scale:
- 5: No hallucinations, all claims fully supported (by context if provided, or factually correct)
- 4: Minor unsupported details, core facts are accurate
- 3: Some hallucinations, but main information is correct
- 2: Multiple hallucinations, significant unsupported or incorrect claims
- 1: Severe hallucinations, mostly fabricated or false information

JSON:
"""

# Chinese Prompt
HALLUCINATION_PROMPT_ZH = """
你是一名专业的数据标注员，负责评估模型输出是否包含幻觉（虚构信息）。你的任务是根据以下标准进行评分：

<评分标准>
无幻觉的回答应该：
- 仅包含可验证事实（如果提供了上下文则应由上下文支持，否则基于已知事实/常识）。
- 不做出无依据的声明或假设。
- 不添加推测性或想象的细节。
- 在日期、数字和具体细节方面完全准确。
- 在信息不完整时适当地表示不确定性。
</评分标准>

<指导>
- 仔细阅读输入问题和输出回答。
- 识别输出中的所有声明。
- 如果提供了上下文：将每个声明与上下文进行交叉核对。
- 如果未提供上下文：根据常识和逻辑一致性验证声明。
- 注意任何无依据、矛盾或事实错误的信息。
- 考虑幻觉的严重程度和数量。
</指导>

<提醒>
仅关注事实准确性。如果提供了上下文，则验证上下文的支持。如果未提供上下文，则基于常识验证事实正确性。评分时不要考虑风格、语法或呈现方式。简短但真实的回答应该比包含无依据声明的较长回答得分更高。
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
    "score": <1到5之间的整数，其中5表示无幻觉，1表示严重幻觉>,
    "reason": "<对所给分数的简要解释，特别提到发现的任何幻觉或确认事实准确性>"
}}

评分标尺：
- 5: 无幻觉，所有声明完全支持（如提供上下文则由上下文支持，否则事实正确）
- 4: 轻微的无依据细节，核心事实准确
- 3: 存在一些幻觉，但主要信息正确
- 2: 多处幻觉，有重大的无依据或错误声明
- 1: 严重幻觉，大部分信息为虚构或错误

JSON:
"""

# Build default template from prompts
DEFAULT_HALLUCINATION_TEMPLATE = PromptTemplate(
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

    Purpose:
        Detects hallucinations in model outputs by verifying that all claims are properly
        grounded in the provided context. A hallucination occurs when the model generates
        information that is not supported by, or contradicts, the given context.

    What it evaluates:
        - Factual Grounding: Every claim must be supported by the input context
        - Claim Verification: All statements must have evidence in provided materials
        - Speculation Detection: Model should not add imagined or assumed details
        - Numerical Accuracy: Dates, numbers, and statistics must be exact
        - Contradiction Avoidance: Output must not contradict the context

    When to use:
        - RAG (Retrieval-Augmented Generation) systems where context is provided
        - Question-answering systems that must stay grounded in given documents
        - Summarization tasks where fidelity to source is critical
        - Fact-checking generated content against reference materials
        - General factual accuracy evaluation (without context, based on common knowledge)

    Scoring:
        - 5: Perfect grounding, no unsupported claims
        - 4: Mostly accurate with minor unsupported details
        - 3: Contains some hallucinations but core facts are correct
        - 2: Multiple hallucinations with significant fabrications
        - 1: Severe hallucinations, mostly fabricated information

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [0, 1] to pass (default: 0.7)
        template: Custom evaluation template (default: DEFAULT_HALLUCINATION_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore object with:
            - score: Score [1, 5] where 5 = no hallucinations, 1 = severe hallucinations
            - reason: Detailed explanation of any hallucinations found
            - metadata: Threshold and evaluation details

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.core.llm_judge import HallucinationGrader
        >>>
        >>> # Initialize model
        >>> model = OpenAIChatModel(
        ...     api_key="sk-...",
        ...     model="qwen3-max",
        ...     temperature=0.1
        ... )
        >>>
        >>> # Create grader
        >>> grader = HallucinationGrader(model=model, threshold=0.7)
        >>>
        >>> # With context: Good output (grounded in context)
        >>> result = await grader.aevaluate(
        ...     query="When was the company founded?",
        ...     response="The company was founded in 2020 in San Francisco.",
        ...     context="The company was founded in 2020 in San Francisco."
        ... )
        >>> print(result.score)  # 5 - no hallucinations
        >>> print(result.reason)  # "Output is fully supported by context"
        >>>
        >>> # With context: Bad output (contains hallucination)
        >>> result = await grader.aevaluate(
        ...     query="When was the company founded?",
        ...     response="The company was founded in 2020 with 100 employees.",
        ...     context="The company was founded in 2020 in San Francisco."
        ... )
        >>> print(result.score)  # 3 - contains unsupported claim about employees
        >>> print(result.reason)  # "Output contains hallucination: '100 employees' not mentioned"
        >>>
        >>> # Without context: Factual verification
        >>> result = await grader.aevaluate(
        ...     query="What is the capital of France?",
        ...     response="The capital of France is Paris."
        ... )
        >>> print(result.score)  # 5 - factually correct
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 0.7,
        template: Optional[PromptTemplate] = DEFAULT_HALLUCINATION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """
        Initialize HallucinationGrader

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Success threshold [0, 1] (default: 0.7)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_HALLUCINATION_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
        """
        super().__init__(
            name="hallucination",
            mode=GraderMode.POINTWISE,
            description="Evaluate whether response contains hallucinations",
            model=model,
            template=template,
            language=language,
        )
        self.threshold = threshold
        self.template = template if template is not None else DEFAULT_HALLUCINATION_TEMPLATE

    async def aevaluate(
        self,
        query: str,
        response: str,
        context: str = "",
        reference_response: str = "",
    ) -> GraderScore:
        """
        Evaluate hallucination in response

        Args:
            query: Input question or prompt
            response: Model response to evaluate
            context: Context information to verify against. If empty string (default),
                    evaluation will be based on general factual consistency and common knowledge.
            reference_response: Reference response for comparison. Defaults to empty string.

        Returns:
            GraderScore: Score with hallucination value [1, 5]
                        where 5 means no hallucinations, 1 means severe hallucinations

        Example:
            >>> # With context
            >>> result = await grader.aevaluate(
            ...     query="When did the product launch?",
            ...     response="The product launched in 2023 with great success.",
            ...     context="The product launched in 2023.",
            ...     reference_response="The product launched in 2023."
            ... )
            >>> # Without context
            >>> result = await grader.aevaluate(
            ...     query="What is the capital of France?",
            ...     response="The capital of France is Paris."
            ... )
        """
        # Prepare context section based on language
        context_section = ""
        if context:
            if self.language == LanguageEnum.ZH:
                context_section = f"""使用以下上下文帮助你评估输出中是否存在幻觉：
<context>
{context}
</context>"""
            else:
                context_section = f"""Use the following context to help you evaluate whether there are hallucinations in the response:
<context>
{context}
</context>"""
        else:
            if self.language == LanguageEnum.ZH:
                context_section = """注意：未提供上下文信息。请基于常识、已知事实和逻辑一致性来评估输出是否包含幻觉、虚假信息或不合理的声明。"""
            else:
                context_section = """Note: No context is provided. Please evaluate whether the response contains hallucinations, false information, or unreasonable claims based on common knowledge, established facts, and logical consistency."""

        # Prepare reference section based on language
        reference_section = ""
        if reference_response:
            if self.language == LanguageEnum.ZH:
                reference_section = f"""如有需要，你也可以使用以下参考输出来帮助识别回答中的幻觉：
<reference_response>
{reference_response}
</reference_response>"""
            else:
                reference_section = f"""If available, you may also use the following reference response to help you identify hallucinations in the response:
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
            logger.error(f"Error evaluating hallucination: {e}")
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "threshold": self.threshold,
        }

        # Generate final reason
        reason = f"Hallucination evaluation score: {score}\n{reason}"

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata=metadata,
        )


__all__ = ["HallucinationGrader", "DEFAULT_HALLUCINATION_TEMPLATE"]
