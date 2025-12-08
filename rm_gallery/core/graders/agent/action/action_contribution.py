# -*- coding: utf-8 -*-
"""
Action Contribution Grader for Individual Tool Call Evaluation
This module provides grader for individual tool call-response pairs
in agent trajectories, assessing how much each action contributes to solving user queries.
"""
import textwrap
from typing import Optional, Union

from loguru import logger

from rm_gallery.core.graders.base_grader import GraderMode, GraderScore
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long
# English Prompt
CONTRIBUTION_PROMPT_EN = """# Task Description
You are an expert evaluator assessing the contribution of individual tool calls in an agent's problem-solving trajectory.
Your task is to evaluate how much a specific tool call-response pair contributes to solving the user's original query, with special attention to factual accuracy and potential hallucinations in the final answer.
Consider these evaluation dimensions:
1. **Relevance**: How relevant is the tool call to the user's question?
2. **Information Quality**: How useful and accurate is the information obtained from the tool response?
3. **Problem Progress**: How much does this action advance toward solving the user's problem?
4. **Efficiency**: Is this tool call necessary or redundant?
5. **Factual Grounding**: How well does this tool call support or contradict the final answer?
# User Query
{query}
# Context Before Tool Call
{context_before}
# Tool Call Information
{tool_calls}
# Tool Response
{tool_responses}
# Context After Tool Call
{context_after}
# Final Answer
{response}
# Evaluation Guidelines
## Scoring Framework
- Score 0.8-1.0: Critical contribution, essential for solving the problem, provides accurate and verifiable information that properly supports the final answer
- Score 0.6-0.8: Significant contribution, provides important and reliable information that aligns with the final answer
- Score 0.4-0.6: Moderate contribution, somewhat helpful but not essential, information appears reliable and consistent
- Score 0.2-0.4: Minor contribution, provides some context but limited value, or reveals inconsistencies with the final answer
- Score 0.0-0.2: No meaningful contribution, irrelevant, redundant, or provides information that contradicts a potentially correct final answer
## Answer Quality Assessment
When evaluating this tool call, specifically consider:
- **Factual Consistency**: Does the information from this tool call align with claims made in the final answer?
- **Evidence Support**: Does this tool call provide concrete evidence that supports specific claims in the final answer?
- **Contradiction Detection**: Does this tool call reveal information that contradicts the final answer? If so, assess whether the tool call or the final answer is more likely to be correct.
- **Completeness**: Does this tool call help address the user's query comprehensively?
- **Reliability**: Can the information from this tool call be considered trustworthy and accurate?
## Hallucination Detection
Pay special attention to:
- Whether valuable information from this tool call was ignored or misrepresented in the final answer
- If this tool call provided evidence that contradicts unsupported claims in the final answer
- Whether this tool call could have prevented potential hallucinations if properly utilized
- If the final answer makes claims not supported by this or other tool calls
## Scoring Adjustments
- **Reduce scores** if this tool call's accurate information was ignored in favor of potentially hallucinated content
- **Increase scores** if this tool call provided crucial factual grounding that prevents hallucinations
- **Consider utility**: Even if the final answer has issues, score based on the inherent value and accuracy of this tool call's contribution
# Output Requirement
Provide your evaluation in the following structured JSON format:
{{
    "score": <0.0-1.0>,
    "reason": "<detailed explanation of contribution score>"
}}
JSON:
"""
# Chinese Prompt
CONTRIBUTION_PROMPT_ZH = """# 任务描述
您是一位专业的评估专家,负责评估智能体问题解决轨迹中各个工具调用的贡献。
您的任务是评估特定工具调用-响应对对解决用户原始查询的贡献程度,特别关注最终答案的事实准确性和潜在的幻觉问题。
请考虑以下评估维度:
1. **相关性**: 工具调用与用户问题的相关程度如何?
2. **信息质量**: 从工具响应中获得的信息有多有用和准确?
3. **问题进展**: 此步骤在解决用户问题方面取得了多大进展?
4. **效率**: 此工具调用是必要的还是冗余的?
5. **事实依据**: 此工具调用对最终答案的支持或矛盾程度如何?
# 用户查询
{query}
# 工具调用前的上下文
{context_before}
# 工具调用信息
{tool_calls}
# 工具响应
{tool_responses}
# 工具调用后的上下文
{context_after}
# 最终答案
{response}
# 评估指南
## 评分框架
- 分数 0.8-1.0: 关键贡献,解决问题必不可少,提供准确可验证的信息,恰当支持最终答案
- 分数 0.6-0.8: 重要贡献,提供重要且可靠的信息,与最终答案一致
- 分数 0.4-0.6: 中等贡献,有一定帮助但非必要,信息看起来可靠且一致
- 分数 0.2-0.4: 轻微贡献,提供一些上下文但价值有限,或揭示与最终答案的不一致之处
- 分数 0.0-0.2: 无意义贡献,不相关、冗余或提供与可能正确的最终答案相矛盾的信息
## 答案质量评估
评估此工具调用时,请特别考虑:
- **事实一致性**: 此工具调用的信息是否与最终答案中的声明一致?
- **证据支持**: 此工具调用是否为最终答案中的特定声明提供了具体证据?
- **矛盾检测**: 此工具调用是否揭示了与最终答案相矛盾的信息?如果是,请评估工具调用或最终答案哪个更可能正确。
- **完整性**: 此工具调用是否有助于全面解决用户查询?
- **可靠性**: 此工具调用的信息是否可以被认为是可信且准确的?
## 幻觉检测
请特别注意:
- 此工具调用的有价值信息是否在最终答案中被忽略或误述
- 此工具调用是否提供了与最终答案中无支持声明相矛盾的证据
- 如果得到适当利用,此工具调用是否可能防止潜在的幻觉
- 最终答案是否做出了此工具调用或其他工具调用不支持的声明
## 评分调整
- 如果此工具调用的准确信息被忽略,转而采用可能的幻觉内容,则**降低分数**
- 如果此工具调用提供了防止幻觉的关键事实依据,则**提高分数**
- **考虑实用性**: 即使最终答案有问题,也要基于此工具调用贡献的内在价值和准确性进行评分
# 输出要求
请以JSON格式输出您的评估结果:
{{
    "score": <0.0-1.0>,
    "reason": "<贡献分数的详细解释>"
}}
JSON:
"""
# Default PromptTemplate
DEFAULT_CONTRIBUTION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(CONTRIBUTION_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(CONTRIBUTION_PROMPT_ZH),
            ),
        ],
    },
)


class ActionContributionGrader(LLMGrader):
    """
    Action Contribution Grader for evaluating individual tool call contributions.
    This grader evaluates each tool call-response pair independently to assess
    how much it contributes to solving the user's query.
    Attributes:
        name: Grader name
        model: ChatModelBase instance for evaluation
        template: Evaluation template
        language: Language for evaluation prompts
    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> api = OpenAIChatModel(api_key="...", model="gpt-4o")
        >>> grader = ActionContributionGrader(model=api)
        >>> result = await grader.aevaluate(
        ...     query="帮我找投资建议",
        ...     tool_calls="Tool: search\\nArguments: {...}",
        ...     tool_responses="搜索结果...",
        ...     context_before="之前的对话...",
        ...     context_after="之后的对话...",
        ...     response="根据分析..."
        ... )
        >>> print(f"Contribution score: {result.score}")
    """

    def __init__(
        self,
        model: Union[BaseChatModel, dict],
        template: Optional[PromptTemplate] = DEFAULT_CONTRIBUTION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="action_contribution",
            mode=GraderMode.POINTWISE,
            description="Action contribution evaluation for individual tool calls",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_CONTRIBUTION_TEMPLATE

    async def aevaluate(
        self,
        query: str,
        tool_calls: str,
        tool_responses: str,
        context_before: str = "No context before",
        context_after: str = "No context after",
        response: str = "",
    ) -> GraderScore:
        """
        Evaluate individual tool call contribution.
        Args:
            query: Original user question
            tool_calls: Tool call information
            tool_responses: Tool response content
            context_before: Context before the tool call (optional)
            context_after: Context after the tool call (optional)
            response: Final answer from the agent (optional)
        Returns:
            GraderScore: Contribution score (0.0-1.0)
                - score: Contribution score
                - reason: Detailed evaluation explanation
                - metadata: Additional evaluation details
        Example:
            >>> result = await grader.aevaluate(
            ...     query="帮我找投资建议",
            ...     tool_calls="Tool: search\\nArguments: {...}",
            ...     tool_responses="搜索结果...",
            ...     response="根据分析..."
            ... )
        """
        try:
            # Call parent evaluation with formatted parameters
            result = await super().aevaluate(
                query=query,
                tool_calls=tool_calls,
                tool_responses=tool_responses,
                context_before=context_before,
                context_after=context_after,
                response=response,
            )
            score = result.score
            reason = result.reason
            # Ensure score is in valid range
            normalized_score = max(0.0, min(1.0, score))
        except Exception as e:
            logger.error(f"Error evaluating {self.name}: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"
        # Prepare metadata
        metadata = {
            "raw_score": score,
            "evaluation_mode": "independent",
            "evaluation_type": "tool_call_contribution",
        }
        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )
