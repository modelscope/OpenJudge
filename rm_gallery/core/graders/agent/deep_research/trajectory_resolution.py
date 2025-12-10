# -*- coding: utf-8 -*-
"""
Trajectory Resolution Grader for Multi-Step Tool Call Evaluation

This module provides holistic evaluation for all tool call steps in an agent trajectory,
assessing each step's contribution and whether the trajectory resolves the user's query.
"""

import json
import textwrap
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from rm_gallery.core.graders.base_grader import GraderMode, GraderScore
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# Chinese Prompt
TRAJECTORY_RESOLUTION_PROMPT_ZH = """# 任务描述
您是一位专业的评估专家,负责评估智能体轨迹中每个工具调用step的贡献以及整体问题解决能力。

您需要为轨迹中的每个工具调用step提供独立且一致的评估,考虑它们各自的贡献和相对重要性。此外,您还需要评估所有工具调用返回是否能够集体解决用户查询。

# 用户查询
{user_query}

# 完整轨迹
{trajectory_messages}

# 最终答案
{final_answer}

# 评估维度与分数分配(总计100分)

## 1. 问题解决能力 (40分)
评估所有工具调用返回是否能集体解决用户查询。

**分数段标准:**
- **36-40分 (优秀)**: 工具调用返回完全解决用户查询,提供全面准确的信息,用户无需额外信息即可理解并采取行动
- **28-35分 (良好)**: 工具调用返回大部分解决用户查询,提供大致完整可靠的信息,仅存在微小空白
- **20-27分 (中等)**: 工具调用返回部分解决用户查询,存在一些空白或不确定信息,但仍有参考价值
- **12-19分 (较差)**: 工具调用返回最低限度解决用户查询,存在重大空白或可疑信息,参考价值有限
- **0-11分 (极差)**: 工具调用返回未能解决用户查询或提供误导/错误信息,无参考价值

**评估要点:**
- **完整性**: 所有工具调用返回是否集体提供足够信息来回答用户查询?
- **一致性**: 工具调用返回之间以及与最终答案是否一致?
- **准确性**: 所有工具调用的综合信息是否准确可靠?
- **充分性**: 信息是否足够用户理解和采取行动?

## 2. 轨迹质量 (35分)
评估整个工具调用轨迹的质量,包括每个step的贡献、相关性和准确性。

**分数段标准:**
- **32-35分 (优秀)**: 每个step都提供关键贡献,解决问题必不可少,信息准确可验证,无冗余步骤
- **25-31分 (良好)**: 大部分step提供重要贡献,信息可靠,可能存在少量冗余或次要步骤
- **18-24分 (中等)**: step贡献中等,有一定帮助但部分非必要,信息基本可靠,存在明显冗余
- **10-17分 (较差)**: step贡献有限,提供的信息价值较低,存在较多冗余或可疑信息
- **0-9分 (极差)**: step大多不相关、冗余或提供误导/错误信息,无实际价值

**评估要点:**
- **step贡献度**: 每个step对解决问题的实际贡献
- **相关性**: 每个step与用户查询的相关程度
- **准确性**: 每个step获取信息的准确性和可靠性
- **必要性**: 是否存在冗余或不必要的工具调用

## 3. 执行效率 (25分)
评估轨迹的执行效率,包括步骤数量、顺序合理性和资源利用。

**分数段标准:**
- **23-25分 (优秀)**: 步骤数量最优,执行顺序高效合理,无冗余调用,资源利用充分
- **18-22分 (良好)**: 步骤数量合理,执行顺序基本合理,存在1-2个可优化的地方
- **13-17分 (中等)**: 步骤数量略多或顺序欠佳,存在明显可优化空间,但不影响最终结果
- **7-12分 (较差)**: 步骤数量过多或顺序混乱,存在较多冗余调用,效率低下
- **0-6分 (极差)**: 步骤严重冗余或顺序错乱,资源浪费严重,执行效率极低

**评估要点:**
- **步骤优化**: 是否用最少的步骤达成目标
- **顺序合理性**: 工具调用的顺序是否合理高效
- **冗余程度**: 是否存在重复或不必要的工具调用
- **时间成本**: 整体执行的时间效率

# Step级评估(可变长度列表)
对于轨迹中的每个工具调用step,提供独立评估。

**Step评估要点:**
- **证据支持**: 此step是否为解决用户问题提供了具体证据?
- **矛盾检测**: 此step是否揭示了与用户问题相矛盾的信息?
- **完整性**: 此step是否有助于全面解决用户查询?
- **准确性验证**: 此工具调用的信息是否可靠准确?
- **效率评价**: 此step的执行是否必要且高效?

# 重要说明
- 整体评估所有step以确保一致性,并与最终答案进行交叉验证
- 识别任何冗余或不必要的工具调用,同时注意可能改善答案质量的缺失工具调用
- 确保评分既反映对解决用户问题的实际贡献,也反映在支持事实准确性方面的作用
- 三个维度的分数相加应约等于100分(允许±2分的误差)
- 使用step_index将您的评估与正确的step匹配

# 输出要求
请以JSON格式输出评估结果:
{{
    "problem_solving": {{
        "score": <int (0-40)>,
        "reason": "<所有工具调用返回是否能集体解决用户查询的详细推理>"
    }},
    "trajectory_quality": {{
        "score": <int (0-35)>,
        "reason": "<轨迹质量评估的详细理由>"
    }},
    "efficiency": {{
        "score": <int (0-25)>,
        "reason": "<轨迹效率评估的详细理由>"
    }},
    "step_evaluations": [
        {{
            "step_index": <int - step索引,从0开始>,
            "step_description": "<简短描述此step的工具调用>",
            "contribution_level": "<关键/重要/中等/轻微/无效>",
            "reason": "<此step的详细评估理由>"
        }}
    ]
}}

JSON:
"""

# English Prompt
TRAJECTORY_RESOLUTION_PROMPT_EN = """# Task Description
You are a professional evaluation expert responsible for assessing the contribution of each tool call step in an agent trajectory and the overall problem-solving capability.

You need to provide independent and consistent evaluation for each tool call step
in the trajectory, considering their respective contributions and relative importance.
Additionally, you must evaluate whether all tool call returns collectively can solve
the user query.

# User Query
{user_query}

# Complete Trajectory
{trajectory_messages}

# Final Answer
{final_answer}

# Evaluation Dimensions and Score Distribution (Total 100 points)

## 1. Problem Solving Capability (40 points)
Evaluate whether all tool call returns collectively can solve the user query.

**Score Ranges:**
- **36-40 points (Excellent)**: Tool call returns fully address the user query with comprehensive and accurate information, user can understand and act without additional information
- **28-35 points (Good)**: Tool call returns largely address the user query with mostly complete and reliable information, only minor gaps exist
- **20-27 points (Fair)**: Tool call returns partially address the user query with some gaps or uncertain information, but still have reference value
- **12-19 points (Poor)**: Tool call returns minimally address the user query with significant gaps or questionable information, limited reference value
- **0-11 points (Very Poor)**: Tool call returns fail to address the user query or provide misleading/incorrect information, no reference value

**Assessment Focus:**
- **Completeness**: Do all tool call returns collectively provide sufficient information to answer the user query?
- **Consistency**: Are the tool call returns consistent with each other and the final answer?
- **Accuracy**: Is the combined information from all tool calls accurate and reliable?
- **Sufficiency**: Is the information sufficient for the user to understand and act upon?

## 2. Trajectory Quality (35 points)
Evaluate the quality of the entire tool call trajectory, including contribution, relevance, and accuracy of each step.

**Score Ranges:**
- **32-35 points (Excellent)**: Every step provides critical contribution, essential for solving the problem, information is accurate and verifiable, no redundant steps
- **25-31 points (Good)**: Most steps provide significant contribution, information is reliable, may have few redundant or secondary steps
- **18-24 points (Fair)**: Steps provide moderate contribution, somewhat helpful but partially unnecessary, information is basically reliable, obvious redundancy exists
- **10-17 points (Poor)**: Steps provide limited contribution, information has low value, considerable redundancy or questionable information
- **0-9 points (Very Poor)**: Most steps are irrelevant, redundant, or provide misleading/incorrect information, no practical value

**Assessment Focus:**
- **Step Contribution**: Actual contribution of each step to solving the problem
- **Relevance**: Relevance of each step to the user query
- **Accuracy**: Accuracy and reliability of information obtained by each step
- **Necessity**: Whether redundant or unnecessary tool calls exist

## 3. Execution Efficiency (25 points)
Evaluate the execution efficiency of the trajectory, including number of steps, sequence rationality, and resource utilization.

**Score Ranges:**
- **23-25 points (Excellent)**: Optimal number of steps, efficient and reasonable execution sequence, no redundant calls, full resource utilization
- **18-22 points (Good)**: Reasonable number of steps, basically reasonable execution sequence, 1-2 areas for optimization exist
- **13-17 points (Fair)**: Slightly too many steps or suboptimal sequence, obvious room for optimization, but doesn't affect final results
- **7-12 points (Poor)**: Too many steps or chaotic sequence, considerable redundant calls, low efficiency
- **0-6 points (Very Poor)**: Severely redundant steps or disordered sequence, serious resource waste, extremely low execution efficiency

**Assessment Focus:**
- **Step Optimization**: Whether the goal is achieved with minimum steps
- **Sequence Rationality**: Whether the tool call sequence is reasonable and efficient
- **Redundancy Level**: Whether duplicate or unnecessary tool calls exist
- **Time Cost**: Overall execution time efficiency

# Step-Level Evaluation (Variable Length List)
Provide independent evaluation for each tool call step in the trajectory.

**Step Assessment Focus:**
- **Evidence Support**: Does this step provide concrete evidence for addressing the user's question?
- **Contradiction Detection**: Does this step reveal information that contradicts the user's question?
- **Completeness**: Does this step help comprehensively address the user query?
- **Accuracy Verification**: Is the information from this tool call reliable and accurate?
- **Efficiency Evaluation**: Is this step execution necessary and efficient?

# Important Notes
- Evaluate all steps as a whole to ensure consistency and cross-validate with the final answer
- Identify any redundant or unnecessary tool calls, while noting missing tool calls that could improve answer quality
- Ensure scoring reflects both actual contribution to solving the user's problem and role in supporting factual accuracy
- The sum of scores from three dimensions should approximately equal 100 points (±2 points tolerance)
- Use step_index to match your evaluations with the correct steps

# Output Requirement
Please output your evaluation in JSON format:
{{
    "problem_solving": {{
        "score": <int (0-40)>,
        "reason": "<detailed reasoning for whether all tool call returns collectively can solve the user query>"
    }},
    "trajectory_quality": {{
        "score": <int (0-35)>,
        "reason": "<detailed reasoning for trajectory quality assessment>"
    }},
    "efficiency": {{
        "score": <int (0-25)>,
        "reason": "<detailed reasoning for trajectory efficiency assessment>"
    }},
    "step_evaluations": [
        {{
            "step_index": <int - step index starting from 0>,
            "step_description": "<brief description of this tool call step>",
            "contribution_level": "<critical/significant/moderate/minor/ineffective>",
            "reason": "<detailed evaluation reasoning for this step>"
        }}
    ]
}}

JSON:
"""

# Build default template from prompts
DEFAULT_TRAJECTORY_RESOLUTION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TRAJECTORY_RESOLUTION_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TRAJECTORY_RESOLUTION_PROMPT_ZH),
            ),
        ],
    },
)


class TrajectoryResolutionGrader(LLMGrader):
    """
    Trajectory Resolution Grader for evaluating complete agent trajectories.

    This grader evaluates complete agent trajectories (multi-turn messages),
    assessing whether the trajectory successfully resolves the user's query.
    It accepts standard messages format and automatically extracts the trajectory
    after removing system prompts.

    Attributes:
        name: Grader name
        model: ChatModelBase instance for evaluation
        language: Language for evaluation prompts
        resolution_threshold: Threshold for determining if the trajectory is resolved (default: 0.9)

    Example:
        >>> from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
        >>> api = OpenAIChatModel(api_key="...", model="qwen-max")
        >>> grader = TrajectoryResolutionGrader(model=api, resolution_threshold=0.85)
        >>> result = await grader.aevaluate(
        ...     messages=[
        ...         {"role": "system", "content": "..."},
        ...         {"role": "user", "content": "帮我找投资建议"},
        ...         {"role": "assistant", "content": "...", "tool_calls": [...]},
        ...         ...
        ...     ]
        ... )
        >>> print(f"Score: {result.score}")
    """

    def __init__(
        self,
        model: Union[BaseChatModel, dict],
        template: Optional[PromptTemplate] = DEFAULT_TRAJECTORY_RESOLUTION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.ZH,
        resolution_threshold: float = 0.9,
    ):
        """
        Initialize the TrajectoryResolutionGrader.

        Args:
            model (Union[BaseChatModel, dict]): The chat model to use for evaluation.
                Can be either a BaseChatModel instance or a dictionary configuration.
            template (Optional[PromptTemplate]): The prompt template for trajectory evaluation.
                Defaults to DEFAULT_TRAJECTORY_RESOLUTION_TEMPLATE.
            language (LanguageEnum): Language for the evaluation prompt.
                Defaults to LanguageEnum.ZH (Chinese).
            resolution_threshold (float): Threshold for determining if the trajectory is resolved.
                Scores greater than or equal to this value are considered resolved.
                Defaults to 0.9 (90%).

        Example:
            >>> from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
            >>> api = OpenAIChatModel(api_key="...", model="qwen-max")
            >>> grader = TrajectoryResolutionGrader(model=api, resolution_threshold=0.85)
        """
        super().__init__(
            name="trajectory_resolution",
            mode=GraderMode.POINTWISE,
            description="Trajectory resolution evaluation for complete agent trajectories",
            model=model,
            template=template,
            language=language,
        )
        self.resolution_threshold = resolution_threshold

    def _extract_trajectory_from_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[str, str, str]:
        """
        Extract user query, trajectory, and final answer from messages.

        Args:
            messages: List of message dicts (standard format)

        Returns:
            Tuple of (user_query, trajectory_messages, final_answer)
        """
        # Filter out system messages
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]

        if not non_system_messages:
            return "", "", ""

        # First non-system message should be user query
        user_query = ""
        first_msg = non_system_messages[0]
        if first_msg.get("role") == "user":
            user_query = first_msg.get("content", "")

        # Format trajectory messages
        trajectory_parts = []
        # pylint: disable=unused-variable
        for i, msg in enumerate(non_system_messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if role == "user":
                trajectory_parts.append(f"**User**: {content}")
            elif role == "assistant":
                if tool_calls:
                    # Format tool calls
                    tool_calls_str = ""
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        tool_name = func.get("name", "unknown")
                        tool_args = func.get("arguments", "{}")
                        try:
                            args_dict = json.loads(tool_args)
                            tool_calls_str += f"  - {tool_name}({json.dumps(args_dict, ensure_ascii=False)})\n"
                        except json.JSONDecodeError:
                            tool_calls_str += f"  - {tool_name}({tool_args})\n"

                    trajectory_parts.append(f"**Assistant Tool Calls**:\n{tool_calls_str}")

                if content:
                    trajectory_parts.append(f"**Assistant**: {content}")
            elif role == "tool":
                tool_name = msg.get("name", "unknown_tool")
                trajectory_parts.append(f"**Tool Response ({tool_name})**: {content}")

        trajectory_messages = "\n\n".join(trajectory_parts)

        # Last assistant message is final answer
        final_answer = ""
        for msg in reversed(non_system_messages):
            if msg.get("role") == "assistant":
                final_answer = msg.get("content", "")
                break

        return user_query, trajectory_messages, final_answer

    async def aevaluate(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate complete agent trajectory.

        Args:
            messages: List of messages (standard format, including system, user, assistant, tool)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Holistic evaluation score for the trajectory (0.0-1.0)
                - score: Normalized score (0-100 from LLM normalized to 0.0-1.0)
                - reason: Detailed evaluation breakdown
                - metadata: Additional evaluation details including is_resolved status

        Example:
            >>> result = await grader.aevaluate(
            ...     messages=[
            ...         {"role": "user", "content": "帮我找投资建议"},
            ...         {"role": "assistant", "content": "...", "tool_calls": [...]},
            ...         ...
            ...     ]
            ... )
        """
        # Extract trajectory from messages
        user_query, trajectory_messages, final_answer = self._extract_trajectory_from_messages(messages)

        if not user_query or not trajectory_messages:
            logger.warning("Empty user query or trajectory, returning zero score")
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="Empty user query or trajectory",
                metadata={
                    "evaluation_mode": "trajectory_resolution",
                    "error": "Empty input",
                },
            )

        try:
            # Call parent evaluation with formatted parameters
            result = await super().aevaluate(
                user_query=user_query,
                trajectory_messages=trajectory_messages,
                final_answer=final_answer,
            )

            # Parse the JSON response and extract scores
            response_dict = {}
            if hasattr(result, "metadata") and isinstance(result.metadata, dict):
                response_dict = result.metadata.get("response", {})

            # Extract evaluation scores from new structure (out of their respective maximums)
            problem_solving_dict = response_dict.get("problem_solving", {})
            raw_problem_solving_score = float(problem_solving_dict.get("score", 0.0))  # 0-40
            overall_problem_solving_reason = problem_solving_dict.get("reason", "")

            quality_dict = response_dict.get("trajectory_quality", {})
            raw_quality_score = float(quality_dict.get("score", 0.0))  # 0-35
            trajectory_quality_reason = quality_dict.get("reason", "")

            efficiency_dict = response_dict.get("efficiency", {})
            raw_efficiency_score = float(efficiency_dict.get("score", 0.0))  # 0-25
            efficiency_reason = efficiency_dict.get("reason", "")

            # Extract step evaluations (variable length list)
            step_evaluations = response_dict.get("step_evaluations", [])

            # Calculate total score (should be approximately 100)
            total_raw_score = raw_problem_solving_score + raw_quality_score + raw_efficiency_score

            # Normalize total score from 0-100 to 0.0-1.0
            normalized_score = total_raw_score / 100.0

            # Ensure score is in valid range
            normalized_score = max(0.0, min(1.0, normalized_score))

            # Normalize individual dimension scores for metadata
            overall_problem_solving_score = raw_problem_solving_score / 40.0  # Normalize to 0-1
            trajectory_quality_score = raw_quality_score / 35.0  # Normalize to 0-1
            efficiency_score = raw_efficiency_score / 25.0  # Normalize to 0-1

            # Determine resolution status using the specified threshold
            is_resolved = normalized_score >= self.resolution_threshold

            reason = (
                f"Trajectory evaluation (total: {total_raw_score:.1f}/100): "
                f"problem_solving={raw_problem_solving_score:.1f}/40, "
                f"quality={raw_quality_score:.1f}/35, "
                f"efficiency={raw_efficiency_score:.1f}/25"
            )

        except Exception as e:
            logger.error(f"Error evaluating {self.name}: {e}")
            normalized_score = 0.0
            total_raw_score = 0.0
            overall_problem_solving_score = 0.0
            raw_problem_solving_score = 0.0
            overall_problem_solving_reason = ""
            trajectory_quality_score = 0.0
            raw_quality_score = 0.0
            trajectory_quality_reason = ""
            efficiency_score = 0.0
            raw_efficiency_score = 0.0
            efficiency_reason = ""
            step_evaluations = []
            is_resolved = False
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "problem_solving": {
                "raw_score": raw_problem_solving_score,
                "max_score": 40.0,
                "normalized_score": overall_problem_solving_score,
                "reason": overall_problem_solving_reason,
            },
            "trajectory_quality": {
                "raw_score": raw_quality_score,
                "max_score": 35.0,
                "normalized_score": trajectory_quality_score,
                "reason": trajectory_quality_reason,
            },
            "efficiency": {
                "raw_score": raw_efficiency_score,
                "max_score": 25.0,
                "normalized_score": efficiency_score,
                "reason": efficiency_reason,
            },
            "step_evaluations": step_evaluations,
            "total_raw_score": total_raw_score,
            "is_resolved": is_resolved,
            "resolution_threshold": self.resolution_threshold,
            "evaluation_type": "trajectory_resolution",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )
