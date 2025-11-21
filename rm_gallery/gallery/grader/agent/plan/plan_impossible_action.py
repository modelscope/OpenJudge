# -*- coding: utf-8 -*-
"""
Plan Impossible Action Grader

Evaluates whether the agent creates a plan that is semantically illogical or infeasible.
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
PLAN_IMPOSSIBLE_ACTION_PROMPT_EN = """
You are an expert in analyzing agent behavior. Your task is to detect whether the agent creates a plan that is semantically illogical or infeasible.

<Error Type: Plan Impossible Action>
The agent creates a plan that is semantically illogical or infeasible, such as violating causal logic, specifying actions in incorrect order, or proposing impossible actions. The plan may ignore physical constraints, logical prerequisites, or the current state of the environment.
</Error Type>

<Rubrics for Detection>
1. The plan violates causal logic (e.g., using an object before obtaining it)
2. The plan specifies actions in an impossible order (e.g., closing before opening)
3. The plan proposes actions that cannot be executed given the current environment state
4. The plan ignores necessary preconditions or prerequisites for actions
5. The plan contains logically contradictory steps or goals
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Apply each rubric: Check if the step matches the error patterns described in each rubric
2. Focus on relevant modules: Only consider plan, observation, and memory modules
3. Provide evidence-based reasoning: Explain whether the step matches the rubric patterns and why
4. Assess confidence: Rate your confidence based on how clearly the patterns are exhibited
</Evaluation Criteria>

{context_section}

<trajectory_steps>
{trajectory_steps}
</trajectory_steps>

# Scoring Instructions
- If the error is detected: score = 0.0 (has problem)
- If no error is detected: score = 1.0 (good quality)

Provide your evaluation in the following structured JSON format:
{{
    "score": <0.0 or 1.0>,
    "reason": "<detailed explanation including error_step if applicable and confidence level>"
}}

JSON:
"""

# Chinese Prompt
PLAN_IMPOSSIBLE_ACTION_PROMPT_ZH = """
你是一名分析智能体行为的专家。你的任务是检测智能体是否创建了语义上不合逻辑或不可行的计划。

<错误类型：计划不可能动作>
智能体创建了语义上不合逻辑或不可行的计划，例如违反因果逻辑、以错误的顺序指定动作，或提出不可能的动作。该计划可能忽略物理约束、逻辑前提或环境的当前状态。
</错误类型>

<检测准则>
1. 计划违反因果逻辑（例如，在获得对象之前使用它）
2. 计划以不可能的顺序指定动作（例如，在打开之前关闭）
3. 计划提出了在当前环境状态下无法执行的动作
4. 计划忽略了动作的必要前提条件或先决条件
5. 计划包含逻辑上相互矛盾的步骤或目标
</检测准则>

<评估标准>
进行分析时：
1. 应用每个准则：检查步骤是否匹配每个准则中描述的错误模式
2. 关注相关模块：仅考虑计划、观察和记忆模块
3. 提供基于证据的推理：解释步骤是否匹配准则模式以及原因
4. 评估置信度：根据模式表现的清晰程度评估你的置信度
</评估标准>

{context_section}

<trajectory_steps>
{trajectory_steps}
</trajectory_steps>

# 评分指令
- 如果检测到错误：score = 0.0（有问题）
- 如果未检测到错误：score = 1.0（质量良好）

请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0.0 或 1.0>,
    "reason": "<详细解释，包括错误步骤（如适用）和置信度水平>"
}}

JSON:
"""

# Build default template from prompts
DEFAULT_PLAN_IMPOSSIBLE_ACTION_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(PLAN_IMPOSSIBLE_ACTION_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(PLAN_IMPOSSIBLE_ACTION_PROMPT_ZH),
            ),
        ],
    },
)


class PlanImpossibleActionGrader(LLMGrader):
    """
    Plan Impossible Action Grader

    Evaluates whether the agent creates a plan that is semantically illogical or infeasible.

    Required modules: plan, observation, memory

    Attributes:
        name: Grader name
        model: ChatModelBase instance for evaluation
        template: Evaluation template
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
        >>> grader = PlanImpossibleActionGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     plan="I will use the key to unlock the door.",
        ...     observation="The drawer is closed. You don't have any items.",
        ...     memory="The key is inside the drawer."
        ... )
        >>> print(f"Score: {result.score}")  # 0.0 (error detected - using key before obtaining it)
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Optional[Template] = DEFAULT_PLAN_IMPOSSIBLE_ACTION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="plan_impossible_action",
            mode=GraderMode.POINTWISE,
            description="Detect plan impossible action errors",
            model=model,
            template=template,
            language=language,
        )
        self.template = (
            template if template is not None else DEFAULT_PLAN_IMPOSSIBLE_ACTION_TEMPLATE
        )

    def _format_trajectory_steps(
        self,
        plan: str,
        observation: str,
        memory: str,
        history_steps: Optional[list] = None,
    ) -> str:
        """Format trajectory steps for evaluation.
        
        Args:
            plan: Agent's planning/reasoning
            observation: Agent's observation from the environment
            memory: Agent's memory content
            history_steps: Optional list of previous step dictionaries
        
        Returns:
            Formatted trajectory string
        """
        lines = []
        
        # Add history steps if provided
        if history_steps:
            for i, hist_step in enumerate(history_steps):
                lines.append(f"Step {i + 1}:")
                for key, value in hist_step.items():
                    if value:
                        lines.append(f"{key.capitalize()}: {value}")
                lines.append("")
        
        # Add current step
        step_number = len(history_steps) + 1 if history_steps else 1
        lines.append(f"Step {step_number}:")
        lines.append(f"Plan: {plan}")
        lines.append(f"Observation: {observation}")
        lines.append(f"Memory: {memory}")
        
        return "\n".join(lines)

    async def aevaluate(
        self,
        plan: str,
        observation: str,
        memory: str,
        history_steps: Optional[list] = None,
        task_context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate plan impossible action

        Args:
            plan: Agent's planning/reasoning
            observation: Agent's observation from the environment
            memory: Agent's memory content
            history_steps: Optional list of previous step dictionaries for context
            task_context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = no error, 0.0 = error detected)

        Example:
            >>> result = await grader.aevaluate(
            ...     plan="I will use the key to unlock the door.",
            ...     observation="The drawer is closed. You don't have any items.",
            ...     memory="The key is inside the drawer.",
            ...     task_context="Task: Unlock the door"
            ... )
        """
        # Format trajectory steps
        trajectory_steps = self._format_trajectory_steps(
            plan=plan,
            observation=observation,
            memory=memory,
            history_steps=history_steps,
        )

        # Prepare context section
        context_section = ""
        if task_context:
            context_section = f"""<task_context>
{task_context}
</task_context>"""

        try:
            result = await super().aevaluate(
                trajectory_steps=trajectory_steps,
                context_section=context_section,
            )
            score = result.score
            reason = result.reason

            # Ensure score is binary (0.0 or 1.0)
            normalized_score = 1.0 if score > 0.5 else 0.0

        except Exception as e:
            logger.error(f"Error evaluating plan impossible action: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "error_type": "plan_impossible_action",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "PlanImpossibleActionGrader",
    "DEFAULT_PLAN_IMPOSSIBLE_ACTION_TEMPLATE",
]

