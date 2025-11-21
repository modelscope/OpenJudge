# -*- coding: utf-8 -*-
"""
Action Misalignment Grader

Evaluates whether the agent executes an action inconsistent with its stated plan or reasoning.
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
ACTION_MISALIGNMENT_PROMPT_EN = """
You are an expert in analyzing agent behavior. Your task is to detect whether the agent executes an action inconsistent with its stated plan or reasoning.

<Error Type: Action Misalignment>
The agent executes an action inconsistent with its stated plan or reasoning. The action contradicts the plan or does not logically follow from the reflection. This suggests a disconnect between the agent's planning and execution modules.
</Error Type>

<Rubrics for Detection>
1. The action directly contradicts the stated plan (e.g., plan says "open" but action is "close")
2. The action targets a different object than specified in the plan
3. The action is unrelated to achieving the goal stated in the plan
4. The action sequence does not follow the logical order outlined in the plan
5. The action ignores important preconditions or constraints mentioned in the plan
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Apply each rubric: Check if the step matches the error patterns described in each rubric
2. Focus on relevant modules: Only consider plan and action modules
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
ACTION_MISALIGNMENT_PROMPT_ZH = """
你是一名分析智能体行为的专家。你的任务是检测智能体是否执行了与其声明的计划或推理不一致的动作。

<错误类型：动作不对齐>
智能体执行了与其声明的计划或推理不一致的动作。该动作与计划相矛盾或不符合反思的逻辑。这表明智能体的计划和执行模块之间存在脱节。
</错误类型>

<检测准则>
1. 动作直接与声明的计划相矛盾（例如，计划说"打开"但动作是"关闭"）
2. 动作针对的对象与计划中指定的不同
3. 动作与实现计划中声明的目标无关
4. 动作序列不遵循计划中概述的逻辑顺序
5. 动作忽略了计划中提到的重要前提条件或约束
</检测准则>

<评估标准>
进行分析时：
1. 应用每个准则：检查步骤是否匹配每个准则中描述的错误模式
2. 关注相关模块：仅考虑计划和动作模块
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
DEFAULT_ACTION_MISALIGNMENT_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(ACTION_MISALIGNMENT_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(ACTION_MISALIGNMENT_PROMPT_ZH),
            ),
        ],
    },
)


class ActionMisalignmentGrader(LLMGrader):
    """
    Action Misalignment Grader

    Evaluates whether the agent executes an action inconsistent with its stated plan or reasoning.

    Required modules: plan, action

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
        >>> grader = ActionMisalignmentGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     plan="I will open drawer 1 to find the key.",
        ...     action="close drawer 1"
        ... )
        >>> print(f"Score: {result.score}")  # 0.0 (error detected)
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Optional[Template] = DEFAULT_ACTION_MISALIGNMENT_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="action_misalignment",
            mode=GraderMode.POINTWISE,
            description="Detect action misalignment errors",
            model=model,
            template=template,
            language=language,
        )
        self.template = (
            template if template is not None else DEFAULT_ACTION_MISALIGNMENT_TEMPLATE
        )

    def _format_trajectory_steps(
        self,
        plan: str,
        action: str,
        history_steps: Optional[list] = None,
    ) -> str:
        """Format trajectory steps for evaluation.
        
        Args:
            plan: Agent's planning/reasoning
            action: Agent's chosen action
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
        lines.append(f"Action: {action}")
        
        return "\n".join(lines)

    async def aevaluate(
        self,
        plan: str,
        action: str,
        history_steps: Optional[list] = None,
        task_context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate action misalignment

        Args:
            plan: Agent's planning/reasoning
            action: Agent's chosen action
            history_steps: Optional list of previous step dictionaries for context
            task_context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = no error, 0.0 = error detected)

        Example:
            >>> result = await grader.aevaluate(
            ...     plan="I will open drawer 1 to find the key.",
            ...     action="close drawer 1",
            ...     task_context="Task: Find the key"
            ... )
        """
        # Format trajectory steps
        trajectory_steps = self._format_trajectory_steps(
            plan=plan,
            action=action,
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
            logger.error(f"Error evaluating action misalignment: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "error_type": "action_misalignment",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ActionMisalignmentGrader",
    "DEFAULT_ACTION_MISALIGNMENT_TEMPLATE",
]

