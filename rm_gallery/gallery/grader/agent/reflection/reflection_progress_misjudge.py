# -*- coding: utf-8 -*-
"""
Reflection Progress Misjudge Grader

Evaluates whether the agent misjudges the progress toward completing the task
in its reflection.
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
REFLECTION_PROGRESS_MISJUDGE_PROMPT_EN = """
You are an expert in analyzing agent behavior. Your task is to detect whether the agent misjudges the progress toward completing the task in its reflection.

<Error Type: Reflection Progress Misjudge>
The agent misjudges the progress toward completing the task in its reflection. This includes overestimating progress (claiming good progress while stuck in a loop) or underestimating progress (believing the task is stalled when actually making progress).
</Error Type>

<Rubrics for Detection>
1. The agent claims to be making progress while observations show it is repeating the same actions
2. The agent believes it is stuck in a loop when it is actually making forward progress
3. The agent overestimates how close it is to task completion based on current state
4. The agent underestimates the progress made when observations show clear advancement
5. The agent fails to recognize it is repeating failed attempts
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Apply each rubric: Check if the step matches the error patterns described in each rubric
2. Focus on relevant modules: Only consider observation and reflection modules
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
REFLECTION_PROGRESS_MISJUDGE_PROMPT_ZH = """
你是一名分析智能体行为的专家。你的任务是检测智能体是否在其反思中误判了完成任务的进度。

<错误类型：反思进度误判>
智能体在其反思中误判了完成任务的进度。这包括高估进度（在陷入循环时声称进展良好）或低估进度（在实际取得进展时认为任务停滞）。
</错误类型>

<检测准则>
1. 智能体声称正在取得进展，而观察显示它正在重复相同的动作
2. 智能体认为自己陷入循环，而实际上正在向前推进
3. 智能体根据当前状态高估了距离任务完成的接近程度
4. 智能体在观察显示明确进展时低估了所取得的进度
5. 智能体未能识别自己正在重复失败的尝试
</检测准则>

<评估标准>
进行分析时：
1. 应用每个准则：检查步骤是否匹配每个准则中描述的错误模式
2. 关注相关模块：仅考虑观察和反思模块
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
DEFAULT_REFLECTION_PROGRESS_MISJUDGE_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(REFLECTION_PROGRESS_MISJUDGE_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(REFLECTION_PROGRESS_MISJUDGE_PROMPT_ZH),
            ),
        ],
    },
)


class ReflectionProgressMisjudgeGrader(LLMGrader):
    """
    Reflection Progress Misjudge Grader

    Evaluates whether the agent misjudges the progress toward completing the task
    in its reflection.

    Required modules: observation, reflection

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
        >>> grader = ReflectionProgressMisjudgeGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     observation="Cabinet 1 is still empty.",
        ...     reflection="Great progress! I'm making excellent headway.",
        ...     task_context="Task: Find apples in cabinets"
        ... )
        >>> print(f"Score: {result.score}")  # 0.0 (error detected)
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Optional[Template] = DEFAULT_REFLECTION_PROGRESS_MISJUDGE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="reflection_progress_misjudge",
            mode=GraderMode.POINTWISE,
            description="Detect reflection progress misjudge errors",
            model=model,
            template=template,
            language=language,
        )
        self.template = (
            template if template is not None else DEFAULT_REFLECTION_PROGRESS_MISJUDGE_TEMPLATE
        )

    def _format_trajectory_steps(
        self,
        observation: str,
        reflection: str,
        history_steps: Optional[list] = None,
    ) -> str:
        """Format trajectory steps for evaluation.
        
        Args:
            observation: Agent's observation from the environment
            reflection: Agent's reflection on the situation
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
        lines.append(f"Observation: {observation}")
        lines.append(f"Reflection: {reflection}")
        
        return "\n".join(lines)

    async def aevaluate(
        self,
        observation: str,
        reflection: str,
        history_steps: Optional[list] = None,
        task_context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate reflection progress misjudge

        Args:
            observation: Agent's observation from the environment
            reflection: Agent's reflection on the situation
            history_steps: Optional list of previous step dictionaries for context
            task_context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = no error, 0.0 = error detected)

        Example:
            >>> result = await grader.aevaluate(
            ...     observation="Cabinet 1 is still empty.",
            ...     reflection="Great progress! I'm making excellent headway.",
            ...     task_context="Task: Find apples in cabinets"
            ... )
        """
        # Format trajectory steps
        trajectory_steps = self._format_trajectory_steps(
            observation=observation,
            reflection=reflection,
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
            logger.error(f"Error evaluating reflection progress misjudge: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "error_type": "reflection_progress_misjudge",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ReflectionProgressMisjudgeGrader",
    "DEFAULT_REFLECTION_PROGRESS_MISJUDGE_TEMPLATE",
]

