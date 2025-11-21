# -*- coding: utf-8 -*-
"""
Memory Over-Simplification Grader

Evaluates whether the agent over-simplifies information when storing it in memory,
losing critical details.
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
MEMORY_OVER_SIMPLIFICATION_PROMPT_EN = """
You are an expert in analyzing agent behavior. Your task is to detect whether the agent over-simplifies information when storing it in memory, losing critical details.

<Error Type: Memory Over-Simplification>
The agent over-simplifies information when storing it in memory, losing critical details such as exact locations, specific values, or important constraints. This makes the stored memory less useful or potentially misleading for future decision-making.
</Error Type>

<Rubrics for Detection>
1. The agent stores vague descriptions when the observation contained specific details
2. The agent omits exact locations, coordinates, or spatial information present in observation
3. The agent loses specific numerical values (quantities, distances, measurements) when storing
4. The agent removes important constraints, conditions, or qualifiers from observed information
5. The agent's memory is too generic to be actionable compared to the detailed observation
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Apply each rubric: Check if the step matches the error patterns described in each rubric
2. Focus on relevant modules: Only consider observation and memory modules
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
MEMORY_OVER_SIMPLIFICATION_PROMPT_ZH = """
你是一名分析智能体行为的专家。你的任务是检测智能体在将信息存储到记忆中时是否过度简化，丢失了关键细节。

<错误类型：记忆过度简化>
智能体在将信息存储到记忆中时过度简化，丢失了关键细节，如确切位置、具体数值或重要约束。这使得存储的记忆变得不太有用，或可能对未来的决策产生误导。
</错误类型>

<检测准则>
1. 智能体在观察包含具体细节时存储了模糊的描述
2. 智能体在存储时省略了观察中存在的确切位置、坐标或空间信息
3. 智能体在存储时丢失了具体的数值（数量、距离、测量值）
4. 智能体从观察到的信息中删除了重要的约束、条件或限定词
5. 与详细的观察相比，智能体的记忆过于笼统，无法采取行动
</检测准则>

<评估标准>
进行分析时：
1. 应用每个准则：检查步骤是否匹配每个准则中描述的错误模式
2. 关注相关模块：仅考虑观察和记忆模块
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
DEFAULT_MEMORY_OVER_SIMPLIFICATION_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(MEMORY_OVER_SIMPLIFICATION_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(MEMORY_OVER_SIMPLIFICATION_PROMPT_ZH),
            ),
        ],
    },
)


class MemoryOverSimplificationGrader(LLMGrader):
    """
    Memory Over-Simplification Grader

    Evaluates whether the agent over-simplifies information when storing it in memory,
    losing critical details.

    Required modules: observation, memory

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
        >>> grader = MemoryOverSimplificationGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     observation="Cabinet 1 at coordinates (3.5, 2.1) contains 5 red apples.",
        ...     memory="Found some apples in a cabinet."
        ... )
        >>> print(f"Score: {result.score}")  # 0.0 (error detected)
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Optional[Template] = DEFAULT_MEMORY_OVER_SIMPLIFICATION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="memory_over_simplification",
            mode=GraderMode.POINTWISE,
            description="Detect memory over-simplification errors",
            model=model,
            template=template,
            language=language,
        )
        self.template = (
            template if template is not None else DEFAULT_MEMORY_OVER_SIMPLIFICATION_TEMPLATE
        )

    def _format_trajectory_steps(
        self,
        observation: str,
        memory: str,
        history_steps: Optional[list] = None,
    ) -> str:
        """Format trajectory steps for evaluation.
        
        Args:
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
        lines.append(f"Observation: {observation}")
        lines.append(f"Memory: {memory}")
        
        return "\n".join(lines)

    async def aevaluate(
        self,
        observation: str,
        memory: str,
        history_steps: Optional[list] = None,
        task_context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate memory over-simplification

        Args:
            observation: Agent's observation from the environment
            memory: Agent's memory content
            history_steps: Optional list of previous step dictionaries for context
            task_context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = no error, 0.0 = error detected)

        Example:
            >>> result = await grader.aevaluate(
            ...     observation="Cabinet 1 at coordinates (3.5, 2.1) contains 5 red apples.",
            ...     memory="Found some apples in a cabinet.",
            ...     task_context="Task: Inventory items with precise locations"
            ... )
        """
        # Format trajectory steps
        trajectory_steps = self._format_trajectory_steps(
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
            logger.error(f"Error evaluating memory over-simplification: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "error_type": "memory_over_simplification",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "MemoryOverSimplificationGrader",
    "DEFAULT_MEMORY_OVER_SIMPLIFICATION_TEMPLATE",
]

