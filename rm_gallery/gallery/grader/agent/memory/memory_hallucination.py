# -*- coding: utf-8 -*-
"""
Memory Hallucination Grader

Evaluates whether the agent stores false or fabricated information in its memory module.
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
MEMORY_HALLUCINATION_PROMPT_EN = """
You are an expert in analyzing agent behavior. Your task is to detect whether the agent stores false or fabricated information in its memory module.

<Error Type: Memory Hallucination>
The agent stores false or fabricated information in its memory module. This includes recording information that was never observed, storing incorrect interpretations as facts, or fabricating details about objects or states.
</Error Type>

<Rubrics for Detection>
1. The agent stores information in memory that was not present in the observation
2. The agent records fabricated details (colors, quantities, locations) not mentioned in observation
3. The agent saves incorrect interpretations of observations as factual memories
4. The agent creates false associations or relationships not supported by observations
5. The agent's memory contains information that contradicts or goes beyond what was observed
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
MEMORY_HALLUCINATION_PROMPT_ZH = """
你是一名分析智能体行为的专家。你的任务是检测智能体是否在其记忆模块中存储了虚假或捏造的信息。

<错误类型：记忆幻觉>
智能体在其记忆模块中存储了虚假或捏造的信息。这包括记录从未观察到的信息、将错误的解释存储为事实，或捏造有关对象或状态的细节。
</错误类型>

<检测准则>
1. 智能体在记忆中存储了观察中不存在的信息
2. 智能体记录了观察中未提及的捏造细节（颜色、数量、位置）
3. 智能体将对观察的错误解释保存为事实记忆
4. 智能体创建了观察不支持的虚假关联或关系
5. 智能体的记忆包含了与观察相矛盾或超出观察范围的信息
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
DEFAULT_MEMORY_HALLUCINATION_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(MEMORY_HALLUCINATION_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(MEMORY_HALLUCINATION_PROMPT_ZH),
            ),
        ],
    },
)


class MemoryHallucinationGrader(LLMGrader):
    """
    Memory Hallucination Grader

    Evaluates whether the agent stores false or fabricated information in its memory module.

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
        >>> grader = MemoryHallucinationGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     observation="You see a closed cabinet.",
        ...     memory="There is a red vase inside the cabinet."
        ... )
        >>> print(f"Score: {result.score}")  # 0.0 (error detected)
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Optional[Template] = DEFAULT_MEMORY_HALLUCINATION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="memory_hallucination",
            mode=GraderMode.POINTWISE,
            description="Detect memory hallucination errors",
            model=model,
            template=template,
            language=language,
        )
        self.template = (
            template if template is not None else DEFAULT_MEMORY_HALLUCINATION_TEMPLATE
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
        Evaluate memory hallucination

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
            ...     observation="You see a closed cabinet.",
            ...     memory="There is a red vase inside the cabinet.",
            ...     task_context="Task: Inventory room objects"
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
            logger.error(f"Error evaluating memory hallucination: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "error_type": "memory_hallucination",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "MemoryHallucinationGrader",
    "DEFAULT_MEMORY_HALLUCINATION_TEMPLATE",
]

