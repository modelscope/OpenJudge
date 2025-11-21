# -*- coding: utf-8 -*-
"""
Memory Retrieval Failure Grader

Evaluates whether the agent fails to retrieve relevant information from memory when needed.
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
MEMORY_RETRIEVAL_FAILURE_PROMPT_EN = """
You are an expert in analyzing agent behavior. Your task is to detect whether the agent fails to retrieve information from memory when needed.

<Error Type: Memory Retrieval Failure>
The agent fails to retrieve information from memory when needed, cannot access information that should be present, or retrieves incorrect/outdated information. This leads to the agent making plans based on incomplete information or repeating already-performed actions.
</Error Type>

<Rubrics for Detection>
1. The agent's plan ignores relevant information that should be in memory from previous observations
2. The agent proposes actions that were already tried, suggesting failure to recall past attempts
3. The agent plans to search for information that was already discovered and should be in memory
4. The agent's plan contradicts facts that should be stored in memory from earlier steps
5. The agent retrieves outdated or incorrect information instead of the most recent memory state
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
MEMORY_RETRIEVAL_FAILURE_PROMPT_ZH = """
你是一名分析智能体行为的专家。你的任务是检测智能体在需要时是否未能从记忆中检索信息。

<错误类型：记忆检索失败>
智能体在需要时未能从记忆中检索信息，无法访问应该存在的信息，或检索了错误/过时的信息。这导致智能体基于不完整的信息制定计划或重复已执行的动作。
</错误类型>

<检测准则>
1. 智能体的计划忽略了记忆中应该存在的来自先前观察的相关信息
2. 智能体提出了已经尝试过的动作，表明未能回忆起过去的尝试
3. 智能体计划搜索已经发现并应该在记忆中的信息
4. 智能体的计划与早期步骤中应该存储在记忆中的事实相矛盾
5. 智能体检索了过时或错误的信息，而不是最新的记忆状态
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
DEFAULT_MEMORY_RETRIEVAL_FAILURE_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(MEMORY_RETRIEVAL_FAILURE_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(MEMORY_RETRIEVAL_FAILURE_PROMPT_ZH),
            ),
        ],
    },
)


class MemoryRetrievalFailureGrader(LLMGrader):
    """
    Memory Retrieval Failure Grader

    Evaluates whether the agent fails to retrieve relevant information from memory when needed.

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
        >>> grader = MemoryRetrievalFailureGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     plan="I will search for the key in drawer 1.",
        ...     observation="You are standing in the room.",
        ...     memory="The key was found in drawer 1 in step 3."
        ... )
        >>> print(f"Score: {result.score}")  # 0.0 (error detected)
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        template: Optional[Template] = DEFAULT_MEMORY_RETRIEVAL_FAILURE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="memory_retrieval_failure",
            mode=GraderMode.POINTWISE,
            description="Detect memory retrieval failure errors",
            model=model,
            template=template,
            language=language,
        )
        self.template = (
            template if template is not None else DEFAULT_MEMORY_RETRIEVAL_FAILURE_TEMPLATE
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
        Evaluate memory retrieval failure

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
            ...     plan="I will search for the key in drawer 1.",
            ...     observation="You are standing in the room.",
            ...     memory="The key was found in drawer 1 in step 3.",
            ...     task_context="Task: Find and use the key"
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
            logger.error(f"Error evaluating memory retrieval failure: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "error_type": "memory_retrieval_failure",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "MemoryRetrievalFailureGrader",
    "DEFAULT_MEMORY_RETRIEVAL_FAILURE_TEMPLATE",
]

