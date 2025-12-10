# -*- coding: utf-8 -*-
"""
Tool Selection Grader

Evaluates the tool selection made by the agent to address the user query.
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

# English Prompt
TOOL_SELECTION_PROMPT_EN = """
You are an expert in analyzing tool selection decisions. Your task is to evaluate the  of tool selection made by an agent to address the user query.

<Evaluation Dimension: Tool Selection >
Evaluate whether the agent selected the most appropriate tool(s) from the available tools to effectively address the user's query. This includes assessing relevance, completeness, and efficiency of tool selection.
</Evaluation Dimension>

<Rubrics for Evaluation>
1. The selected tool is highly relevant to the user's query and request
2. The tool selection directly addresses the core intent of the query
3. All necessary tools are selected (no critical tools missing)
4. No unnecessary or irrelevant tools are selected
5. The tool selection is efficient (avoids redundant tool calls)
6. The tool is capable of providing the information or performing the action requested
7. The tool selection demonstrates good understanding of tool capabilities and limitations
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Assess relevance: Does the selected tool match the query intent?
2. Check completeness: Are all necessary tools selected?
3. Evaluate efficiency: Are there unnecessary or redundant tool selections?
4. Verify capability: Can the selected tool(s) fulfill the user's request?
5. Consider alternatives: Are there better tool choices available?
</Evaluation Criteria>

<user_query>
{user_query}
</user_query>

<available_tools>
{available_tools}
</available_tools>

<selected_tools>
{selected_tools}
</selected_tools>

{context_section}

# Scoring Instructions
Use a scale from 1 to 5:
- 5: Perfect tool selection - highly relevant, complete, and efficient
- 4: Good tool selection - relevant with minor inefficiencies
- 3: Acceptable tool selection - partially addresses the query
- 2: Poor tool selection - mostly irrelevant or incomplete
- 1: Completely wrong tool selection - fails to address the query

Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 1 and 5>,
    "reason": "<detailed explanation of tool selection quality, including strengths and weaknesses>"
}}

JSON:
"""

# Chinese Prompt
TOOL_SELECTION_PROMPT_ZH = """
你是一名分析工具选择决策的专家。你的任务是评估智能体为解决用户查询而做出的工具选择的质量。

<评估维度：工具选择质量>
评估智能体是否从可用工具中选择了最合适的工具来有效地解决用户的查询。这包括评估工具选择的相关性、完整性和效率。
</评估维度>

<评估准则>
1. 选择的工具与用户的查询和请求高度相关
2. 工具选择直接针对查询的核心意图
3. 选择了所有必要的工具（没有遗漏关键工具）
4. 没有选择不必要或不相关的工具
5. 工具选择是高效的（避免冗余的工具调用）
6. 工具能够提供请求的信息或执行请求的操作
7. 工具选择表现出对工具能力和局限性的良好理解
</评估准则>

<评估标准>
进行分析时：
1. 评估相关性：选择的工具是否与查询意图匹配？
2. 检查完整性：是否选择了所有必要的工具？
3. 评估效率：是否有不必要或冗余的工具选择？
4. 验证能力：选择的工具是否能够满足用户的请求？
5. 考虑替代方案：是否有更好的工具选择可用？
</评估标准>

<user_query>
{user_query}
</user_query>

<available_tools>
{available_tools}
</available_tools>

<selected_tools>
{selected_tools}
</selected_tools>

{context_section}

# 评分指令
使用 1 到 5 的评分标准：
- 5：完美的工具选择 - 高度相关、完整且高效
- 4：良好的工具选择 - 相关但有轻微的低效
- 3：可接受的工具选择 - 部分解决了查询
- 3：较差的工具选择 - 大多不相关或不完整
- 1：完全错误的工具选择 - 无法解决查询

请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <1 到 5 之间的整数>,
    "reason": "<关于工具选择质量的详细解释，包括优点和缺点>"
}}

JSON:
"""

# Build default template from prompts
DEFAULT_TOOL_SELECTION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TOOL_SELECTION_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TOOL_SELECTION_PROMPT_ZH),
            ),
        ],
    },
)


class ToolSelectionGrader(LLMGrader):
    """
    Tool Selection Grader

    Evaluates the tool selection made by the agent to address the user query.

    Attributes:
        name: Grader name
        model: BaseChatModel instance for evaluation
        template: Evaluation template
        language: Language for evaluation prompts (default: LanguageEnum.EN)
        threshold: threshold [0, 1] (default: 0.7)

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.core.schema.template import LanguageEnum
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",  # pragma: allowlist secret
        ...     model="qwen3-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>>
        >>> grader = ToolSelectionGrader(
        ...     model=api,
        ...     threshold=0.7,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     user_query="Find all Python files modified in the last week",
        ...     available_tools="search_files, list_directory, get_file_info, git_log",
        ...     selected_tools="search_files, git_log"
        ... )
        >>> print(f"Score: {result.score}")  # High score for good selection
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 0.7,
        template: Optional[PromptTemplate] = DEFAULT_TOOL_SELECTION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="tool_selection",
            mode=GraderMode.POINTWISE,
            description="Evaluate tool selection ",
            model=model,
            template=template,
            language=language,
        )
        self.threshold = threshold
        self.template = template if template is not None else DEFAULT_TOOL_SELECTION_TEMPLATE

    async def aevaluate(
        self,
        query: Union[str, List[Dict[str, Any]]],
        tool_definitions: Union[Dict[str, Any], List[Dict[str, Any]]],
        tool_calls: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> GraderScore:
        """
        Evaluate tool selection

        Args:
            query: Query or chat history. Can be a string for simple queries or a list of
                  message dictionaries for conversation history.
            tool_definitions: List of all available tool definitions.
                             Each definition includes name, description, and parameters.
            tool_calls: List of tool calls actually made by the agent.
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score from 0.0 to 1.0 indicating tool selection quality

        Example:
            >>> conversation = [
            ...     {"role": "user", "content": "Search for configuration files"}
            ... ]
            >>> tool_defs = [
            ...     {"name": "search_files", "description": "Search files"},
            ...     {"name": "read_file", "description": "Read a file"}
            ... ]
            >>> tool_calls = [{"name": "search_files", "arguments": {...}}]
            >>> result = await grader.aevaluate(
            ...     query=conversation,
            ...     tool_definitions=tool_defs,
            ...     tool_calls=tool_calls
            ... )
        """

        # Ensure tool_calls and tool_definitions are lists
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        if not isinstance(tool_definitions, list):
            tool_definitions = [tool_definitions] if tool_definitions else []

        # Format query as string for the prompt
        if isinstance(query, list):
            user_query = "\n".join(
                [f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in query],
            )
        else:
            user_query = str(query)

        # Format available tools
        available_tools = json.dumps(tool_definitions, indent=2)

        # Format selected tools (extract tool names from tool_calls)
        selected_tools = json.dumps(
            [{"name": tc.get("name"), "arguments": tc.get("arguments", {})} for tc in tool_calls],
            indent=2,
        )

        try:
            result = await super().aevaluate(
                user_query=user_query,
                available_tools=available_tools,
                selected_tools=selected_tools,
                context_section="",
            )
            score = result.score
            reason = result.reason

            # Normalize score to [0, 1] range if needed
            normalized_score = max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Error evaluating tool selection: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "threshold": self.threshold,
            "raw_score": score,
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ToolSelectionGrader",
    "DEFAULT_TOOL_SELECTION_TEMPLATE",
]
