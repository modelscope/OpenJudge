# -*- coding: utf-8 -*-
"""Tool call success grader for evaluating agent tool call success."""

import json
import re
from typing import Any, Dict, List, Union

from rm_gallery.core.grader.base import GraderMode, GraderScore, LLMGrader
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template

# Tool call success evaluation system prompt
TOOL_CALL_SUCCESS_SYSTEM_PROMPT = """You are an expert evaluator with strong software \
development background. You are required to extract the tool result for every tool call \
then decide for each tool result whether it indicates that the tool call succeeded or failed.

ROLE
====
You are a judge on tool call success who assesses **each tool call made by an AI agent and \
decide if the result of the tool call indicates a success or failure**. You only care about \
technical errors , failures and exceptions , not the business correctness of the tool \
implementation.

You are NOT evaluating:
- The parameters passed to the tool
- The rationale behind choosing this tool
- Whether the successfully returned result from the tool is correct or not business-wise \
given the tool name and definition

You **ARE ONLY** evaluating:
-Whether tool results indicate the presence of a technical error

EVALUATION FRAMEWORK
====================

A tool call is considered successful if:
1. The tool executed without technical errors
2. The tool returned a non-empty result
3. The result is not an error message or exception

A tool call is considered failed if any of the following occurs:
1. The tool result is empty or null
2. The tool result contains explicit error messages
3. The tool result indicates timeout or exception
4. The tool result is an empty object when a populated object is expected"""

# Tool call success evaluation user prompt
TOOL_CALL_SUCCESS_USER_PROMPT = """INPUT
=====

TOOL_DEFINITIONS: {tool_definitions}
TOOL_CALLS: {tool_calls}

TOOL_CALLS is a list of tool calls that were produced by the AI agent. It includes calls \
together with the result of every tool call.
TOOL_DEFINITIONS is a list of definitions for the tools that were called. This definition \
can contain a description of functionality provided by the tool, the parameters that the \
tool accepts and the expected return of the tool. This definition can contribute to the \
assessment of whether a tool call succeeded or failed.

EXPECTED OUTPUT
===============
Generate a JSON object with the following structure:
```json
{{
  "reason": "Brief explanation of why the tool calls succeeded or failed",
  "score": 1.0 for success or 0.0 for failure
}}
```"""


class ToolCallSuccessGrader(LLMGrader):
    """Evaluates whether tool calls done by an AI agent includes failures or not.

    This evaluator focuses solely on tool call results and tool definitions, disregarding user's query to
    the agent, conversation history and agent's final response. Although tool definitions is optional,
    providing them can help the evaluator better understand the context of the tool calls made by the
    agent. Please note that this evaluator validates tool calls for potential technical failures like
    errors, exceptions, timeouts and empty results (only in cases where empty results could indicate a
    failure). It does not assess the correctness or the tool result itself, like mathematical errors and
    unrealistic field values like name="668656".

    Scoring is binary:
    - TRUE (1.0): All tool calls were successful
    - FALSE (0.0): At least one tool call failed

    Example:
        >>> import asyncio
        >>> grader = ToolCallSuccessGrader()
        >>> tool_definitions = [
        ...     {
        ...         "name": "get_weather",
        ...         "description": "Get weather information for a location",
        ...         "parameters": {
        ...             "location": "City name"
        ...         }
        ...     }
        ... ]
        >>> tool_calls = [
        ...     {
        ...         "name": "get_weather",
        ...         "arguments": {"location": "New York"},
        ...         "result": {"temperature": 25, "condition": "sunny"}
        ...     }
        ... ]
        >>> result = asyncio.run(grader.aevaluate(tool_definitions=tool_definitions, tool_calls=tool_calls))
        >>> print(result.score)
        1.0
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        **kwargs: Any,
    ):
        """Initialize the ToolCallSuccessGrader.

        Args:
            model: The language model used for evaluation. Can be either a ChatModelBase
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            **kwargs: Additional keyword arguments.
        """

        super().__init__(
            name="tool_call_success",
            mode=GraderMode.POINTWISE,
            description="Evaluates whether tool calls done by an AI agent includes failures or not",
            template=Template(
                messages=[
                    ChatMessage(
                        role="system",
                        content=TOOL_CALL_SUCCESS_SYSTEM_PROMPT,
                    ),
                    ChatMessage(
                        role="user",
                        content=TOOL_CALL_SUCCESS_USER_PROMPT,
                    ),
                ],
            ),
            model=model,
            **kwargs,
        )

    def _parse_tools_from_response(
        self,
        response: str,
    ) -> List[Dict[str, Any]]:
        """Extract tool calls from the response.

        Args:
            response: The response string to extract tool calls from.

        Returns:
            List of parsed tool calls.
        """
        tool_calls = []

        # Pattern to match tool calls in JSON format
        tool_call_pattern = (
            r'\{\s*"name"\s*:\s*"[^"]*"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}'
        )
        matches = re.findall(tool_call_pattern, response, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                # Skip invalid JSON
                continue

        return tool_calls

    async def aevaluate(
        self,
        tool_definitions: Union[Dict[str, Any], List[Dict[str, Any]]],
        tool_calls: Union[Dict[str, Any], List[Dict[str, Any]]],
        **kwargs: Any,
    ) -> GraderScore:
        """Evaluate tool call success. Accepts tool definitions and tool calls for evaluation.

        This method evaluates whether all tool calls were technically successful, focusing only on
        technical aspects like errors, exceptions, timeouts, and empty results. It does not assess
        the correctness of the tool results themselves.

        Args:
            tool_definitions: List of tool definitions whose calls are being evaluated.
                             Each definition typically includes name, description, and parameters.
            tool_calls: List of tool calls with results to evaluate. Each call should include
                       the tool name, arguments, and result.
            **kwargs: Additional keyword arguments passed to the underlying evaluation model.

        Returns:
            GraderScore with the evaluation result containing:
                - score: A numerical score (1.0 for success, 0.0 for failure)
                - reason: Explanation of how the score was determined
                - metadata: Additional evaluation information including failed tool names

        Example:
            >>> import asyncio
            >>> grader = ToolCallSuccessGrader()
            >>> tool_defs = [{"name": "calculator", "description": "Performs calculations"}]
            >>> tool_calls = [
            ...     {
            ...         "name": "calculator",
            ...         "arguments": {"expression": "2+2"},
            ...         "result": {"value": 4}
            ...     }
            ... ]
            >>> result = asyncio.run(grader.aevaluate(tool_definitions=tool_defs, tool_calls=tool_calls))
            >>> print(f"Score: {result.score}, Reason: {result.reason}")
            Score: 1.0, Reason: All tool calls were successful
        """
        # Ensure tool_calls and tool_definitions are lists
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        if not isinstance(tool_definitions, list):
            tool_definitions = [tool_definitions] if tool_definitions else []

        # Call parent evaluate method with the structured data
        result = await super().aevaluate(
            tool_calls=json.dumps(tool_calls, indent=2),
            tool_definitions=json.dumps(tool_definitions, indent=2),
            **kwargs,
        )

        # Process and normalize the result
        if hasattr(result, "score"):
            # Ensure score is binary (0.0 or 1.0)
            score = 1.0 if result.score >= 0.5 else 0.0
            result.score = score

        return result
