"""
Check tool call format including think, answer and tool_call tags with JSON validation.
"""

import json
import re
from typing import Any

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderScore


class ReasoningToolCallFormatGrader(BaseGrader):
    """
    Check tool call format including think, answer and tool_call tags with JSON validation.

    This reward verifies if the response content follows the required format
    with proper <think>, <answer> and <tool_call> tags, including JSON validation
    for tool calls.
    """

    def __init__(self) -> None:
        super().__init__(
            name="tool_call_format",
            mode=GraderMode.POINTWISE,
            description="Check tool call format including think, answer and tool_call tags with JSON validation.",
        )

    # pylint: disable=too-many-statements
    async def aevaluate(self, response: str, **kwargs: Any) -> GraderScore:
        """
        Check tool call format and calculate reward score.

        This method evaluates if the given answer follows the required format with proper
        , <answer> and  tags. It validates two possible formats:
        1.  + <answer> - Reasoning with final answer format
        2.  +  - Reasoning with tool calls format

        For the tool call format, it also validates that the content within the  tags
        is valid JSON with required 'name' and 'arguments' fields.

        Args:
            response: The response text to evaluate for proper formatting.
            **kwargs: Additional keyword arguments (not used in this implementation).

        Returns:
            GraderScore: A GraderScore object containing:
                - score: 1.0 if format is valid, 0.0 otherwise
                - reason: Explanation of the evaluation result
                - extra_data: Dictionary with detailed metadata including:
                    * has_think_tag: Whether  tags are present
                    * has_answer_tag: Whether <answer> tags are present
                    * has_tool_call_tag: Whether  tags are present
                    * valid_format: Whether overall format is valid
                    * valid_tool_call_json: Whether tool call JSON is valid
                    * tool_call_count: Number of tool calls found
                    * reward: The calculated reward score

        Examples:
            >>> grader = ReasoningToolCallFormatGrader()
            >>> result = await grader.aevaluate("思考过程</think>\\n<answer>最终答案</answer>")
            >>> print(result.score)
            1.0

            >>> result = await grader.aevaluate("思考过程</think>\\n<function_call>
            {\\"name\\": \\"func\\", \\"arguments\\": {\\"arg1\\": \\"value1\\"}}</function_call>")
            >>> print(result.score)
            1.0
        """

        # Extract tag contents
        think_pattern = r"<think>(.*?)</think>"
        answer_pattern = r"<answer>(.*?)</answer>"
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"

        think_matches = re.search(think_pattern, response, re.DOTALL)
        answer_matches = re.search(answer_pattern, response, re.DOTALL)
        tool_call_matches = re.findall(tool_call_pattern, response, re.DOTALL)

        has_think_tag = think_matches is not None
        has_answer_tag = answer_matches is not None
        has_tool_call_tag = len(tool_call_matches) > 0

        valid_format = False
        valid_tool_call_json = False
        reasons = []

        if has_think_tag:
            # Case 1: <think></think> + <answer></answer>
            if has_answer_tag and not has_tool_call_tag:
                # Check overall format
                format_pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
                valid_format = bool(
                    re.match(format_pattern, response, re.DOTALL),
                )

                # Check tag occurrence count
                if valid_format:
                    valid_format = (
                        response.count("<think>") == 1
                        and response.count("</think>") == 1
                        and response.count("<answer>") == 1
                        and response.count("</answer>") == 1
                    )

                if valid_format:
                    reasons.append(
                        "Valid <think></think> + <answer></answer> format",
                    )
                else:
                    reasons.append(
                        "Invalid <think></think> + <answer></answer> format",
                    )

            # Case 2: <think></think> + <tool_call></tool_call>
            elif has_tool_call_tag and not has_answer_tag:
                # Check overall format
                format_pattern = r"^\s*<think>.*?</think>\s*(?:<tool_call>.*?</tool_call>\s*)+$"
                valid_format = bool(
                    re.match(format_pattern, response, re.DOTALL),
                )

                # Check <think> tag occurrence count
                if valid_format:
                    valid_format = response.count("<think>") == 1 and response.count("</think>") == 1

                # Check if <tool_call> and </tool_call> tags appear in pairs
                if valid_format:
                    if response.count("<tool_call>") != response.count(
                        "</tool_call>",
                    ):
                        valid_format = False

                # Check for consecutive duplicate tags
                if valid_format:
                    if re.search(
                        r"</tool_call>\s*</tool_call>",
                        response,
                    ) or re.search(
                        r"<tool_call>\s*<tool_call>",
                        response,
                    ):
                        valid_format = False

                # Check tool_call JSON format
                valid_tool_call_json = True
                tool_calls = []
                if valid_format:
                    for tool_call_content in tool_call_matches:
                        try:
                            tool_call_json = json.loads(
                                tool_call_content.strip(),
                            )
                            # Check if JSON contains required fields
                            if not ("name" in tool_call_json and "arguments" in tool_call_json):
                                valid_tool_call_json = False
                                break
                            tool_calls.append(
                                {
                                    "function": {
                                        "name": tool_call_json["name"],
                                        "arguments": json.dumps(
                                            tool_call_json["arguments"],
                                            ensure_ascii=False,
                                        ),
                                    },
                                },
                            )
                        except json.JSONDecodeError:
                            valid_tool_call_json = False
                            break

                valid_format = valid_format and valid_tool_call_json

                if valid_format:
                    reasons.append(
                        "Valid <think></think> + <tool_call></tool_call> format with valid JSON",
                    )
                else:
                    if not valid_tool_call_json:
                        reasons.append(
                            "Invalid JSON format in <tool_call> tags",
                        )
                    else:
                        reasons.append(
                            "Invalid <think></think> + <tool_call></tool_call> format",
                        )
            else:
                # Has both answer and tool_call, or neither
                reasons.append(
                    "Invalid combination: should have either <answer> or <tool_call> tags, not both or neither",
                )
        else:
            reasons.append("Missing <think></think> tags")

        # Calculate reward score
        reward = 1.0 if valid_format else 0.0
        return GraderScore(
            name=self.name,
            score=reward,
            reason="; ".join(reasons),
            metadata={
                "has_think_tag": has_think_tag,
                "has_answer_tag": has_answer_tag,
                "has_tool_call_tag": has_tool_call_tag,
                "valid_format": valid_format,
                "valid_tool_call_json": valid_tool_call_json,
                "tool_call_count": len(tool_call_matches),
                "reward": reward,
            },
        )
