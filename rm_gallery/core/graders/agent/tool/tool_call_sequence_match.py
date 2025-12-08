# -*- coding: utf-8 -*-
"""
Tool Call Sequence Grader
This module provides graders for evaluating tool call sequences against
ground truth references, supporting both strict and loose matching modes.
"""
import json
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from loguru import logger

from rm_gallery.core.graders.base_grader import BaseGrader, GraderMode, GraderScore


class ToolCallSequenceMatchGrader(BaseGrader):
    """
    Tool call sequence ground truth matching grader.
    This grader evaluates whether the model's tool call sequence matches the ground truth
    expected sequence by comparing predicted tool calls against reference tool calls.
    **Strict mode**: Matches both tool name and parameters, using F1 score calculation.
    **Loose mode**: Only matches tool name, checking whether model tool list is a subset of ground truth.
    Attributes:
        strict_mode: If True, matches both tool name and arguments; if False, only matches tool name
        use_jaccard_similarity: If True, use Jaccard similarity for loose mode (ignores step order)
    Example:
        >>> grader = ToolCallSequenceMatchGrader(strict_mode=True)
        >>> result = await grader.aevaluate(
        ...     messages=[...],  # Model's messages with tool calls
        ...     ground_truth_tool_calls=[...]  # Ground truth reference tool calls
        ... )
        >>> print(f"Sequence match score: {result.score}")
    """

    def __init__(
        self,
        strict_mode: bool = True,
        use_jaccard_similarity: bool = True,
        **kwargs,
    ):
        super().__init__(
            name="tool_call_sequence",
            mode=GraderMode.POINTWISE,
            description="Evaluate tool call sequence matching against ground truth",
            **kwargs,
        )
        self.strict_mode = strict_mode
        self.use_jaccard_similarity = use_jaccard_similarity

    def extract_predicted_tool_sequence(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract the predicted tool call sequence from model messages, organized by steps.
        Args:
            messages: List of message dicts containing the model's messages, including tool calls
        Returns:
            A dictionary mapping step numbers to lists of tool call information within that step
        """
        step_tools = {}
        step_idx = 0
        for message in messages:
            role = message.get("role", "")
            tool_calls = message.get("tool_calls", [])
            # Process tool calls only from messages with role 'assistant'
            if role == "assistant" and tool_calls:
                step_tools[step_idx] = []
                for chat_tool in tool_calls:
                    # Parse the tool call arguments
                    function = chat_tool.get("function", {})
                    raw_args = function.get("arguments", "")
                    try:
                        params = json.loads(raw_args)
                    except json.JSONDecodeError:
                        params = {}
                    # Prepare the tool information
                    tool_info = {
                        "name": function.get("name", ""),
                        "parameters": params,
                    }
                    step_tools[step_idx].append(tool_info)
                # Increment the step index for the next step
                step_idx += 1
        return step_tools

    def extract_ground_truth_tool_sequence(
        self,
        ground_truth_tool_calls: List[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extract ground truth tool call sequence from reference tool calls, organized by steps.
        Args:
            ground_truth_tool_calls: Ground truth tool call list in format:
                [
                    {
                        "step": 0,
                        "tool": [
                            {
                                "name": "search",
                                "parameters": {...}
                            }
                        ]
                    }
                ]
        Returns:
            Dictionary mapping step numbers to lists of tool calls within that step
        """
        step_tools = {}
        for step_info in ground_truth_tool_calls:
            step = step_info.get("step", 0)
            tools = step_info.get("tool", [])
            if step not in step_tools:
                step_tools[step] = []
            for tool in tools:
                tool_info = {
                    "name": tool.get("name", ""),
                    "parameters": tool.get("parameters", {}),
                }
                step_tools[step].append(tool_info)
        return step_tools

    def create_tool_elements(self, tool: Dict[str, Any]) -> Set[str]:
        """
        Create elements for tool matching based on matching mode.
        Args:
            tool: Tool information dictionary
        Returns:
            Set of elements representing the tool
        """
        elements = set()
        # Add flattened parameters
        params = tool.get("parameters", {})
        for key, value in params.items():
            elements.add(str(key))
            elements.add(str(value))
        return elements

    def calculate_param_similarity(
        self,
        ref_tool: Dict[str, Any],
        model_tool: Dict[str, Any],
    ) -> float:
        """
        Calculate F1 score for complete tool parameters matching.
        Args:
            ref_tool: Ground truth tool with name and parameters
            model_tool: Model tool with name and parameters
        Returns:
            Float similarity score between 0 and 1
        """
        ref_elements = self.create_tool_elements(ref_tool)
        model_elements = self.create_tool_elements(model_tool)
        # Handle edge cases
        if not ref_elements and not model_elements:
            return 1.0
        if not ref_elements or not model_elements:
            return 0.0
        # Calculate precision, recall, and F1
        true_positives = len(ref_elements & model_elements)
        precision = true_positives / len(model_elements) if model_elements else 0.0
        recall = true_positives / len(ref_elements) if ref_elements else 0.0
        if precision + recall == 0:
            return 0.0
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def is_subsequence_unordered(
        self,
        list1: List[str],
        list2: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Check if all elements of list1 are present in list2, regardless of order.
        Args:
            list1: Target list (ground truth)
            list2: Source list (model output)
        Returns:
            Tuple of (is_subsequence, missing_elements)
        """
        counter1 = Counter(list1)
        counter2 = Counter(list2)
        missing_elements = []
        for elem, count in counter1.items():
            if counter2.get(elem, 0) < count:
                missing_elements.extend([elem] * (count - counter2.get(elem, 0)))
        is_subsequence = len(missing_elements) == 0
        return is_subsequence, missing_elements

    def calculate_step_matching_score(
        self,
        predicted_tool_steps: Dict[int, List[Dict[str, Any]]],
        ground_truth_tool_steps: Dict[int, List[Dict[str, Any]]],
    ) -> float:
        """
        Calculate step matching score by comparing each step between predicted and ground truth tool calls.
        Uses F1 score for parameter matching in strict mode and improved scoring based on missing tools.
        Args:
            predicted_tool_steps: Model's predicted tool calls organized by steps
            ground_truth_tool_steps: Ground truth reference tool calls organized by steps
        Returns:
            Step matching score (0.0 - 1.0) based on proportion of matched steps
        """
        if not ground_truth_tool_steps:
            return 1.0 if not predicted_tool_steps else 0.0
        total_score = 0.0
        total_steps = len(ground_truth_tool_steps)
        # Iterate through each step in ground_truth_tool_steps
        for step_index, ground_truth_tools in ground_truth_tool_steps.items():
            step_score = 0.0
            # Check if model has the corresponding step
            if step_index in predicted_tool_steps:
                predicted_tools = predicted_tool_steps[step_index]
                # Extract tool names
                gt_tool_names = [tool.get("name", "") for tool in ground_truth_tools]
                pred_tool_names = [tool.get("name", "") for tool in predicted_tools]
                # Use unordered matching to check tools in this step
                is_match, missing = self.is_subsequence_unordered(
                    gt_tool_names,
                    pred_tool_names,
                )
                if self.strict_mode:
                    if not is_match:
                        # Tool names don't match, score is 0
                        step_score = 0.0
                    else:
                        # Tool names match, calculate F1 score for parameters
                        # For each ground truth tool, find matching predicted tool by name and calculate F1
                        tool_scores = []
                        for gt_tool in ground_truth_tools:
                            gt_name = gt_tool.get("name", "")
                            # Find matching predicted tool by name
                            matching_pred_tool = None
                            for pred_tool in predicted_tools:
                                if pred_tool.get("name", "") == gt_name:
                                    matching_pred_tool = pred_tool
                                    break

                            if matching_pred_tool is not None:
                                # Calculate F1 score for this tool pair
                                tool_f1 = self.calculate_param_similarity(
                                    gt_tool,
                                    matching_pred_tool,
                                )
                                tool_scores.append(tool_f1)
                            else:
                                # Ground truth tool not found in predictions, score is 0
                                tool_scores.append(0.0)

                        # Average F1 score across all ground truth tools in this step
                        step_score = sum(tool_scores) / len(tool_scores) if tool_scores else 0.0
                else:
                    # In loose mode, calculate step score based on the ratio of matched tools
                    if len(gt_tool_names) > 0:
                        matched_count = len(gt_tool_names) - len(missing)
                        step_score = matched_count / len(gt_tool_names)
                    else:
                        step_score = 1.0
            else:
                step_score = 0.0  # No matching step in model
            total_score += step_score
        return total_score / total_steps if total_steps > 0 else 0.0

    def calculate_jaccard_similarity_score(
        self,
        predicted_tool_steps: Dict[int, List[Dict[str, Any]]],
        ground_truth_tool_steps: Dict[int, List[Dict[str, Any]]],
    ) -> Tuple[float, Set[str], Set[str]]:
        """
        Calculate Jaccard similarity score for tool calls, ignoring step order.
        Treats all tool calls as a set and computes intersection over union.
        Args:
            predicted_tool_steps: Model's predicted tool calls organized by steps
            ground_truth_tool_steps: Ground truth reference tool calls organized by steps
        Returns:
            Tuple of (jaccard_score, intersection_set, union_set)
        """
        if self.strict_mode:
            # In strict mode, we need to consider both tool names and parameters
            predicted_tool_names = set()
            for tools in predicted_tool_steps.values():
                for tool in tools:
                    predicted_tool_names.add(
                        f"{tool.get('name', '')}: {tool.get('parameters', {})}",
                    )
            ground_truth_tool_names = set()
            for tools in ground_truth_tool_steps.values():
                for tool in tools:
                    ground_truth_tool_names.add(
                        f"{tool.get('name', '')}: {tool.get('parameters', {})}",
                    )
        else:
            # Extract all tool names from predicted steps
            predicted_tool_names = set()
            for tools in predicted_tool_steps.values():
                for tool in tools:
                    predicted_tool_names.add(tool.get("name", ""))
            # Extract all tool names from ground truth steps
            ground_truth_tool_names = set()
            for tools in ground_truth_tool_steps.values():
                for tool in tools:
                    ground_truth_tool_names.add(tool.get("name", ""))
        # Handle edge cases
        if not ground_truth_tool_names and not predicted_tool_names:
            return 1.0, set(), set()  # Both empty sets
        if not ground_truth_tool_names or not predicted_tool_names:
            return 0.0, set(), set()  # One empty set
        # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
        intersection_set = predicted_tool_names & ground_truth_tool_names
        union_set = predicted_tool_names | ground_truth_tool_names
        score = len(intersection_set) / len(union_set) if union_set else 0.0
        return score, intersection_set, union_set

    async def aevaluate(
        self,
        messages: List[Dict[str, Any]],
        ground_truth_tool_calls: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate tool call sequence matching against ground truth.
        Args:
            messages: List of message dicts containing model's predicted tool calls
            ground_truth_tool_calls: Ground truth reference tool call sequence
            **kwargs: Additional evaluation parameters
        Returns:
            GraderScore: Tool call sequence matching score and details
        """
        # Extract sequences
        try:
            predicted_tool_steps = self.extract_predicted_tool_sequence(messages)
            ground_truth_tool_steps = self.extract_ground_truth_tool_sequence(ground_truth_tool_calls)
        except Exception as e:
            logger.error(f"Sequence extraction failed: {e}")
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=f"Sequence extraction failed: {str(e)}",
                metadata={"error": str(e)},
            )
        if not predicted_tool_steps and not ground_truth_tool_steps:
            return GraderScore(
                name=self.name,
                score=1.0,
                reason="Both predicted and ground truth have no tool calls",
                metadata={
                    "predicted_tool_count": 0,
                    "ground_truth_tool_count": 0,
                },
            )
        # Calculate scores using the appropriate function
        if self.use_jaccard_similarity:
            (
                jaccard_score,
                intersection_set,
                union_set,
            ) = self.calculate_jaccard_similarity_score(predicted_tool_steps, ground_truth_tool_steps)
            final_score = jaccard_score
            score_type = "jaccard_similarity"
        else:
            step_matching_score = self.calculate_step_matching_score(
                predicted_tool_steps,
                ground_truth_tool_steps,
            )
            final_score = step_matching_score
            score_type = "step_matching"
        # Generate detailed reason
        mode_str = "strict" if self.strict_mode else "loose"
        method_str = "jaccard" if self.use_jaccard_similarity else "step-by-step"
        reason = f"Tool call sequence evaluation ({mode_str} mode, {method_str}): {score_type}={final_score:.3f}"
        # Count tools for metadata
        predicted_tool_count = sum(len(tools) for tools in predicted_tool_steps.values())
        ground_truth_tool_count = sum(len(tools) for tools in ground_truth_tool_steps.values())
        # Prepare metadata
        metadata = {
            "strict_mode": self.strict_mode,
            "use_jaccard_similarity": self.use_jaccard_similarity,
            "predicted_tool_count": predicted_tool_count,
            "ground_truth_tool_count": ground_truth_tool_count,
            "predicted_tool_steps": predicted_tool_steps,
            "ground_truth_tool_steps": ground_truth_tool_steps,
        }
        # Add the appropriate score to metadata
        if self.use_jaccard_similarity:
            metadata["jaccard_similarity_score"] = final_score
            metadata["intersection_set"] = list(intersection_set)
            metadata["union_set"] = list(union_set)
        else:
            metadata["step_matching_score"] = final_score
        return GraderScore(
            name=self.name,
            score=final_score,
            reason=reason,
            metadata=metadata,
        )
