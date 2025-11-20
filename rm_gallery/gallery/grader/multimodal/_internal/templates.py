# -*- coding: utf-8 -*-
"""
Multimodal G-Eval Template

Template for flexible multimodal evaluation using the G-Eval framework.
"""

import textwrap
from typing import List, Optional, Tuple

from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import Template


class MultimodalGEvalTemplate:
    """
    Templates for Multimodal G-Eval evaluation

    Supports chain-of-thought evaluation with custom criteria and rubrics.
    """

    @staticmethod
    def generate_evaluation_steps(parameters: str, criteria: str) -> Template:
        """
        Generate template for creating evaluation steps from criteria

        Args:
            parameters: Description of parameters being evaluated
            criteria: Evaluation criteria description

        Returns:
            Template for generating evaluation steps
        """
        content = textwrap.dedent(
            """Given an evaluation criteria which outlines how you should judge the \
{parameters}, generate 3-4 concise evaluation steps based on the criteria below. \
You MUST make it clear how to evaluate {parameters} in relation to one another.

            Evaluation Criteria:
            {criteria}

            **
            IMPORTANT: Please make sure to only return in JSON format, with the "steps" \
key as a list of strings. No words or explanation is needed.
            Example JSON:
            {{
                "steps": <list_of_strings>
            }}
            **

            JSON:
            """,
        )

        return Template(
            messages=[ChatMessage(role="user", content=content, name="User")],
        )

    @staticmethod
    def generate_evaluation_results(
        evaluation_steps: str,
        test_case_list: List,
        parameters: str,
        rubric: Optional[str] = None,
        score_range: Tuple[int, int] = (0, 10),
        additional_context: Optional[str] = None,
    ) -> List:
        """
        Generate prompt for evaluating test case using G-Eval framework

        Args:
            evaluation_steps: Formatted string of evaluation steps
            test_case_list: List of test case components (text and images)
            parameters: Description of parameters being evaluated
            rubric: Optional formatted rubric string
            score_range: Score range tuple (min, max)
            additional_context: Optional additional context

        Returns:
            List of prompt parts (can include text and images)
        """
        rubric_text = f"Rubric:\n{rubric}\n" if rubric else ""
        dependencies = "evaluation steps and rubric" if rubric else "evaluation steps"
        score_explanation = (
            "based on the rubric provided"
            if rubric
            else f"with {score_range[1]} indicating strong alignment with the evaluation steps and {score_range[0]} indicating no alignment"
        )
        reasoning_expectation = (
            "Be specific and grounded in the evaluation steps and rubric."
            if rubric
            else "Be specific and grounded in the evaluation steps."
        )
        additional_context_text = (
            f"\n\nAdditional Context:\n{additional_context}\n"
            if additional_context
            else ""
        )

        return (
            [
                textwrap.dedent(
                    f"""You are an evaluator. Given the following {dependencies}, assess the \
response below and return a JSON object with two fields:

                - `"score"`: an integer between {score_range[0]} and {score_range[1]}, \
{score_explanation}.
                - `"reason"`: a brief explanation for why the score was given. This must \
mention specific strengths or shortcomings, referencing relevant details from the input. \
Do **not** quote the score itself in the explanation.

                Your explanation should:
                - {reasoning_expectation}
                - Mention key details from the test case parameters.
                - Be concise, clear, and focused on the evaluation logic.

                Only return valid JSON. Do **not** include any extra commentary or text.

                ---

                Evaluation Steps:
                {evaluation_steps}

                {rubric_text}
                Test Case:
                ************************
                """,
                ),
            ]
            + test_case_list
            + [
                textwrap.dedent(
                    f"""
                ************************
                \n\n\n
                Parameters:
                {parameters}
                {additional_context_text}

                ---
                **Example JSON:**
                {{
                    "score": {score_range[0]},
                    "reason": "your concise and informative reason here"
                }}

                JSON:
                """,
                ),
            ]
        )

    @staticmethod
    def generate_strict_evaluation_results(
        evaluation_steps: str,
        test_case_list: List,
        parameters: str,
        additional_context: Optional[str] = None,
    ) -> List:
        """
        Generate prompt for strict binary evaluation (pass/fail)

        Args:
            evaluation_steps: Formatted string of evaluation steps
            test_case_list: List of test case components (text and images)
            parameters: Description of parameters being evaluated
            additional_context: Optional additional context

        Returns:
            List of prompt parts (can include text and images)
        """
        additional_context_text = (
            f"\n\nAdditional Context:\n{additional_context}\n"
            if additional_context
            else ""
        )
        return (
            [
                textwrap.dedent(
                    f"""Given the evaluation steps, return a JSON with two keys: 1) a `score` key that is STRICTLY EITHER 1 (follows the criteria 100% outlined in the evaluation steps), OR 0 (does not follow the criteria), and 2) a `reason` key, a reason for the given score, but DO NOT QUOTE THE SCORE in your reason. Please mention specific information from {parameters} in your reason, but be very concise with it!

                Evaluation Steps:
                {evaluation_steps}
                ************************
                """,
                ),
            ]
            + test_case_list
            + [
                textwrap.dedent(
                    f"""
                ************************
                {additional_context_text}
                **
                IMPORTANT: Please make sure to only return in JSON format, with the "score" and "reason" key. No words or explanation is needed.

                Example JSON:
                {{
                    "score": 0,
                    "reason": "The text does not follow the evaluation steps provided."
                }}
                **

                JSON:
                """,
                ),
            ]
        )


__all__ = ["MultimodalGEvalTemplate"]
