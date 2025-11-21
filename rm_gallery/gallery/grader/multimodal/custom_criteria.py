# -*- coding: utf-8 -*-
"""
Custom Criteria Grader

Flexible evaluation with custom criteria for multimodal content.
Supports automatic evaluation step generation and custom scoring rubrics.
"""

import textwrap
from typing import Any, List, Optional, Tuple

from loguru import logger

from rm_gallery.core.grader.base import Grader
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.grader import GraderMode, GraderScore
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import LanguageEnum, Template
from rm_gallery.gallery.grader.multimodal._internal import (
    EvaluationSteps,
    MLLMImage,
    MLLMTestCaseParams,
    Rubric,
    construct_params_string,
    format_image_content,
    format_rubrics,
    validate_and_sort_rubrics,
    validate_criteria_and_evaluation_steps,
)

# pylint: disable=line-too-long

# English Prompts
CUSTOM_CRITERIA_GENERATE_STEPS_PROMPT_EN = """Given an evaluation criteria which outlines how you should judge the {parameters}, generate 3-4 concise evaluation steps based on the criteria below. You MUST make it clear how to evaluate {parameters} in relation to one another.

Evaluation Criteria:
{criteria}

**
IMPORTANT: Please make sure to only return in JSON format, with the "steps" key as a list of strings. No words or explanation is needed.
Example JSON:
{{
    "steps": <list_of_strings>
}}
**

JSON:
"""

# Chinese Prompts
CUSTOM_CRITERIA_GENERATE_STEPS_PROMPT_ZH = """给定一个评估标准，概述你应该如何评估{parameters}，根据以下标准生成3-4个简明的评估步骤。你必须明确如何相互关联地评估{parameters}。

评估标准：
{criteria}

**
重要：请确保只返回JSON格式，使用"steps"键作为字符串列表。不需要其他文字或解释。
示例JSON：
{{
    "steps": <字符串列表>
}}
**

JSON:
"""

# Build default template for generating evaluation steps
DEFAULT_CUSTOM_CRITERIA_GENERATE_STEPS_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(CUSTOM_CRITERIA_GENERATE_STEPS_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(CUSTOM_CRITERIA_GENERATE_STEPS_PROMPT_ZH),
            ),
        ],
    },
)


class CustomCriteriaGrader(Grader):
    """
    Custom Criteria Grader

    Flexible evaluation with custom criteria for multimodal content.
    Supports:
    - Chain-of-Thought evaluation with step-by-step reasoning
    - Automatic evaluation step generation from criteria
    - Custom rubrics for detailed scoring standards
    - Flexible scoring (0-10 scale, normalized to 0-1)

    Attributes:
        name: Grader name
        model: OpenAIChatModel instance for evaluation
        evaluation_name: Name for this evaluation
        evaluation_params: List of parameters to evaluate (e.g., input, actual_output)
        criteria: Evaluation criteria description
        evaluation_steps: Explicit evaluation steps (optional, auto-generated if not provided)
        rubric: Detailed scoring rubric (optional)
        threshold: Success threshold [0, 1] (default: 0.7)

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.gallery.grader.multimodal import MLLMTestCaseParams, MLLMImage
        >>>
        >>> vlm_api = VisionModelAdapter.from_qwen(model_name="qwen-vl-plus")
        >>> grader = CustomCriteriaGrader(
        ...     model=vlm_api,
        ...     evaluation_name="Image Caption Quality",
        ...     evaluation_params=[
        ...         MLLMTestCaseParams.INPUT,
        ...         MLLMTestCaseParams.ACTUAL_OUTPUT
        ...     ],
        ...     criteria="Evaluate the quality of image captions based on accuracy and detail",
        ...     threshold=0.7
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     input=[MLLMImage(url="..."), "Describe this image"],
        ...     actual_output=["A cat sitting on a mat"]
        ... )
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        evaluation_name: str,
        evaluation_params: List[MLLMTestCaseParams],
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        rubric: Optional[List[Rubric]] = None,
        threshold: float = 0.7,
        score_range: Tuple[int, int] = (0, 10),
        generate_steps_template: Template = DEFAULT_CUSTOM_CRITERIA_GENERATE_STEPS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="custom_criteria",
            grader_mode=GraderMode.POINTWISE,
            description="Custom criteria flexible evaluation",
        )
        self.model = (
            model if isinstance(model, ChatModelBase) else OpenAIChatModel(**model)
        )
        self.evaluation_name = evaluation_name
        self.evaluation_params = evaluation_params
        self.criteria = criteria
        self.evaluation_steps = evaluation_steps
        self.rubric = rubric
        self.threshold = threshold
        self.score_range = score_range
        self.generate_steps_template = generate_steps_template
        self.language = language
        self._generated_steps: Optional[List[str]] = None

        # Validate criteria and evaluation steps
        validate_criteria_and_evaluation_steps(criteria, evaluation_steps)

        # Validate and sort rubrics
        if rubric:
            self.rubric = validate_and_sort_rubrics(rubric, score_range)

    async def _agenerate_evaluation_steps(
        self,
        _params_dict: dict,
    ) -> List[str]:
        """Generate evaluation steps from criteria (async)"""
        if self._generated_steps is not None:
            return self._generated_steps

        if self.criteria is None:
            raise ValueError(
                "Cannot generate evaluation steps without criteria",
            )

        # Build parameters string for context
        params_str = construct_params_string(self.evaluation_params)

        # Generate steps using VLM
        messages = self.generate_steps_template.get(self.language)
        prompt = messages[0].content.format(
            parameters=params_str,
            criteria=self.criteria,
        )

        try:
            response = await self.model.achat(
                messages=[{"role": "user", "content": prompt}],
                structured_model=EvaluationSteps,
            )

            # Extract structured output
            if response.metadata:
                result = EvaluationSteps(**response.metadata)
            else:
                # Fallback: parse from text content
                text_content = "".join(
                    [
                        block.text
                        for block in response.content
                        if hasattr(block, "text")
                    ],
                )
                import json

                result = EvaluationSteps(**json.loads(text_content))

            self._generated_steps = result.steps
            return self._generated_steps

        except Exception as e:
            logger.error(f"Error generating evaluation steps: {e}")
            # Fallback to default steps
            return [
                f"Analyze the {param.value}" for param in self.evaluation_params
            ] + ["Evaluate based on the given criteria"]

    def _generate_evaluation_prompt_parts(
        self,
        evaluation_steps: str,
        test_case_list: List,
        parameters: str,
        rubric_str: Optional[str] = None,
    ) -> List:
        """Generate evaluation prompt parts with bilingual support"""
        if self.language == LanguageEnum.ZH:
            dependencies = "评估步骤和评分标准" if rubric_str else "评估步骤"
            score_explanation = (
                "基于提供的评分标准"
                if rubric_str
                else f"{self.score_range[1]}表示与评估步骤高度一致，{self.score_range[0]}表示不一致"
            )
            reasoning_expectation = "具体且基于评估步骤和评分标准。" if rubric_str else "具体且基于评估步骤。"
            rubric_text = f"评分标准：\n{rubric_str}\n" if rubric_str else ""

            prompt_start = textwrap.dedent(
                f"""你是一名评估员。根据以下{dependencies}，评估下面的回答并返回一个包含两个字段的JSON对象：

- `"score"`：一个介于{self.score_range[0]}和{self.score_range[1]}之间的整数，{score_explanation}。
- `"reason"`：对给出分数的简要解释。必须提及具体的优点或缺点，引用输入中的相关细节。不要在解释中引用分数本身。

你的解释应该：
- {reasoning_expectation}
- 提及测试用例参数中的关键细节。
- 简洁、清晰、专注于评估逻辑。

只返回有效的JSON。不要包含任何额外的评论或文本。

---

评估步骤：
{evaluation_steps}

{rubric_text}
测试用例：
************************
""",
            )

            prompt_end = textwrap.dedent(
                f"""
************************


参数：
{parameters}

---
**示例JSON：**
{{
    "score": {self.score_range[0]},
    "reason": "你的简洁且信息丰富的理由"
}}

JSON:
""",
            )
        else:
            dependencies = (
                "evaluation steps and rubric" if rubric_str else "evaluation steps"
            )
            score_explanation = (
                "based on the rubric provided"
                if rubric_str
                else f"with {self.score_range[1]} indicating strong alignment with the evaluation steps and {self.score_range[0]} indicating no alignment"
            )
            reasoning_expectation = (
                "Be specific and grounded in the evaluation steps and rubric."
                if rubric_str
                else "Be specific and grounded in the evaluation steps."
            )
            rubric_text = f"Rubric:\n{rubric_str}\n" if rubric_str else ""

            prompt_start = textwrap.dedent(
                f"""You are an evaluator. Given the following {dependencies}, assess the response below and return a JSON object with two fields:

- `"score"`: an integer between {self.score_range[0]} and {self.score_range[1]}, {score_explanation}.
- `"reason"`: a brief explanation for why the score was given. This must mention specific strengths or shortcomings, referencing relevant details from the input. Do **not** quote the score itself in the explanation.

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
            )

            prompt_end = textwrap.dedent(
                f"""
************************


Parameters:
{parameters}

---
**Example JSON:**
{{
    "score": {self.score_range[0]},
    "reason": "your concise and informative reason here"
}}

JSON:
""",
            )

        return [prompt_start] + test_case_list + [prompt_end]

    async def _aevaluate_with_criteria(
        self,
        params_dict: dict,
    ) -> Tuple[float, str]:
        """Evaluate using custom criteria framework (asynchronous)"""
        # Get or generate evaluation steps
        steps = (
            self.evaluation_steps
            if self.evaluation_steps
            else await self._agenerate_evaluation_steps(params_dict)
        )

        # Format evaluation steps as string
        steps_str = "\n".join(
            [f"{i+1}. {step}" for i, step in enumerate(steps)],
        )

        # Build parameters string
        params_str = construct_params_string(self.evaluation_params)

        # Format rubric if provided
        rubric_str = format_rubrics(self.rubric) if self.rubric else None

        # Build test case list (text and images)
        test_case_list = []
        for param in self.evaluation_params:
            param_value = params_dict.get(param.value, [])
            if isinstance(param_value, list):
                test_case_list.extend(param_value)
            else:
                test_case_list.append(param_value)

        # Generate evaluation prompt using helper method
        prompt_parts = self._generate_evaluation_prompt_parts(
            evaluation_steps=steps_str,
            test_case_list=test_case_list,
            parameters=params_str,
            rubric_str=rubric_str,
        )

        try:
            # Extract text and images from prompt parts
            prompt_text = "".join(
                [p for p in prompt_parts if isinstance(p, str)],
            )
            prompt_images = [p for p in prompt_parts if isinstance(p, MLLMImage)]

            # Format content with images
            if prompt_images:
                content = format_image_content(prompt_text, prompt_images)
            else:
                content = prompt_text

            response = await self.model.achat(
                messages=[{"role": "user", "content": content}],
            )

            # Parse response from text content
            import json

            text_content = "".join(
                [block.text for block in response.content if hasattr(block, "text")],
            )

            # Parse JSON response
            result_data = json.loads(text_content.strip())
            score_data = result_data.get("score", 0)
            score = (
                float(score_data)
                if not isinstance(score_data, list)
                else float(score_data[0])
            )
            reason = result_data.get("reason", "No reason provided")

            return score, reason

        except Exception as e:
            logger.error(f"Error in custom criteria evaluation: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

    async def _acompute(
        self,
        **params_dict: Any,
    ) -> Tuple[float, dict]:
        """
        Compute custom criteria score (asynchronous)

        Args:
            **params_dict: Dictionary containing evaluation parameters

        Returns:
            tuple[float, dict]: (normalized_score [0,1], details)
        """

        # Validate required parameters
        for param in self.evaluation_params:
            if param.value not in params_dict:
                return 0.0, {
                    "error": f"Missing required parameter: {param.value}",
                }

        # Evaluate
        raw_score, reason = await self._aevaluate_with_criteria(params_dict)

        # Normalize score to [0, 1]
        score_min, score_max = self.score_range
        normalized_score = (raw_score - score_min) / (score_max - score_min)
        normalized_score = max(0.0, min(1.0, normalized_score))

        details = {
            "raw_score": raw_score,
            "score_range": self.score_range,
            "reason": reason,
            "evaluation_name": self.evaluation_name,
            "evaluation_params": [p.value for p in self.evaluation_params],
            "evaluation_steps": (self.evaluation_steps or self._generated_steps),
            "threshold": self.threshold,
        }

        return normalized_score, details

    async def aevaluate(
        self,
        **params_dict: Any,
    ) -> GraderScore:
        """
        Evaluate using custom criteria framework

        Args:
            **params_dict: Dictionary containing evaluation parameters
                Expected keys depend on evaluation_params, e.g.:
                - input: List[Union[str, MLLMImage]]
                - actual_output: List[Union[str, MLLMImage]]
                - expected_output: List[Union[str, MLLMImage]]
                - context: List[Union[str, MLLMImage]]
                etc.

        Returns:
            GraderScore: Score with normalized evaluation value [0, 1]

        Example:
            >>> result = await grader.aevaluate(
            ...     input=[MLLMImage(url="..."), "Describe this"],
            ...     actual_output=["A cat sitting"]
            ... )
        """
        score, details = await self._acompute(**params_dict)

        if "error" in details:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=details["error"],
                metadata=details,
            )

        raw_score = details["raw_score"]
        max_score = self.score_range[1]
        reason = f"""{self.evaluation_name}: {score:.4f} (raw: {raw_score:.2f}/{max_score})

{details['reason']}

Evaluation Steps:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(details['evaluation_steps'] or []))}
"""

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason.strip(),
            metadata=details,
        )


__all__ = ["CustomCriteriaGrader", "DEFAULT_CUSTOM_CRITERIA_GENERATE_STEPS_TEMPLATE"]
