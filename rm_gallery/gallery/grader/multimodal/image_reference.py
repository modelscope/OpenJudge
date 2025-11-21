# -*- coding: utf-8 -*-
"""
Image Reference Grader

Evaluates the accuracy and clarity of image references in surrounding text.
Restructured to work with Grader framework.
"""

import asyncio
import textwrap
from typing import Any, List, Optional, Tuple, Union

from loguru import logger

from rm_gallery.core.grader.base import LLMGrader
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.grader import GraderMode, GraderScore, _GraderScore
from rm_gallery.core.schema.message import ChatMessage
from rm_gallery.core.schema.template import LanguageEnum, Template
from rm_gallery.gallery.grader.multimodal._internal import (
    MLLMImage,
    format_image_content,
    get_image_context,
    get_image_indices,
)

# pylint: disable=line-too-long

# English Prompt
IMAGE_REFERENCE_PROMPT_EN = """
# Task Description
You are a multi-modal document quality assessment assistant. You will receive an image and its accompanying textual context.
Your task is to determine whether the image is explicitly referenced or explained within the surrounding text (both above and below the image).

# Context Above
{context_above}

# Context Below
{context_below}

# Image
[The image is provided below this section.]

# Scoring Criteria
Evaluate the extent to which the image is referenced or explained in the text, assigning a score from 0 to 10:
- 0: The image is not mentioned or referenced in the context.
- 1-3: The image is referenced implicitly, and the reference is improper or incorrect.
- 4-6: The image is referenced explicitly but in an improper manner, or it is referenced implicitly.
- 7-9: The image is referenced explicitly, with the reference being generally proper and correct.
- 10: The image is referenced explicitly, with the placement and explanation being completely proper and correct.

Be rigorous and discerning when assigning your score.

# Output Instructions
Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 0 and 10>,
    "reason": "<brief explanation for the assigned score>"
}}

# Image
[Insert Image Here]
"""

# Chinese Prompt
IMAGE_REFERENCE_PROMPT_ZH = """
# 任务描述
你是一名多模态文档质量评估助手。你将收到一张图片及其伴随的文本背景。
你的任务是判断图片是否在其周围文本（图片上方和下方）中被明确引用或解释。

# 上文
{context_above}

# 下文
{context_below}

# 图片
[图片将在本节下方提供。]

# 评分标准
评估图片在文本中被引用或解释的程度，给出0到10的分数：
- 0：图片在上下文中未被提及或引用。
- 1-3：图片被隐式引用，且引用不当或不正确。
- 4-6：图片被明确引用但方式不当，或仅被隐式引用。
- 7-9：图片被明确引用，引用总体上恰当且正确。
- 10：图片被明确引用，位置和解释完全恰当且正确。

请严格审慎地评分。

# 输出指令
请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0到10之间的整数>,
    "reason": "<对所给分数的简要解释>"
}}

# 图片
[在此插入图片]
"""

# Build default template from prompts
DEFAULT_IMAGE_REFERENCE_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(IMAGE_REFERENCE_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(IMAGE_REFERENCE_PROMPT_ZH),
            ),
        ],
    },
)


class ImageReferenceGrader(LLMGrader):
    """
    Image Reference Grader

    Evaluates whether images are properly referenced in the surrounding text.
    Assesses the clarity, accuracy, and appropriateness of image references.

    Key evaluation aspects:
    - Reference clarity: Is the reference explicit and clear?
    - Reference accuracy: Does the reference description match the image?
    - Reference necessity: Is the reference at an appropriate location?

    Common reference types:
    - Explicit: "As shown in Figure 1...", "The diagram above..."
    - Implicit: "This shows...", "We can see..."
    - None: No reference to the image

    Attributes:
        name: Grader name
        model: OpenAIChatModel instance for evaluation
        max_context_size: Maximum characters to extract from context (default: 500)
        threshold: Success threshold [0, 1] (default: 0.7)

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.gallery.grader.multimodal import MLLMImage
        >>>
        >>> api = VisionModelAdapter.from_qwen(api_key="your-key", model_name="qwen-vl-plus")
        >>> grader = ImageReferenceGrader(model=api, threshold=0.7)
        >>>
        >>> result = await grader.aevaluate(
        ...     actual_output=[
        ...         "The sales data is presented below.",
        ...         MLLMImage(url="https://example.com/sales_chart.jpg"),
        ...         "As shown in the chart above, Q3 had the highest sales."
        ...     ]
        ... )
        >>> print(f"Reference quality score: {result.score:.2f}")
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        max_context_size: int = 500,
        threshold: float = 0.7,
        template: Template = DEFAULT_IMAGE_REFERENCE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="image_reference",
            grader_mode=GraderMode.POINTWISE,
            description="Evaluate image reference quality in text",
            model=model,
            template=template,
            language=language,
        )
        self.max_context_size = max_context_size
        self.threshold = threshold

    async def _aevaluate_single_image(
        self,
        image: MLLMImage,
        context_above: Optional[str],
        context_below: Optional[str],
    ) -> Tuple[float, str]:
        """Async evaluation of single image reference"""
        messages = self.template.to_messages(self.language)
        prompt = (
            messages[0]
            .format(
                context_above=context_above or "",
                context_below=context_below or "",
            )
            .content
        )

        try:
            content = format_image_content(prompt, [image])
            response = await self.model.achat(
                messages=[{"role": "user", "content": content}],
                structured_model=_GraderScore,
            )
            score = response.metadata["score"]
            reason = response.metadata["reason"]
            return score, reason
        except Exception as e:
            logger.error(f"Error evaluating image reference: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

    async def _acompute(
        self,
        actual_output: List[Union[str, MLLMImage]],
        **_kwargs: Any,
    ) -> Tuple[float, dict]:
        """Compute image reference score (asynchronous)"""

        image_indices = get_image_indices(actual_output)

        if not image_indices:
            return 0.0, {
                "error": "No images found in actual_output",
                "num_images": 0,
            }

        tasks = []
        for image_index in image_indices:
            context_above, context_below = get_image_context(
                image_index,
                actual_output,
                self.max_context_size,
            )
            image = actual_output[image_index]
            tasks.append(
                self._aevaluate_single_image(
                    image,
                    context_above,
                    context_below,
                ),
            )

        results = await asyncio.gather(*tasks)

        scores = []
        reasons = []
        for raw_score, reason in results:
            normalized_score = raw_score / 10.0
            scores.append(normalized_score)
            reasons.append(reason)

        final_score = sum(scores) / len(scores) if scores else 0.0

        details = {
            "num_images": len(image_indices),
            "individual_scores": scores,
            "individual_reasons": reasons,
            "threshold": self.threshold,
        }

        return final_score, details

    async def aevaluate(
        self,
        actual_output: List[Union[str, MLLMImage]],
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate image reference quality

        Args:
            actual_output: List containing text and images (mixed)
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized reference quality value [0, 1]

        Example:
            >>> result = await grader.aevaluate(
            ...     actual_output=[
            ...         "See the chart below.",
            ...         MLLMImage(url="chart.jpg"),
            ...         "The chart shows growth trends."
            ...     ]
            ... )
        """
        score, details = await self._acompute(actual_output, **kwargs)

        if "error" in details:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=details["error"],
                metadata=details,
            )

        # Generate combined reason
        if len(details["individual_reasons"]) == 1:
            reason = details["individual_reasons"][0]
        else:
            reason_parts = []
            for i, (s, r) in enumerate(
                zip(
                    details["individual_scores"],
                    details["individual_reasons"],
                ),
                1,
            ):
                reason_parts.append(f"Image {i} (score: {s:.2f}): {r}")
            reason = "\n".join(reason_parts)

        return GraderScore(
            name=self.name,
            score=score,
            reason=f"Image reference quality score: {score:.4f}\n{reason}",
            metadata=details,
        )


__all__ = ["ImageReferenceGrader"]
