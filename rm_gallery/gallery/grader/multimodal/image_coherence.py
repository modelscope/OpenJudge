# -*- coding: utf-8 -*-
"""
Image Coherence Grader

Evaluates the coherence between images and their surrounding text context.
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
    get_image_context,
    get_image_indices,
)

# pylint: disable=line-too-long

# English Prompt
IMAGE_COHERENCE_PROMPT_EN = """
# Task Description
You are a multi-modal document evaluation assistant. You will receive an image and its textual context.
Your task is to evaluate the coherence between the image and the text (context above and below) it accompanies.

# Context Above
{context_above}

# Context Below
{context_below}

# Image
[The image is provided below this section.]

# Scoring Criteria
Assess how coherent the image is in relation to its accompanying text, assigning a score from 0 to 10.
A higher score indicates stronger coherence between the image and the text. Be precise when assigning the score.

- A score from 0-3 means that the image is minimally or not at all coherent with the text.
- A score from 4-6 indicates that the image shows some coherence with the text but may include unrelated elements.
- A score from 7-9 indicates that the image is highly coherent with the text.
- A score of 10 indicates perfect coherence, where the image completely corresponds with and enhances the text.

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
IMAGE_COHERENCE_PROMPT_ZH = """
# 任务描述
你是一名多模态文档评估助手。你将收到一张图片及其文本背景。
你的任务是评估图片与其伴随文本（上下文）之间的连贯性。

# 上文
{context_above}

# 下文
{context_below}

# 图片
[图片将在本节下方提供。]

# 评分标准
评估图片与其伴随文本的连贯性，给出0到10的分数。
分数越高表示图片与文本之间的连贯性越强。请精确地给出分数。

- 0-3分表示图片与文本的连贯性极低或完全不连贯。
- 4-6分表示图片与文本有一定连贯性，但可能包含无关元素。
- 7-9分表示图片与文本高度连贯。
- 10分表示完美连贯，图片完全对应并增强文本内容。

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
DEFAULT_IMAGE_COHERENCE_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(IMAGE_COHERENCE_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(IMAGE_COHERENCE_PROMPT_ZH),
            ),
        ],
    },
)


class ImageCoherenceGrader(LLMGrader):
    """
    Image Coherence Grader

    Evaluates how well images cohere with their surrounding textual context.
    Considers both the text appearing above and below each image.

    For multiple images, computes the average coherence score.

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
        >>> grader = ImageCoherenceGrader(model=api, threshold=0.7)
        >>>
        >>> result = await grader.aevaluate(
        ...     actual_output=[
        ...         "Our sales grew significantly in Q3.",
        ...         MLLMImage(url="https://example.com/sales_chart.jpg"),
        ...         "This was driven by new product launches."
        ...     ]
        ... )
        >>> print(f"Coherence score: {result.score:.2f}")
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        max_context_size: int = 500,
        threshold: float = 0.7,
        template: Template = DEFAULT_IMAGE_COHERENCE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="image_coherence",
            grader_mode=GraderMode.POINTWISE,
            description="Evaluate image-text coherence",
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
        """Async evaluation of single image coherence"""
        messages = self.template.to_messages(self.language)
        prompt = messages[0].content.format(
            context_above=context_above or "",
            context_below=context_below or "",
        )

        try:
            # Format image content for OpenAI API
            content = [{"type": "text", "text": prompt}]

            if image.url:
                content.append({"type": "image_url", "image_url": {"url": image.url}})
            elif image.base64:
                # Format base64 image with data URL scheme
                image_format = image.format or "jpeg"
                data_url = f"data:image/{image_format};base64,{image.base64}"
                content.append({"type": "image_url", "image_url": {"url": data_url}})

            # Call model without structured output
            response = await self.model.achat(
                messages=[{"role": "user", "content": content}],
                structured_model=_GraderScore,
            )
            score = response.metadata["score"]
            reason = response.metadata["reason"]
            return score, reason

        except Exception as e:
            logger.error(f"Error evaluating image coherence: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

    async def _acompute(
        self,
        actual_output: List[Union[str, MLLMImage]],
        **_kwargs: Any,
    ) -> Tuple[float, dict]:
        """
        Compute image coherence score (asynchronous)

        Args:
            actual_output: List containing text and images

        Returns:
            tuple[float, dict]: (normalized_score [0,1], details)
        """

        # Find all images
        image_indices = get_image_indices(actual_output)

        if not image_indices:
            return 0.0, {
                "error": "No images found in actual_output",
                "num_images": 0,
            }

        # Prepare evaluation tasks
        tasks = []
        for image_index in image_indices:
            context_above, context_below = get_image_context(
                image_index,
                actual_output,
                self.max_context_size,
            )
            image = actual_output[image_index]
            tasks.append(
                self._aevaluate_single_image(image, context_above, context_below),
            )

        # Evaluate all images in parallel
        results = await asyncio.gather(*tasks)

        # Process results
        scores = []
        reasons = []
        for raw_score, reason in results:
            normalized_score = raw_score / 10.0
            scores.append(normalized_score)
            reasons.append(reason)

        # Compute average score
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
        Evaluate image coherence

        Args:
            actual_output: List containing text and images (mixed)
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized coherence value [0, 1]

        Example:
            >>> result = await grader.aevaluate(
            ...     actual_output=[
            ...         "Sales data for Q3:",
            ...         MLLMImage(url="chart.jpg"),
            ...         "Shows 20% growth"
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
                zip(details["individual_scores"], details["individual_reasons"]),
                1,
            ):
                reason_parts.append(f"Image {i} (score: {s:.2f}): {r}")
            reason = "\n".join(reason_parts)

        return GraderScore(
            name=self.name,
            score=score,
            reason=f"Image coherence score: {score:.4f}\n{reason}",
            metadata=details,
        )


__all__ = ["ImageCoherenceGrader", "DEFAULT_IMAGE_COHERENCE_TEMPLATE"]
