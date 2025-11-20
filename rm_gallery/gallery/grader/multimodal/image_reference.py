# -*- coding: utf-8 -*-
"""
Image Reference Grader

Evaluates the accuracy and clarity of image references in surrounding text.
Restructured to work with Grader framework.
"""

import asyncio
from typing import Any, List, Optional, Tuple, Union

from loguru import logger

from rm_gallery.core.grader.base import Grader
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.grader import GraderMode, GraderScore, _GraderScore
from rm_gallery.gallery.grader.multimodal._internal import (
    ImageReferenceTemplate,
    MLLMImage,
    format_image_content,
    get_image_context,
    get_image_indices,
)


class ImageReferenceGrader(Grader):
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
        model: OpenAIChatModel,
        name: str = "image_reference",
        max_context_size: int = 500,
        threshold: float = 0.7,
        description: str = "Evaluate image reference quality in text",
    ):
        super().__init__(
            name=name,
            grader_mode=GraderMode.POINTWISE,
            description=description,
        )
        self.model = model
        self.max_context_size = max_context_size
        self.threshold = threshold
        self.evaluation_cost = 0.0

    async def _aevaluate_single_image(
        self,
        image: MLLMImage,
        context_above: Optional[str],
        context_below: Optional[str],
    ) -> Tuple[float, str]:
        """Async evaluation of single image reference"""
        template = ImageReferenceTemplate.evaluate_image_reference()
        messages = template.to_messages()
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
        self.evaluation_cost = 0.0

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
            "evaluation_cost": self.evaluation_cost,
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
