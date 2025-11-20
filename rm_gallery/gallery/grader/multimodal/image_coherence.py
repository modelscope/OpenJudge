# -*- coding: utf-8 -*-
"""
Image Coherence Grader

Evaluates the coherence between images and their surrounding text context.
Restructured to work with Grader framework.
"""

import asyncio
from typing import Any, List, Optional, Tuple, Union

from loguru import logger

from rm_gallery.core.grader.base import Grader
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.grader import GraderMode, GraderScore
from rm_gallery.gallery.grader.multimodal._internal import (
    ImageCoherenceTemplate,
    MLLMImage,
    get_image_context,
    get_image_indices,
)


class ImageCoherenceGrader(Grader):
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
        model: OpenAIChatModel,
        name: str = "image_coherence",
        max_context_size: int = 500,
        threshold: float = 0.7,
        description: str = "Evaluate image-text coherence",
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
        """Async evaluation of single image coherence"""
        template = ImageCoherenceTemplate.evaluate_image_coherence()
        messages = template.get()
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
            response = await self.model(
                messages=[{"role": "user", "content": content}],
            )

            # Parse response from text content
            import json

            text_content = "".join(
                [block.text for block in response.content if hasattr(block, "text")],
            )

            # Parse JSON response
            result_data = json.loads(text_content.strip())
            score = float(result_data.get("score", 0))
            reasoning = result_data.get("reasoning", "No reasoning provided")

            # Track usage if available
            if response.usage:
                # Estimate cost (example rates, adjust as needed)
                # For Qwen VL, approximate cost tracking
                pass

            return score, reasoning
        except Exception as e:
            logger.error(f"Error evaluating image coherence: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

    async def _a_compute(
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
        self.evaluation_cost = 0.0

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
        score, details = await self._a_compute(actual_output, **kwargs)

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


__all__ = ["ImageCoherenceGrader"]
