# -*- coding: utf-8 -*-
"""
Image Helpfulness Grader

Evaluates how helpful images are in understanding surrounding text.
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
IMAGE_HELPFULNESS_PROMPT_EN = """
# Task Description
You are a multi-modal document evaluation assistant. You will receive an image and its textual context.
Your task is to evaluate the helpfulness of the image in enabling human readers to comprehend the text (context above and below) it accompanies.

# Context Above
{context_above}

# Context Below
{context_below}

# Image
[The image is provided below this section.]

# Scoring Criteria
Evaluate how well the image helps human readers understand the content of its accompanying text, assigning a score from 0 to 10.
A higher score indicates that the image significantly enhances comprehension of the text. Be precise when assigning the score.

- A score from 0-3 means the image is minimally or not at all helpful for comprehension.
- A score from 4-6 indicates the image provides some helpful context or information but may contain extraneous or less relevant details.
- A score from 7-9 indicates the image is highly helpful in enabling comprehension of the text.
- A score of 10 indicates the image perfectly enhances and clarifies the information provided in the text.

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
IMAGE_HELPFULNESS_PROMPT_ZH = """
# 任务描述
你是一名多模态文档评估助手。你将收到一张图片及其文本背景。
你的任务是评估图片对于帮助人类读者理解其伴随文本（上下文）的有用性。

# 上文
{context_above}

# 下文
{context_below}

# 图片
[图片将在本节下方提供。]

# 评分标准
评估图片对于帮助人类读者理解伴随文本内容的有用程度，给出0到10的分数。
分数越高表示图片越能显著增强对文本的理解。请精确地给出分数。

- 0-3分表示图片对理解文本的帮助极小或完全没有帮助。
- 4-6分表示图片提供了一些有用的背景或信息，但可能包含多余或关联性较弱的细节。
- 7-9分表示图片对理解文本非常有帮助。
- 10分表示图片完美地增强并澄清了文本中提供的信息。

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
DEFAULT_IMAGE_HELPFULNESS_TEMPLATE = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(IMAGE_HELPFULNESS_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(IMAGE_HELPFULNESS_PROMPT_ZH),
            ),
        ],
    },
)


class ImageHelpfulnessGrader(LLMGrader):
    """
    Image Helpfulness Grader

    Evaluates how helpful images are in enabling readers to understand the
    surrounding text. Unlike coherence (which measures matching), helpfulness
    measures whether the image provides additional value for comprehension.

    Key evaluation dimensions:
    - Information enhancement: Does the image provide new information?
    - Understanding assistance: Does it make concepts clearer?
    - Practical value: Is it genuinely useful for the reader?

    Attributes:
        name: Grader name
        model: OpenAIChatModel instance for evaluation
        max_context_size: Maximum characters to extract from context (default: 500)
        threshold: Success threshold [0, 1] (default: 0.7)

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.gallery.grader.multimodal import MLLMImage
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",  # pragma: allowlist secret
        ...     model_name="gpt-4o",
        ...     generate_kwargs={"temperature": 0.1},
        ... )
        >>> grader = ImageHelpfulnessGrader(model=api, threshold=0.7)
        >>>
        >>> result = await grader.aevaluate(
        ...     actual_output=[
        ...         "The architecture consists of multiple layers.",
        ...         MLLMImage(url="https://example.com/architecture_diagram.jpg"),
        ...         "Each layer serves a specific purpose."
        ...     ]
        ... )
        >>> print(f"Helpfulness score: {result.score:.2f}")
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        max_context_size: int = 500,
        threshold: float = 0.7,
        template: Template = DEFAULT_IMAGE_HELPFULNESS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="image_helpfulness",
            grader_mode=GraderMode.POINTWISE,
            description="Evaluate image helpfulness for understanding text",
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
        """Async evaluation of single image helpfulness"""
        messages = self.template.to_messages()
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
            logger.error(f"Error evaluating image helpfulness: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

    async def _acompute(
        self,
        actual_output: List[Union[str, MLLMImage]],
        **_kwargs: Any,
    ) -> Tuple[float, dict]:
        """Compute image helpfulness score (asynchronous)"""

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
        Evaluate image helpfulness

        Args:
            actual_output: List containing text and images (mixed)
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized helpfulness value [0, 1]

        Example:
            >>> result = await grader.aevaluate(
            ...     actual_output=[
            ...         "The system architecture:",
            ...         MLLMImage(url="diagram.jpg"),
            ...         "shows the component interactions"
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
            reason=f"Image helpfulness score: {score:.4f}\n{reason}",
            metadata=details,
        )


__all__ = ["ImageHelpfulnessGrader"]
