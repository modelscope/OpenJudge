# -*- coding: utf-8 -*-
"""
Simple test script for ImageCoherenceGrader

"""

import asyncio
import base64
import os

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.gallery.grader.multimodal._internal import MLLMImage
from rm_gallery.gallery.grader.multimodal.image_coherence import ImageCoherenceGrader


async def test_with_base64_image():
    """Test with base64 encoded local image"""

    # Initialize model
    model = OpenAIChatModel(
        model_name="qwen-vl-max",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        generate_kwargs={"temperature": 0.1},
    )

    # Create grader
    grader = ImageCoherenceGrader(model=model, threshold=0.7)

    # Load local test image
    test_image_path = (
        "/Users/boyin.liu/Desktop/code/RM-Gallery-git/data/test_images/html_good_1.png"
    )

    if os.path.exists(test_image_path):
        # Convert to base64
        with open(test_image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Test coherent content
        test_content = [
            "This is a good HTML rendering example showing a financial comparison.",
            MLLMImage(base64=image_data, format="png"),
            "The layout is clean and well-structured with clear metrics.",
        ]

        print("Evaluating image coherence...")
        result = await grader.aevaluate(actual_output=test_content)

        print(f"\nScore: {result.score:.4f}")
        print(
            f"Passed (threshold {grader.threshold}): {result.score >= grader.threshold}",
        )
        print(f"\nReason:\n{result.reason[:300]}...")
        print(f"\nMetadata: {result.metadata}")
    else:
        print(f"Test image not found: {test_image_path}")


async def test_with_public_image():
    """Test with a stable public image URL"""

    model = OpenAIChatModel(
        model_name="qwen-vl-max",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    grader = ImageCoherenceGrader(model=model)

    # Use a stable image URL (example: GitHub avatar or similar)
    # Note: Some URLs may not work with DashScope due to network restrictions
    test_content = [
        "Here is a sample image for testing.",
        MLLMImage(
            url="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
        ),
        "This is just a test image.",
    ]

    print("\nTesting with public URL...")
    try:
        result = await grader.aevaluate(actual_output=test_content)
        print(f"Score: {result.score:.4f}")
        print(f"Reason: {result.reason[:200]}...")
    except Exception as e:
        print(f"Error (may be network restriction): {str(e)[:200]}")


if __name__ == "__main__":
    print("=" * 80)
    print("ImageCoherenceGrader Simple Test")
    print("=" * 80)

    asyncio.run(test_with_base64_image())
    # asyncio.run(test_with_public_image())  # Uncomment to test URL images

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)
