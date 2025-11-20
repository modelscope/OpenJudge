# -*- coding: utf-8 -*-
"""
Test script for MultimodalGEvalGrader

This script tests the MultimodalGEvalGrader using Qwen VL model through DashScope API.
"""

import asyncio
import os

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.gallery.grader.multimodal._internal import (
    MLLMImage,
    MLLMTestCaseParams,
    Rubric,
)
from rm_gallery.gallery.grader.multimodal.multimodal_geval import MultimodalGEvalGrader


# pylint: disable=line-too-long
async def test_geval_basic():
    """Test basic G-Eval with auto-generated evaluation steps"""
    print("=" * 80)
    print("Test 1: Basic G-Eval with Auto-Generated Steps")
    print("=" * 80)

    # Initialize model with environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    print(f"Using API Key: {api_key[:10]}...")
    print(f"Using Base URL: {base_url}")

    # Create model
    model = OpenAIChatModel(
        model_name="qwen-vl-max",
        api_key=api_key,
        base_url=base_url,
        generate_kwargs={"temperature": 0.1},
    )

    # Create grader with auto-generated steps
    grader = MultimodalGEvalGrader(
        model=model,
        evaluation_name="Image Caption Quality",
        evaluation_params=[MLLMTestCaseParams.INPUT, MLLMTestCaseParams.ACTUAL_OUTPUT],
        criteria="Evaluate the quality and accuracy of image descriptions. A good caption should be detailed, accurate, and describe the key elements visible in the image.",
        threshold=0.7,
    )

    # Test case: Image with caption (using DashScope official example image)
    print("\n--- Test Case 1: Image with Caption ---")
    test_input = [
        MLLMImage(
            url="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
        ),
        "Describe this image in detail",
    ]
    test_output = ["A young girl with a friendly dog in an outdoor setting"]

    try:
        result = await grader.aevaluate(input=test_input, actual_output=test_output)

        print(f"\nScore: {result.score:.4f}")
        print(f"\nReason:\n{result.reason}")
        print(f"\nMetadata keys: {list(result.metadata.keys())}")
        print(f"Raw score: {result.metadata.get('raw_score')}")
        print(f"Generated steps: {result.metadata.get('evaluation_steps')}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)


async def test_geval_with_explicit_steps():
    """Test G-Eval with explicitly provided evaluation steps"""
    print("=" * 80)
    print("Test 2: G-Eval with Explicit Evaluation Steps")
    print("=" * 80)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    model = OpenAIChatModel(
        model_name="qwen-vl-max",
        api_key=api_key,
        base_url=base_url,
        generate_kwargs={"temperature": 0.1},
    )

    # Create grader with explicit steps
    grader = MultimodalGEvalGrader(
        model=model,
        evaluation_name="Visual Question Answering",
        evaluation_params=[
            MLLMTestCaseParams.INPUT,
            MLLMTestCaseParams.ACTUAL_OUTPUT,
            MLLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        criteria="Evaluate the accuracy and completeness of the answer to the visual question",
        evaluation_steps=[
            "Analyze the content of the image",
            "Understand the question being asked",
            "Compare the actual answer with the expected answer",
            "Evaluate accuracy and completeness of the response",
        ],
        threshold=0.6,
    )

    print("\n--- Test Case 2: VQA Evaluation ---")
    test_input = [
        MLLMImage(url="https://dashscope.oss-cn-beijing.aliyuncs.com/images/cat.png"),
        "What animal can you see in this image?",
    ]
    test_output = ["I can see a cat"]
    expected_output = ["The image shows a cat"]

    try:
        result = await grader.aevaluate(
            input=test_input,
            actual_output=test_output,
            expected_output=expected_output,
        )

        print(f"\nScore: {result.score:.4f}")
        print(f"\nReason:\n{result.reason}")
        print(
            f"Raw score: {result.metadata.get('raw_score')}/{result.metadata.get('score_range')[1]}",
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)


async def test_geval_with_rubric():
    """Test G-Eval with detailed scoring rubric"""
    print("=" * 80)
    print("Test 3: G-Eval with Detailed Rubric")
    print("=" * 80)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    model = OpenAIChatModel(
        model_name="qwen-vl-max",
        api_key=api_key,
        base_url=base_url,
        generate_kwargs={"temperature": 0.1},
    )

    # Create grader with rubric
    grader = MultimodalGEvalGrader(
        model=model,
        evaluation_name="Image Description Quality",
        evaluation_params=[MLLMTestCaseParams.INPUT, MLLMTestCaseParams.ACTUAL_OUTPUT],
        criteria="Evaluate the quality of image descriptions based on detail, accuracy, and clarity",
        rubric=[
            Rubric(
                score_range=(0, 3),
                expected_outcome="Poor quality: Description is inaccurate, vague, or missing key elements",
            ),
            Rubric(
                score_range=(4, 6),
                expected_outcome="Moderate quality: Description captures some elements but lacks detail or has minor inaccuracies",
            ),
            Rubric(
                score_range=(7, 8),
                expected_outcome="Good quality: Description is detailed and mostly accurate with minor omissions",
            ),
            Rubric(
                score_range=(9, 10),
                expected_outcome="Excellent quality: Description is comprehensive, accurate, and well-articulated",
            ),
        ],
        threshold=0.7,
    )

    print("\n--- Test Case 3: With Rubric ---")
    test_input = [
        MLLMImage(
            url="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
        ),
        "Describe this image",
    ]
    test_output = [
        "This image shows a young girl playing with a golden retriever dog outdoors on a sunny day",
    ]

    try:
        result = await grader.aevaluate(input=test_input, actual_output=test_output)

        print(f"\nScore: {result.score:.4f}")
        print(f"\nReason:\n{result.reason}")
        print(f"Raw score: {result.metadata.get('raw_score')}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)


async def test_geval_with_local_image():
    """Test G-Eval with local base64 encoded image"""
    print("=" * 80)
    print("Test 4: G-Eval with Local Base64 Image")
    print("=" * 80)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    model = OpenAIChatModel(
        model_name="qwen-vl-max",
        api_key=api_key,
        base_url=base_url,
    )

    grader = MultimodalGEvalGrader(
        model=model,
        evaluation_name="HTML Rendering Quality",
        evaluation_params=[MLLMTestCaseParams.INPUT, MLLMTestCaseParams.ACTUAL_OUTPUT],
        criteria="Evaluate the quality of HTML rendering including layout, design, and visual appeal",
        threshold=0.6,
    )

    # Check if local test image exists
    test_image_path = (
        "/Users/boyin.liu/Desktop/code/RM-Gallery-git/data/test_images/html_good_1.png"
    )

    print("\n--- Test Case 4: Local Image ---")
    print(f"Checking for test image: {test_image_path}")

    if os.path.exists(test_image_path):
        import base64

        with open(test_image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        test_input = ["Create a well-designed HTML page"]
        test_output = [
            "Here is the rendered HTML page:",
            MLLMImage(base64=image_data, format="png"),
        ]

        try:
            result = await grader.aevaluate(input=test_input, actual_output=test_output)

            print(f"\nScore: {result.score:.4f}")
            print(f"\nReason (first 300 chars):\n{result.reason[:300]}...")
            print(f"Raw score: {result.metadata.get('raw_score')}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("Test image not found, skipping local image test")

    print("\n" + "=" * 80)


async def test_geval_multiple_images():
    """Test G-Eval with multiple images in input"""
    print("=" * 80)
    print("Test 5: G-Eval with Multiple Images")
    print("=" * 80)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    model = OpenAIChatModel(
        model_name="qwen-vl-max",
        api_key=api_key,
        base_url=base_url,
    )

    grader = MultimodalGEvalGrader(
        model=model,
        evaluation_name="Multi-Image Comparison",
        evaluation_params=[MLLMTestCaseParams.INPUT, MLLMTestCaseParams.ACTUAL_OUTPUT],
        criteria="Evaluate how well the description captures the similarities and differences between multiple images",
        evaluation_steps=[
            "Analyze each image individually",
            "Compare the images to identify similarities",
            "Compare the images to identify differences",
            "Evaluate if the description accurately captures both",
        ],
        threshold=0.6,
    )

    print("\n--- Test Case 5: Multiple Images ---")
    test_input = [
        "Compare these two images:",
        MLLMImage(
            url="https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg",
        ),
        MLLMImage(url="https://dashscope.oss-cn-beijing.aliyuncs.com/images/cat.png"),
        "What are the similarities and differences?",
    ]
    test_output = [
        "The first image shows a dog with a girl outdoors, while the second image shows a cat. Both feature domestic animals but in different settings and contexts.",
    ]

    try:
        result = await grader.aevaluate(input=test_input, actual_output=test_output)

        print(f"\nScore: {result.score:.4f}")
        print(f"\nReason:\n{result.reason}")
        print(f"Raw score: {result.metadata.get('raw_score')}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)


async def test_geval_error_handling():
    """Test error handling for missing parameters"""
    print("=" * 80)
    print("Test 6: Error Handling - Missing Parameters")
    print("=" * 80)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    model = OpenAIChatModel(
        model_name="qwen-vl-max",
        api_key=api_key,
        base_url=base_url,
    )

    grader = MultimodalGEvalGrader(
        model=model,
        evaluation_name="Test",
        evaluation_params=[MLLMTestCaseParams.INPUT, MLLMTestCaseParams.ACTUAL_OUTPUT],
        criteria="Test criteria",
    )

    print("\n--- Test Case 6: Missing Required Parameter ---")

    try:
        # Call without providing actual_output
        result = await grader.aevaluate(
            input=["Test input"],
            # Missing: actual_output
        )

        print(f"Score: {result.score:.4f}")
        print(f"Reason: {result.reason}")
        print(f"Has error: {'error' in result.metadata}")
    except Exception as e:
        print(f"Exception raised (expected): {e}")

    print("\n" + "=" * 80)


async def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("MultimodalGEvalGrader Test Suite")
    print("=" * 80)

    # Check environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("\nError: OPENAI_API_KEY environment variable not set")
        print("Please set it using:")
        print('export OPENAI_API_KEY="your-api-key"')  # pragma: allowlist secret
        return

    if not base_url:
        print("\nWarning: OPENAI_BASE_URL not set, will use default OpenAI endpoint")

    print("\nConfiguration:")
    print(f"  API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"  Base URL: {base_url or 'default'}")
    print("")

    # Run tests
    try:
        await test_geval_basic()
        await test_geval_with_explicit_steps()
        await test_geval_with_rubric()
        await test_geval_with_local_image()
        await test_geval_multiple_images()
        await test_geval_error_handling()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test Suite Complete")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
