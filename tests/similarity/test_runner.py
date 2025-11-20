# -*- coding: utf-8 -*-
"""
Test Runner for Grader Architecture

Quick test to verify core Grader functionality works before running full test suite.
"""

import pytest

# pylint: disable=line-too-long


def test_basic_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    # Test imports for Grader architecture - avoid circular import by importing from specific modules
    # pylint: disable=unused-import
    from rm_gallery.gallery.grader.format.json_match import (  # noqa: F401
        JsonMatchGrader,
    )
    from rm_gallery.gallery.grader.text.similarity import SimilarityGrader  # noqa: F401
    from rm_gallery.gallery.grader.text.string_match import (  # noqa: F401
        StringMatchGrader,
    )

    print("✓ All imports successful")


@pytest.mark.asyncio
async def test_fuzzy_match():
    """Test fuzzy match grader"""
    print("\nTesting Fuzzy Match...")

    from rm_gallery.gallery.grader.text.similarity import SimilarityGrader

    grader = SimilarityGrader()
    result = await grader.aevaluate(
        reference="hello world",
        candidate="hello world",
        algorithm="fuzzy_match",
    )

    if result.score == 1.0:
        print(f"✓ Fuzzy match works: {result.score}")
        return True
    else:
        print(f"✗ Unexpected score: {result.score}")
        return False


@pytest.mark.asyncio
async def test_exact_match():
    """Test exact match grader"""
    print("\nTesting Exact Match...")

    from rm_gallery.gallery.grader.text.string_match import StringMatchGrader

    grader = StringMatchGrader()
    result = await grader.aevaluate(reference="test", candidate="test")

    if result.score == 1.0:
        print(f"✓ Exact match works: {result.score}")
        return True
    else:
        print(f"✗ Unexpected score: {result.score}")
        return False


@pytest.mark.asyncio
async def test_json_match():
    """Test JSON match grader"""
    print("\nTesting JSON Match...")

    from rm_gallery.gallery.grader.format.json_match import JsonMatchGrader

    grader = JsonMatchGrader()
    result = await grader.aevaluate(
        reference='{"name": "Alice"}',
        candidate='{"name": "Alice"}',
    )

    if result.score == 1.0:
        print(f"✓ JSON match works: {result.score}")
        return True
    else:
        print(f"✗ Unexpected score: {result.score}")
        return False


@pytest.mark.asyncio
async def test_multiple_graders():
    """Test using multiple graders"""
    print("\nTesting Multiple Graders...")

    from rm_gallery.gallery.grader.text.similarity import SimilarityGrader
    from rm_gallery.gallery.grader.text.string_match import StringMatchGrader

    reference = "the cat sat on the mat"
    candidate = "the cat sat on the mat"

    # Test similarity grader with different algorithms
    similarity_grader = SimilarityGrader()
    result1 = await similarity_grader.aevaluate(
        reference=reference,
        candidate=candidate,
        algorithm="fuzzy_match",
    )
    print(f"  - fuzzy_match: {result1.score:.4f}")

    result2 = await similarity_grader.aevaluate(
        reference=reference,
        candidate=candidate,
        algorithm="bleu",
    )
    print(f"  - bleu: {result2.score:.4f}")

    # Test exact match grader
    exact_grader = StringMatchGrader()
    result3 = await exact_grader.aevaluate(reference=reference, candidate=candidate)
    print(f"  - exact_match: {result3.score:.4f}")

    print("✓ Multiple graders work")
    return True


def main():
    """Run all quick tests"""
    import asyncio

    print("=" * 60)
    print("Grader Architecture Test Runner")
    print("=" * 60)

    # Run async tests
    loop = asyncio.get_event_loop()

    tests = [
        test_basic_imports,
    ]

    async_tests = [
        test_fuzzy_match,
        test_exact_match,
        test_json_match,
        test_multiple_graders,
    ]

    passed = 0
    failed = 0

    # Run sync tests
    for test in tests:
        try:
            if test() is not False:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} raised exception: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    # Run async tests
    for test in async_tests:
        try:
            if loop.run_until_complete(test()) is not False:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} raised exception: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
