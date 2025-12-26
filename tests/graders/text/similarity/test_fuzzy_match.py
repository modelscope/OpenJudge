# -*- coding: utf-8 -*-
"""
Unit Tests for Fuzzy Match Grader

Test fuzzy matching functionality including exact match, partial match, and token sorting.
"""

import pytest

from openjudge.graders.text.similarity import SimilarityGrader


class TestFuzzyMatchBasic:
    """Basic fuzzy match functionality tests"""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test exact match returns perfect score"""
        grader = SimilarityGrader(algorithm="fuzzy_match")
        result = await grader.aevaluate(
            reference_response="hello world",
            response="hello world",
        )

        assert result.score == 1.0
        assert "matched" in result.metadata
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_complete_mismatch(self):
        """Test completely different strings return low score"""
        grader = SimilarityGrader(algorithm="fuzzy_match")
        result = await grader.aevaluate(
            reference_response="hello world",
            response="goodbye universe",
        )

        assert 0.0 <= result.score < 0.5
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_partial_match(self):
        """Test partial matching"""
        grader = SimilarityGrader(
            algorithm="fuzzy_match",
        )
        result = await grader.aevaluate(
            reference_response="hello world",
            response="hello worl",
        )

        assert 0.9 < result.score < 1.0

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        """Test case insensitive matching"""
        # Note: FuzzyMatchGrader doesn't have normalize_text parameter
        # Case sensitivity is handled by the fuzzy matching algorithm
        grader = SimilarityGrader(algorithm="fuzzy_match")
        result = await grader.aevaluate(
            reference_response="Hello World",
            response="hello world",
        )

        # Fuzzy match is case-sensitive, so won't be 1.0, but should be high
        assert result.score > 0.8


class TestFuzzyMatchMethods:
    """Test different fuzzy matching methods"""

    @pytest.mark.asyncio
    async def test_ratio_method(self):
        """Test standard ratio method"""
        grader = SimilarityGrader(algorithm="fuzzy_match")
        result = await grader.aevaluate(
            reference_response="the quick brown fox",
            response="the quick brown fox",
            method="ratio",
        )

        assert result.score == 1.0
        assert result.metadata["method"] == "ratio"

    @pytest.mark.asyncio
    async def test_partial_ratio_method(self):
        """Test partial ratio for substring matching"""
        grader = SimilarityGrader(algorithm="fuzzy_match")
        result = await grader.aevaluate(
            reference_response="the quick brown fox jumps",
            response="quick brown fox",
            method="partial_ratio",
        )

        # Partial ratio should give high score for substring match
        assert result.score > 0.8

    @pytest.mark.asyncio
    async def test_token_sort_ratio_method(self):
        """Test token sort ratio for order-independent matching"""
        grader = SimilarityGrader(algorithm="fuzzy_match")
        result = await grader.aevaluate(
            reference_response="brown quick the fox",
            response="the quick brown fox",
            method="token_sort_ratio",
        )

        # Token sort should give perfect score for same words different order
        assert result.score == 1.0


class TestFuzzyMatchMultipleReferences:
    """Test fuzzy matching with multiple reference texts"""

    @pytest.mark.asyncio
    async def test_multiple_references_exact_match(self):
        """Test exact match with multiple references"""
        # Note: Grader architecture evaluates single reference at a time
        # For multiple references, we evaluate each separately and take the best
        grader = SimilarityGrader(algorithm="fuzzy_match")

        references = ["hello world", "hi there", "greetings"]
        candidate = "hello world"

        scores = []
        for ref in references:
            result = await grader.aevaluate(
                reference_response=ref,
                response=candidate,
            )
            scores.append(result.score)

        assert max(scores) == 1.0
        assert len(scores) == 3

    @pytest.mark.asyncio
    async def test_multiple_references_best_match(self):
        """Test that best matching reference is selected"""
        grader = SimilarityGrader(algorithm="fuzzy_match")

        references = ["completely different text", "hello world", "another text"]
        candidate = "hello world"

        scores = []
        for ref in references:
            result = await grader.aevaluate(
                reference_response=ref,
                response=candidate,
            )
            scores.append(result.score)

        # Should match the second reference perfectly
        assert max(scores) == 1.0

    @pytest.mark.asyncio
    async def test_multiple_references_partial_match(self):
        """Test partial matching with multiple references"""
        grader = SimilarityGrader(algorithm="fuzzy_match")

        references = ["hello", "world", "foo bar"]
        candidate = "hello world"

        scores = []
        for ref in references:
            result = await grader.aevaluate(
                reference_response=ref,
                response=candidate,
            )
            scores.append(result.score)

        # Should have some match
        assert max(scores) > 0.0


class TestFuzzyMatchEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_strings(self):
        """Test handling of empty strings"""
        grader = SimilarityGrader(algorithm="fuzzy_match")
        result = await grader.aevaluate(
            reference_response="",
            response="",
        )

        # Empty strings should match perfectly
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_empty_reference_non_empty_candidate(self):
        """Test empty reference with non-empty candidate"""
        grader = SimilarityGrader(algorithm="fuzzy_match")
        result = await grader.aevaluate(
            reference_response="",
            response="hello",
        )

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_very_long_strings(self):
        """Test performance with long strings"""
        grader = SimilarityGrader(algorithm="fuzzy_match")
        long_text = "word " * 1000  # 1000 words
        result = await grader.aevaluate(
            reference_response=long_text,
            response=long_text,
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test handling of special characters"""
        grader = SimilarityGrader(algorithm="fuzzy_match")
        result = await grader.aevaluate(
            reference_response="hello@world#2024!",
            response="hello@world#2024!",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_unicode_characters(self):
        """Test handling of Unicode characters"""
        grader = SimilarityGrader(algorithm="fuzzy_match")
        result = await grader.aevaluate(
            reference_response="你好世界",
            response="你好世界",
        )

        assert result.score == 1.0


class TestFuzzyMatchThreshold:
    """Test threshold functionality"""

    @pytest.mark.asyncio
    async def test_threshold_matching(self):
        """Test threshold-based matching decision"""
        grader = SimilarityGrader(
            algorithm="fuzzy_match",
            threshold=0.9,
        )

        # High similarity - should match
        result = await grader.aevaluate(
            reference_response="the quick brown fox",
            response="the quick brown fo",
        )

        if result.score >= 0.9:
            assert result.metadata["matched"] is True
        else:
            assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_different_thresholds(self):
        """Test different threshold values"""

        # Strict threshold
        grader_strict = SimilarityGrader(
            algorithm="fuzzy_match",
            threshold=0.99,
        )
        result_strict = await grader_strict.aevaluate(
            reference_response="hello world",
            response="hello worl",
        )

        # Lenient threshold
        grader_lenient = SimilarityGrader(
            algorithm="fuzzy_match",
            threshold=0.80,
        )
        result_lenient = await grader_lenient.aevaluate(
            reference_response="hello world",
            response="hello worl",
        )

        # Scores should be the same, but matching decision may differ
        assert result_strict.score == result_lenient.score
