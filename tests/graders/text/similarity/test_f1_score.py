# -*- coding: utf-8 -*-
"""
Unit Tests for F1 Score Grader

Test token-based F1 score calculation following OpenAI Evals implementation.
"""

import pytest

from rm_gallery.core.graders.text.similarity import SimilarityGrader


class TestF1ScoreBasic:
    """Basic F1 score functionality tests"""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test exact match returns perfect F1 score"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="hello world",
            response="hello world",
            algorithm="f1_score",
        )

        assert result.score == 1.0
        assert result.metadata["precision"] == 1.0
        assert result.metadata["recall"] == 1.0

    @pytest.mark.asyncio
    async def test_no_overlap(self):
        """Test completely different strings return 0 F1 score"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="hello world",
            response="goodbye universe",
            algorithm="f1_score",
        )

        assert result.score == 0.0
        assert result.metadata["precision"] == 0.0
        assert result.metadata["recall"] == 0.0

    @pytest.mark.asyncio
    async def test_partial_overlap(self):
        """Test partial token overlap"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat is on the mat",
            response="cat on mat",
            algorithm="f1_score",
        )

        # Tokens: reference has more tokens, candidate has subset
        # The exact score depends on normalization
        assert 0.5 < result.score < 1.0
        assert result.metadata["precision"] >= 0.5
        assert result.metadata["recall"] >= 0.5

    @pytest.mark.asyncio
    async def test_word_order_matters(self):
        """Test that word order doesn't affect F1 (token-based)"""
        grader = SimilarityGrader()

        result1 = await grader.aevaluate(
            reference="the quick brown fox",
            response="fox brown quick the",
            algorithm="f1_score",
        )
        result2 = await grader.aevaluate(
            reference="the quick brown fox",
            response="the quick brown fox",
            algorithm="f1_score",
        )

        # Both should have similar F1 (same tokens, may differ due to normalization)
        assert abs(result1.score - result2.score) < 0.1


class TestF1ScoreNormalization:
    """Test normalization effects on F1 score"""

    @pytest.mark.asyncio
    async def test_with_normalization(self):
        """Test with normalization enabled"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="Hello World",
            response="hello world",
            algorithm="f1_score",
            normalize=True,
        )

        # With normalization, case differences shouldn't matter
        assert result.score > 0.9

    @pytest.mark.asyncio
    async def test_without_normalization(self):
        """Test that disabling normalization preserves case"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="Hello World",
            response="hello world",
            algorithm="f1_score",
            normalize=False,
        )

        # Without normalization, case differences matter
        assert result.score < 1.0


class TestF1ScoreEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_strings(self):
        """Test handling of empty strings"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="",
            response="",
            algorithm="f1_score",
        )

        # Both empty - perfect match
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_empty_reference(self):
        """Test empty reference with non-empty candidate"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="",
            response="hello",
            algorithm="f1_score",
        )

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_empty_candidate(self):
        """Test non-empty reference with empty candidate"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="hello",
            response="",
            algorithm="f1_score",
        )

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_single_token(self):
        """Test single token matching"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="hello",
            response="hello",
            algorithm="f1_score",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_duplicate_tokens(self):
        """Test handling of duplicate tokens"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="hello hello world",
            response="hello world world",
            algorithm="f1_score",
        )

        # Should handle token counts correctly
        assert 0.5 < result.score < 0.8


class TestF1ScorePrecisionRecall:
    """Test precision and recall calculations"""

    @pytest.mark.asyncio
    async def test_high_precision_low_recall(self):
        """Test case with high precision but low recall"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the quick brown fox jumps over the lazy dog",
            response="quick brown fox",
            algorithm="f1_score",
        )

        # All candidate tokens should be in reference (high precision)
        # But only fraction of reference tokens in candidate (low recall)
        assert result.metadata["precision"] > result.metadata["recall"]

    @pytest.mark.asyncio
    async def test_low_precision_high_recall(self):
        """Test case with low precision but high recall"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="quick brown fox",
            response="the quick brown fox jumps over the lazy dog",
            algorithm="f1_score",
        )

        # All reference tokens should be in candidate (high recall)
        # But many extra candidate tokens (low precision)
        assert result.metadata["recall"] > result.metadata["precision"]


class TestTokenF1Alias:
    """Test TokenF1Grader alias"""

    @pytest.mark.asyncio
    async def test_alias_works(self):
        """Test that TokenF1Grader works"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="hello world",
            response="hello world",
            algorithm="token_f1",
        )

        assert result.score == 1.0
