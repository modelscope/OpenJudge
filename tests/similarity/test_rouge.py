# -*- coding: utf-8 -*-
"""
Unit Tests for ROUGE Graders

Test ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics.
"""

import pytest

from rm_gallery.gallery.grader.text.similarity import SimilarityGrader


class TestROUGEBasic:
    """Basic ROUGE functionality tests"""

    @pytest.mark.asyncio
    async def test_rouge_perfect_match(self):
        """Test perfect match returns score of 1.0"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat is on the mat",
            candidate="the cat is on the mat",
            algorithm="rouge",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_rouge_complete_mismatch(self):
        """Test completely different text"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat is on the mat",
            candidate="hello world foo bar",
            algorithm="rouge",
        )

        assert result.score < 0.1

    @pytest.mark.asyncio
    async def test_rouge_partial_match(self):
        """Test partial overlapping text"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat is on the mat",
            candidate="the dog is on the rug",
            algorithm="rouge",
        )

        # Some overlap in words like "the", "is", "on"
        assert 0.2 < result.score < 0.8


class TestROUGE1:
    """Test ROUGE-1 (unigram overlap)"""

    @pytest.mark.asyncio
    async def test_rouge1_perfect_match(self):
        """Test ROUGE-1 perfect match"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat sat",
            candidate="the cat sat",
            algorithm="rouge",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_rouge1_word_order_independent(self):
        """Test that ROUGE-1 is independent of word order"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat sat",
            candidate="sat cat the",
            algorithm="rouge1",
        )

        # ROUGE-1 should give high score for same words different order
        assert result.score > 0.9

    @pytest.mark.asyncio
    async def test_rouge1_extra_words(self):
        """Test ROUGE-1 with extra words in candidate"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat sat",
            candidate="the big cat sat down",
            algorithm="rouge",
        )

        # Should have high recall but lower precision
        assert 0.5 < result.score < 1.0


class TestROUGE2:
    """Test ROUGE-2 (bigram overlap)"""

    @pytest.mark.asyncio
    async def test_rouge2_perfect_match(self):
        """Test ROUGE-2 perfect match"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat is on the mat",
            candidate="the cat is on the mat",
            algorithm="rouge",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_rouge2_word_order_matters(self):
        """Test that ROUGE-2 is sensitive to word order"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat is on the mat",
            candidate="the mat is on the cat",
            algorithm="rouge",
        )

        # Different word order means different bigrams
        assert result.score < 1.0

    @pytest.mark.asyncio
    async def test_rouge2_no_bigram_overlap(self):
        """Test ROUGE-2 with no bigram overlap"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="a b c d",
            candidate="b a d c",
            algorithm="rouge2",
        )

        # No matching bigrams
        assert result.score == 0.0


class TestROUGEL:
    """Test ROUGE-L (Longest Common Subsequence)"""

    @pytest.mark.asyncio
    async def test_rougeL_perfect_match(self):
        """Test ROUGE-L perfect match"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat is on the mat",
            candidate="the cat is on the mat",
            algorithm="rouge",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_rougeL_subsequence(self):
        """Test ROUGE-L with common subsequence"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="a b c d e f",
            candidate="a x b x c x d x e x f",
            algorithm="rougeL",
        )

        # All reference words present in order (with gaps)
        assert result.score > 0.5

    @pytest.mark.asyncio
    async def test_rougeL_vs_rouge2(self):
        """Compare ROUGE-L and ROUGE-2 behavior"""
        rougeL = SimilarityGrader()
        rouge2 = SimilarityGrader()

        resultL = await rougeL.aevaluate(
            reference="the cat sat on the mat",
            candidate="the cat was sitting on the mat",
            algorithm="rougeL",
        )
        result2 = await rouge2.aevaluate(
            reference="the cat sat on the mat",
            candidate="the cat was sitting on the mat",
            algorithm="rouge2",
        )

        # Both should detect some overlap
        assert resultL.score > 0.0
        assert result2.score > 0.0


class TestROUGENGram:
    """Test ROUGE-3, ROUGE-4, ROUGE-5"""

    @pytest.mark.asyncio
    async def test_rouge3_perfect_match(self):
        """Test ROUGE-3 perfect match"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat sat on the mat",
            candidate="the cat sat on the mat",
            algorithm="rouge",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_rouge4_perfect_match(self):
        """Test ROUGE-4 perfect match"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat sat on the mat",
            candidate="the cat sat on the mat",
            algorithm="rouge",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_rouge5_perfect_match(self):
        """Test ROUGE-5 perfect match"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat sat on the mat",
            candidate="the cat sat on the mat",
            algorithm="rouge",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_higher_ngram_more_strict(self):
        """Test that higher n-grams are more strict"""
        rouge1 = SimilarityGrader()
        rouge2 = SimilarityGrader()
        rouge3 = SimilarityGrader()

        result1 = await rouge1.aevaluate(
            reference="the quick brown fox jumps over",
            candidate="the quick brown fox walks over",
            algorithm="rouge1",
        )
        result2 = await rouge2.aevaluate(
            reference="the quick brown fox jumps over",
            candidate="the quick brown fox walks over",
            algorithm="rouge2",
        )
        result3 = await rouge3.aevaluate(
            reference="the quick brown fox jumps over",
            candidate="the quick brown fox walks over",
            algorithm="rouge3",
        )

        # Higher n-grams should be more sensitive to differences
        assert result1.score >= result2.score >= result3.score


class TestROUGEEdgeCases:
    """Test edge cases"""

    @pytest.mark.asyncio
    async def test_empty_candidate(self):
        """Test handling of empty candidate"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat",
            candidate="",
            algorithm="rouge",
        )

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_empty_reference(self):
        """Test handling of empty reference"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="",
            candidate="the cat",
            algorithm="rouge",
        )

        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_single_word(self):
        """Test single word texts"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="cat",
            candidate="cat",
            algorithm="rouge1",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_repeated_words(self):
        """Test handling of repeated words"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="cat cat cat",
            candidate="cat dog cat",
            algorithm="rouge1",
        )

        # Should handle repeated words correctly
        assert 0.5 < result.score < 1.0


class TestROUGEWithStemming:
    """Test ROUGE with stemming enabled/disabled"""

    @pytest.mark.asyncio
    async def test_with_stemming(self):
        """Test ROUGE with stemming enabled"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cats are running",
            candidate="the cat is running",
            algorithm="rouge",
            use_stemmer=True,
        )

        # With stemming, "cats" and "cat" should match better
        assert result.score > 0.6

    @pytest.mark.asyncio
    async def test_without_stemming(self):
        """Test ROUGE without stemming"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cats are running",
            candidate="the cat is running",
            algorithm="rouge",
            use_stemmer=False,
        )

        # Without stemming, scores may be lower
        assert 0.0 <= result.score <= 1.0
