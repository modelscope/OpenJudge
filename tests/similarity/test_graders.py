# -*- coding: utf-8 -*-
"""
Unit Tests for New Grader Architecture

Tests for all Grader classes after Core class merging.
"""

import pytest

from rm_gallery.gallery.grader.format.json_match import (
    JsonMatchGrader,
    JsonValidatorGrader,
)
from rm_gallery.gallery.grader.text.similarity import SimilarityGrader
from rm_gallery.gallery.grader.text.string_match import StringMatchGrader


class TestFuzzyMatchGrader:
    """Test FuzzyMatchGrader"""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test exact match returns perfect score"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="hello world",
            candidate="hello world",
            algorithm="fuzzy_match",
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_partial_match(self):
        """Test partial matching"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="hello world",
            candidate="hello worl",
            algorithm="fuzzy_match",  # Missing 'd'
        )

        assert 0.9 < result.score < 1.0

    @pytest.mark.asyncio
    async def test_different_methods(self):
        """Test different fuzzy matching methods"""
        # Ratio method
        grader_ratio = SimilarityGrader()
        result = await grader_ratio.aevaluate(
            reference="the quick brown fox",
            candidate="the quick brown fox",
            algorithm="fuzzy_match",
            method="ratio",
        )
        assert result.score == 1.0
        assert result.metadata["method"] == "ratio"

        # Token sort ratio
        grader_token = SimilarityGrader()
        result = await grader_token.aevaluate(
            reference="brown quick the fox",
            candidate="the quick brown fox",
            algorithm="fuzzy_match",
            method="token_sort_ratio",
        )
        assert result.score == 1.0


class TestEditDistanceGrader:
    """Test EditDistanceGrader"""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test exact match"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="hello",
            candidate="hello",
            algorithm="edit_distance",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_one_char_difference(self):
        """Test one character difference"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="hello",
            candidate="helo",
            algorithm="edit_distance",
        )

        assert 0.7 < result.score < 1.0
        assert result.metadata["raw_distance"] == 1


class TestJsonMatchGrader:
    """Test JsonMatchGrader"""

    @pytest.mark.asyncio
    async def test_exact_match_simple(self):
        """Test exact match for simple JSON objects"""
        grader = JsonMatchGrader()
        result = await grader.aevaluate(
            reference='{"name": "Alice", "age": 30}',
            candidate='{"name": "Alice", "age": 30}',
        )

        assert result.score == 1.0
        assert result.metadata["matched"] is True

    @pytest.mark.asyncio
    async def test_different_key_order(self):
        """Test that dict key order doesn't matter"""
        grader = JsonMatchGrader()
        result = await grader.aevaluate(
            reference='{"name": "Alice", "age": 30}',
            candidate='{"age": 30, "name": "Alice"}',
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_match(self):
        """Test no match when values differ"""
        grader = JsonMatchGrader()
        result = await grader.aevaluate(
            reference='{"name": "Alice"}',
            candidate='{"name": "Bob"}',
        )

        assert result.score == 0.0
        assert result.metadata["matched"] is False

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Test handling of invalid JSON"""
        grader = JsonMatchGrader()
        result = await grader.aevaluate(
            reference='{"name": "Alice"}',
            candidate="not valid json",
        )

        assert result.score == 0.0
        assert "error" in result.metadata


class TestJsonValidatorGrader:
    """Test JsonValidatorGrader"""

    @pytest.mark.asyncio
    async def test_valid_json(self):
        """Test valid JSON object"""
        grader = JsonValidatorGrader()
        result = await grader.aevaluate(candidate='{"name": "Alice", "age": 30}')

        assert result.score == 1.0
        assert result.metadata["is_valid"] is True

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Test invalid JSON (malformed)"""
        grader = JsonValidatorGrader()
        result = await grader.aevaluate(candidate='{"name": "Alice"')  # Missing }

        assert result.score == 0.0
        assert result.metadata["is_valid"] is False

    @pytest.mark.asyncio
    async def test_valid_json_array(self):
        """Test valid JSON array"""
        grader = JsonValidatorGrader()
        result = await grader.aevaluate(candidate='[1, 2, 3, "test"]')

        assert result.score == 1.0
        assert result.metadata["is_valid"] is True


class TestStringMatchGrader:
    """Test StringMatchGrader"""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test exact match"""
        grader = StringMatchGrader()
        result = await grader.aevaluate(
            reference="hello",
            candidate="hello",
            algorithm="exact_match",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        """Test case insensitive matching"""
        grader = StringMatchGrader()
        result = await grader.aevaluate(
            reference="Hello",
            candidate="hello",
            algorithm="exact_match",
            case_sensitive=False,
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_match(self):
        """Test no match"""
        grader = StringMatchGrader()
        result = await grader.aevaluate(
            reference="hello",
            candidate="world",
            algorithm="exact_match",
        )

        assert result.score == 0.0


class TestSubstringMatchGrader:
    """Test SubstringMatchGrader"""

    @pytest.mark.asyncio
    async def test_substring_found(self):
        """Test substring found"""
        grader = StringMatchGrader()
        result = await grader.aevaluate(
            reference="cat",
            candidate="The cat sat on the mat",
            algorithm="substring_match",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_substring_not_found(self):
        """Test substring not found"""
        grader = StringMatchGrader()
        result = await grader.aevaluate(
            reference="dog",
            candidate="The cat sat on the mat",
            algorithm="substring_match",
        )

        assert result.score == 0.0


class TestCosineSimilarityGrader:
    """Test CosineSimilarityGrader"""

    @pytest.mark.asyncio
    async def test_identical_texts(self):
        """Test identical texts have similarity 1.0"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat sat on the mat",
            candidate="the cat sat on the mat",
            algorithm="cosine",
        )

        assert result.score > 0.99

    @pytest.mark.asyncio
    async def test_similar_texts(self):
        """Test similar texts have high similarity"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="the cat sat on the mat",
            candidate="the dog sat on the mat",
            algorithm="cosine",
        )

        assert 0.5 < result.score < 1.0


class TestJaccardSimilarityGrader:
    """Test JaccardSimilarityGrader"""

    @pytest.mark.asyncio
    async def test_identical_texts(self):
        """Test identical texts"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="hello world",
            candidate="hello world",
            algorithm="jaccard",
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_overlap(self):
        """Test no overlap"""
        grader = SimilarityGrader()
        result = await grader.aevaluate(
            reference="hello world",
            candidate="goodbye universe",
            algorithm="jaccard",
        )

        assert result.score == 0.0


class TestAllGraderBasics:
    """Test that all graders work with basic inputs"""

    @pytest.mark.asyncio
    async def test_all_graders_can_evaluate(self):
        """Test that all graders can be instantiated and evaluate"""
        reference = "the cat sat on the mat"
        candidate = "the cat sat on the mat"

        # Test all similarity algorithms
        similarity_grader = SimilarityGrader()
        algorithms = [
            "bleu",
            "sentence_bleu",
            "gleu",
            "chrf",
            "meteor",
            "rouge",
            "rouge1",
            "rouge2",
            "rougeL",
            "rouge3",
            "rouge4",
            "rouge5",
            "f1_score",
            "token_f1",
            "fuzzy_match",
            "edit_distance",
            "cosine",
            "jaccard",
        ]

        for algorithm in algorithms:
            result = await similarity_grader.aevaluate(
                reference=reference,
                candidate=candidate,
                algorithm=algorithm,
            )
            assert hasattr(result, "score")
            assert hasattr(result, "reason")
            assert hasattr(result, "metadata")
            assert 0.0 <= result.score <= 1.0
            print(f"✓ {algorithm}: score={result.score:.2f}")

        # Test StringMatchGrader with different algorithms
        string_match_grader = StringMatchGrader()
        string_algorithms = [
            "exact_match",
            "prefix_match",
            "suffix_match",
            "regex_match",
            "substring_match",
            "word_overlap",
            "char_overlap",
        ]

        for algorithm in string_algorithms:
            result = await string_match_grader.aevaluate(
                reference=reference,
                candidate=candidate,
                algorithm=algorithm,
            )
            assert hasattr(result, "score")
            assert hasattr(result, "reason")
            assert hasattr(result, "metadata")
            assert 0.0 <= result.score <= 1.0
            print(f"✓ {algorithm}: score={result.score:.2f}")

        # Test contains_all and contains_any with substrings parameter
        result = await string_match_grader.aevaluate(
            reference="",
            candidate=candidate,
            algorithm="contains_all",
            substrings=["test"],
        )
        assert hasattr(result, "score")
        print(f"✓ contains_all: score={result.score:.2f}")

        result = await string_match_grader.aevaluate(
            reference="",
            candidate=candidate,
            algorithm="contains_any",
            substrings=["test"],
        )
        assert hasattr(result, "score")
        print(f"✓ contains_any: score={result.score:.2f}")

        # Test other graders
        json_grader = JsonMatchGrader()
        result = await json_grader.aevaluate(reference=reference, candidate=candidate)
        assert hasattr(result, "score")
        assert hasattr(result, "reason")
        assert hasattr(result, "metadata")
        assert 0.0 <= result.score <= 1.0
        print(f"✓ {json_grader.name}: score={result.score:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
