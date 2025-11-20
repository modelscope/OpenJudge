# -*- coding: utf-8 -*-
"""
Tests for StringMatchGrader
"""

import pytest

from rm_gallery.gallery.grader.text.string_match import StringMatchGrader

# pylint: disable=too-many-public-methods


@pytest.mark.asyncio
class TestStringMatchGrader:
    """Test cases for unified StringMatchGrader"""

    @pytest.fixture
    def grader(self):
        """Create grader instance"""
        return StringMatchGrader()

    async def test_exact_match_case_sensitive(self, grader):
        """Test exact match with case sensitivity"""
        result = await grader.aevaluate(
            reference="Hello World",
            candidate="Hello World",
            algorithm="exact_match",
            case_sensitive=True,
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_exact_match_case_insensitive(self, grader):
        """Test exact match without case sensitivity"""
        result = await grader.aevaluate(
            reference="Hello World",
            candidate="hello world",
            algorithm="exact_match",
            case_sensitive=False,
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_exact_match_ignore_whitespace(self, grader):
        """Test exact match ignoring whitespace"""
        result = await grader.aevaluate(
            reference="Hello World",
            candidate="HelloWorld",
            algorithm="exact_match",
            ignore_whitespace=True,
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_prefix_match_success(self, grader):
        """Test prefix match success"""
        result = await grader.aevaluate(
            reference="Hello",
            candidate="Hello World",
            algorithm="prefix_match",
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_prefix_match_failure(self, grader):
        """Test prefix match failure"""
        result = await grader.aevaluate(
            reference="World",
            candidate="Hello World",
            algorithm="prefix_match",
        )
        assert result.score == 0.0
        assert result.metadata["matched"] is False

    async def test_suffix_match_success(self, grader):
        """Test suffix match success"""
        result = await grader.aevaluate(
            reference="World",
            candidate="Hello World",
            algorithm="suffix_match",
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_suffix_match_failure(self, grader):
        """Test suffix match failure"""
        result = await grader.aevaluate(
            reference="Hello",
            candidate="Hello World",
            algorithm="suffix_match",
        )
        assert result.score == 0.0
        assert result.metadata["matched"] is False

    async def test_regex_match_success(self, grader):
        """Test regex match success"""
        result = await grader.aevaluate(
            reference=r"\d{3}-\d{4}",
            candidate="My phone is 123-4567",
            algorithm="regex_match",
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_regex_match_with_pattern_param(self, grader):
        """Test regex match with pattern parameter"""
        result = await grader.aevaluate(
            reference="",
            candidate="test@example.com",
            algorithm="regex_match",
            pattern=r"[\w.-]+@[\w.-]+\.\w+",
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_regex_match_invalid_pattern(self, grader):
        """Test regex match with invalid pattern"""
        result = await grader.aevaluate(
            reference="[invalid(",
            candidate="test",
            algorithm="regex_match",
        )
        assert result.score == 0.0
        assert "error" in result.metadata

    async def test_substring_match_success(self, grader):
        """Test substring match success"""
        result = await grader.aevaluate(
            reference="cat",
            candidate="The cat sat on the mat",
            algorithm="substring_match",
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_substring_match_failure(self, grader):
        """Test substring match failure"""
        result = await grader.aevaluate(
            reference="dog",
            candidate="The cat sat on the mat",
            algorithm="substring_match",
        )
        assert result.score == 0.0
        assert result.metadata["matched"] is False

    async def test_substring_match_bidirectional(self, grader):
        """Test substring match bidirectional"""
        result = await grader.aevaluate(
            reference="The cat sat on the mat",
            candidate="cat",
            algorithm="substring_match",
            bidirectional=True,
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True

    async def test_contains_all_success(self, grader):
        """Test contains all success"""
        result = await grader.aevaluate(
            reference="",
            candidate="The cat sat on the mat",
            algorithm="contains_all",
            substrings=["cat", "mat"],
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True
        assert len(result.metadata["missing_substrings"]) == 0

    async def test_contains_all_partial(self, grader):
        """Test contains all partial match"""
        result = await grader.aevaluate(
            reference="",
            candidate="The cat sat on the mat",
            algorithm="contains_all",
            substrings=["cat", "dog", "mat"],
        )
        assert result.score == pytest.approx(2.0 / 3.0)
        assert result.metadata["matched"] is False
        assert "dog" in result.metadata["missing_substrings"]

    async def test_contains_any_success(self, grader):
        """Test contains any success"""
        result = await grader.aevaluate(
            reference="",
            candidate="The cat sat on the mat",
            algorithm="contains_any",
            substrings=["dog", "cat"],
        )
        assert result.score == 1.0
        assert result.metadata["matched"] is True
        assert "cat" in result.metadata["matched_substrings"]

    async def test_contains_any_failure(self, grader):
        """Test contains any failure"""
        result = await grader.aevaluate(
            reference="",
            candidate="The cat sat on the mat",
            algorithm="contains_any",
            substrings=["dog", "bird"],
        )
        assert result.score == 0.0
        assert result.metadata["matched"] is False

    async def test_word_overlap(self, grader):
        """Test word overlap"""
        result = await grader.aevaluate(
            reference="the cat sat on the mat",
            candidate="the dog sat on the rug",
            algorithm="word_overlap",
        )
        # Overlapping words: "the", "sat", "on" = 3 out of 5 unique words in reference
        # Reference has: {"the", "cat", "sat", "on", "mat"} = 5 unique words
        # Overlap: {"the", "sat", "on"} = 3 words
        assert result.score == pytest.approx(3.0 / 5.0)

    async def test_char_overlap(self, grader):
        """Test character overlap"""
        result = await grader.aevaluate(
            reference="hello",
            candidate="helo",
            algorithm="char_overlap",
        )
        # All characters in "hello" {h, e, l, o} are in "helo"
        assert result.score == 1.0

    async def test_invalid_algorithm(self, grader):
        """Test invalid algorithm"""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            await grader.aevaluate(
                reference="test",
                candidate="test",
                algorithm="invalid_algorithm",
            )

    async def test_algorithm_metadata(self, grader):
        """Test that algorithm is included in metadata"""
        result = await grader.aevaluate(
            reference="test",
            candidate="test",
            algorithm="exact_match",
        )
        assert result.metadata["algorithm"] == "exact_match"
