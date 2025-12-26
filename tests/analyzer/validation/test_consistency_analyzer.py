# -*- coding: utf-8 -*-
"""Tests for the ConsistencyAnalyzer."""

import pytest

from openjudge.analyzer.statistical import (
    ConsistencyAnalysisResult,
    ConsistencyAnalyzer,
)
from openjudge.graders.schema import GraderScore


@pytest.mark.unit
class TestConsistencyAnalyzer:
    """Test suite for ConsistencyAnalyzer."""

    def test_initialization(self):
        """Test successful initialization."""
        analyzer = ConsistencyAnalyzer()
        assert analyzer.name == "Consistency Analysis"

    def test_analyze_perfect_consistency(self):
        """Test analyze method with perfect consistency."""
        # Prepare test data - two identical sets of results
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        another_grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        # Create analyzer and run analysis
        analyzer = ConsistencyAnalyzer()
        result = analyzer.analyze([], grader_results, another_grader_results)

        # Assertions
        assert isinstance(result, ConsistencyAnalysisResult)
        assert result.name == "Consistency Analysis"
        assert result.consistency == 1.0
        assert "explanation" in result.metadata
        assert "4 paired evaluations" in result.metadata["explanation"]

    def test_analyze_partial_consistency(self):
        """Test analyze method with partial consistency."""
        # Prepare test data with some differences
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        another_grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=1.0, reason="Incorrect"),  # Different from first run
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        # Create analyzer and run analysis
        analyzer = ConsistencyAnalyzer()
        result = analyzer.analyze([], grader_results, another_grader_results)

        # Assertions
        assert isinstance(result, ConsistencyAnalysisResult)
        assert result.name == "Consistency Analysis"
        assert "explanation" in result.metadata
        assert "4 paired evaluations" in result.metadata["explanation"]

    def test_analyze_empty_data(self):
        """Test analyze method with empty data."""
        # Create analyzer and run analysis with empty data
        analyzer = ConsistencyAnalyzer()
        result = analyzer.analyze([], [], [])

        # Assertions
        assert isinstance(result, ConsistencyAnalysisResult)
        assert result.name == "Consistency Analysis"
        assert result.consistency == 0.0
        assert "explanation" in result.metadata
        assert "No results provided for consistency calculation" in result.metadata["explanation"]

    def test_analyze_mismatched_lengths(self):
        """Test analyze method with mismatched lengths."""
        # Prepare test data with different lengths
        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        another_grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
            GraderScore(name="test", score=1.0, reason="Extra"),  # Extra result
        ]

        # Create analyzer and run analysis
        analyzer = ConsistencyAnalyzer()
        result = analyzer.analyze([], grader_results, another_grader_results)

        # Assertions - should only compare first 2 results
        assert isinstance(result, ConsistencyAnalysisResult)
        assert result.name == "Consistency Analysis"
        assert result.consistency == pytest.approx(1.0)  # First 2 results are consistent
        assert "explanation" in result.metadata
        assert "2 paired evaluations" in result.metadata["explanation"]
