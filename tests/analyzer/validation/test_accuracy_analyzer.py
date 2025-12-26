# -*- coding: utf-8 -*-
"""Tests for the AccuracyAnalyzer."""

import pytest

from openjudge.analyzer.validation import AccuracyAnalysisResult, AccuracyAnalyzer
from openjudge.graders.schema import GraderScore


@pytest.mark.unit
class TestAccuracyAnalyzer:
    """Test suite for AccuracyAnalyzer."""

    def test_initialization(self):
        """Test successful initialization."""
        analyzer = AccuracyAnalyzer()
        assert analyzer.name == "Accuracy Analysis"

    def test_analyze_perfect_accuracy(self):
        """Test analyze method with perfect accuracy."""
        # Prepare test data
        dataset = [
            {"label": 1},
            {"label": 0},
            {"label": 1},
            {"label": 0},
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        # Create analyzer and run analysis
        analyzer = AccuracyAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions
        assert isinstance(result, AccuracyAnalysisResult)
        assert result.name == "Accuracy Analysis"
        assert result.accuracy == 1.0
        assert "explanation" in result.metadata
        assert "4 out of 4" in result.metadata["explanation"]

    def test_analyze_partial_accuracy(self):
        """Test analyze method with partial accuracy."""
        # Prepare test data
        dataset = [
            {"label": 1},
            {"label": 0},
            {"label": 1},
            {"label": 0},
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=1.0, reason="Incorrect"),  # Wrong prediction
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Incorrect"),
        ]

        # Create analyzer and run analysis
        analyzer = AccuracyAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions
        assert isinstance(result, AccuracyAnalysisResult)
        assert result.name == "Accuracy Analysis"
        assert result.accuracy == 0.75  # 3 out of 4 correct
        assert "explanation" in result.metadata
        assert "3 out of 4" in result.metadata["explanation"]

    def test_analyze_empty_data(self):
        """Test analyze method with empty data."""
        # Create analyzer and run analysis with empty data
        analyzer = AccuracyAnalyzer()
        result = analyzer.analyze([], [], label_path="label")

        # Assertions
        assert isinstance(result, AccuracyAnalysisResult)
        assert result.name == "Accuracy Analysis"
        assert result.accuracy == 0.0
        assert "explanation" in result.metadata
        assert "No data or grader results" in result.metadata["explanation"]

    def test_analyze_with_none_values(self):
        """Test analyze method handling None values."""
        # Prepare test data with None values
        dataset = [
            {"label": 1},
            {"label": None},  # This sample should be skipped
            {"label": 1},
            {"label": 0},
        ]

        grader_results = [
            GraderScore(name="test", score=1.0, reason="Correct"),
            GraderScore(name="test", score=0.0, reason="Skipped"),  # Will be skipped due to None label
            GraderScore(name="test", score=0.0, reason="Incorrect"),  # Wrong prediction
            GraderScore(name="test", score=0.0, reason="Correct"),
        ]

        # Create analyzer and run analysis
        analyzer = AccuracyAnalyzer()
        result = analyzer.analyze(dataset, grader_results, label_path="label")

        # Assertions - only 3 samples should be considered (one with None label is skipped)
        # 2 out of 3 predictions are correct (first and last ones)
        assert isinstance(result, AccuracyAnalysisResult)
        assert result.name == "Accuracy Analysis"
        assert result.accuracy == pytest.approx(2 / 3)  # 2 out of 3 correct
        assert "explanation" in result.metadata
