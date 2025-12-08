#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete demo test for HelpfulnessGrader showing all types of tests.

This file demonstrates all types of tests recommended in the GRADER_TESTING_STRATEGY.md
using HelpfulnessGrader as an example of LLMGrader:

1. Unit tests (offline testing with mocks)
2. Integration tests (testing with real components)
3. Regression tests (testing previously fixed bugs)
4. Performance tests (benchmarking)
5. Quality tests (evaluation against gold standard datasets)

Example:
    Run all tests:
    ```bash
    poetry run pytest tests/graders/common/test_helpfulness_complete_demo.py -v
    ```

    Run only unit tests:
    ```bash
    poetry run pytest tests/graders/common/test_helpfulness_complete_demo.py -m unit
    ```

    Run only integration tests:
    ```bash
    poetry run pytest tests/graders/common/test_helpfulness_complete_demo.py -m integration
    ```

    Run only regression tests:
    ```bash
    poetry run pytest tests/graders/common/test_helpfulness_complete_demo.py -m regression
    ```

    Run performance tests:
    ```bash
    poetry run pytest tests/graders/common/test_helpfulness_complete_demo.py -m performance
    ```

    Run quality tests (only if API keys are configured):
    ```bash
    poetry run pytest tests/graders/common/test_helpfulness_complete_demo.py -m quality
    ```
"""

import os
import time
from unittest.mock import AsyncMock, patch

import pytest

from rm_gallery.core.analyzer.validation import (
    AccuracyAnalyzer,
    ConsistencyAnalyzer,
    CorrelationAnalyzer,
    F1ScoreAnalyzer,
    FalseNegativeAnalyzer,
    FalsePositiveAnalyzer,
    PrecisionAnalyzer,
    RecallAnalyzer,
)
from rm_gallery.core.graders.common.helpfulness import HelpfulnessGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.runner.grading_runner import GraderConfig, GradingRunner

# ==================== UNIT TESTS ====================
# These tests verify the basic functionality of the grader in isolation
# All external services are mocked to enable offline testing


@pytest.mark.unit
@pytest.mark.asyncio
class TestHelpfulnessGraderUnit:
    """Unit tests for HelpfulnessGrader - testing isolated functionality"""

    def test_initialization(self):
        """Test successful initialization"""
        mock_model = AsyncMock()
        grader = HelpfulnessGrader(
            model=mock_model,
            threshold=0.8,
        )
        assert grader.name == "helpfulness"
        assert grader.threshold == 0.8
        assert grader.model == mock_model

    async def test_successful_evaluation(self):
        """Test successful evaluation with valid inputs"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.score = 4
        mock_response.reason = "Response is helpful and well-structured"
        mock_response.metadata = {}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # Create grader
        grader = HelpfulnessGrader(model=mock_model)

        # Execute test
        result = await grader.aevaluate(
            query="What is Python?",
            response="Python is a high-level programming language.",
        )

        # Assertions
        assert isinstance(result, mock_model.structured_model)
        assert result.score == 4
        assert "helpful" in result.reason.lower()

    async def test_evaluation_with_context_and_reference(self):
        """Test evaluation with context and reference answer"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.score = 5
        mock_response.reason = "Response perfectly addresses the query with context"
        mock_response.metadata = {}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # Create grader
        grader = HelpfulnessGrader(model=mock_model)

        # Execute test
        result = await grader.aevaluate(
            query="What is the capital of France?",
            response="The capital of France is Paris.",
            context="Geography of European countries",
            ground_truth="Paris is the capital and largest city of France.",
        )

        # Assertions
        assert result.score == 5
        assert "helpfulness evaluation score: 5" in result.reason

        # Verify model was called correctly
        mock_model.achat.assert_called_once()

    async def test_error_handling(self):
        """Test graceful error handling"""
        # Setup mock to raise exception
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(side_effect=Exception("API Error"))

        # Create grader
        grader = HelpfulnessGrader(model=mock_model)

        # Execute test
        result = await grader.aevaluate(
            query="What is Python?",
            response="Python is a high-level programming language.",
        )

        # Assertions
        assert result.score == 0.0
        assert "Evaluation error: API Error" in result.reason
        assert result.metadata["threshold"] == 0.7  # Default threshold


# ==================== INTEGRATION TESTS ====================
# These tests verify the grader works correctly with other components


@pytest.mark.integration
@pytest.mark.asyncio
class TestHelpfulnessGraderIntegration:
    """Integration tests for HelpfulnessGrader - testing with other components"""

    async def test_grader_with_runner_pipeline(self):
        """Test HelpfulnessGrader in a Runner pipeline"""
        # Mock the model response
        mock_response = AsyncMock()
        mock_response.score = 4
        mock_response.reason = "Response is helpful"
        mock_response.metadata = {}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # Create grader
        grader = HelpfulnessGrader(model=mock_model)

        # Create runner with mapper
        grader_configs = {
            "helpfulness": GraderConfig(
                grader=grader,
                mapper={
                    "query": "question",
                    "response": "answer",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Test data
        test_data = [
            {
                "question": "What is Python?",
                "answer": "Python is a high-level programming language.",
            },
        ]

        # Run evaluation
        results = await runner.arun(dataset=test_data)

        # Verify results
        assert len(results) == 1
        assert "helpfulness" in results
        assert len(results["helpfulness"]) == 1
        assert results["helpfulness"][0].score == 4

    async def test_grader_with_real_model_config(self):
        """Test HelpfulnessGrader with real model configuration (but still mocked)"""
        # Mock the model response
        mock_response = AsyncMock()
        mock_response.score = 5
        mock_response.reason = "Excellent response"
        mock_response.metadata = {}

        with patch("rm_gallery.core.models.openai_chat_model.OpenAIChatModel") as mock_model_class:
            mock_instance = AsyncMock()
            mock_instance.achat = AsyncMock(return_value=mock_response)
            mock_model_class.return_value = mock_instance

            # Create grader with model configuration
            model_config = {
                "model": "gpt-3.5-turbo",
                "api_key": "test-key",
            }

            grader = HelpfulnessGrader(model=model_config)
            result = await grader.aevaluate(
                query="What is the capital of France?",
                response="The capital of France is Paris.",
                context="Geography question",
            )

            # Verify model was correctly initialized
            mock_model_class.assert_called_once_with(**model_config)

            # Verify results
            assert result.score == 5
            assert "helpfulness evaluation score: 5" in result.reason


# ==================== REGRESSION TESTS ====================
# These tests ensure previously fixed bugs don't reappear


@pytest.mark.regression
@pytest.mark.asyncio
class TestHelpfulnessGraderRegression:
    """Regression tests for previously identified bugs"""

    async def test_unicode_handling_regression(self):
        """Test Unicode handling issue regression"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.score = 4
        mock_response.reason = "Response correctly handles unicode characters"
        mock_response.metadata = {}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # Create grader
        grader = HelpfulnessGrader(model=mock_model)

        # Execute test with unicode characters
        result = await grader.aevaluate(
            query="What is café?",
            response="A café is a coffee house.",
        )

        # Assertions
        assert result.score == 4
        assert isinstance(result.reason, str)

    async def test_empty_response_handling(self):
        """Test handling of empty response"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.score = 1
        mock_response.reason = "Response is empty and unhelpful"
        mock_response.metadata = {}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # Create grader
        grader = HelpfulnessGrader(model=mock_model)

        # Execute test with empty response
        result = await grader.aevaluate(
            query="What is Python?",
            response="",
        )

        # Assertions
        assert result.score == 1
        assert "helpfulness evaluation score: 1" in result.reason


# ==================== PERFORMANCE TESTS ====================
# These tests measure execution efficiency and resource consumption


@pytest.mark.performance
@pytest.mark.asyncio
class TestHelpfulnessGraderPerformance:
    """Performance tests for HelpfulnessGrader"""

    async def test_evaluation_speed(self):
        """Test evaluation speed performance"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.score = 4
        mock_response.reason = "Standard response"
        mock_response.metadata = {}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # Create grader
        grader = HelpfulnessGrader(model=mock_model)

        # Measure execution time
        start_time = time.time()

        # Execute multiple evaluations
        for _ in range(10):
            await grader.aevaluate(
                query="What is Python?",
                response="Python is a high-level programming language.",
            )

        end_time = time.time()
        total_time = end_time - start_time

        # Assert reasonable performance (should complete within 1 second for 10 mocked calls)
        assert total_time < 1.0

    async def test_memory_efficiency(self):
        """Test memory efficiency during repeated evaluations"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.score = 4
        mock_response.reason = "Standard response"
        mock_response.metadata = {}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # Create grader
        grader = HelpfulnessGrader(model=mock_model)

        # Execute many evaluations
        for _ in range(100):
            result = await grader.aevaluate(
                query="What is Python?",
                response="Python is a high-level programming language.",
            )
            assert result.score == 4


# ==================== QUALITY TESTS ====================
# These tests verify the quality of the grader's evaluations

# Check for API keys to determine if live tests should run
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)

pytestmark = pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")


@pytest.mark.quality
class TestHelpfulnessGraderQuality:
    """Quality tests for HelpfulnessGrader - testing evaluation quality"""

    @pytest.fixture
    def dataset(self):
        """Load gold standard dataset"""
        return [
            {
                "query": "What is photosynthesis?",
                "response": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This process occurs in the chloroplasts of plant cells and is fundamental to life on Earth as it produces oxygen and forms the base of the food chain.",
                "context": "Biology basics",
                "human_score": 5,  # High quality response
            },
            {
                "query": "What is photosynthesis?",
                "response": "It's how plants make food.",
                "context": "Biology basics",
                "human_score": 2,  # Low quality response
            },
        ]

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {"model": "gpt-3.5-turbo", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            # This shouldn't happen because tests are skipped if keys aren't configured
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_discriminative_power_with_runner(self, dataset, model):
        """Test the grader's ability to distinguish between high and low quality responses (using Runner)"""
        # Create grader with real model
        grader = HelpfulnessGrader(model=model)

        # Use mapper to configure data transformation
        grader_configs = {
            "helpfulness": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Prepare test data
        test_data = dataset
        human_scores = [item["human_score"] for item in dataset]

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset=test_data)

        # Use AccuracyAnalyzer to calculate accuracy metrics
        accuracy_analyzer = AccuracyAnalyzer()
        accuracy_result = accuracy_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness"],
            label_path="human_score",
        )

        # Use PrecisionAnalyzer to calculate precision metrics
        precision_analyzer = PrecisionAnalyzer()
        precision_result = precision_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness"],
            label_path="human_score",
        )

        # Use RecallAnalyzer to calculate recall metrics
        recall_analyzer = RecallAnalyzer()
        recall_result = recall_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness"],
            label_path="human_score",
        )

        # Use F1ScoreAnalyzer to calculate F1 score metrics
        f1_analyzer = F1ScoreAnalyzer()
        f1_result = f1_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness"],
            label_path="human_score",
        )

        # Use CorrelationAnalyzer to calculate correlation metrics
        correlation_analyzer = CorrelationAnalyzer()
        correlation_result = correlation_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness"],
            label_path="human_score",
        )

        # Assert that quality metrics meet expected thresholds
        assert accuracy_result.accuracy >= 0.7, f"Accuracy below threshold: {accuracy_result.accuracy}"
        assert precision_result.precision >= 0.7, f"Precision below threshold: {precision_result.precision}"
        assert recall_result.recall >= 0.7, f"Recall below threshold: {recall_result.recall}"
        assert f1_result.f1_score >= 0.7, f"F1 score below threshold: {f1_result.f1_score}"
        assert correlation_result.correlation >= 0.7, f"Correlation below threshold: {correlation_result.correlation}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in accuracy_result.metadata
        assert "explanation" in precision_result.metadata
        assert "explanation" in recall_result.metadata
        assert "explanation" in f1_result.metadata
        assert "explanation" in correlation_result.metadata

        assert accuracy_result.name == "Accuracy Analysis"
        assert precision_result.name == "Precision Analysis"
        assert recall_result.name == "Recall Analysis"
        assert f1_result.name == "F1 Score Analysis"
        assert correlation_result.name == "Correlation Analysis"

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """Test grader evaluation consistency (using Runner)"""
        # Create grader with real model
        grader = HelpfulnessGrader(model=model)

        # Use duplicate configuration to implement consistency testing
        grader_configs = {
            "helpfulness_run1": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context",
                },
            ),
            "helpfulness_run2": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Prepare test data
        test_data = dataset

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset=test_data)

        # Use ConsistencyAnalyzer to calculate consistency metrics
        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            first_run_results=results["helpfulness_run1"],
            second_run_results=results["helpfulness_run2"],
        )

        # Assert that consistency metrics meet expected thresholds
        assert (
            consistency_result.consistency >= 0.9
        ), f"Evaluation consistency insufficient: {consistency_result.consistency}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in consistency_result.metadata
        assert consistency_result.name == "Consistency Analysis"


@pytest.mark.quality
class TestHelpfulnessGraderAdversarial:
    """Adversarial tests for HelpfulnessGrader - testing robustness against adversarial examples"""

    @pytest.fixture
    def dataset(self):
        """Load adversarial dataset"""
        return [
            {
                "query": "What is the Earth?",
                "correct_response": "Earth is one of the planets in the solar system.",
                "incorrect_response": "Earth is the only planet in the universe with life.",
                "context": "Basic astronomy knowledge",
                "correct_label": 5,  # High quality score
                "incorrect_label": 2,  # Low quality score
            },
            {
                "query": "What is the chemical formula of water?",
                "correct_response": "The chemical formula of water is H2O.",
                "incorrect_response": "The chemical formula of water is CO2.",
                "context": "Basic chemistry knowledge",
                "correct_label": 5,  # High quality score
                "incorrect_label": 2,  # Low quality score
            },
        ]

    @pytest.fixture
    def model(self):
        """Return OpenAIChatModel instance based on environment variables"""
        if OPENAI_API_KEY:
            config = {"model": "gpt-3.5-turbo", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            # This shouldn't happen because tests are skipped if keys aren't configured
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_adversarial_helpfulness_with_runner(self, dataset, model):
        """Test the grader's ability to identify adversarial examples (using Runner)"""
        # Create grader with real model
        grader = HelpfulnessGrader(model=model)

        # Use mapper to configure data transformation
        # Configure GraderConfig to evaluate both correct and incorrect answers simultaneously
        grader_configs = {
            "helpfulness_correct": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "correct_response",
                    "context": "context",
                },
            ),
            "helpfulness_incorrect": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "incorrect_response",
                    "context": "context",
                },
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # Prepare test data
        test_data = dataset

        # Use Runner to perform batch evaluation
        results = await runner.arun(dataset=test_data)

        # Use FalsePositiveAnalyzer to calculate false positive rate metrics
        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness_incorrect"],  # Evaluation results for incorrect responses
            label_path="incorrect_label",
        )

        # Use FalseNegativeAnalyzer to calculate false negative rate metrics
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness_correct"],  # Evaluation results for correct responses
            label_path="correct_label",
        )

        # Assert that false positive and false negative rates meet expected thresholds
        assert fp_result.false_positive_rate <= 0.3, f"False positive rate too high: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.3, f"False negative rate too high: {fn_result.false_negative_rate}"

        # Verify analysis results contain necessary metadata
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
