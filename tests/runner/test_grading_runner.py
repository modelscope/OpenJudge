# -*- coding: utf-8 -*-
"""
Test Grading Runner

Tests for the GradingRunner class functionality.
"""

import pytest
from jsonschema import validate

from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderScore
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.runner.grading_runner import GradingRunner


class MockGrader(BaseGrader):
    """Mock grader for testing purposes."""

    def __init__(self, name="mock_grader", score_value=1.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.score_value = score_value
        self.call_args_list = []
        self.call_count = 0

    async def aevaluate(self, **kwargs):
        """Mock evaluation that returns a fixed score."""
        self.call_count += 1
        self.call_args_list.append(kwargs)
        return GraderScore(
            name=self.name,
            score=self.score_value,
            reason=f"Mock score of {self.score_value}",
            metadata={"called_with": kwargs, "call_count": self.call_count},
        )


@pytest.mark.asyncio
async def test_grading_runner_with_mock_grader():
    """Test the grading runner with mock graders to verify functionality without external dependencies."""

    # Create mock graders
    mock_grader1 = MockGrader(name="accuracy_grader", score_value=0.9)
    mock_grader2 = MockGrader(name="relevance_grader", score_value=0.8)

    # Create runner with mock graders
    runner = GradingRunner(
        grader_configs={
            "accuracy": mock_grader1,
            "relevance": mock_grader2,
        },
        max_concurrency=2,
    )

    # Test data
    dataset = [
        {"query": "What is the capital of France?", "answer": "Paris"},
        {"query": "What is the capital of Germany?", "answer": "Berlin"},
    ]

    # Run the evaluation
    results = await runner.arun(dataset)

    # Verify the structure of results
    assert "accuracy" in results
    assert "relevance" in results
    assert len(results["accuracy"]) == 2
    assert len(results["relevance"]) == 2

    # Verify that each result is a GraderScore
    for grader_results in results.values():
        for result in grader_results:
            assert isinstance(result, GraderScore)

    # Verify specific scores
    assert results["accuracy"][0].score == 0.9
    assert results["relevance"][1].score == 0.8

    # Verify that graders were called correctly
    assert mock_grader1.call_count == 2
    assert mock_grader2.call_count == 2

    # Verify the data passed to graders for each call
    assert "query" in mock_grader1.call_args_list[0]
    assert "answer" in mock_grader1.call_args_list[0]
    assert "query" in mock_grader1.call_args_list[1]
    assert "answer" in mock_grader1.call_args_list[1]

    # Verify specific values for first and second calls
    assert mock_grader1.call_args_list[0]["query"] == "What is the capital of France?"
    assert mock_grader1.call_args_list[0]["answer"] == "Paris"
    assert mock_grader1.call_args_list[1]["query"] == "What is the capital of Germany?"
    assert mock_grader1.call_args_list[1]["answer"] == "Berlin"


@pytest.mark.asyncio
async def test_grading_runner_with_mappers():
    """Test the grading runner with data mappers."""

    # Create mock grader
    mock_grader = MockGrader(name="mapped_grader")

    # Create runner with mapper
    # The mapper format is {new_field_name: path_in_original_data}
    runner = GradingRunner(
        grader_configs={
            "mapped_test": (mock_grader, {"query": "question", "answer": "response"}),
        },
    )

    # Test data with different field names
    dataset = [
        {"question": "What is 2+2?", "response": "4"},
        {"question": "What is the sky color?", "response": "blue"},
    ]

    # Run the evaluation
    results = await runner.arun(dataset)

    # Verify results structure
    assert "mapped_test" in results
    assert len(results["mapped_test"]) == 2

    # Verify that the mapper worked - the grader should have been called with mapped field names
    assert len(mock_grader.call_args_list) == 2
    assert "query" in mock_grader.call_args_list[0]
    assert "answer" in mock_grader.call_args_list[0]
    assert "query" in mock_grader.call_args_list[1]
    assert "answer" in mock_grader.call_args_list[1]

    # Verify that the data was correctly mapped
    assert mock_grader.call_args_list[0]["query"] == "What is 2+2?"
    assert mock_grader.call_args_list[0]["answer"] == "4"
    assert mock_grader.call_args_list[1]["query"] == "What is the sky color?"
    assert mock_grader.call_args_list[1]["answer"] == "blue"
