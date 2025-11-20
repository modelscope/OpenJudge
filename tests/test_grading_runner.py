# -*- coding: utf-8 -*-
"""
Test Grading Runner

Tests for the GradingRunner class functionality.
"""

import asyncio

import pytest

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.runner.grading.base import GradingRunner
from rm_gallery.core.schema.data import validate_eval_cases
from rm_gallery.gallery.grader.alignment.honesty.factuality import FactualityGrader


def test_grading_runner_example():
    """Test the example code from grading.py __main__ section"""
    eval_case_schema = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
            "samples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        },
        "required": ["data", "samples"],
    }
    eval_cases = [
        {
            "data": {
                "query": "What is the capital of France?",
            },
            "samples": [{"answer": "Paris"}, {"answer": "Marseille"}],
        },
        {
            "data": {
                "query": "What is the capital of Germany?",
            },
            "samples": [{"answer": "Berlin"}, {"answer": "Munich"}],
        },
    ]
    eval_cases = validate_eval_cases(eval_cases, eval_case_schema)
    model = OpenAIChatModel(model_name="qwen-plus")

    runner = GradingRunner(
        grading_configs={
            "factual_grader": {
                "grader": FactualityGrader(model=model),
                "weight": 1.0,
            },
        },
    )
    # This would normally be run but we're just testing creation
    assert runner is not None
    assert len(runner.grading_configs) == 1


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_grading_runner_execution():
    """Test executing the grading runner with actual model call"""
    eval_case_schema = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
            "samples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        },
        "required": ["data", "samples"],
    }
    eval_cases = [
        {
            "data": {
                "query": "What is the capital of France?",
            },
            "samples": [{"answer": "Paris"}, {"answer": "Marseille"}],
        },
        {
            "data": {
                "query": "What is the capital of Germany?",
            },
            "samples": [{"answer": "Berlin"}, {"answer": "Munich"}],
        },
    ]
    eval_cases = validate_eval_cases(eval_cases, eval_case_schema)
    model = OpenAIChatModel(model_name="qwen-plus")

    runner = GradingRunner(
        grading_configs={
            "factual_grader": {
                "grader": FactualityGrader(model=model),
                "weight": 1.0,
            },
        },
    )
    # Run using async method
    result = await runner.aevaluate_batch(eval_cases=eval_cases)
    assert result is not None
