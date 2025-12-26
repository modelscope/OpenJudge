#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for data mapping utilities.
"""

from typing import Any, Dict

import pytest

from openjudge.utils.mapping import (
    get_value_by_mapping,
    get_value_by_path,
    parse_data_with_mapper,
)


class TestGetValueByPath:
    """Test cases for get_value_by_path function."""

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Sample data for testing."""
        return {
            "user": {
                "name": "Alice",
                "age": 30,
                "emails": ["alice@example.com", "alice@work.com"],
                "address": {
                    "street": "123 Main St",
                    "city": "Wonderland",
                },
            },
            "items": [
                {"id": 1, "name": "Item A", "price": 10.0},
                {"id": 2, "name": "Item B", "price": 20.0},
                {"id": 3, "name": "Item C", "price": 30.0},
            ],
            "simple_list": [1, 2, 3, 4, 5],
            "mixed_list": [
                {"product": "A", "tags": ["red", "small"]},
                {"product": "B", "tags": ["blue", "large"]},
                {"product": "C", "tags": ["green", "medium"]},
            ],
        }

    def test_basic_path_access(self, sample_data: Dict[str, Any]):
        """Test basic dictionary path access."""
        assert get_value_by_path(sample_data, "user.name") == "Alice"
        assert get_value_by_path(sample_data, "user.age") == 30
        assert get_value_by_path(sample_data, "user.address.city") == "Wonderland"

    def test_list_indexing(self, sample_data: Dict[str, Any]):
        """Test accessing list elements by index."""
        assert get_value_by_path(sample_data, "user.emails.0") == "alice@example.com"
        assert get_value_by_path(sample_data, "user.emails.1") == "alice@work.com"
        assert get_value_by_path(sample_data, "items.0.name") == "Item A"
        assert get_value_by_path(sample_data, "items.1.price") == 20.0
        assert get_value_by_path(sample_data, "simple_list.2") == 3

    def test_automatic_list_traversal(self, sample_data: Dict[str, Any]):
        """Test automatic traversal of lists to extract fields."""
        # Test extracting names from all items
        names = get_value_by_path(sample_data, "items.name")
        assert names == ["Item A", "Item B", "Item C"]

        # Test extracting prices from all items
        prices = get_value_by_path(sample_data, "items.price")
        assert prices == [10.0, 20.0, 30.0]

        # Test extracting products from mixed list
        products = get_value_by_path(sample_data, "mixed_list.product")
        assert products == ["A", "B", "C"]

    def test_nested_list_traversal(self, sample_data: Dict[str, Any]):
        """Test traversal of nested lists."""
        # Test extracting tags from all items in mixed list
        tags = get_value_by_path(sample_data, "mixed_list.tags")
        assert tags == [["red", "small"], ["blue", "large"], ["green", "medium"]]

    def test_nonexistent_paths(self, sample_data: Dict[str, Any]):
        """Test behavior with nonexistent paths."""
        assert get_value_by_path(sample_data, "user.nonexistent") is None
        assert get_value_by_path(sample_data, "nonexistent") is None
        assert get_value_by_path(sample_data, "user.emails.5") is None
        assert get_value_by_path(sample_data, "items.10.name") is None

    def test_empty_path(self, sample_data: Dict[str, Any]):
        """Test behavior with empty path."""
        assert get_value_by_path(sample_data, "") == sample_data

    def test_invalid_index(self, sample_data: Dict[str, Any]):
        """Test behavior with invalid list indices."""
        assert get_value_by_path(sample_data, "items.abc.name") is None
        assert get_value_by_path(sample_data, "items.-1.name") is None


class TestGetValueByMapping:
    """Test cases for get_value_by_mapping function."""

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Sample data for testing."""
        return {
            "task": {
                "question": "What is 2+2?",
                "reference_answer": "4",
            },
            "workflow_output": [
                {"metadata": {"final_answer": "4", "confidence": 0.9}},
                {"metadata": {"final_answer": "four", "confidence": 0.7}},
                {"metadata": {"final_answer": "2+2=4", "confidence": 0.8}},
            ],
        }

    @pytest.fixture
    def sample_mapping(self) -> Dict[str, str]:
        """Sample mapping for testing."""
        return {
            "query": "task.question",
            "reference_response": "task.reference_answer",
            "responses": "workflow_output.metadata.final_answer",
            "confidences": "workflow_output.metadata.confidence",
        }

    def test_mapping_application(self, sample_data: Dict[str, Any], sample_mapping: Dict[str, str]):
        """Test applying a mapping to data."""
        result = get_value_by_mapping(sample_data, sample_mapping)

        expected = {
            "query": "What is 2+2?",
            "reference_response": "4",
            "responses": ["4", "four", "2+2=4"],
            "confidences": [0.9, 0.7, 0.8],
        }

        assert result == expected

    def test_partial_mapping(self, sample_data: Dict[str, Any]):
        """Test mapping with some fields missing."""
        partial_mapping = {
            "query": "task.question",
            "answers": "workflow_output.metadata.final_answer",
        }

        result = get_value_by_mapping(sample_data, partial_mapping)

        expected = {
            "query": "What is 2+2?",
            "answers": ["4", "four", "2+2=4"],
        }

        assert result == expected

    def test_empty_mapping(self, sample_data: Dict[str, Any]):
        """Test applying an empty mapping."""
        result = get_value_by_mapping(sample_data, {})
        assert result == {}


class TestParseDataWithMapper:
    """Test cases for parse_data_with_mapper function."""

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Sample data for testing."""
        return {
            "user": {
                "name": "Bob",
                "preferences": ["reading", "swimming", "coding"],
            },
            "products": [
                {"name": "Laptop", "price": 1000},
                {"name": "Mouse", "price": 25},
                {"name": "Keyboard", "price": 75},
            ],
        }

    def test_dict_mapper(self, sample_data: Dict[str, Any]):
        """Test parsing data with dictionary mapper."""
        mapper = {
            "username": "user.name",
            "hobbies": "user.preferences",
            "product_names": "products.name",
            "product_prices": "products.price",
        }

        result = parse_data_with_mapper(sample_data, mapper)

        expected = {
            "username": "Bob",
            "hobbies": ["reading", "swimming", "coding"],
            "product_names": ["Laptop", "Mouse", "Keyboard"],
            "product_prices": [1000, 25, 75],
        }

        assert result == expected

    def test_callable_mapper(self, sample_data: Dict[str, Any]):
        """Test parsing data with callable mapper."""

        def custom_mapper(data):
            return {
                "user_info": f"{data['user']['name']} has {len(data['user']['preferences'])} hobbies",
                "expensive_products": [p for p in data["products"] if p["price"] > 50],
            }

        result = parse_data_with_mapper(sample_data, custom_mapper)

        expected = {
            "user_info": "Bob has 3 hobbies",
            "expensive_products": [
                {"name": "Laptop", "price": 1000},
                {"name": "Keyboard", "price": 75},
            ],
        }

        assert result == expected

    def test_none_mapper(self, sample_data: Dict[str, Any]):
        """Test parsing data with None mapper."""
        result = parse_data_with_mapper(sample_data, None)
        assert result == sample_data

    def test_empty_mapper(self, sample_data: Dict[str, Any]):
        """Test parsing data with empty dictionary mapper."""
        result = parse_data_with_mapper(sample_data, {})
        assert result == {}

    # OpenJudge specific test cases
    def test_open_judge_task_mapping(self):
        """Test mapping typical OpenJudge task data."""
        task_data = {
            "task": {
                "question": "How do I sort a list in Python?",
                "metadata": {
                    "difficulty": "intermediate",
                    "category": "programming",
                },
            },
            "workflow_output": [
                {"metadata": {"final_answer": "Use sorted() function", "score": 0.9}},
                {"metadata": {"final_answer": "Use list.sort() method", "score": 0.8}},
                {"metadata": {"final_answer": "Both sorted() and sort()", "score": 0.95}},
            ],
        }

        mapper = {
            "query": "task.question",
            "difficulty": "task.metadata.difficulty",
            "answers": "workflow_output.metadata.final_answer",
            "scores": "workflow_output.metadata.score",
        }

        result = parse_data_with_mapper(task_data, mapper)

        expected = {
            "query": "How do I sort a list in Python?",
            "difficulty": "intermediate",
            "answers": [
                "Use sorted() function",
                "Use list.sort() method",
                "Both sorted() and sort()",
            ],
            "scores": [0.9, 0.8, 0.95],
        }

        assert result == expected

    def test_single_workflow_output_mapping(self):
        """Test mapping with a single workflow output (pointwise case)."""
        task_data = {
            "task": {
                "question": "What is the capital of France?",
            },
            "workflow_output": {
                "metadata": {
                    "final_answer": "Paris",
                    "confidence": 0.98,
                },
            },
        }

        mapper = {
            "query": "task.question",
            "answer": "workflow_output.metadata.final_answer",
            "confidence": "workflow_output.metadata.confidence",
        }

        result = parse_data_with_mapper(task_data, mapper)

        expected = {
            "query": "What is the capital of France?",
            "answer": "Paris",
            "confidence": 0.98,
        }

        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__])
