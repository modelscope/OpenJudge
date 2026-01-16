# -*- coding: utf-8 -*-
"""Configuration module for OpenJudge Studio."""

from config.constants import DEFAULT_MODELS, EXAMPLE_DATA, GRADER_CATEGORIES
from config.grader_registry import GRADER_REGISTRY, get_graders_by_category

__all__ = [
    "DEFAULT_MODELS",
    "EXAMPLE_DATA",
    "GRADER_CATEGORIES",
    "GRADER_REGISTRY",
    "get_graders_by_category",
]
