# -*- coding: utf-8 -*-
"""Evaluation strategy module for graders."""

from .base_evaluation_strategy import BaseEvaluationStrategy
from .local_evaluation_strategy import LocalEvaluationStrategy

__all__ = ["BaseEvaluationStrategy", "LocalEvaluationStrategy"]
