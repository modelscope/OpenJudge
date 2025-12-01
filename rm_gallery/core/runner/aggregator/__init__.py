# -*- coding: utf-8 -*-
"""
Aggregator module for combining results from multiple graders.
"""

from .base_aggregator import BaseAggregator
from .weighted_sum_aggregator import WeightedSumAggregator

__all__ = [
    "BaseAggregator",
    "WeightedSumAggregator",
]
