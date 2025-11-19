# -*- coding: utf-8 -*-
"""
Gallery evaluation implementations.

This module contains concrete evaluation implementations for various benchmarks:
- ConflictDetector: Pairwise comparison conflict detection
- RewardBench2: Four-way comparison with Ties mode
- RMBench: 3x3 comparison matrix evaluation

Example:
    ```python
    from rm_gallery.gallery.evaluation.conflict_detector import PairwiseComparisonRunner
    from rm_gallery.core.runner.evaluation import AccuracyMetric

    # Create runner with metrics
    runner = PairwiseComparisonRunner(
        model=my_model,
        metrics=[AccuracyMetric()]
    )

    # Run evaluation
    report = await runner(data_samples)
    print(report.model_name, report.metrics)
    ```
"""

# Note: Import specific runners as needed
# This file is intentionally minimal to avoid heavy imports

__all__ = []
