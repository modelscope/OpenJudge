# -*- coding: utf-8 -*-
"""
Conflict detection metric for evaluation.
"""
from typing import List

import numpy as np

from rm_gallery.core.runner.evaluation.metric.base import BaseMetric
from rm_gallery.core.runner.evaluation.schema import EvaluationResult, MetricResult


class ConflictMetric(BaseMetric):
    """
    Conflict rate metric that detects logical inconsistencies in pairwise comparisons.

    Uses Strongly Connected Components (SCC) algorithm to detect cycles in
    comparison graphs, which indicate conflicting preferences.

    Examples of conflicts:
    - Pairwise cycle: A > B, B > A
    - Multi-cycle: A > B > C > A
    """

    def __init__(self, name: str = "conflict_rate") -> None:
        """
        Initialize conflict metric.

        Args:
            name: Name of the metric (default: "conflict_rate")
        """
        super().__init__(name)

    def _tarjan_scc(self, graph: List[List[int]]) -> List[List[int]]:
        """
        Tarjan's algorithm for strongly connected components detection.

        Args:
            graph: Adjacency list representation of directed graph

        Returns:
            List of strongly connected components
        """
        n = len(graph)
        index = 0
        stack = []
        indices = [-1] * n
        lowlinks = [-1] * n
        on_stack = [False] * n
        sccs = []

        def strongconnect(v: int) -> None:
            nonlocal index
            indices[v] = index
            lowlinks[v] = index
            index += 1
            stack.append(v)
            on_stack[v] = True

            for w in graph[v]:
                if indices[w] == -1:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif on_stack[w]:
                    lowlinks[v] = min(lowlinks[v], indices[w])

            if lowlinks[v] == indices[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        for v in range(n):
            if indices[v] == -1:
                strongconnect(v)

        return sccs

    def _has_conflict(self, matrix: np.ndarray) -> bool:
        """
        Detect if a comparison matrix has conflicts using SCC.

        Args:
            matrix: Comparison matrix where matrix[i][j] > 0 means i beats j

        Returns:
            True if conflicts are detected, False otherwise
        """
        n = matrix.shape[0]
        if n < 2:
            return False

        # Build directed graph from comparison matrix
        graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i][j] > 0:
                    graph[i].append(j)

        # Find strongly connected components
        sccs = self._tarjan_scc(graph)

        # If any SCC has more than 1 node, there's a cycle (conflict)
        for scc in sccs:
            if len(scc) > 1:
                return True

        return False

    def compute(self, results: List[EvaluationResult]) -> MetricResult:
        """
        Compute conflict rate from evaluation results.

        Args:
            results: List of evaluation results with comparison matrices

        Returns:
            MetricResult with conflict rate and detailed statistics
        """
        if not results:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={
                    "samples_with_conflicts": 0,
                    "total_samples": 0,
                    "valid_samples": 0,
                },
            )

        # Filter valid results with comparison matrices
        valid_results = [
            r for r in results if r.is_valid and r.comparison_matrix is not None
        ]

        if not valid_results:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                details={
                    "samples_with_conflicts": 0,
                    "total_samples": len(results),
                    "valid_samples": 0,
                    "no_matrix_count": len(results),
                },
            )

        # Count samples with conflicts
        samples_with_conflicts = 0
        conflict_details = []

        for result in valid_results:
            # Convert comparison matrix to numpy array
            matrix = np.array(result.comparison_matrix)

            # Check for conflicts
            has_conflict = self._has_conflict(matrix)

            if has_conflict:
                samples_with_conflicts += 1
                conflict_details.append(
                    {
                        "unique_id": result.unique_id,
                        "matrix_size": matrix.shape[0],
                    },
                )

        # Calculate conflict rate
        valid_count = len(valid_results)
        conflict_rate = samples_with_conflicts / valid_count if valid_count > 0 else 0.0

        return MetricResult(
            metric_name=self.name,
            value=float(conflict_rate),
            details={
                "samples_with_conflicts": samples_with_conflicts,
                "total_samples": len(results),
                "valid_samples": valid_count,
                "conflict_details": conflict_details[:10],  # Limit to first 10
            },
        )
