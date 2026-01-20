# -*- coding: utf-8 -*-
"""Feature modules for OpenJudge Studio.

Each feature module provides a self-contained UI for a specific functionality.
Features are registered with the FeatureRegistry and can be accessed via the
navigation system.
"""

from features.grader import GraderFeature
from features.zero_shot import ZeroShotFeature

__all__ = ["GraderFeature", "ZeroShotFeature"]
