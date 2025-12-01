# AgentScope Integration

This document explains how to integrate RM-Gallery with AgentScope through the [RMGalleryMetric](file:///Users/zhuohua/workspace/RM-Gallery/tutorials/integrations/agentscope/agentscope.py#L23-L106) wrapper class.

## Overview

The integration enables RM-Gallery graders to be used as AgentScope metrics. This allows you to leverage RM-Gallery's extensive collection of evaluation methods within the AgentScope framework.

## Core Component

### RMGalleryMetric Class

The [RMGalleryMetric](file:///Users/zhuohua/workspace/RM-Gallery/tutorials/integrations/agentscope/agentscope.py#L23-L106) is a wrapper that bridges RM-Gallery's grading system with AgentScope's metric system.

```python
from rm_gallery.core.graders.base_grader import Grader
from tutorials.integrations.agentscope.agentscope import RMGalleryMetric

# Create or obtain an RM-Gallery grader
grader = YourCustomGrader()

# Wrap it as an AgentScope metric
metric = RMGalleryMetric(grader)
```

## Implementation Requirements

To use [RMGalleryMetric](file:///Users/zhuohua/workspace/RM-Gallery/tutorials/integrations/agentscope/agentscope.py#L23-L106), you must implement two abstract methods in a subclass:

1. `_convert_solution_to_dict`: Converts AgentScope solutions to RM-Gallery input format
2. `_convert_grader_result_to_metric_result`: Converts RM-Gallery outputs to AgentScope format

Example implementation:
```python
class CustomRMGalleryMetric(RMGalleryMetric):
    async def _convert_solution_to_dict(self, solution):
        # Your conversion logic here
        return {"query": solution.query, "response": solution.response}

    async def _convert_grader_result_to_metric_result(self, grader_result):
        # Your conversion logic here
        return {"score": grader_result.score, "reason": grader_result.reason}
```

## Usage

Once implemented, you can use the metric in AgentScope workflows:

```python
# Evaluate a solution
result = await metric(solution)
```

This executes the RM-Gallery grader internally and returns the result in AgentScope's expected format.