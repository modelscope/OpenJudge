# Using RM Gallery Graders in astune

## Overview

RMGraderJudge is a component in the astune system that integrates RM Gallery Graders. It allows you to use predefined and custom evaluators (Graders) provided by RM Gallery within astune workflows. Through this integration, you can conveniently evaluate AgentScope workflow outputs and obtain corresponding scores.

## Key Features

- Supports both Pointwise and Listwise evaluation modes
- Flexible data mapping mechanism for field transformation
- Seamless integration with astune workflow
- Supports all predefined evaluators from RM Gallery
- Supports integration of custom evaluators

## Configuration Guide

To use RMGraderJudge in astune, you need to configure it in your configuration file as follows:

```yaml
astuner:
  task_judge:
    judge_type: customized_protocol  # Must be set to customized_protocol
    judge_protocol: astuner.task_judge.rm_grader_judge->RMGraderJudge  # Specify to use RMGraderJudge
    grader:
      class_name: "LLMGrader"
      module_path: "rm_gallery.core.graders.llm_grader"
      kwargs:
        name: "helpfulness_grader"
        mode: "pointwise"
        description: "Evaluates helpfulness of answers using LLM as a judge"
        model:
          model: "qwen-plus"
          base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
        template:
          messages:
            - role: "system"
              content: "You are an expert evaluator of AI assistant responses..."
            - role: "user"
              content: "Question: {query}\nAnswer: {answer}\n\nPlease rate the helpfulness..."
        rubrics: "Consider factors like accuracy, completeness, and clarity..."
    # Optional data mapper to transform data fields
    mapper:
      query: "task.main_query"  # Maps task.main_query to 'query' parameter
      answer: "workflow_output.metadata.final_answer"  # Maps workflow_output.metadata.final_answer to 'answer' parameter
```

### Key Configuration Items

1. `judge_type`: Must be set to `customized_protocol`
2. `judge_protocol`: Specifies using RMGraderJudge, format is `module_path->class_name`
3. `grader`: Specific configuration for the RM Gallery evaluator
   - `class_name`: Evaluator class name
   - `module_path`: Module path where the evaluator is located
   - `kwargs`: Evaluator initialization parameters
4. `mapper`: Optional data mapping configuration to map task and workflow output data to evaluator parameters

## Using Predefined Graders

RM Gallery provides a rich set of predefined evaluators covering multiple domains. Here are some common usage examples:

### Text Similarity Evaluation

```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: astuner.task_judge.rm_grader_judge->RMGraderJudge
    grader:
      class_name: "SimilarityGrader"
      module_path: "rm_gallery.core.graders.text.similarity"
      kwargs:
        name: "f1_score_evaluator"
        normalize: true
    mapper:
      reference: "task.reference_answer"
      response: "workflow_output.metadata.final_answer"
      algorithm: "f1_score"
```

### LLM-based Evaluation

```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: astuner.task_judge.rm_grader_judge->RMGraderJudge
    grader:
      class_name: "HelpfulnessGrader"
      module_path: "rm_gallery.core.graders.alignment.helpfulness.helpfulness"
      kwargs:
        name: "helpfulness_evaluator"
        mode: "pointwise"
        model:
          model: "qwen-plus"
        template:
          messages:
            - role: "system"
              content: "You are an expert evaluator of AI assistant responses."
            - role: "user"
              content: |
                Please evaluate the helpfulness of the following Q&A pair:

                Question: {query}
                Answer: {answer}

                Please rate on a scale of 1-10 based on accuracy, completeness, and clarity.
        rubrics: "Consider factors such as accuracy, completeness, and clarity"
    mapper:
      query: "task.main_query"
      answer: "workflow_output.metadata.final_answer"
```

### Mathematical Problem Evaluation

```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: astuner.task_judge.rm_grader_judge->RMGraderJudge
    grader:
      class_name: "MathVerifyGrader"
      module_path: "rm_gallery.core.graders.math.math"
      kwargs:
        name: "math_evaluator"
        mode: "pointwise"
    mapper:
      problem: "task.question"
      solution: "workflow_output.metadata.final_answer"
      reference: "task.answer"
```

## Creating and Using Custom Graders

If you need to create your own evaluator, follow these steps:

### 1. Create a Custom Grader Class

Create a new Python file, for example, `my_custom_grader.py`:

```python
from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderMode, GraderResult

class MyCustomGrader(BaseGrader):
    def __init__(self, name: str, mode: GraderMode = GraderMode.POINTWISE, **kwargs):
        super().__init__(name=name, mode=mode)
        # Initialize your custom parameters

    async def aevaluate(self, **kwargs) -> GraderResult:
        # Implement your evaluation logic
        # Return results according to mode

        if self.mode == GraderMode.POINTWISE:
            # Pointwise evaluation mode, return score
            score = self._calculate_score(**kwargs)
            return GraderResult(score=score)
        else:
            # Listwise evaluation mode, return ranking
            ranks = self._calculate_ranks(**kwargs)
            return GraderResult(rank=ranks)

    def _calculate_score(self, **kwargs):
        # Implement pointwise evaluation scoring logic
        pass

    def _calculate_ranks(self, **kwargs):
        # Implement listwise evaluation ranking logic
        pass
```

### 2. Use Custom Grader in Configuration

```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: astuner.task_judge.rm_grader_judge->RMGraderJudge
    grader:
      class_name: "MyCustomGrader"
      module_path: "path.to.my_custom_grader"
      kwargs:
        name: "my_custom_evaluator"
        mode: "pointwise"
        # Your custom parameters
    mapper:
      # Map fields according to your evaluator's requirements
      input_field: "task.input_data"
      output_field: "workflow_output.result"
```

## Data Mapping (Mapper)

The data mapping feature allows you to map data fields from workflows to parameters required by evaluators. This is a very useful feature because the workflow data structure and evaluator expected parameters may not match.

### Mapping Syntax

Mapping configuration uses dot notation to access nested object properties:

```yaml
mapper:
  target_param: "source_object.field.subfield"
```

### Example

Assuming your task data structure is as follows:

```json
{
  "task": {
    "question": "What is 2+2?",
    "reference_answer": "4"
  },
  "workflow_output": {
    "metadata": {
      "final_answer": "The answer is 4"
    }
  }
}
```

You can map it like this:

```yaml
mapper:
  query: "task.question"
  reference: "task.reference_answer"
  answer: "workflow_output.metadata.final_answer"
```

### Listwise Mode Data Mapping

For listwise evaluation mode, the data mapping works with a list of workflow outputs. When using listwise mode, the `workflow_output` parameter passed to the mapper is a list of workflow outputs rather than a single output.

The mapper configuration uses the same syntax and automatically handles list traversal:

```yaml
astuner:
  task_judge:
    judge_type: customized_protocol
    judge_protocol: astuner.task_judge.rm_grader_judge->RMGraderJudge
    grader:
      class_name: "SomeListwiseGrader"
      module_path: "path.to.listwise_grader"
      kwargs:
        name: "listwise_evaluator"
        mode: "listwise"
        # Other parameters
    mapper:
      query: "task.question"                          # Maps the same task question to all outputs
      answers: "workflow_output.metadata.final_answer" # Automatically maps final_answer from all workflow outputs
```

In listwise mode, the mapper receives a data dictionary with the following structure:
- `task`: A single task object
- `workflow_output`: A list of workflow output objects

When the mapper processes this data:
1. For `query: "task.question"` - it extracts the question field from the task object
2. For `answers: "workflow_output.metadata.final_answer"` - it automatically traverses the list of workflow outputs and extracts the `final_answer` field from each output's metadata, returning a list of answers

This automatic list traversal means you don't need to manually specify indices like `workflow_output.0.metadata.final_answer` and `workflow_output.1.metadata.final_answer`. Instead, the mapper will automatically collect the specified field from all items in the list and return them as a list.

## Evaluation Modes

### Pointwise Evaluation

Pointwise evaluation mode scores individual outputs, returning a numeric score and success flag.

Return format: `(score: float, is_success: bool)`

### Listwise Evaluation

Listwise evaluation mode ranks a group of outputs, returning a ranking list and success flag.

Return format: `(rank: list, is_success: bool)`

## Error Handling

When errors occur during evaluation, RMGraderJudge returns default values:

- Pointwise mode: returns `(0.0, False)`
- Listwise mode: returns `([0.0] * count, False)`, where count is the number of outputs

## Best Practices

1. **Choose appropriate evaluators**: Select the most suitable predefined evaluator based on your task type
2. **Configure mapping properly**: Ensure data mapping is correct so evaluators can access required data
3. **Optimize prompts**: For LLM-based evaluators, carefully designed prompts can significantly improve evaluation quality
4. **Handle exceptions**: Consider potential exception scenarios during evaluation and handle them appropriately
5. **Performance considerations**: For large-scale evaluation tasks, pay attention to evaluator performance
