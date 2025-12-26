# Code & Math Graders

Specialized graders for evaluating code generation and mathematical problem-solving capabilities. These graders assess syntax correctness, execution results, code style, and mathematical expression accuracy.


## Overview

| Grader | Purpose | Type | Score Range | Key Use Case |
|--------|---------|------|-------------|--------------|
| `CodeExecutionGrader` | Tests code against test cases | Code-Based | [0, 1] | Coding challenges, algorithm tasks |
| `SyntaxCheckGrader` | Validates Python syntax | Code-Based | {0, 1} | Code generation quality control |
| `CodeStyleGrader` | Checks code style compliance | Code-Based | [0, 1] | Code formatting evaluation |
| `PatchSimilarityGrader` | Measures patch similarity | Code-Based | [0, 1] | Code modification, bug fixes |
| `MathExpressionVerifyGrader` | Verifies math expressions | Code-Based | {0, 1} | Math problem solving, symbolic verification |


## Code Graders

### CodeExecutionGrader

Executes generated code against test cases to verify functional correctness. Evaluates whether code produces expected outputs for given inputs.

**When to use:**
- Coding challenge evaluation
- Algorithm correctness verification
- Programming education systems
- Code competition platforms

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | The code to evaluate (may include markdown fences) |
| `continuous` | bool | No | If True, returns partial score based on test pass rate |
| `timeout` | int | No | Maximum execution time in seconds (default: 10) |

**Scoring:**
- **1.0**: All test cases passed
- **0.0-1.0** (continuous mode): Ratio of passed tests to total tests
- **0.0**: No tests passed or execution failed

!!! note "When to Use"
    `CodeExecutionGrader` is designed for benchmark datasets that provide test cases. For quick code quality checks without test cases, use `SyntaxCheckGrader` or `CodeStyleGrader` instead.

**Example:**

=== "Quick Code Quality Check"

    ````python
    import asyncio
    from openjudge.graders.code import SyntaxCheckGrader, CodeStyleGrader
    from openjudge.runner.grading_runner import GradingRunner

    async def main():
        # For code quality evaluation without test cases
        runner = GradingRunner(grader_configs={
            "syntax": SyntaxCheckGrader(),
            "style": CodeStyleGrader(),
        })

        code_response = '''
        ```python
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        ```
        '''

        results = await runner.arun([{"response": code_response}])

        print(f"Syntax Score: {results['syntax'][0].score}")
        print(f"Style Score: {results['style'][0].score}")

    asyncio.run(main())
    ````

=== "Benchmark Execution Testing"

    ```python
    import asyncio
    from openjudge.graders.code import CodeExecutionGrader
    from openjudge.runner.grading_runner import GradingRunner

    async def main():
        # CodeExecutionGrader is used with datasets containing test cases
        # Typical usage in benchmark evaluation contexts (HumanEval, APPS, etc.)
        grader = CodeExecutionGrader(continuous=True, timeout=10)
        runner = GradingRunner(grader_configs={"execution": grader})

        # Dataset format from benchmarks
        # Each sample should include test cases in the expected format
        dataset = [
            {
                "response": "def add(a, b):\n    return a + b",
                # Test cases are typically pre-loaded from benchmark datasets
            }
        ]

        results = await runner.arun(dataset)
        print(f"Execution Score: {results['execution'][0].score}")

    asyncio.run(main())
    ```


### SyntaxCheckGrader

Validates Python code syntax using Abstract Syntax Tree (AST) parsing. Extracts code blocks from markdown and checks for syntax errors.

**When to use:**
- Pre-execution code validation
- Code generation quality assurance
- Syntax error detection
- Basic code correctness screening

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | Response containing code blocks (markdown fenced) |

**Scoring:**
- **1.0**: All code blocks are syntactically valid
- **0.5**: Some valid blocks (ratio - 0.5 penalty for errors)
- **0.0**: No code blocks found
- **Negative**: Invalid syntax with penalty applied

**Example:**

=== "Valid Syntax Check"

    ````python
    import asyncio
    from openjudge.graders.code import SyntaxCheckGrader

    async def main():
        grader = SyntaxCheckGrader()

        response = '''
        Here's a function:
        ```python
        def greet(name):
            return f"Hello, {name}!"
        ```
        '''

        result = await grader.aevaluate(response=response)
        print(f"Score: {result.score}")   # 1.0
        print(f"Reason: {result.reason}") # Syntax check: 1/1 blocks valid, 0 errors

    asyncio.run(main())
    ````

=== "Invalid Syntax Detection"

    ````python
    import asyncio
    from openjudge.graders.code import SyntaxCheckGrader

    async def main():
        grader = SyntaxCheckGrader()

        # Missing colon after function definition
        invalid_response = '''
        ```python
        def greet(name)
            return f"Hello, {name}!"
        ```
        '''

        result = await grader.aevaluate(response=invalid_response)
        print(f"Score: {result.score}")   # -0.5
        print(f"Errors: {result.metadata['syntax_errors']}")

    asyncio.run(main())
    ````

!!! tip "Syntax Validation Best Practice"
    Use `SyntaxCheckGrader` as a fast pre-filter before running more expensive execution graders. This saves computation time by catching obvious errors early.


### CodeStyleGrader

Evaluates code style including indentation consistency and naming conventions. Checks for adherence to Python style guidelines.

**When to use:**
- Code quality assessment
- Style guide compliance checking
- Educational feedback on code formatting
- Code review automation

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | Response containing code blocks to evaluate |

**What it checks:**
- **Indentation**: Consistent use of spaces or tabs
- **Naming conventions**: snake_case for functions and variables
- **Code consistency**: Overall style compliance

**Scoring:**
- **1.0**: Excellent style (consistent indentation + proper naming)
- **0.5**: Moderate style (some issues detected)
- **0.0**: No code blocks found or poor style

**Example:**

=== "Good Code Style"

    ````python
    import asyncio
    from openjudge.graders.code import CodeStyleGrader

    async def main():
        grader = CodeStyleGrader()

        # Proper snake_case naming and consistent indentation
        good_code = '''
        ```python
        def calculate_sum(numbers):
            total = 0
            for num in numbers:
                total += num
            return total
        ```
        '''

        result = await grader.aevaluate(response=good_code)
        print(f"Score: {result.score}")   # 1.0
        print(f"Reason: {result.reason}")

    asyncio.run(main())
    ````

=== "Poor Code Style"

    ````python
    import asyncio
    from openjudge.graders.code import CodeStyleGrader

    async def main():
        grader = CodeStyleGrader()

        # Incorrect PascalCase naming (should be snake_case)
        poor_code = '''
        ```python
        def CalculateSum(Numbers):
            Total = 0
            for Num in Numbers:
                Total += Num
            return Total
        ```
        '''

        result = await grader.aevaluate(response=poor_code)
        print(f"Score: {result.score}")   # Lower score
        print(f"Details: {result.metadata['details']}")

    asyncio.run(main())
    ````

!!! info "Style Checks"
    The grader evaluates:

    - **Naming**: Functions and variables should use `snake_case`
    - **Indentation**: Consistent use of spaces (4 spaces per level recommended)
    - **Consistency**: Overall adherence to Python style conventions


### PatchSimilarityGrader

Calculates similarity between generated code patches and ground truth patches using sequence matching algorithms.

**When to use:**
- Code modification evaluation
- Bug fix assessment
- Code refactoring quality measurement
- Diff-based code generation tasks

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | The generated code patch or modification |
| `reference_response` | str | Yes | The correct (oracle) patch |

**Scoring:**
- **1.0**: Perfect match with ground truth
- **0.8-0.99**: High similarity
- **0.5-0.79**: Moderate similarity
- **0.0-0.49**: Low similarity

**Example:**

```python
import asyncio
from openjudge.graders.code import PatchSimilarityGrader

async def main():
    grader = PatchSimilarityGrader()

    ground_truth = """
def calculate_area(radius):
    import math
    return math.pi * radius ** 2
"""

    response = """
def calculate_area(r):
    import math
    return math.pi * r ** 2
"""

    result = await grader.aevaluate(
        response=response,
        reference_response=ground_truth
    )

    print(f"Score: {result.score}")           # ~0.95
    print(f"Similarity: {result.metadata['similarity']:.3f}")
    print(f"Reason: {result.reason}")

asyncio.run(main())
```


## Math Graders

### MathExpressionVerifyGrader

Verifies mathematical expressions for correctness using symbolic mathematics. Supports both LaTeX and plain mathematical notation.

**When to use:**
- Math problem solving evaluation
- Educational math tutoring systems
- Mathematical reasoning assessment
- Symbolic math verification

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `response` | str | Yes | The mathematical expression to verify |
| `reference_response` | str | Yes | The correct mathematical expression |
| `timeout_score` | float | No | Score on timeout/error (default: 1.0) |

**Supported formats:**
- Plain expressions: `2+2`, `x^2+3x+2`
- LaTeX notation: `\frac{1}{2}`, `x^{2}+3x+2`
- Mixed expressions with both formats

**Scoring:**
- **1.0**: Expressions are mathematically equivalent
- **0.0**: Expressions are not equivalent
- **timeout_score**: On parsing error or timeout (default: 1.0)

**Example:**

=== "Numeric Comparison"

    ```python
    import asyncio
    from openjudge.graders.math import MathExpressionVerifyGrader

    async def main():
        grader = MathExpressionVerifyGrader()

        # Simple numeric comparison
        result = await grader.aevaluate(
            response="4",
            reference_response="2+2"
        )
        print(f"Score: {result.score}")   # 1.0
        print(f"Reason: {result.reason}") # Expressions are equivalent

    asyncio.run(main())
    ```

=== "LaTeX Expressions"

    ```python
    import asyncio
    from openjudge.graders.math import MathExpressionVerifyGrader

    async def main():
        grader = MathExpressionVerifyGrader()

        # LaTeX notation comparison
        result = await grader.aevaluate(
            response=r"\frac{1}{2}",
            reference_response="0.5"
        )
        print(f"Score: {result.score}")   # 1.0
        print(f"Reason: {result.reason}")

    asyncio.run(main())
    ```

=== "Non-Equivalent Detection"

    ```python
    import asyncio
    from openjudge.graders.math import MathExpressionVerifyGrader

    async def main():
        grader = MathExpressionVerifyGrader()

        # Non-equivalent values
        result = await grader.aevaluate(
            response="5",
            reference_response="3"
        )
        print(f"Score: {result.score}")   # 0.0
        print(f"Reason: {result.reason}") # Expressions are not equivalent

    asyncio.run(main())
    ```

!!! note "Supported Expression Types"
    The math expression verifier works best with numeric values and LaTeX expressions. Complex algebraic expressions may have limited support depending on the underlying math_verify library.

???+ tip "Advanced: Custom Timeout Handling"
    You can configure how the grader handles parsing errors or timeouts:

    ```python
    import asyncio
    from openjudge.graders.math import MathExpressionVerifyGrader

    async def main():
        # Set timeout_score to 0.0 to be stricter on errors
        grader = MathExpressionVerifyGrader(timeout_score=0.0)

        result = await grader.aevaluate(
            response="6",
            reference_response="2*3"
        )
        print(f"Score: {result.score}")   # 1.0
        print(f"Reason: {result.reason}")

    asyncio.run(main())
    ```

    **Default behavior** (`timeout_score=1.0`): Gives benefit of doubt on parsing errors
    **Strict mode** (`timeout_score=0.0`): Penalizes any errors or timeouts


## Best Practices

**For Code Evaluation:**
- Use `SyntaxCheckGrader` first as a fast pre-filter before execution
- Combine `CodeStyleGrader` with execution graders for comprehensive assessment
- Set appropriate timeout values for `CodeExecutionGrader` to prevent infinite loops
- Use `continuous=True` for partial credit on coding challenges

**For Math Evaluation:**
- Normalize mathematical expressions before comparison when possible
- Use `MathExpressionVerifyGrader` for symbolic equivalence, not string matching
- Consider both LaTeX and plain notation in your evaluation pipeline
- Set `timeout_score` based on your application's error tolerance

**For Combined Workflows:**
- Weight different graders based on your use case priorities
- Use execution graders only after syntax validation passes
- Combine code style with correctness for educational applications
- Cache evaluation results for repeated assessments


## Next Steps

- [Text Graders](text.md) — Evaluate text similarity and linguistic quality
- [Format Graders](format.md) — Validate output format and structure
- [Create Custom Graders](../building_graders/create_custom_graders.md) — Build specialized code graders






