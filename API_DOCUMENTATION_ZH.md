# -*- coding: utf-8 -*-

# 基础API

### 1. Grader定义

评估器是负责评估样本并分配分数的核心组件。有两种主要类型的评估器：通用（基于函数）评估器和基于LLM的评估器。

#### Grader基类

所有评估器都继承自Grader基类，它定义了评估器的基本接口和属性。

```python
class Grader(ABC):
    """Base class for graders.

    This abstract base class defines the interface for all graders.
    Subclasses must implement the evaluate method.

    Attributes:
        name (str): The name of the grader.
        mode (GraderMode): The grader mode (pointwise or listwise).
        description: The description of the grader.
    """

    async def aevaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate method to be implemented by subclasses.

        This abstract method must be implemented by all Grader subclasses. It performs
        the actual evaluation logic and returns either a score or a ranking based on
        the grader's mode (pointwise or listwise).

        In pointwise mode, each sample is evaluated independently, returning a
        GraderScore with a numerical value and explanation. In listwise mode, all
        samples are evaluated together, returning a GraderRank with a ranked list and
        explanation.

        Args:
            **kwargs: Arbitrary keyword arguments containing the data to be evaluated.
                     The specific arguments depend on the grader implementation but
                     typically include fields like 'query', 'answer', 'context', etc.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical score (typically 0.0-1.0 or 1-5 scale)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of items (e.g., [1, 3, 2] means first
                      item is best, third item is second best, second item is worst)
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise grader
            >>> class AccuracyGrader(Grader):
            ...     def __init__(self):
            ...         super().__init__(
            ...             name="accuracy",
            ...             mode=GraderMode.POINTWISE,
            ...             description="Evaluates factual accuracy of answers"
            ...         )
            ...
            ...     async def aevaluate(self, query: str, answer: str, **kwargs) -> GraderScore:
            ...         # Implementation would evaluate accuracy
            ...         return GraderScore(
            ...             name=self.name,
            ...             score=0.8,
            ...             reason="Answer is mostly accurate but missing some details"
            ...         )
            ...
            >>> # Example for listwise grader
            >>> class RelevanceRanker(Grader):
            ...     def __init__(self):
            ...         super().__init__(
            ...             name="relevance_ranking",
            ...             mode=GraderMode.LISTWISE,
            ...             description="Ranks answers by relevance"
            ...         )
            ...
            ...     async def aevaluate(self, query: str, answer_1: str, answer_2: str, **kwargs) -> GraderRank:
            ...         # Implementation would rank answers by relevance
            ...         return GraderRank(
            ...             rank=[1, 2],
            ...             reason="First answer is more relevant to the query than the second"
            ...         )
        """
        ...

    async def aevaluate_eval_case(
        self,
        eval_case: EvalCase,
        parser: EvalCaseParser | None = None,
        *args,
        **kwargs: Any,
    ) -> List[GraderScore]:
        """Main entry point to evaluate eval case.

        Evaluates one eval cases using the  grader.

        Args:
            eval_case (EvalCase):
                The eval case to evaluate.
                EvalCase consists of:
                    - data: A dictionary containing shared data for all samples
                    - samples: A list of dictionaries, each representing an individual
                    sample to evaluate

            parser (EvalCaseParser | Callable | None, optional):
                Optional parser to transform the eval case before evaluation. This
                allows for mapping field names between the data structure and what the
                grader expects. Can be:
                1. A dictionary with direct field mappings where paths start with "data" or "sample"
                2. A callable function that takes a EvalCase and returns a EvalCase
                3. None, in which case the eval case is used as is
                Defaults to None.
            *args:
                Additional positional arguments to pass to the grader.
            **kwargs:
                Additional keyword arguments to pass to the grader.

        Returns:
            List[GraderScore] | List[List[GraderScore]]:
                For a single EvalCase: a list of GraderScore objects, one for each
                sample within the EvalCase.

                For a list of EvalCases: a list of lists of GraderScore objects,
                where each inner list contains the scores for the corresponding
                EvalCase in the input list.

                Each GraderScore contains:
                    - score: A numerical score assigned by the grader
                    - reason: Explanation of how the score was determined
                    - metadata: Optional additional information from the evaluation

        Raises:
            ValueError: If grader function signature is invalid.

        Example:
            >>> from rm_gallery.core.schema.data import EvalCase
            >>> from rm_gallery.core.grader.base import LLMGrader, evaluate
            >>>
            >>> # Create eval case
            >>> eval_case = EvalCase(
            ...     input={"query": "What is the capital of France?"},
            ...     outputs=[
            ...         {"answer": "Paris"},
            ...         {"answer": "London"}
            ...     ]
            ... )
            >>>
            >>> # Create grader
            >>> grader = LLMGrader(
            ...     name="factual_accuracy",
            ...     template=[
            ...         {
            ...             "role": "system",
            ...             "content": "You are evaluating factual accuracy."
            ...         },
            ...         {
            ...             "role": "user",
            ...             "content": "Question: {query}\\nAnswer: {answer}\\nRate accuracy (0-1):"
            ...         }
            ...     ],
            ...     model={"model_name": "qwen-plus"}
            ... )
            >>>
            >>> # Evaluate
            >>> results = await grader.aevaluate(eval_case=eval_case)
            >>> print(results)
        """
```

#### 通用评估器

通用评估器是基于函数的评估器，执行确定性评估。它们具有以下特点：
- 快速高效
- 确定性和可重现
- 适用于基于规则的评估

##### 方法1：继承Grader基类

```python
from rm_gallery.core.grader.base import Grader, GraderMode, GraderScore

class ExactMatchGrader(Grader):
    """Exact match grader for comparing strings."""

    def __init__(self):
        super().__init__(
            name="exact_match",
            mode=GraderMode.POINTWISE,
            description="通过精确字符串匹配进行评估"
        )

    async def aevaluate(self, reference: str, prediction: str) -> GraderScore:
        """Evaluate through exact string matching.

        Args:
            reference: The reference string to compare against
            prediction: The prediction string to evaluate

        Returns:
            GraderScore with 1.0 if matched, otherwise 0.0
        """
        score = 1.0 if reference.strip() == prediction.strip() else 0.0
        return GraderScore(
            name=self.name,
            score=score,
            reason="精确匹配" if score == 1.0 else "非精确匹配"
        )
```

##### 方法2：使用FunctionGrader包装器

```python
from rm_gallery.core.grader.base import FunctionGrader, GraderScore

async def exact_match_function(reference: str, prediction: str) -> GraderScore:
    """Exact match checking function.

    Args:
        reference: The reference string to compare against
        prediction: The prediction string to evaluate

    Returns:
        GraderScore with 1.0 if matched, otherwise 0.0
    """
    score = 1.0 if reference.strip() == prediction.strip() else 0.0
    return GraderScore(
        name=self.name,
        score=score,
        reason="精确匹配" if score == 1.0 else "非精确匹配"
    )

# 将函数包装为评估器
exact_match_grader = FunctionGrader(
    name="exact_match",
    func=exact_match_function,
    description="通过精确字符串匹配进行评估"
)
```

#### LLM 评估器

基于LLM的评估器使用大语言模型作为评判。它们具有以下特点：
- 更加灵活和细致
- 适用于复杂、主观的评估
- 可以处理自然语言推理

创建LLM评估器需要提供Template和模型配置

##### Template 类

Template类是LLM Template系统的核心，用于定义对话模板和必需字段。

```python
class Template(BaseModel):
    """Template for generating chat messages.

    Attributes:
        messages (List[ChatMessage] | Dict[LanguageEnum, List[ChatMessage]]):
            定义对话消息的列表或字典。如果是字典，键为语言枚举（如EN、ZH），值为该语言的消息列表
    """

    messages: List[ChatMessage] | Dict[
        LanguageEnum,
        List[ChatMessage],
    ] = Field(
        default_factory=list,
        description="messages for generating chat"
    )
```

##### 使用示例

```python
from rm_gallery.core.schema.template import Template
from rm_gallery.core.schema.message import ChatMessage

# 单语言模板
template = Template(
    messages=[
        ChatMessage(
            role="system",
            content="你是一个 helpful assistant，负责评估回答的质量。"
        ),
        ChatMessage(
            role="user",
            content="问题: {query}\n回答: {answer}\n请评估这个回答的质量，给出0-1之间的分数。"
        )
    ]
)

# 多语言模板
multilingual_template = Template(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content="You are a helpful assistant for evaluating answer quality."
            ),
            ChatMessage(
                role="user",
                content="Question: {query}\nAnswer: {answer}\nPlease evaluate the quality of this answer on a scale of 0-1."
            )
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content="你是一个 helpful assistant，负责评估回答的质量。"
            ),
            ChatMessage(
                role="user",
                content="问题: {query}\n回答: {answer}\n请评估这个回答的质量，给出0-1之间的分数。"
            )
        ]
    }
)
```

### 2. EvalCase定义

EvalCase是评估任务的基本数据结构。它由共享的上下文数据和要评估的独立样本组成。

```python
class EvalCase(BaseModel):
    """EvalCase containing shared data and individual samples.

    EvalCase is the basic data structure for evaluation tasks. It consists of
    shared context data and independent samples to be evaluated.

    For pointwise evaluation: Each EvalCase contains one sample in the samples list.
    For listwise evaluation: Each EvalCase contains multiple samples in the samples list.

    Attributes:
        data (dict): A dictionary containing shared data for all samples
                    (e.g., query, reference answer).
        samples (List[dict]): A list of dictionaries, each representing an
                             individual sample to evaluate.
    """
    data: dict = Field(default_factory=dict, description="Shared data for all samples")
    samples: List[dict] = Field(
        default_factory=list,
        description="List of individual samples to evaluate",
    )
```

#### 使用示例

##### 点对点评估
```python
# 点对点评估（每个EvalCase一个样本）
eval_case = EvalCase(
    input={"query": "解释量子计算"},
    outputs=[{"answer": "量子计算使用量子比特..."}]
)
```

##### 列表式评估
```python
# 列表式评估（每个EvalCase多个样本）
eval_case = EvalCase(
    input={"query": "解释量子计算"},
    outputs=[
        {"answer": "量子计算使用量子比特..."},
        {"answer": "量子计算是一种计算类型..."}
    ]
)
```

### 3. EvalCaseParser定义

EvalCaseParser 是一个联合类型，可以是字典映射或可调用函数：

- Dict[str, str]: 字段映射字典，键为目标字段名，值为源路径。路径以"data"或"sample"开头
- Callable[[EvalCase], EvalCase]: 自定义处理函数，接收EvalCase并返回处理后的EvalCase

#### 使用示例

```python
# 当您的数据字段与评估器期望不匹配时
# 使用字典映射，路径以"data"或"sample"开头
parser = {
    "query": "data.question",      # 将数据中的"data.question"映射为评估器的"query"
    "answer": "sample.response"    # 将数据中的"sample.response"映射为评估器的"answer"
}

# 或者使用自定义函数处理
def custom_parser(eval_case: EvalCase) -> EvalCase:
    # 自定义处理逻辑
    return processed_eval_case
```

### 4. Grader使用方法

#### 基本使用示例

```python
import asyncio
from rm_gallery.core.schema.data import EvalCase, EvalCaseParser
from rm_gallery.core.grader.base import LLMGrader, FunctionGrader, GraderScore
from rm_gallery.core.model.base import ChatModelBase

# 创建数据样本
eval_cases = [
    EvalCase(
        input={"query": "法国的首都是什么?"},
        outputs=[
            {"answer": "巴黎"},
            {"answer": "伦敦"}
        ]
    )
]

# 创建基于函数的评估器
async def contains_capital_function(query: str, answer: str) -> GraderScore:
    """检查答案是否包含正确的首都。"""
    capitals = {
        "法国": "巴黎",
        "英国": "伦敦",
        "德国": "柏林"
    }

    for country, capital in capitals.items():
        if country in query and capital.lower() in answer.lower():
            return GraderScore(name=self.name, score=1.0, reason=f"包含首都 {capital}")

    return GraderScore(name=self.name, score=0.0, reason="不包含正确首都")

contains_capital_grader = FunctionGrader(
    name="contains_capital",
    func=contains_capital_function,
    description="检查答案是否包含正确的首都"
)

# 创建基于LLM的评估器
llm_grader = LLMGrader(
    name="comprehensive_quality",
    template=[
        {
            "role": "system",
            "content": "您正在评估答案的质量和准确性。"
        },
        {
            "role": "user",
            "content": "问题: {query}\n答案: {answer}\n综合评价 (0-1):"
        }
    ],
    model=ChatModelBase(model_name="qwen-plus"),
    description="综合评估答案的质量和准确性"
)

# 使用函数评估器评估
func_results = await contains_capital_grader.evaluate_eval_case(eval_case=eval_cases[0])

# 使用LLM评估器评估
llm_results = await llm_grader.evaluate_eval_case(eval_case=eval_cases[0])

print("函数评估器结果:", func_results)
print("LLM评估器结果:", llm_results)
```

# 进阶API

## GradingRunner

GradingRunner是一个高级评估运行器，允许同时使用多个评估器对同一批数据进行评估。它特别适用于需要从多个维度评估模型输出的场景。

### 结构

```python
class GradingRunner(BaseRunner):
    """Runner for grading by graders."""

    def __init__(
        self,
        grading_configs: Dict[str, GradingConfig],
        max_concurrent: int = 32,
    ):
        """Initialize the EvaluationRunner.

        Args:
            grading_configs: Dictionary of grading configurations where keys are dimension names
                           and values are grading configurations
            max_concurrent: Maximum number of concurrent evaluations (default: 32)
        """

    async def aevaluate(self, eval_case: EvalCase) -> GradingResult:
        """Run experiment for a single sample.

        Args:
            eval_case: The eval case to evaluate

        Returns:
            Grading result with scores for each dimension
        """

    async def aevaluate_batch(
        self,
        eval_cases: List[EvalCase],
        *args,
        **kwargs: Any,
    ) -> dict:
        """Run experiment.

        Args:
            eval_cases: The evaluation eval cases

        Returns:
            Evaluation result containing scores for all dimensions
        """
```

### 使用示例

```python
from rm_gallery.core.runner.grading import GradingRunner
from rm_gallery.core.schema.data import EvalCase

# 准备数据样本
eval_cases = [
    EvalCase(
        input={"query": "如何制作巧克力蛋糕？"},
        outputs=[
            {"answer": "首先预热烤箱至180度，然后准备面粉、糖、鸡蛋和巧克力..."},
            {"answer": "巧克力蛋糕的制作方法：1.预热烤箱；2.混合材料；3.烘烤30分钟。"}
        ]
    )
]

# 定义评估配置
grading_configs = {
    "helpfulness": {
        "grader": helpfulness_grader,  # 已定义的帮助性评估器
        "weight": 1.0
    },
    "accuracy": {
        "grader": accuracy_grader,     # 已定义的准确性评估器
        "weight": 1.0
    },
    "completeness": {
        "grader": completeness_grader, # 已定义的完整性评估器
        "weight": 0.8
    }
}

# 创建GradingRunner
runner = GradingRunner(
    grading_configs=grading_configs,
    max_concurrent=10  # 最大并发数
)

# 运行评估
results = await runner(eval_cases)

# 结果将包含每个维度的评分以及总分
print(results)
```

## AutoRubric/AutoGrader

AutoRubric和AutoGrader是一组自动化评估工具，可以根据提供的数据样本自动生成评分标准（评分细则）和评估器。

### AutoRubrics

AutoRubrics能够根据给定的数据样本自动生成评估标准（评分细则）。它支持两种模式：

1. **Single Mode（单样本模式）**：为每个样本独立生成评分细则
2. **Batch Mode（批处理模式）**：使用MCR²算法选择最优的评分细则集合

#### 结构

```python
class AutoRubrics(BaseRunner):
    """
    Dual-Mode AutoRubrics Generator

    Supports two generation modes:
    1. Single Mode: Generate rubrics for each sample independently
    2. Batch Mode: Use MCR-based selection and aggregation for optimal rubric sets
    """

    def __init__(
        self,
        model: OpenAIChatModel,
        parser: EvalCaseParser | Callable | None = None,
        config: AutoRubricsConfig | None = None,
    ):
        """
        Initialize AutoRubrics

        Args:
            model: Language model used for rubric generation
            parser: Optional parser to transform eval cases
            config: AutoRubrics configuration
        """
```

#### 使用示例

```python
from rm_gallery.core.runner.auto_rubrics import AutoRubrics
from rm_gallery.core.model.openai_llm import OpenAIChatModel

# 准备数据样本
eval_cases = [
    EvalCase(
        input={"query": "解释量子计算的基本原理"},
        outputs=[
            {"answer": "量子计算利用量子比特的叠加和纠缠特性进行计算..."},
            {"answer": "量子计算是一种基于量子力学的计算方式..."}
        ]
    ),
    # 更多数据样本...
]

# 创建语言模型
model = OpenAIChatModel(model_name="gpt-4")

# 创建AutoRubrics实例
auto_rubrics = AutoRubrics(
    model=model,
    config=AutoRubricsConfig(
        generation_mode=GenerationMode.SINGLE,  # 或 BATCH
        grader_mode=GraderMode.POINTWISE,
        language=LanguageEnum.ZH,
        generate_number=5  # 为每个样本生成5条评分细则
    )
)

# 生成评分细则
rubrics_result = await auto_rubrics(eval_cases)

# 获取生成的评分细则
final_rubrics = rubrics_result["final_rubrics"]
print("自动生成的评分细则:")
for i, rubric in enumerate(final_rubrics):
    print(f"{i+1}. {rubric}")
```

### AutoGrader

AutoGrader在AutoRubrics的基础上进一步生成完整的评估器(LLMGrader)。它首先使用AutoRubrics生成评分细则，然后基于这些细则创建一个可以直接使用的评估器。

#### 结构

```python
class AutoGrader(BaseRunner):
    def __init__(
        self,
        model: OpenAIChatModel,
        parser: EvalCaseParser | Callable | None = None,
        config: AutoGraderConfig | None = None,
    ):
        """AutoGrader"""

    async def aevaluate_batch(
        self,
        eval_cases: List[EvalCase],
        *args,
        **kwargs: Any,
    ) -> LLMGrader:
        """Generate an LLMGrader based on automatically generated rubrics

        Args:
            eval_cases: Training eval cases used to generate rubrics

        Returns:
            LLMGrader: An automatically generated grader that can be used for evaluation
        """
```

#### 使用示例

```python
from rm_gallery.core.runner.auto_rubrics import AutoGrader

# 使用相同的数据样本和模型
auto_grader = AutoGrader(
    model=model,
    config=AutoGraderConfig(
        generation_mode=GenerationMode.SINGLE,
        grader_mode=GraderMode.POINTWISE,
        language=LanguageEnum.ZH
    )
)

# 自动生成评估器
generated_grader = await auto_grader.aevaluate_batch(eval_cases)

# 使用自动生成的评估器进行评估
evaluation_results = await generated_grader.evaluate_eval_case(eval_case=test_eval_case)  # 测试数据

print("自动评估结果:", evaluation_results)
```

### 完整工作流程示例

```python
import asyncio
from rm_gallery.core.runner.auto_rubrics import AutoGrader, AutoRubrics
from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.data import EvalCase

async def main():
    # 准备训练数据
    train_data = [
        EvalCase(
            input={"query": "什么是人工智能？"},
            outputs=[
                {"answer": "人工智能是计算机科学的一个分支，致力于创造智能机器..."},
                {"answer": "AI是使机器能够执行通常需要人类智能的任务的技术..."}
            ]
        )
        # 更多训练样本...
    ]

    # 准备测试数据
    test_data = [
        EvalCase(
            input={"query": "机器学习和深度学习有什么区别？"},
            outputs=[
                {"answer": "机器学习是人工智能的一个子集，而深度学习是机器学习的一种特殊形式..."},
                {"answer": "深度学习使用神经网络，而机器学习使用各种算法..."}
            ]
        )
    ]

    # 创建模型
    model = OpenAIChatModel(model_name="gpt-4")

    # 1. 使用AutoRubrics生成评分细则
    auto_rubrics = AutoRubrics(model=model)
    rubrics_result = await auto_rubrics(train_data)
    print("生成的评分细则:")
    for rubric in rubrics_result["final_rubrics"]:
        print(f"- {rubric}")

    # 2. 使用AutoGrader生成完整评估器
    auto_grader = AutoGrader(model=model)
    auto_generated_grader = await auto_grader.aevaluate_batch(train_data)

    # 3. 使用自动生成的评估器评估测试数据
    results = await auto_generated_grader.evaluate_eval_case(eval_case=test_data)
    print("\n自动评估结果:")
    for i, result in enumerate(results):
        print(f"样本 {i+1}: 分数={result.score}, 理由={result.reason}")

# 运行示例
# asyncio.run(main())
```