# Graders 开发规范

本文档定义了 RM-Gallery 中 Grader 的开发规范和最佳实践，旨在确保所有 Grader 的一致性、可靠性和易用性。

## 1. Grader 基本结构

所有 Grader 必须继承 [BaseGrader](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/base_grader.py#L27-L124) 或其子类（如 [LLMGrader](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/llm_grader.py#L36-L114)），并实现 [aevaluate](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/base_grader.py#L71-L116) 方法。

常见基类包括：
1. [BaseGrader](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/base_grader.py#L27-L124) - 所有 Grader 的基类
2. [LLMGrader](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/llm_grader.py#L36-L114) - 基于大语言模型的评估器基类
3. [FunctionGrader](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/function_grader.py#L26-L75) - 基于函数的评估器基类

每个 Grader和[aevaluate](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/base_grader.py#L71-L116) 方法都应该包含：

1. 清晰的文档字符串，说明其用途和工作原理
2. 参数说明和使用示例
3. 返回值的详细说明

### 1.1 LLMGrader 结构规范

对于基于 LLM 的评估器（继承自 [LLMGrader](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/llm_grader.py#L36-L114)），Grader 文件内的代码组织应遵循 PROMPT + Grader 的形式，具体顺序如下：

1. **模块文档字符串**：描述模块的功能和用途
2. **导入语句**：按标准 Python 导入顺序排列
3. **常量定义**：定义英文 Prompt 模板字符串
4. **中文 Prompt 模板**（如果需要）：定义中文版本的 Prompt
5. **PromptTemplate 定义**：使用常量定义 [PromptTemplate](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/models/schema/prompt_template.py#L36-L96) 实例
6. **Grader 类定义**：包含类文档字符串、`__init__` 方法和 [aevaluate](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/base_grader.py#L71-L116) 方法

对于基于 LLM 的评估器，文件结构应包含必要的文档字符串和类定义：

```python
"""
Module docstring describing the purpose of this LLM grader.
"""

import json
import textwrap
from typing import Any, Optional

from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.graders.schema import GraderMode, GraderScore
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate
from rm_gallery.core.models.schema.message import ChatMessage

# English Prompt
EVALUATION_PROMPT_EN = """
Your English prompt content here with {variables}.
"""

# Chinese Prompt (if needed)
EVALUATION_PROMPT_ZH = """
你的中文提示内容，包含 {variables}。
"""

# Template definition
DEFAULT_EVALUATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(role="user", content=textwrap.dedent(EVALUATION_PROMPT_EN)),
        ],
        LanguageEnum.ZH: [
            ChatMessage(role="user", content=textwrap.dedent(EVALUATION_PROMPT_ZH)),
        ],
    },
)


class MyLLMGrader(LLMGrader):
    """
    Class docstring for the LLM grader.

    Attributes:
        name: 评估器名称
        mode: 评估模式

    Example:
        >>> grader = MyLLMGrader()
        >>> result = await grader.aevaluate(
        ...     query="What is 2+2?",
        ...     response="4"
        ... )
        >>> print(result.score)
        1.0
    """

    def __init__(
        self,
        model: Any,
        template: Optional[PromptTemplate] = DEFAULT_EVALUATION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="my_llm_grader",
            mode=GraderMode.POINTWISE,
            description="Description of this LLM grader",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_EVALUATION_TEMPLATE

    async def aevaluate(self, **kwargs: Any) -> GraderScore:
        """
        Evaluate using LLM.

        Args:
            query (str): 用户的查询或指令
            response (str): 模型生成的响应
            reference (Optional[str]): 参考答案或标准答案
            **kwargs: 其他参数

        Returns:
            GraderScore: 评估结果分数

        Example:
            >>> result = await grader.aevaluate(
            ...     query="What is 2+2?",
            ...     response="4",
            ...     reference="4"
            ... )
            >>> print(result.score)
            1.0
        """
        # Implementation here
        pass
```

### 1.2 非 LLMGrader 结构规范

对于非 LLM 类的评估器，文件结构可以更加灵活，但仍需包含必要的文档字符串和类定义：

```python
"""
Module docstring describing the purpose of this LLM grader.
"""

from rm_gallery.core.graders.base_grader import BaseGrader

class MyGrader(BaseGrader):
    """
    MyGrader 评估器

    该评估器用于评估响应是否符合特定标准。

    Attributes:
        name: 评估器名称
        mode: 评估模式

    Example:
        >>> grader = MyGrader()
        >>> result = await grader.aevaluate(
        ...     query="What is 2+2?",
        ...     response="4",
        ...     reference="4"
        ... )
        >>> print(result.score)
        1.0
    """

    def __init__(self):
        super().__init__(
            name="my_grader",
            mode=GraderMode.POINTWISE,  # or GraderMode.LISTWISE
            description="Description of what this grader evaluates"
        )

    async def aevaluate(self, **kwargs) -> GraderScore | GraderRank:
        """
        评估模型响应是否符合特定标准。

        Args:
            query (str): 用户的查询或指令
            response (str): 模型生成的响应
            reference (Optional[str]): 参考答案或标准答案
            **kwargs: 其他参数

        Returns:
            GraderScore | GraderRank: 评估结果，根据Grader的模式返回GraderScore或GraderRank

        Example:
            >>> result = await grader.aevaluate(
            ...     query="What is 2+2?",
            ...     response="4",
            ...     reference="4"
            ... )
            >>> print(result.score)
            1.0
        """
        # Implementation here
        pass
```

## 2. Prompt 规范

对于基于 LLM 的评估器（继承自 [LLMGrader](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/llm_grader.py#L36-L114)），Prompt 的设计和组织需要遵循以下规范：

### 2.1 Prompt 设计与模板规范

LLM 类 Grader 的 Prompt 应该包含以下几个关键部分：

1. **评估类型说明**：明确说明评估的任务类型
2. **评分标准**：详细说明评估的准则和标准
3. **评估准则**：列出具体的评估要点
4. **输入内容**：清晰展示需要评估的内容
5. **评分指令**：明确如何进行评分

Prompt 设计应遵循以下最佳实践：

1. 使用 [PromptTemplate](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/models/schema/prompt_template.py#L36-L96) 来管理 Prompt 模板
2. 支持多语言，使用 [LanguageEnum](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/models/schema/prompt_template.py#L21-L27) 来区分语言版本
3. 模板变量使用花括号 `{variable}` 格式
4. **清晰的结构**：使用标记（如 `<section>`）来组织 Prompt 的不同部分
5. **明确的指令**：评分指令应非常明确，避免歧义
6. **示例输出**：提供期望的输出格式示例
7. **多语言支持**：如有需要，提供中英文版本的 Prompt

### 2.2 输出格式规范

LLM 类 Grader 的输出必须遵循 JSON Schema 规范，以确保输出的一致性和可解析性：

1. **结构化输出**：LLM 的输出必须是结构化的 JSON 格式
2. **固定字段**：对于 Pointwise 模式，必须包含 `score` 和 `reason` 字段；对于 Listwise 模式，必须包含 `rank` 和 `reason` 字段
3. **类型约束**：
   - `score` 字段必须是数值类型（整数或浮点数）
   - `rank` 字段必须是整数数组
   - `reason` 字段必须是字符串类型
4. **示例格式**：

Pointwise 模式输出示例：
```json
{
  "score": 0.8,
  "reason": "Response is mostly accurate but lacks some details"
}
```

Listwise 模式输出示例：
```json
{
  "rank": [2, 1, 3],
  "reason": "The second response is most relevant, followed by the first, and the third is least relevant"
}
```

## 3. aevaluate 方法参数规范

为了确保一致性和易用性，Grader 的 [aevaluate](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/base_grader.py#L71-L116) 方法应遵循以下参数命名规范：

### 3.1 标准参数名称

#### 3.1.1 核心文本参数

| 参数名 | 用途 | 替代名称（不推荐） | 说明 |
|--------|------|-------------------|------|
| `query` | 用户查询 | `user_query`, `input_query`, `question`, `instruction` | 用户的原始问题或指令 |
| `response` | 模型响应 | `answer`, `generated`, `candidate`, `output`, `generated_text`, `prediction`, `generated_response`, `completion` | 模型生成的回答或输出 |
| `ground_truth` | 参考答案 | `expected`, `reference` | 标准答案或期望的输出 |
| `context` | 上下文信息 | `input`, `input_text`, `text` | 评估所需的额外上下文 |

#### 3.1.2 工具相关参数

| 参数名 | 用途 | 替代名称（不推荐） | 说明 |
|--------|------|-------------------|------|
| `tool_definitions` | 工具定义 | `tool_definition` | 可用工具的定义列表 |
| `tool_calls` | 工具调用 | `generated_tool_call` | 实际的工具调用记录 |

### 3.2 通用规则

1. 所有参数命名采用小写字母加下划线的 snake_case 格式
2. 避免使用易混淆的别名或缩写
3. 在基类中明确定义参数签名，子类继承并复用
4. 新增参数应遵循现有命名体系，保持一致性

### 3.3 参数使用示例

```python
async def aevaluate(
    self,
    query: str,
    response: str,
    reference: Optional[str] = None,
    context: Optional[str] = None,
    **kwargs
) -> GraderScore:
    # 实现评估逻辑
    pass
```

## 4. 返回值规范

Grader 必须返回 [GraderScore](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/schema.py#L70-L94) 或 [GraderRank](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/schema.py#L127-L153) 对象，具体取决于 Grader 的模式（Pointwise 或 Listwise）。

### 4.1 GraderScore 规范

#### 4.1.1 分数值规范

为了保持简单和一致性，Grader 返回的分数值应遵循以下规范：

1. **推荐使用二元分数**：
   - 通过/失败：`1.0` 表示通过，`0.0` 表示失败
   - 是/否：`1.0` 表示是，`0.0` 表示否

2. **如需分级评分**：
   - 最多不超过 5 个级别
   - 使用整数值：`1`, `2`, `3`, `4`, `5`
   - 避免使用浮点数，除非有特殊需要

#### 4.1.2 GraderScore 示例

```python
# 二元评分示例
return GraderScore(
    name=self.name,
    score=1.0,  # 通过
    reason="Response meets all requirements"
)

# 分级评分示例
return GraderScore(
    name=self.name,
    score=4,  # 整数值
    reason="Response is mostly correct with minor issues"
)
```

### 4.2 GraderRank 规范

[GraderRank](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/schema.py#L127-L153) 用于 Listwise 模式，对多个响应进行排序。

#### 4.2.1 排名规范

1. 排名使用从 1 开始的正整数序列
2. 数字越小表示排名越高（1 为最好）
3. 排名数量应与输入的响应数量一致
4. 每个排名只能出现一次（不允许并列排名）

#### 4.2.2 GraderRank 示例

```python
# 排名示例：3个响应的排序结果
return GraderRank(
    name=self.name,
    rank=[2, 1, 3],  # 第二个响应最好，第一个次之，第三个最差
    reason="Ranked based on relevance and completeness"
)
```

## 5. 错误处理

Grader 应该妥善处理可能发生的异常，并返回有意义的错误信息：

```python
try:
    # 评估逻辑
    pass
except Exception as e:
    return GraderScore(
        name=self.name,
        score=0.0,
        reason=f"Evaluation failed: {str(e)}"
    )
```

## 6. 测试

每个 Grader 都应该有相应的单元测试，测试用例应覆盖：

1. 正常情况下的评分
2. 边界条件
3. 错误处理
4. 各种输入组合
