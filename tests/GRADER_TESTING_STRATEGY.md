# Grader 测试策略

本文档概述了 RM-Gallery 框架中 Grader 的综合测试策略，涵盖了测试类型、测试方法和质量保证要求。

## 1. 概述

为了确保 RM-Gallery 中的 Graders 能够稳定可靠地运行，我们需要建立一套完整的测试体系。该体系包括多种类型的测试，每种测试都有其特定的目标和方法。

## 2. 测试类型

### 2.1 功能测试 (Functional Testing)

功能测试验证 Grader 是否按预期工作，包括单元测试、集成测试和回归测试。

#### 2.1.1 单元测试 (Unit Testing)

单元测试是最基础的测试类型，针对 Grader 的单个功能进行验证。

##### 测试原则
- **隔离性**: 每个测试应独立运行，不依赖其他测试
- **针对性**: 每个测试应专注于验证一个特定功能
- **自动化**: 测试应能自动运行并给出明确结果

##### 离线测试框架
所有外部服务（特别是 LLM API）必须进行模拟以实现离线测试：

```python
from unittest.mock import AsyncMock, patch
import pytest

# 模拟 LLM 调用的示例
@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_based_grader_offline():
    # 设置模拟
    mock_response = AsyncMock()
    mock_response.score = 5
    mock_response.reason = "Perfect"
    mock_response.metadata = {}

    mock_model = AsyncMock()
    mock_model.achat = AsyncMock(return_value=mock_response)

    # 使用模拟初始化 grader
    grader = MyLLMGrader(model=mock_model)

    # 执行测试
    result = await grader.aevaluate(query="test", response="test response")

    # 断言
    assert result.score == 5
    assert result.reason == "Perfect"
```

##### 标准测试结构
每个 Grader 都应按以下方式组织测试：

```python
# tests/graders/[category]/test_[grader_name].py
import pytest
from unittest.mock import AsyncMock

from rm_gallery.core.graders.[category].[grader_module] import [GraderClass]

@pytest.mark.unit
@pytest.mark.asyncio
class Test[GraderClass]:
    """Test suite for [GraderClass]"""

    def test_initialization(self):
        """测试成功初始化"""
        mock_model = AsyncMock()
        grader = [GraderClass](model=mock_model, **kwargs)
        assert grader.name == "[expected_name]"
        assert grader.mode == [expected_mode]

    @pytest.mark.asyncio
    async def test_successful_evaluation(self):
        """测试使用有效输入的成功评估"""
        # 设置模拟
        mock_response = AsyncMock()
        mock_response.score = 5
        mock_response.reason = "Good response"
        mock_response.metadata = {}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        grader = [GraderClass](model=mock_model)
        result = await grader.aevaluate(**valid_inputs)
        assert isinstance(result, [ExpectedResultType])
        assert 0 <= result.score <= 5  # 或适当范围

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """测试边缘情况，如空输入、极值"""
        mock_model = AsyncMock()
        grader = [GraderClass](model=mock_model)
        result = await grader.aevaluate(**edge_case_inputs)
        # 适当的断言

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """测试优雅的错误处理"""
        # 设置模拟以引发异常
        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(side_effect=Exception("API Error"))

        grader = [GraderClass](model=mock_model)
        result = await grader.aevaluate(**invalid_inputs)
        # 应返回有意义的错误结果，而不是抛出异常
        assert result.score == 0  # 或其他适当的错误值
```

#### 2.1.2 集成测试 (Integration Testing)

集成测试验证 Grader 在与其他组件协同工作时的表现。这些测试确保 Grader 能够正确地与模型、数据处理管道和其他系统组件交互。

##### 测试目标
- 验证 Grader 与实际模型 API 的集成
- 确保数据在不同组件间正确流动
- 验证错误处理和恢复机制

##### 测试场景
1. **模型集成测试**：
   - 测试 Grader 与不同模型提供商（如 OpenAI、DashScope）的集成
   - 验证模型响应的正确解析和处理
   - 检查超时和重试机制

2. **数据流测试**：
   - 验证输入数据从源头到 Grader 的完整流程
   - 测试不同类型和格式的输入数据处理
   - 确保输出数据格式符合预期

##### 示例测试代码
```python
# tests/integration/test_grader_integration.py
import pytest
from unittest.mock import patch, AsyncMock

from rm_gallery.core.graders.common.helpfulness import HelpfulnessGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.runner.grading_runner import GradingRunner, GraderConfig

@pytest.mark.integration
@pytest.mark.asyncio
class TestGraderIntegration:
    """Grader 集成测试"""

    @pytest.mark.asyncio
    async def test_grader_with_real_model_config(self):
        """测试使用真实模型配置的 Grader"""
        # 使用实际的模型配置（但仍模拟 API 调用）
        model_config = {
            "model": "gpt-3.5-turbo",
            "api_key": "test-key"
        }

        with patch('rm_gallery.core.models.openai_chat_model.OpenAIChatModel') as mock_model_class:
            mock_instance = AsyncMock()
            mock_response = AsyncMock()
            mock_response.score = 4
            mock_response.reason = "Good response with minor issues"
            mock_response.metadata = {}
            mock_instance.achat = AsyncMock(return_value=mock_response)
            mock_model_class.return_value = mock_instance

            grader = HelpfulnessGrader(model=model_config)
            result = await grader.aevaluate(
                query="What is the capital of France?",
                response="The capital of France is Paris.",
                context="European geography"
            )

            # 验证模型被正确初始化
            mock_model_class.assert_called_once_with(**model_config)

            # 验证结果
            assert result.score == 4
            assert "Good response" in result.reason

    @pytest.mark.asyncio
    async def test_grader_in_runner_pipeline(self):
        """测试 Grader 在 Runner 管道中的集成"""
        # 模拟模型响应
        mock_response = AsyncMock()
        mock_response.score = 5
        mock_response.reason = "Excellent"
        mock_response.metadata = {}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # 创建 Grader 并将其添加到 Runner
        grader = HelpfulnessGrader(model=mock_model)

        # 使用 mapper 配置数据转换
        grader_configs = {
            "helpfulness": GraderConfig(
                grader=grader,
                mapper={
                    "query": "question",
                    "response": "answer"
                }
            )
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # 执行评估
        results = await runner.arun(
            dataset=[{
                "question": "What is 2+2?",
                "answer": "4",
                "context": "Basic mathematics"
            }]
        )

        # 验证结果
        assert len(results) == 1
        assert "helpfulness" in results
        assert len(results["helpfulness"]) == 1
        assert results["helpfulness"][0].score == 5
```

#### 2.1.3 回归测试 (Regression Testing)

回归测试确保修复的 bug 不会再次出现。

维护先前识别的 bug 案例集合：

```python
# tests/regression/test_grader_bugs.py
import pytest
from unittest.mock import AsyncMock

from rm_gallery.core.graders.text.string_match import StringMatchGrader

@pytest.mark.regression
@pytest.mark.asyncio
class TestGraderRegression:
    """先前识别的 bug 的回归测试"""

    @pytest.mark.asyncio
    async def test_github_issue_123_unicode_handling(self):
        """测试 Unicode 处理问题 #123 的修复"""
        # 设置模拟
        mock_response = AsyncMock()
        mock_response.score = 1
        mock_response.reason = "Response handles unicode correctly"
        mock_response.metadata = {}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        grader = StringMatchGrader(model=mock_model)
        result = await grader.aevaluate(
            ground_truth="café",
            response="I love café",
            algorithm="substring_match"
        )
        # 以前由于编码问题会失败
        assert result.score > 0

    @pytest.mark.asyncio
    async def test_github_issue_125_case_sensitivity(self):
        """测试大小写敏感性问题 #125 的修复"""
        # 设置模拟
        mock_response = AsyncMock()
        mock_response.score = 1
        mock_response.reason = "Matches case insensitively"
        mock_response.metadata = {}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        grader = StringMatchGrader(model=mock_model, case_sensitive=False)
        result = await grader.aevaluate(
            ground_truth="Hello World",
            response="HELLO WORLD",
            algorithm="exact_match"
        )
        # 尽管大小写不同也应该匹配
        assert result.score == 1.0
```

### 2.2 性能测试 (Performance Testing)

性能测试衡量 Grader 的执行效率和资源消耗。

#### 基准测试
```python
# tests/performance/test_grader_performance.py
import pytest
import time
from unittest.mock import AsyncMock

from rm_gallery.core.graders.common.helpfulness import HelpfulnessGrader

@pytest.mark.performance
@pytest.mark.asyncio
class TestHelpfulnessGraderPerformance:
    """Performance tests for HelpfulnessGrader"""

    async def test_evaluation_speed(self):
        """Test evaluation speed performance"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.score = 4
        mock_response.reason = "Standard response"
        mock_response.metadata = {}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # Create grader
        grader = HelpfulnessGrader(model=mock_model)

        # Measure execution time
        start_time = time.time()

        # Execute multiple evaluations
        for _ in range(10):
            await grader.aevaluate(
                query="What is Python?",
                response="Python is a high-level programming language."
            )

        end_time = time.time()
        total_time = end_time - start_time

        # Assert reasonable performance (should complete within 1 second for 10 mocked calls)
        assert total_time < 1.0

    async def test_memory_efficiency(self):
        """Test memory efficiency during repeated evaluations"""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.score = 4
        mock_response.reason = "Standard response"
        mock_response.metadata = {}

        mock_model = AsyncMock()
        mock_model.achat = AsyncMock(return_value=mock_response)

        # Create grader
        grader = HelpfulnessGrader(model=mock_model)

        # Execute many evaluations
        for _ in range(100):
            result = await grader.aevaluate(
                query="What is Python?",
                response="Python is a high-level programming language."
            )
            assert result.score == 4
```

### 2.3 质量测试 (Quality Testing)

质量测试用于确保 Grader 的评估质量符合使用要求。这类测试关注的是 Grader 是否能够正确地区分高质量和低质量的响应。质量测试必须配置 API_KEY，基于真实数据进行在线验证评估质量。通过与 [GradingRunner](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/runner/grading_runner.py#L31-L175) 结合使用，可以显著提高测试执行效率。

#### 环境变量检测

质量测试需要配置真实的模型 API 密钥，测试脚本应检测环境变量以确认是否执行测试：

```python
import os
import pytest

# 检测是否配置了必要的 API 密钥和基础URL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 根据环境变量决定是否运行质量测试
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)

# 如果没有配置 API 密钥和基础URL，则跳过所有质量测试
pytestmark = pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")
```

#### 基于黄金标准数据集的质量测试

```python
# tests/quality/test_grader_quality.py
import os
import pytest
import pandas as pd
import numpy as np

from rm_gallery.core.runner.grading_runner import GradingRunner, GraderConfig
from rm_gallery.core.analyzer.validation import (
    AccuracyAnalyzer,
    F1ScoreAnalyzer,
    PrecisionAnalyzer,
    RecallAnalyzer,
    CorrelationAnalyzer
)
from rm_gallery.core.graders.common.helpfulness import HelpfulnessGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel

# 检测是否配置了必要的 API 密钥和基础URL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 根据环境变量决定是否运行质量测试
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)

# 如果没有配置 API 密钥和基础URL，则跳过所有质量测试
pytestmark = pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")

@pytest.mark.quality
class TestGraderQuality:
    """测试 Grader 的评估质量"""

    @pytest.fixture
    def dataset(self):
        """加载黄金标准数据集"""
        # 这里应该加载一个包含查询、响应和人工标注质量分数的数据集
        # 数据格式示例：
        # [
        #   {
        #     "query": "什么是光合作用？",
        #     "response": "光合作用是植物利用阳光将二氧化碳和水转化为葡萄糖的过程。",
        #     "human_score": 5  # 5表示高质量（1-5评分）
        #   },
        #   ...
        # ]
        return [
            {
                "query": "什么是光合作用？",
                "response": "光合作用是植物利用阳光将二氧化碳和水转化为葡萄糖的过程，这是生态系统中非常重要的一个环节。",
                "context": "生物学基础知识",
                "human_score": 5  # 5表示高质量（1-5评分）
            },
            {
                "query": "什么是光合作用？",
                "response": "光合作用是一种化学反应。",
                "context": "生物学基础知识",
                "human_score": 2  # 2表示低质量
            }
        ]

    @pytest.fixture
    def model(self):
        """根据环境变量返回OpenAIChatModel实例"""
        if OPENAI_API_KEY:
            config = {"model": "gpt-3.5-turbo", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            # 这种情况不应该发生，因为测试已经被跳过了
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_discriminative_power_with_runner(self, dataset, model):
        """测试 Grader 区分高低质量响应的能力（使用 Runner）"""
        # 创建使用真实模型的 Grader，传入OpenAIChatModel实例
        grader = HelpfulnessGrader(model=model)

        # 使用 mapper 配置数据转换
        grader_configs = {
            "helpfulness": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context"
                }
            )
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # 准备测试数据
        test_data = dataset
        human_scores = [item["human_score"] for item in dataset]

        # 使用 Runner 执行批量评估
        results = await runner.arun(dataset=test_data)

        # 使用 AccuracyAnalyzer 计算准确率指标
        accuracy_analyzer = AccuracyAnalyzer()
        accuracy_result = accuracy_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness"],
            label_path="human_score"
        )

        # 使用 PrecisionAnalyzer 计算精确率指标
        precision_analyzer = PrecisionAnalyzer()
        precision_result = precision_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness"],
            label_path="human_score"
        )

        # 使用 RecallAnalyzer 计算召回率指标
        recall_analyzer = RecallAnalyzer()
        recall_result = recall_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness"],
            label_path="human_score"
        )

        # 使用 F1ScoreAnalyzer 计算F1分数指标
        f1_analyzer = F1ScoreAnalyzer()
        f1_result = f1_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness"],
            label_path="human_score"
        )

        # 使用 CorrelationAnalyzer 计算相关性指标
        correlation_analyzer = CorrelationAnalyzer()
        correlation_result = correlation_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness"],
            label_path="human_score"
        )

        # 断言质量指标达到预期阈值
        assert accuracy_result.accuracy >= 0.7, f"准确率低于阈值: {accuracy_result.accuracy}"
        assert precision_result.precision >= 0.7, f"精确率低于阈值: {precision_result.precision}"
        assert recall_result.recall >= 0.7, f"召回率低于阈值: {recall_result.recall}"
        assert f1_result.f1_score >= 0.7, f"F1分数低于阈值: {f1_result.f1_score}"
        assert correlation_result.correlation >= 0.7, f"相关性低于阈值: {correlation_result.correlation}"

        # 验证分析结果包含必要的元数据
        assert "explanation" in accuracy_result.metadata
        assert "explanation" in precision_result.metadata
        assert "explanation" in recall_result.metadata
        assert "explanation" in f1_result.metadata
        assert "explanation" in correlation_result.metadata

        assert accuracy_result.name == "Accuracy Analysis"
        assert precision_result.name == "Precision Analysis"
        assert recall_result.name == "Recall Analysis"
        assert f1_result.name == "F1 Score Analysis"
        assert correlation_result.name == "Correlation Analysis"

    @pytest.mark.asyncio
    async def test_consistency_with_runner(self, dataset, model):
        """测试 Grader 评估的一致性（使用 Runner）"""
        # 创建使用真实模型的 Grader，传入OpenAIChatModel实例
        grader = HelpfulnessGrader(model=model)

        # 使用重复配置实现一致性测试
        grader_configs = {
            "helpfulness_run1": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context"
                }
            ),
            "helpfulness_run2": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "response",
                    "context": "context"
                }
            )
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # 准备测试数据
        test_data = dataset

        # 使用 Runner 执行批量评估
        results = await runner.arun(dataset=test_data)

        # 使用 ConsistencyAnalyzer 计算一致性指标
        from rm_gallery.core.analyzer.validation import ConsistencyAnalyzer
        consistency_analyzer = ConsistencyAnalyzer()
        consistency_result = consistency_analyzer.analyze(
            first_run_results=results["helpfulness_run1"],
            second_run_results=results["helpfulness_run2"]
        )

        # 断言一致性指标达到预期阈值
        assert consistency_result.consistency >= 0.9, f"评估一致性不足: {consistency_result.consistency}"

        # 验证分析结果包含必要的元数据
        assert "explanation" in consistency_result.metadata
        assert consistency_result.name == "Consistency Analysis"
```

#### 基于对抗样本的质量测试

```python
# tests/quality/test_adversarial_examples.py
import os
import pytest
from unittest.mock import AsyncMock

from rm_gallery.core.runner.grading_runner import GradingRunner, GraderConfig
from rm_gallery.core.analyzer.validation import (
    FalsePositiveAnalyzer,
    FalseNegativeAnalyzer
)
from rm_gallery.core.graders.common.helpfulness import HelpfulnessGrader
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel

# 检测是否配置了必要的 API 密钥和基础URL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 根据环境变量决定是否运行质量测试
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)

# 如果没有配置 API 密钥和基础URL，则跳过所有质量测试
pytestmark = pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires API keys and base URL to run quality tests")

@pytest.mark.quality
class TestAdversarialExamples:
    """测试 Grader 对对抗样本的鲁棒性"""

    @pytest.fixture
    def dataset(self):
        """加载对抗样本数据集"""
        # 这里应该加载一个包含查询、正确响应、错误响应及其标签的数据集
        # 数据格式示例：
        # [
        #   {
        #     "query": "地球是什么？",
        #     "correct_response": "地球是太阳系中的行星之一",
        #     "incorrect_response": "地球是宇宙中唯一有生命的星球",
        #     "context": "基础天文学知识",
        #     "correct_label": 5,
        #     "incorrect_label": 2
        #   },
        #   ...
        # ]
        return [
            {
                "query": "地球是什么？",
                "correct_response": "地球是太阳系中的行星之一",
                "incorrect_response": "地球是宇宙中唯一有生命的星球",
                "context": "基础天文学知识",
                "correct_label": 5,   # 高质量评分
                "incorrect_label": 2  # 低质量评分
            },
            {
                "query": "水的化学分子式是什么？",
                "correct_response": "水的化学分子式是H2O",
                "incorrect_response": "水的化学分子式是CO2",
                "context": "基础化学知识",
                "correct_label": 5,   # 高质量评分
                "incorrect_label": 2  # 低质量评分
            }
        ]

    @pytest.fixture
    def model(self):
        """根据环境变量返回OpenAIChatModel实例"""
        if OPENAI_API_KEY:
            config = {"model": "gpt-3.5-turbo", "api_key": OPENAI_API_KEY}
            if OPENAI_BASE_URL:
                config["base_url"] = OPENAI_BASE_URL
            return OpenAIChatModel(**config)
        else:
            # 这种情况不应该发生，因为测试已经被跳过了
            raise RuntimeError("No API key configured")

    @pytest.mark.asyncio
    async def test_adversarial_helpfulness_with_runner(self, dataset, model):
        """测试对帮助性对抗样本的识别能力（使用 Runner）"""
        # 创建使用真实模型的 Grader，传入OpenAIChatModel实例
        grader = HelpfulnessGrader(model=model)

        # 使用 mapper 配置数据转换
        # 通过配置GraderConfig实现同时评估正确和错误答案
        grader_configs = {
            "helpfulness_correct": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "correct_response",
                    "context": "context"
                }
            ),
            "helpfulness_incorrect": GraderConfig(
                grader=grader,
                mapper={
                    "query": "query",
                    "response": "incorrect_response",
                    "context": "context"
                }
            )
        }
        runner = GradingRunner(grader_configs=grader_configs)

        # 准备测试数据
        test_data = dataset

        # 使用 Runner 执行批量评估
        results = await runner.arun(dataset=test_data)

        # 使用 FalsePositiveAnalyzer 计算误报率指标
        fp_analyzer = FalsePositiveAnalyzer()
        fp_result = fp_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness_incorrect"],  # 错误响应的评估结果
            label_path="incorrect_label"
        )

        # 使用 FalseNegativeAnalyzer 计算漏报率指标
        fn_analyzer = FalseNegativeAnalyzer()
        fn_result = fn_analyzer.analyze(
            dataset=test_data,
            grader_results=results["helpfulness_correct"],  # 正确响应的评估结果
            label_path="correct_label"
        )

        # 断言误报率和漏报率达到预期阈值
        assert fp_result.false_positive_rate <= 0.3, f"误报率过高: {fp_result.false_positive_rate}"
        assert fn_result.false_negative_rate <= 0.3, f"漏报率过高: {fn_result.false_negative_rate}"

        # 验证分析结果包含必要的元数据
        assert "explanation" in fp_result.metadata
        assert "explanation" in fn_result.metadata

        assert fp_result.name == "False Positive Analysis"
        assert fn_result.name == "False Negative Analysis"
```

## 3. 测试执行

### 3.1 本地开发环境

```bash
# 运行所有 grader 测试
poetry run pytest tests/graders/ -v

# 运行特定类别
poetry run pytest tests/graders/text/ -v

# 运行带覆盖率
poetry run pytest tests/graders/ --cov=rm_gallery.core.graders --cov-report=html

# 运行性能测试
poetry run pytest tests/performance/ --benchmark-only

# 运行质量测试（仅模拟测试）
poetry run pytest tests/quality/ -v -m quality

# 运行所有质量测试（包括真实模型测试，如果配置了API密钥）
poetry run pytest tests/quality/ -v -m quality --run-live

# 运行特定类型的测试（使用标记）
poetry run pytest tests/graders/ -m unit       # 运行单元测试
poetry run pytest tests/graders/ -m integration # 运行集成测试
poetry run pytest tests/graders/ -m regression  # 运行回归测试
poetry run pytest tests/graders/ -m performance # 运行性能测试
poetry run pytest tests/graders/ -m quality # 运行质量测试
```

### 3.2 持续集成 (CI/CD)

每次推送和拉取请求时，测试都会通过 GitHub Actions 自动运行，结果报告给 Codecov 以进行覆盖率跟踪。

#### GitHub Actions 工作流

```
# .github/workflows/grader-tests.yml
name: Grader Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install poetry
        poetry install

    - name: Run unit tests
      run: |
        poetry run pytest tests/graders/ -v --cov=rm_gallery.core.graders -m unit

    - name: Run integration tests
      run: |
        poetry run pytest tests/integration/ -v -m integration

    - name: Run quality tests (simulation only)
      run: |
        poetry run pytest tests/quality/ -v -m quality

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
```

## 4. 质量保证要求

### 4.1 代码覆盖率

- **行覆盖率**：90%
- **分支覆盖率**：80%
- **函数覆盖率**：100%

### 4.2 测试用例要求

每个 Grader 必须有涵盖以下内容的测试用例：

1. **正常操作**：产生预期输出的典型输入
2. **边界条件**：边缘情况，如空字符串、最大长度输入
3. **错误条件**：无效输入、缺少参数
4. **配置变化**：不同的初始化参数
5. **序列化**：to_dict/from_config 往返

## 5. 测试维护

### 5.1 添加新 Grader 时

1. 在 `tests/graders/[category]/` 中创建相应的测试文件
2. 实现标准测试套件（初始化、评估、边缘情况、错误处理）
3. 确保达到最低覆盖率阈值
4. 为质量测试准备黄金标准数据集
5. 更新文档

### 5.2 修改现有 Grader 时

1. 更新现有测试以匹配新行为
2. 为任何 bug 修复添加回归测试
3. 验证向后兼容性或相应更新测试
4. 如果行为有显著改变，更新质量测试
5. 重新运行完整测试套件以确保没有回归