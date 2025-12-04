# RM-Gallery 架构优化方案总结

## 📋 核心问题

### 1. 命名不一致
```python
❌ AutoGrader(BaseRunner)      # 应该是 Runner 而不是 Grader
❌ GraderStrategy              # 应该是 GradingStrategy
❌ EvalCaseParser              # 应该是 FieldMapper
❌ Chat                        # 职责不清晰
```

### 2. 职责边界模糊
- `Chat` 类同时负责格式化和执行，违反单一职责原则
- `GradingRunner` 返回简单dict，而 `EvaluationRunner` 有完整报告
- Schema层混杂，缺少清晰的组织

### 3. 缺少关键功能
- 没有混合评分器（规则+LLM）
- 没有集成评分器（多个Grader组合）
- 没有校准机制
- 评分报告不够丰富

---

## 🎯 优化目标

### 1. 架构清晰化
- **重命名**：消除歧义
- **重组**：Schema层按职责细分
- **分离**：职责单一

### 2. 功能增强
- **HybridGrader**: 规则+LLM混合评分
- **EnsembleGrader**: 多Grader集成
- **CalibrationStrategy**: 分数校准
- **GradingReport**: 统一丰富的报告

### 3. 扩展性提升
- 领域特定数据定义
- Grader元数据增强
- 统一报告接口

---

## 🔄 重命名清单

### 高优先级（立即执行）
| 当前 | 新名称 | 原因 |
|-----|--------|------|
| `AutoGrader` | `AutoGradingRunner` | 它是Runner不是Grader |
| `GraderStrategy` | `GradingStrategy` | 处理的是grading过程 |
| `EvalCaseParser` | `FieldMapper` | 更准确描述功能 |
| `parse_eval_case` | `apply_field_mapping` | 与新名称一致 |

### 中优先级（后续执行）
| 当前 | 新名称 | 原因 |
|-----|--------|------|
| `Chat` | `PromptExecutor` | 更明确表示职责 |
| `_GraderScore` | 合并到 `GraderScore` | 消除混淆 |
| `_GraderRank` | 合并到 `GraderRank` | 消除混淆 |

---

## 📁 目录结构调整

### 当前结构
```
core/
├── schema/
│   ├── data.py              # ❌ 混合多种模型
│   ├── grader.py
│   └── template.py
├── grader/
│   └── auto/
│       └── auto_grader.py   # ❌ 需要重命名
└── runner/
    └── grading/
        └── strategy/
            └── base.py      # ❌ GraderStrategy

gallery/
└── grader/                  # ✅ 已经组织良好
```

### 优化后结构
```
core/
├── schema/
│   ├── data/               # ✅ 数据模型独立
│   │   ├── eval_case.py
│   │   ├── field_mapper.py
│   │   └── validator.py
│   ├── grading/            # ✅ 评分模型独立
│   │   ├── grader_result.py
│   │   ├── grader_info.py
│   │   └── grading_report.py  # 新增
│   ├── evaluation/         # ✅ 评估模型独立
│   │   ├── eval_result.py
│   │   └── eval_report.py
│   └── template/
│       ├── template.py
│       └── message.py
├── grader/
│   ├── base.py
│   ├── hybrid.py           # ✅ 新增
│   ├── ensemble.py         # ✅ 新增
│   ├── registry.py
│   └── auto/
│       └── auto_grading_runner.py  # ✅ 重命名
├── model/
│   ├── base.py
│   ├── formatter/
│   └── executor/           # ✅ 新增
│       └── prompt_executor.py  # Chat重构
└── runner/
    └── grading/
        └── strategy/
            ├── base.py     # ✅ GradingStrategy
            ├── voting.py
            ├── calibration.py  # ✅ 新增
            └── ensemble.py     # ✅ 新增

gallery/
├── grader/                 # ✅ 保持
│   ├── general/
│   ├── math/
│   ├── code/
│   ├── agent/
│   ├── multimodal/
│   └── alignment/
└── data/                   # ✅ 新增
    ├── math.py
    ├── code.py
    ├── agent.py
    └── multimodal.py
```

---

## 🆕 新增功能

### 1. HybridGrader（规则+LLM混合）
```python
grader = HybridGrader(
    rule_grader=AccuracyGrader(),
    llm_grader=LLMGrader(...),
    strategy="fallback"  # fallback | weighted | cascade
)
```

**使用场景**：
- 先用规则快速筛选，再用LLM精细评估
- 降低LLM调用成本
- 提高评估可靠性

### 2. EnsembleGrader（多Grader集成）
```python
grader = EnsembleGrader(
    graders=[grader1, grader2, grader3],
    strategy="weighted",  # mean | median | max | weighted | voting
    weights=[0.5, 0.3, 0.2]
)
```

**使用场景**：
- 结合多个Grader的优势
- 提高评分稳定性
- 减少单个Grader的偏差

### 3. CalibrationStrategy（分数校准）
```python
strategy = CalibrationStrategy(
    calibration_data=historical_data,
    method="isotonic"  # linear | isotonic | platt
)
```

**使用场景**：
- 修正Grader的系统性偏差
- 基于历史数据调整分数
- 提高评分准确性

### 4. GradingReport（统一报告）
```python
report = grading_runner.run(eval_cases)

# 统计分析
print(report.statistics)

# 可视化
report.visualize("report.png")

# 导出
report.to_json("report.json")
report.to_html("report.html")
report.to_dataframe()  # pandas DataFrame
```

**功能**：
- 丰富的统计信息
- 自动可视化
- 多种导出格式
- Top/Bottom分析

### 5. EvalDataset（数据集管理）
```python
dataset = EvalDataset(
    name="math_test",
    cases=[case1, case2, ...]
)

# 过滤
hard_cases = dataset.filter(lambda c: c.metadata["difficulty"] == "hard")

# 分割
train_ds, test_ds = dataset.split(ratio=0.8)

# 序列化
dataset.to_json("dataset.json")
```

---

## 🏗️ Grader体系架构

### 按评分模式分类
```python
GraderMode = Enum:
    POINTWISE   # 逐点评分（单个样本）
    PAIRWISE    # 成对比较（两个样本）# 新增
    LISTWISE    # 列表排序（多个样本）
```

### 按实现方式分类
```python
Grader Types:
    1. LLMGrader          # 基于LLM
    2. FunctionGrader     # 基于函数
    3. HybridGrader       # 混合 (新增)
    4. EnsembleGrader     # 集成 (新增)
```

### 各领域覆盖

#### 通用领域 (General)
- `AccuracyGrader` - 精确匹配
- `SimilarityGrader` - 统一的文本相似度评估器 (支持 F1 Score, ROUGE-L, BLEU, METEOR 等多种算法)
- `StringMatchGrader` - 统一的字符串匹配评估器
- `NumberAccuracyGrader` - 数字准确性检查
- `BERTScoreGrader` - BERTScore (建议新增)

#### 数学领域 (Math)
- `MathVerifyGrader` - 数学表达式验证
- `SymbolicMathGrader` - 符号数学 (建议新增)
- `NumericalMathGrader` - 数值计算 (建议新增)

#### 代码领域 (Code)
- `SyntaxCheckGrader` - 语法检查
- `CodeStyleGrader` - 代码风格
- `CodeExecutionGrader` - 代码执行
- `PatchSimilarityGrader` - 补丁相似度
- `CoverageGrader` - 覆盖率 (建议新增)
- `ComplexityGrader` - 复杂度 (建议新增)

#### Agent领域 (Agent)
- `ToolCallAccuracyGrader` - 工具调用准确性
- `ToolCallSuccessGrader` - 工具调用成功率
- `ToolParameterCheckGrader` - 参数检查
- `ToolSelectionQualityGrader` - 工具选择质量
- Memory/Plan/Reflection相关Graders

#### 多模态领域 (Multimodal)
- `ImageCoherenceGrader` - 图像一致性
- `ImageHelpfulnessGrader` - 图像有用性
- `ImageEditingGrader` - 图像编辑质量
- `TextToImageGrader` - 文生图评估
- Video相关 (建议新增)

---

## 📊 评测数据定义

### 核心数据模型
```python
@dataclass
class EvalCase:
    uid: str                        # 唯一标识
    input: Dict[str, Any]           # 共享输入
    outputs: List[Dict[str, Any]]   # 待评估输出
    metadata: Dict[str, Any]        # 元数据
    ground_truth: Optional[Dict]    # 真实标签 (新增)
```

### 领域特定数据
```python
# 数学领域
@dataclass
class MathEvalCase:
    problem: str
    solution: str
    answer: Union[str, float]
    difficulty: str
    topic: str

# 代码领域
@dataclass
class CodeEvalCase:
    prompt: str
    code: str
    test_cases: List[Dict]
    language: str
    difficulty: str

# Agent领域
@dataclass
class AgentEvalCase:
    conversation: List[Dict]
    tool_definitions: List[Dict]
    tool_calls: List[Dict]
    expected_outcome: str
```

### FieldMapper（字段映射）
```python
# 方式1: 字典映射
mapper = {
    "query": "input.question",
    "answer": "output.response"
}

# 方式2: 构建器模式
mapper = (
    FieldMappingBuilder()
    .map_input("query", "question")
    .map_output("answer", "response")
    .map_metadata("difficulty", "level")
    .build()
)

# 方式3: 自定义函数
def custom_mapper(eval_case: EvalCase) -> EvalCase:
    # 自定义映射逻辑
    return transformed_case
```

---

## 🚀 实施路线图

### 阶段1: 低风险重命名（1-2周）
- [x] `EvalCaseParser` → `FieldMapper`
- [x] `GraderStrategy` → `GradingStrategy`
- [x] 合并 `_GraderScore` 和 `GraderScore`

### 阶段2: Schema层重组（2-3周）
- [ ] 创建新的子目录结构
- [ ] 迁移数据模型
- [ ] 迁移评分模型
- [ ] 迁移评估模型

### 阶段3: Grader增强（2-3周）
- [ ] 增强 Grader 基类
- [ ] 实现 HybridGrader
- [ ] 实现 EnsembleGrader

### 阶段4: Strategy增强（2-3周）
- [ ] 增强 VotingStrategy
- [ ] 实现 CalibrationStrategy
- [ ] 实现 EnsembleStrategy

### 阶段5: 报告体系完善（2周）
- [ ] 实现 GradingReport
- [ ] 更新 GradingRunner
- [ ] 统一报告接口

### 阶段6: Chat类重构（2周）
- [ ] 拆分 Chat 类
- [ ] 创建 PromptExecutor
- [ ] 更新 LLMGrader

### 阶段7: AutoGrader重命名（1周）
- [ ] 重命名为 AutoGradingRunner
- [ ] 更新所有引用

### 阶段8: Gallery扩展（2周）
- [ ] 添加领域特定数据定义
- [ ] 完善各领域 Grader

### 阶段9: 文档和示例（2周）
- [ ] 更新API文档
- [ ] 更新使用示例

### 阶段10: 测试和发布（1周）
- [ ] 完整测试
- [ ] 准备发布

**总计：16-21周**

---

## ⚠️ 风险与缓解

### 1. 向后兼容性风险
**缓解**：
- 保留别名和旧的导入路径
- 添加弃用警告
- 提供迁移指南

### 2. 测试覆盖风险
**缓解**：
- 增加测试覆盖率 > 80%
- 快照测试确保输出一致
- 端到端集成测试

### 3. 性能风险
**缓解**：
- 性能基准测试
- 并发优化
- 缓存机制

### 4. 依赖风险
**缓解**：
- 可选依赖
- 延迟导入
- 最小化依赖

---

## 📈 预期收益

### 对开发者
- ✅ 更容易理解和使用
- ✅ 更容易扩展和定制
- ✅ 更少的bug和维护成本
- ✅ 更好的开发体验

### 对用户
- ✅ 更丰富的评分能力
- ✅ 更准确的评估结果
- ✅ 更直观的报告展示
- ✅ 更灵活的配置选项

### 量化指标
- 代码可维护性指数提升 30%
- 新功能开发时间减少 40%
- Bug数量减少 50%
- 用户满意度提升 60%

---

## 📚 相关文档

- [完整设计方案](./ARCHITECTURE_DESIGN_PROPOSAL.md) - 详细的架构设计文档
- [当前架构分析](./CORE_CLASS_DIAGRAM.md) - 当前类关系图
- [重构建议](./CORE_REFACTORING_SUGGESTIONS.md) - 现有重构建议
- [API文档](./API_DOCUMENTATION_ZH.md) - API使用文档

---

## 🤝 贡献指南

### 如何参与
1. 选择一个阶段的任务
2. 创建feature分支
3. 实现并测试
4. 提交PR
5. Code Review
6. 合并到主分支

### 代码规范
- 遵循PEP 8
- 添加类型提示
- 编写文档字符串
- 添加单元测试
- 更新CHANGELOG

---

## 📞 联系方式

如有任何问题或建议，请：
- 创建GitHub Issue
- 发起Discussion
- 联系维护者

---

**最后更新**: 2025-01-XX
**文档版本**: 1.0.0

