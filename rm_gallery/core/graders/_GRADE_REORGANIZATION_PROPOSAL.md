# Reorganization Proposal

## Current Structure

Based on the actual directory structure in `rm_gallery/core/graders/gallery/`:

```
gallery/
├── README.md
├── agent/                                                     # Agent capability evaluation module: Evaluates agent abilities in action execution, memory management, planning and reflection
│   ├── action/                                                # Action execution evaluation: Evaluates the accuracy and consistency of agent action execution
│   │   ├── action_misalignment.py                              # ActionMisalignmentGrader: Evaluates if agent actions align with the intended goals and tasks
│   ├── memory/                                                # Memory management evaluation: Evaluates agent's memory storage, retrieval and processing capabilities
│   │   ├── memory_hallucination.py                             # MemoryHallucinationGrader: Detects hallucinations in memory-related tasks performed by agents
│   │   ├── memory_over_simplification.py                       # MemoryOverSimplificationGrader: Identifies cases where agent memory processing leads to oversimplification of information
│   │   ├── memory_retrieval_failure.py                         # MemoryRetrievalFailureGrader: Evaluates the effectiveness of agent memory retrieval mechanisms
│   ├── plan/                                                  # Planning capability evaluation: Evaluates agent's ability to develop and execute plans
│   │   ├── plan_impossible_action.py                           # PlanImpossibleActionGrader: Assesses the feasibility of planned actions by agents
│   ├── reflection/                                            # Reflection capability evaluation: Evaluates agent's self-reflection and outcome analysis abilities
│   │   ├── reflection_hallucination.py                         # ReflectionHallucinationGrader: Detects hallucinations that occur during agent reflection processes
│   │   ├── reflection_outcome_misinterpretation.py             # ReflectionOutcomeMisinterpretationGrader: Evaluates misinterpretation of outcomes during agent reflection
│   │   ├── reflection_progress_misjudge.py                     # ReflectionProgressMisjudgeGrader: Assesses cases where agents misjudge their progress during reflection
│   ├── tool_call_accuracy.py                                   # ToolCallAccuracyGrader: Evaluates the accuracy of tool parameter extraction and usage by agents
│   ├── tool_call_success.py                                    # ToolCallSuccessGrader: Measures the success rate of tool executions by agents
│   ├── tool_parameter_check.py                                 # ToolParameterCheckGrader: Validates the correctness of tool parameters used by agents
│   └── tool_selection.py                                       # ToolSelectionGrader: Assesses the appropriateness of tool selection by agents
├── alignment/                                                  # Alignment evaluation module: Evaluates how well model outputs align with human values, safety requirements, and helpfulness standards
│   ├── base_alignment.py                                       # Base class for implementing alignment evaluators
│   ├── harmlessness/                                          # Harmlessness evaluation: Evaluates the model's ability to avoid producing harmful, toxic or dangerous content
│   │   ├── detoxification.py                                   # DetoxificationGrader: Detects offensive or toxic content in model outputs
│   │   ├── honesty.py                                          # HonestyGrader: Evaluates the honesty and truthfulness of model responses
│   │   ├── safety.py                                           # SafetyGrader: Evaluates compliance with safety policies and refusal of harmful requests
│   ├── helpfulness/                                           # Helpfulness evaluation: Evaluates the model's ability to provide useful, relevant and high-quality responses
│   │   ├── brainstorming.py                                    # BrainstormingGrader: Evaluates the quality of creative idea generation
│   │   ├── chat.py                                             # ChatGrader: Assesses the quality of conversational interactions
│   │   ├── classification.py                                   # ClassificationGrader: Evaluates performance on classification tasks
│   │   ├── closed_qa.py                                        # ClosedQAGrader: Evaluates the quality of closed question answering responses
│   │   ├── code.py                                             # CodeGrader: Assesses code problem-solving quality
│   │   ├── focus.py                                            # FocusGrader: Evaluates the maintenance of focus during action execution
│   │   ├── generation.py                                       # GenerationGrader: Evaluates general content generation quality
│   │   ├── math.py                                             # MathGrader: Assesses math problem-solving quality
│   │   ├── open_qa.py                                          # OpenQAGrader: Evaluates the quality of open question answering
│   │   ├── precise_if.py                                       # PreciseIFGrader: Evaluates precise instruction following
│   │   ├── reasoning.py                                        # ReasoningGrader: Assesses logical reasoning quality
│   │   ├── rewrite.py                                          # RewriteGrader: Evaluates the quality of text rewriting tasks
│   │   ├── role_playing.py                                     # RolePlayingGrader: Evaluates performance in role-playing scenarios
│   │   ├── summarization.py                                    # SummarizationGrader: Evaluates text summarization quality
│   │   └── translation.py                                      # TranslationGrader: Assesses translation quality
│   └── honesty/                                               # Honesty evaluation: Evaluates the truthfulness and factual accuracy of model responses
│       ├── factuality.py                                       # FactualityGrader: Assesses truthfulness and factual accuracy of model responses
├── code/                                                      # Code evaluation module: Evaluates the model's ability to generate and execute code
│   ├── code.py                                                 # SyntaxCheckGrader: Check code syntax using Abstract Syntax Tree to validate Python code blocks
│   │                                                           # ExecutionVerificationGrader: Verify code execution and check output correctness
│   └── prime_code/                                            # Code evaluation tools: Auxiliary tools for code testing and evaluation
│       ├── testing_util.py                                     # Utility functions for code testing
│       └── utils.py                                            # Helper functions for code evaluation
├── cramo.py                                                    # CramoGrader: Implements the Combined Reward Modeling Approach (CRAMO) framework
├── format/                                                     # Format evaluation module: Evaluates the format standardization and compliance of model outputs
│   ├── format.py                                               # ReasoningFormatGrader: Checks for proper thinking and answer tags
│   │                                                           # ReasoningToolCallFormatGrader: Validates tool call format with JSON validation
│   │                                                           # LengthPenaltyGrader: Applies penalties for text that is too short or too long
│   │                                                           # NgramRepetitionPenaltyGrader: Calculates N-gram repetition penalty with Chinese support
│   │                                                           # PrivacyLeakageGrader: Detects privacy information leakage in generated content
│   ├── json_match.py                                           # JsonMatchGrader: Recursively compares JSON structures element by element
│   │                                                           # JsonValidatorGrader: Validates if the candidate text is valid JSON
├── general.py                                                  # AccuracyGrader: Calculates exact match accuracy between generated and reference text
├── llm_judge/                                                 # LLM judge evaluation module: Uses large language models as judges to evaluate model output quality
│   ├── hallucination.py                                        # HallucinationGrader: Detects hallucinations or fabricated information not supported by context
│   ├── harmfulness.py                                          # HarmfulnessGrader: Evaluates harmfulness in model responses
│   ├── helpfulness.py                                          # HelpfulnessGrader: Provides overall helpfulness evaluation of model responses
│   ├── instruction_adherence.py                                # InstructionAdherenceGrader: Assesses adherence to specified constraints and instructions
│   └── reference_adherence.py                                  # ReferenceAdherenceGrader: Checks alignment between model outputs and reference materials
├── math/                                                       # Mathematics evaluation module: Evaluates the model's ability to solve mathematical problems
│   └── math.py                                                 # MathVerifyGrader: Evaluates mathematical problem-solving capabilities using math verification libraries
├── multimodal/                                                # Multimodal evaluation module: Evaluates the model's ability to process and generate multiple types of content (such as text and images)
│   ├── _internal/                                             # Internal tools: Internal support tools and implementations for multimodal evaluation
│   │   ├── image_utils.py
│   │   ├── metrics.py
│   │   ├── model_utils.py
│   │   ├── registry.py
│   │   └── types.py
│   ├── custom_criteria.py                                      # CustomCriteriaGrader: Flexible framework for custom multimodal evaluation
│   ├── image_coherence.py                                      # ImageCoherenceGrader: Evaluates coherence between images and text context
│   ├── image_editing.py                                        # ImageEditingGrader: Evaluates quality of image editing tasks
│   ├── image_helpfulness.py                                    # ImageHelpfulnessGrader: Assesses helpfulness of images in understanding text
│   ├── image_reference.py                                      # ImageReferenceGrader: Checks if images are properly referenced in text
│   ├── text_to_image.py                                        # TextToImageGrader: Evaluates quality of text-to-image generation
├── text/                                                       # Text evaluation module: Evaluates various capabilities of the model in text understanding and generation
│   ├── similarity.py                                           # SimilarityGrader: Unified text similarity grader supporting multiple algorithms
│   │                                                           # Algorithms: bleu, sentence_bleu, gleu, chrf, meteor, rouge, rouge1, rouge2, rougeL, rouge_ngram, rouge3, rouge4, rouge5, f1_score, token_f1, fuzzy_match, edit_distance, cosine, jaccard
│   ├── string_match.py                                         # StringMatchGrader: Unified string matching grader supporting multiple algorithms
│   │                                                           # Algorithms: exact_match, prefix_match, suffix_match, regex_match, substring_match, contains_all, contains_any, word_overlap, char_overlap
│   └── utils/                                                 # Text evaluation tools: Supporting tools for text similarity calculations
│       ├── bleu.py
│       ├── chrf.py
│       ├── cosine.py
│       ├── edit_distance.py
│       ├── fuzzy_match.py
│       ├── jaccard.py
│       └── meteor.py
```

## Current Issues

在分析当前结构后，我们可以将问题归纳为四个主要类别：

### 1. 结构不一致
项目存在不一致的组织模式：
- 混合的目录命名约定（单数与复数）
- 模块间嵌套级别不一致（有些像[agent/action/](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/gallery/agent/action/)一样深度嵌套，有些像[Math/](file:///mnt3/huangsen.huang/codes/RM-Gallery/examples/train/bradley-terry/Math/)一样扁平）
- 文件命名不一致，单个文件包含多个不相关的类（例如，[general.py](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/gallery/general.py)包含多个不同的评估器）
- 空的或不完整的[__init__.py](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/__init__.py)文件，未正确暴露模块API

### 2. 功能重叠和错位
几个模块存在重叠或错位的功能：
- [llm_judge/](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/gallery/llm_judge/)和[alignment/](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/gallery/alignment/)模块之间存在重复，都包含类似的基于LLM的评估方法
- [general.py](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/gallery/general.py)中误放了文本评估指标，这些应该属于[text/](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/gallery/text/)模块
- 模块名称与其内容混淆（例如，[helpfulness/](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/gallery/alignment/helpfulness/)包含许多与帮助性无关的评估器）
- [cramo.py](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/gallery/cramo.py)作为一个独立的方法，难以自然融入现有的分类

### 3. 架构问题
存在影响可维护性和可用性的基本架构问题：
- 缺乏清晰的继承层次结构（例如，[FactualityGrader](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/gallery/alignment/honesty/factuality.py#L79-L81)继承自[BaseHelpfulnessGrader](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/gallery/alignment/helpfulness/__init__.py#L215-L216)）
- 缺少或不完整的模块导出，导致评估器难以使用
- 不同评估器之间实现模式不一致
- pointwise和listwise评估模式之间缺乏良好的分离

### 4. 命名和文档问题
一些问题影响了可发现性和理解性：
- 文件名不能清楚地表明其中包含多个类
- 模块名称并不总是反映其实际内容（例如，[helpfulness/](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/gallery/alignment/helpfulness/)包含许多与帮助性无关的评估器）
- 一些模块中缺少完整或缺失的文档
- 结构文档中引用了不存在的文件

## Functional Overview and Use Cases

为了更好地理解当前Grader的功能范围和应用场景，我们需要从多个维度对其进行分析，这将直接影响未来的重组策略。

### 主要功能维度

通过对现有Grader的分析，可以识别出以下几个主要功能维度：

1. **能力领域 (Capability Domains)**
   - AI对齐 (Alignment)：评估模型输出与人类价值观、安全要求和实用性的一致性
   - 代理能力 (Agent)：评估代理在行动执行、记忆管理、规划和反思方面的能力
   - 文本处理 (Text Processing)：评估模型在文本理解和生成方面的能力
   - 代码能力 (Code)：评估模型生成和执行代码的能力
   - 数学推理 (Mathematical Reasoning)：评估模型解决数学问题的能力
   - 格式规范 (Format Compliance)：评估模型输出的格式标准化和合规性
   - 多模态处理 (Multimodal)：评估模型处理和生成多种类型内容的能力
   - 复合评估 (Composite)：结合多种评估方法的框架

2. **评估方法 (Evaluation Methods)**
   - 基于规则 (Rule-based)：使用确定性算法进行评估
   - 基于LLM (LLM-based)：使用大型语言模型作为评委进行评估
   - 混合方法 (Hybrid)：结合多种评估方法以获得更全面的结果

3. **评估粒度 (Evaluation Granularity)**
   - 点评 (Pointwise)：对单个样本进行评分
   - 排序 (Listwise)：对多个样本进行排序

### 主要应用场景

根据功能分析，这些Grader主要应用于以下场景：

1. **模型开发和优化**
   - 在训练过程中评估模型性能
   - 使用奖励建模进行RLHF（人类反馈强化学习）
   - 比较不同模型版本的性能

2. **质量保证和测试**
   - 对生产环境中的模型输出进行自动质量检查
   - 确保模型响应符合特定标准和要求
   - 检测模型退化或异常行为

3. **研究和基准测试**
   - 在标准数据集上评估模型能力
   - 进行模型间比较研究
   - 支持学术研究和论文发表

4. **特定任务评估**
   - 代码生成和评审
   - 数学问题求解
   - 多语言翻译质量
   - 创意内容生成

### 功能与场景的映射关系

| 能力领域 | 评估方法 | 评估粒度 | 主要应用场景 |
|---------|---------|---------|-------------|
| AI对齐 | 基于规则/基于LLM | 点评/排序 | 模型开发、质量保证 |
| 代理能力 | 基于规则 | 点评 | 模型开发、研究基准 |
| 文本处理 | 基于规则 | 点评 | 质量保证、特定任务 |
| 代码能力 | 基于规则 | 点评 | 特定任务、质量保证 |
| 数学推理 | 基于规则 | 点评 | 特定任务、研究基准 |
| 格式规范 | 基于规则 | 点评 | 质量保证 |
| 多模态处理 | 基于规则/基于LLM | 点评 | 特定任务、研究基准 |
| 复合评估 | 混合方法 | 点评/排序 | 模型开发、研究基准 |

### 影响重组策略的关键因素

基于以上分析，未来的重组策略应考虑以下因素：

1. **按能力领域组织**：将相关功能的Grader组织在一起，便于用户查找和使用
2. **按评估方法区分**：区分基于规则和基于LLM的评估方法，让用户更容易选择合适的方法
3. **保持语义一致性**：确保模块名称准确反映其内容和功能
4. **简化嵌套结构**：避免过深的嵌套层级，提高可访问性
5. **明确定义边界**：减少模块间功能重叠，确保每个模块有明确的职责

## Proposed Structure

基于上述分析，我们提出以下几种重构方案备选：

### 方案一：按功能领域分类（推荐）

```
gallery/
├── README.md
├── __init__.py
├── alignment/              # AI对齐评估模块
├── agent/                  # 代理能力评估模块
├── text/                   # 文本处理能力评估模块
├── code/                   # 代码能力评估模块
├── math/                   # 数学能力评估模块
├── format/                 # 格式和结构评估模块
├── multimodal/             # 多模态评估模块
└── composite/              # 复合评估模块
```

### 方案二：按评估方法分类

```
gallery/
├── README.md
├── __init__.py
├── rule_based/             # 基于规则的评估器
├── llm_judge/              # 基于LLM的评估器
└── composite/              # 复合评估器
```

### 方案三：混合分类法

```
gallery/
├── README.md
├── __init__.py
├── functional/             # 按功能领域分类
│   ├── alignment/
│   ├── agent/
│   ├── text/
│   ├── code/
│   ├── math/
│   ├── format/
│   ├── multimodal/
│   └── composite/
└── methodological/         # 按评估方法分类
    ├── rule_based/
    ├── llm_judge/
    └── composite/
```

### 方案四：按应用场景分类

```
gallery/
├── README.md
├── __init__.py
├── model_development/      # 模型开发和优化场景
├── quality_assurance/      # 质量保证和测试场景
├── research_benchmark/     # 研究和基准测试场景
└── task_specific/          # 特定任务评估场景
```

### 推荐方案说明

我们推荐使用**方案一：按功能领域分类**，理由如下：

1. **直观性**：用户可以根据评估目标快速定位到相关模块
2. **扩展性**：新增评估器可以很容易地归入合适的功能模块
3. **一致性**：与当前大多数机器学习库的组织方式一致
4. **可维护性**：每个模块职责明确，便于维护和管理
5. **文档友好**：便于编写和维护相关文档

在该方案下，每个功能模块内部可以进一步按子功能或评估方法组织，以兼顾功能领域和实现方式的分类需求。

## Detailed Target Directory Structure

Based on the selected Option 1 (Functional Domain Classification), here is the detailed target directory structure with specific file mappings, ensuring that each grader is placed in a directory that matches its functionality:

```
gallery/
├── README.md
├── __init__.py
├── agent/                                                     # Agent capability evaluation module
│   ├── __init__.py
│   ├── action/                                                # Action execution evaluation
│   │   ├── __init__.py
│   │   └── action_misalignment.py                              # ActionMisalignmentGrader
│   ├── memory/                                                # Memory management evaluation
│   │   ├── __init__.py
│   │   ├── memory_hallucination.py                             # MemoryHallucinationGrader
│   │   ├── memory_over_simplification.py                       # MemoryOverSimplificationGrader
│   │   └── memory_retrieval_failure.py                         # MemoryRetrievalFailureGrader
│   ├── plan/                                                  # Planning capability evaluation
│   │   ├── __init__.py
│   │   └── plan_impossible_action.py                           # PlanImpossibleActionGrader
│   ├── reflection/                                            # Reflection capability evaluation
│   │   ├── __init__.py
│   │   ├── reflection_hallucination.py                         # ReflectionHallucinationGrader
│   │   ├── reflection_outcome_misinterpretation.py             # ReflectionOutcomeMisinterpretationGrader
│   │   └── reflection_progress_misjudge.py                     # ReflectionProgressMisjudgeGrader
│   └── tool/                                                  # Tool usage evaluation
│       ├── __init__.py
│       ├── tool_call_accuracy.py                               # ToolCallAccuracyGrader
│       ├── tool_call_success.py                                # ToolCallSuccessGrader
│       ├── tool_parameter_check.py                             # ToolParameterCheckGrader
│       └── tool_selection.py                                  # ToolSelectionGrader
├── alignment/                                                 # Alignment evaluation module (Safety & Ethical Alignment)
│   ├── __init__.py
│   ├── harmlessness/                                          # Harmlessness evaluation
│   │   ├── __init__.py
│   │   ├── detoxification.py                                   # DetoxificationGrader
│   │   ├── hallucination.py                                    # HallucinationGrader (moved from llm_judge/)
│   │   ├── harmfulness.py                                      # HarmfulnessGrader (moved from llm_judge/)
│   │   ├── honesty.py                                          # HonestyGrader
│   │   ├── privacy.py                                          # Privacy protection evaluation
│   │   │   ├── __init__.py
│   │   │   └── privacy_leakage.py                              # PrivacyLeakageGrader (moved from format/format.py)
│   │   ├── reference_adherence.py                              # ReferenceAdherenceGrader (moved from llm_judge/)
│   │   └── safety.py                                           # SafetyGrader
│   ├── helpfulness/                                           # Helpfulness evaluation
│   │   ├── __init__.py
│   │   ├── brainstorming.py                                    # BrainstormingGrader
│   │   ├── chat.py                                             # ChatGrader
│   │   ├── classification.py                                   # ClassificationGrader
│   │   ├── closed_qa.py                                        # ClosedQAGrader
│   │   ├── code.py                                             # CodeGrader
│   │   ├── focus.py                                            # FocusGrader
│   │   ├── generation.py                                       # GenerationGrader
│   │   ├── helpfulness.py                                      # HelpfulnessGrader (moved from llm_judge/)
│   │   ├── instruction_adherence.py                            # InstructionAdherenceGrader (moved from llm_judge/)
│   │   ├── math.py                                             # MathGrader
│   │   ├── open_qa.py                                          # OpenQAGrader
│   │   ├── precise_if.py                                       # PreciseIFGrader
│   │   ├── reasoning.py                                        # ReasoningGrader
│   │   ├── rewrite.py                                          # RewriteGrader
│   │   ├── role_playing.py                                     # RolePlayingGrader
│   │   ├── summarization.py                                    # SummarizationGrader
│   │   └── translation.py                                      # TranslationGrader
│   └── honesty/                                               # Honesty evaluation
│       ├── __init__.py
│       └── factuality.py                                       # FactualityGrader
├── text/                                                      # Text processing evaluation module (Text Similarity & Matching)
│   ├── __init__.py
│   ├── matching/                                              # String matching algorithms
│   │   ├── __init__.py
│   │   └── string_match.py                                     # StringMatchGrader (all algorithms)
│   ├── similarity/                                            # Text similarity metrics
│   │   ├── __init__.py
│   │   ├── similarity.py                                       # SimilarityGrader (all algorithms)
│   │   └── utils/                                             # Text evaluation utilities
│   │       ├── __init__.py
│   │       ├── bleu.py
│   │       ├── chrf.py
│   │       ├── cosine.py
│   │       ├── edit_distance.py
│   │       ├── fuzzy_match.py
│   │       ├── jaccard.py
│   │       └── meteor.py
│   ├── matching/                                              # String matching and number accuracy
│   │   ├── __init__.py
│   │   ├── string_match.py                                     # StringMatchGrader
│   │   └── number_accuracy.py                                  # NumberAccuracyGrader (moved from general/)
│   ├── similarity/                                            # Text similarity metrics
│   │   ├── __init__.py
│   │   └── similarity.py                                       # SimilarityGrader (unified similarity grader)
│   │                                                           # Note: ROUGE-L is available via SimilarityGrader with algorithm="rougeL"
│   │                                                           # Note: F1 Score is available via SimilarityGrader with algorithm="f1_score"
│   └── utils/                                                 # Shared utilities for text graders
│       ├── __init__.py
│       ├── compute.py                                          # Core computation functions
│       ├── string_match_compute.py                             # String matching computations
│       ├── normalization.py                                    # Text normalization utilities
│       ├── tokenization.py                                     # Tokenization utilities
│       └── setup_nltk_data.py                                  # NLTK data setup
├── code/                                                      # Code evaluation module (Code capability)
│   ├── __init__.py
│   ├── code.py                                                 # SyntaxCheckGrader, ExecutionVerificationGrader
│   └── prime_code/                                            # Code evaluation tools
│       ├── __init__.py
│       ├── testing_util.py                                     # Utility functions for code testing
│       └── utils.py                                            # Helper functions for code evaluation
├── math/                                                      # Mathematics evaluation module (Mathematical reasoning)
│   ├── __init__.py
│   └── math.py                                                 # MathVerifyGrader
├── format/                                                    # Format evaluation module (Format compliance)
│   ├── __init__.py
│   ├── format.py                                               # ReasoningFormatGrader, ReasoningToolCallFormatGrader
│   │                                                           # LengthPenaltyGrader, NgramRepetitionPenaltyGrader
│   └── json/                                                  # JSON evaluation
│       ├── __init__.py
│       ├── json_match.py                                       # JsonMatchGrader
│       └── json_validator.py                                   # JsonValidatorGrader
├── multimodal/                                                # Multimodal evaluation module (Multimodal evaluation)
│   ├── __init__.py
│   ├── _internal/                                             # Internal tools
│   │   ├── __init__.py
│   │   ├── image_utils.py
│   │   ├── metrics.py
│   │   ├── model_utils.py
│   │   ├── registry.py
│   │   └── types.py
│   ├── custom_criteria.py                                      # CustomCriteriaGrader
│   ├── image_coherence.py                                      # ImageCoherenceGrader
│   ├── image_editing.py                                        # ImageEditingGrader
│   ├── image_helpfulness.py                                    # ImageHelpfulnessGrader
│   ├── image_reference.py                                      # ImageReferenceGrader
│   └── text_to_image.py                                        # TextToImageGrader
└── composite/                                                 # Composite evaluation module (Custom & Composite Evaluation)
    ├── __init__.py
    └── cramo.py                                                # CramoGrader
```

## Benefits of the New Structure

1. **Improved Discoverability**: Users can easily find relevant graders by functionality
2. **Better Organization**: Related graders are grouped together logically with clear functional boundaries
3. **Enhanced Maintainability**: Each module has a clear, focused responsibility aligned with its functional purpose
4. **Reduced Redundancy**: Eliminates overlap between [llm_judge/](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/gallery/llm_judge/) and [alignment/](file:///mnt3/huangsen.huang/codes/RM-Gallery/rm_gallery/core/graders/gallery/alignment/) modules
5. **Consistent Naming**: All directories and files follow consistent naming conventions
6. **Proper Module Exports**: All `__init__.py` files properly export contained classes for easy importing
7. **Aligned with Approved Approach**: Structure follows the approved Option 1 classification which ensures stakeholder alignment
8. **Functional Consistency**: Each grader is placed in a directory that directly matches its core functionality
9. **Correct Categorization**: PrivacyLeakageGrader is now correctly placed in alignment/harmlessness/privacy as it's a safety and ethical alignment evaluator
10. **Eliminates Method-Based Organization**: Removes the separate llm_judge directory in favor of functional domain organization
