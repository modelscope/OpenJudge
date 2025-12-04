# Graders Survey

## Openai Graders

以下是对OpenAI平台上Graders功能的整理：

| Grader类型 | Grader能力 | 所需数据字段 |
|------------|------------|--------------|
| string_check | 字符串检查Grader，用于处理答案为确定字符串的场景，例如城市名称、是/否判断等 | input, reference |
| text_similarity | 文本相似度Grader，计算两个文本之间的相似度 | input, reference |
| score_model | 模型评分Grader，让大模型作为裁判，评估输出在各项维度上的表现 | input, reference |
| label_model | 模型标签Grader，使用模型为评估项分配标签 | input_data, labels, model |
| python | Python代码执行Grader，执行Python代码并评估结果 | source, item |
| multi | 多Grader组合，将多个评分标准组合以实现更复杂的评分逻辑 | 多个子Grader的输入参数 |

[Openai Grader](https://platform.openai.com/docs/api-reference/fine-tuning/alpha/graders)


## Azure AI Evaluation

以下是Azure AI Evaluation SDK提供的评估器整理：

| Evaluator名称 | Evaluator能力 | 所需数据字段 |
|---------------|---------------|--------------|
| GroundednessEvaluator | 评估模型输出是否基于给定的上下文，检查输出是否包含未经证实的信息 | context, answer |
| RelevanceEvaluator | 评估模型输出与输入查询的相关性 | query, answer |
| CoherenceEvaluator | 评估模型输出的逻辑连贯性和结构性 | query, answer |
| FluencyEvaluator | 评估模型输出的语言流畅性和自然度 | query, answer |
| ResponseCompletenessEvaluator | 评估模型输出是否完整回答了问题 | query, answer |
| SimilarityEvaluator | 计算两个文本之间的相似度 | ground_truth, answer |
| F1ScoreEvaluator | 计算基于token的F1分数 | ground_truth, answer |
| RougeScoreEvaluator | 计算ROUGE分数，评估文本摘要质量 | ground_truth, answer |
| GleuScoreEvaluator | 计算GLEU分数，适用于语法纠错等任务 | ground_truth, answer |
| BleuScoreEvaluator | 计算BLEU分数，评估机器翻译质量 | ground_truth, answer |
| MeteorScoreEvaluator | 计算METEOR分数，评估机器翻译质量 | ground_truth, answer |
| ViolenceEvaluator | 评估模型输出是否包含暴力相关内容 | query, answer |
| SexualEvaluator | 评估模型输出是否包含性相关内容 | query, answer |
| SelfHarmEvaluator | 评估模型输出是否包含自残相关内容 | query, answer |
| HateUnfairnessEvaluator | 评估模型输出是否包含仇恨或不公平相关内容 | query, answer |
| ProtectedMaterialEvaluator | 评估模型输出是否包含受版权保护的材料 | query, answer |
| ContentSafetyEvaluator | 综合评估模型输出的内容安全性 | query, answer |
| UngroundedAttributesEvaluator | 评估模型输出是否包含未经证实的属性信息 | context, answer |
| CodeVulnerabilityEvaluator | 评估代码输出是否存在安全漏洞 | code |
| IndirectAttackEvaluator | 检测间接攻击或越狱尝试 | query, answer |
| QAEvaluator | 综合评估问答质量，包括准确性、相关性等方面 | query, answer, ground_truth |
| IntentResolutionEvaluator | 评估代理意图识别的准确性 | intent, resolved_intent |
| ToolCallAccuracyEvaluator | 评估工具调用的准确性 | expected_tool_calls, actual_tool_calls |
| TaskAdherenceEvaluator | 评估代理是否遵循了给定任务 | task_description, execution_trace |
| RetrievalEvaluator | 评估检索组件的效果 | query, retrieved_documents, relevant_documents |
| DocumentRetrievalEvaluator | 评估文档检索的准确性 | query, retrieved_documents |
| AzureOpenAILabelGrader | 使用自定义提示指示模型根据定义的标签对输出进行分类 | input_data, labels |
| AzureOpenAIStringCheckGrader | 检查输出是否包含特定字符串 | output_text, check_strings |
| AzureOpenAITextSimilarityGrader | 计算文本相似度 | reference_text, candidate_text |
| AzureOpenAIPythonGrader | 执行自定义Python代码进行评估 | code_snippet, input_data |

[Azure AI Evaluation](https://learn.microsoft.com/zh-cn/azure/ai-foundry/how-to/develop/evaluate-sdk)

## LangSmith

以下是LangSmith提供的具体评估器整理：

| Evaluator名称 | Evaluator能力 | 所需数据字段 |
|---------------|---------------|--------------|
| Exact Match Evaluator | 评估预测输出与参考答案之间的字符级完全匹配程度，适用于有唯一正确答案的场景 | prediction, reference |
| Factuality Evaluator | 使用LLM作为评判者来评估输出的事实准确性，确保内容与参考信息一致 | input, prediction, reference_context |
| Relevance Evaluator | 评估输出内容与输入查询的相关程度，确保回答直接解决问题 | input, prediction |
| Coherence Evaluator | 评估输出内容的逻辑连贯性和结构性 | input, prediction |
| Completeness Evaluator | 评估输出是否完整覆盖了问题的所有要点 | input, prediction |
| Harmfulness Evaluator | 检测输出中是否存在有害、不当或违反规定的内容 | input, prediction |
| Toxicity Evaluator | 评估输出内容的毒性水平，检测攻击性或不当语言 | input, prediction |
| Fluency Evaluator | 评估输出语言的流畅程度和自然度 | input, prediction |
| Sentiment Evaluator | 分析输出内容的情感倾向，评估正面或负面情绪 | input, prediction |
| Context Recall Evaluator | 评估模型在输出中使用了多少相关上下文信息 | input, prediction, reference_context |
| Context Precision Evaluator | 评估模型使用的上下文信息的准确性 | input, prediction, reference_context |
| Faithfulness Evaluator | 评估输出相对于源材料或上下文的忠实程度 | input, prediction, reference_context |
| Embedding Distance Evaluator | 使用嵌入向量计算预测和参考之间的语义距离 | prediction, reference |
| Custom Evaluator | 允许开发者根据特定需求创建自定义评估逻辑 | 根据具体实现而定 |
| QA Evaluator | 综合问答质量评估器，评估答案的正确性 | input, prediction, reference |
| Labeled Criteria Evaluator | 根据指定标准评估输出，如帮助性、简洁性等 | input, prediction, criteria |
| Pairwise Comparison Evaluator | 比较两个输出并确定哪个更好 | input, prediction_list |
| Trajectory Evaluator | 评估代理执行轨迹的质量 | input, trajectory |
| Json Equality Evaluator | 比较两个JSON对象是否相等 | prediction, reference |
| Json Edit Distance Evaluator | 计算两个JSON对象之间的编辑距离 | prediction, reference |
| Json Schema Evaluator | 验证JSON输出是否符合指定的Schema | prediction, schema |
| Regex Evaluator | 使用正则表达式验证输出格式 | prediction, regex_pattern |
| SQL Evaluator | 评估SQL查询的正确性 | input, prediction, reference_sql |
| Python Function Evaluator | 通过执行Python函数验证输出 | prediction, test_function |

## COZE

以下是COZE平台提供的评估能力整理：

| Evaluator名称 | Evaluator能力 | 所需数据字段 |
|---------------|---------------|--------------|
| Relevance Evaluator | 评估回答与问题的重叠度和相关性 | question, answer |
| Factual Accuracy Evaluator | 检查关键信息的正确性，避免编造内容 | question, answer, reference_material |
| Context Consistency Evaluator | 检查多轮对话中是否遗忘上文信息 | conversation_history, answer |
| Format Compliance Evaluator | 验证生成内容是否符合指定格式要求 | answer, expected_format |
| Hallucination Evaluator | 检测AI是否编造知识库外的信息 | question, answer |
| Bias Evaluator | 检查AI回答是否保持中立，避免偏见 | question, answer |
| Multimodal Evaluator | 评估图文输入场景下的理解准确性 | image_input, question, answer |
| Response Latency Evaluator | 测量响应延迟时间 | query_timestamp, response_timestamp |
| Task Completion Evaluator | 评估任务完成率和成功率 | task_description, execution_result |
| Intent Recognition Evaluator | 分析高频问题的意图识别准确率 | user_input, recognized_intent, correct_intent |
| User Satisfaction Evaluator | 基于用户反馈评估满意度 | user_feedback, interaction_log |
| Quality Scoring Evaluator | 通过1-5分制对输出质量打分 | question, answer |

## Google ADK

以下是Google Agent Development Kit (ADK) 提供的评估能力整理：

| Evaluator名称 | Evaluator能力 | 所需数据字段 |
|---------------|---------------|--------------|
| Trajectory Evaluator | 评估智能体的推理路径，使用precision、recall和match-based指标 | expected_trajectory, actual_trajectory |
| Final Response Evaluator | 评估最终用户响应的语义正确性、事实准确性和整体质量 | input, expected_response, actual_response |
| Tool Call Accuracy Evaluator | 评估工具调用的准确性 | expected_tool_calls, actual_tool_calls |
| Action Sequence Evaluator | 评估智能体的动作序列与预期行为的匹配度 | expected_actions, actual_actions |
| Instruction Following Evaluator | 评估智能体遵循指令的能力 | instruction, execution_result |
| Factuality Evaluator | 检查答案的正确性及是否可由观察步骤证实 | input, response, observation_data |
| Helpfulness Evaluator | 评估响应对用户需求的满足程度和适当性 | input, response, user_context |
| Completeness Evaluator | 评估响应是否包含所有必要信息 | input, response, required_elements |
| LLM-as-Judge Evaluator | 使用LLM作为评判者进行评分 | input, response, evaluation_criteria |
| Human-in-the-loop Evaluator | 结合人工反馈进行高质量评估 | input, response, human_rating |
| Tool Failure Rate Evaluator | 监控工具调用失败率 | tool_calls, failure_events |
| User Feedback Evaluator | 分析用户反馈分数 | user_interactions, feedback_scores |
| Execution Latency Evaluator | 测量端到端延迟 | start_time, end_time |
| ReAct Cycle Evaluator | 评估每个任务的ReAct循环次数 | task_description, react_cycles |

## Arize AI Phoenix

以下是Arize AI Phoenix平台提供的评估能力整理：

| Evaluator名称 | Evaluator能力 | 所需数据字段 |
|---------------|---------------|--------------|
| Accuracy Evaluator | 评估模型输出的准确性，包括事实正确性和逻辑一致性 | input, response, ground_truth |
| Relevance Evaluator | 评估输出内容与输入查询的相关程度 | input, response |
| Consistency Evaluator | 评估输出的一致性，特别是在多轮对话中 | conversation_history, response |
| Drift Detection Evaluator | 检测模型输出随时间的漂移情况 | historical_data, current_data |
| Bias Detection Evaluator | 检测模型输出中的偏见问题 | input, response, protected_attributes |
| Safety Metrics Evaluator | 评估模型输出的安全性指标 | input, response |
| Hallucination Evaluator | 检测模型是否产生虚假或编造信息 | input, response, reference_context |
| Toxicity Evaluator | 检测模型输出中的有害或毒性内容 | response |
| Faithfulness Evaluator | 评估输出相对于源材料的忠实程度 | response, reference_context |
| Context Recall Evaluator | 评估模型在输出中使用相关上下文的程度 | input, response, context |
| Context Precision Evaluator | 评估模型使用上下文信息的准确性 | input, response, context |
| Retrieval Evaluator | 评估检索组件的效果和准确性 | query, retrieved_documents, relevant_documents |
| RAG E2E Evaluator | 端到端评估RAG系统的性能 | input, context, response, ground_truth |
| Custom Evaluator | 允许开发者根据特定需求创建自定义评估逻辑 | 根据具体实现而定 |
| LLM-as-Judge Evaluator | 使用大模型作为裁判进行自动化评估 | input, response, evaluation_criteria |
| Human Feedback Evaluator | 集成人工反馈进行评估 | input, response, human_ratings |
| Test Variant Generator | 生成测试变体用于评估 | base_test_cases |
| Trace-based Evaluator | 基于执行轨迹的评估 | execution_traces |

## DeepEval

以下是DeepEval框架提供的评估能力整理：

| Evaluator名称 | Evaluator能力 | 所需数据字段 |
|---------------|---------------|--------------|
| AnswerRelevancyMetric | 评估模型输出与输入问题的相关性 | input, actual_output |
| FaithfulnessMetric | 评估模型输出相对于检索上下文的忠实度，检测幻觉 | input, actual_output, retrieval_context |
| ContextualRelevancyMetric | 评估检索到的上下文与输入问题的相关性 | input, retrieval_context |
| ContextualPrecisionMetric | 评估检索到的上下文的精确度 | input, expected_output, retrieval_context |
| ContextualRecallMetric | 评估检索到的上下文的召回率 | input, expected_output, retrieval_context |
| HallucinationMetric | 检测模型输出中的虚假或编造信息 | input, actual_output, context |
| BiasMetric | 检测模型输出中的偏见内容 | actual_output |
| ToxicityMetric | 检测模型输出中的有害或毒性内容 | actual_output |
| GEval | 通用评估器，可根据自定义标准评估输出 | 根据自定义标准而定 |
| SummarizationMetric | 评估文本摘要的质量 | input, actual_output, expected_output |
| ConcisenessMetric | 评估输出的简洁性 | input, actual_output |
| CorrectnessMetric | 评估输出的正确性 | input, actual_output, expected_output |
| CoherenceMetric | 评估输出的连贯性 | actual_output |
| CompetencyMetric | 评估模型在特定领域的专业能力 | input, actual_output |
| RAGAS Metrics | 一套专门针对RAG系统的评估指标 | input, actual_output, retrieval_context, expected_output |
| TaskCompletionMetric | 评估AI代理完成指定任务的能力 | input, actual_output, expected_output |
| ToolCorrectnessMetric | 评估AI代理使用工具的正确性 | tools_called |
| KnowledgeRetentionMetric | 评估多轮对话中知识保持的能力 | turns |
| ConversationRelevancyMetric | 评估对话中回复的相关性 | conversation_history, input, actual_output |
| RoleAdherenceMetric | 评估AI代理在多轮对话中遵守角色设定的情况 | conversation_history, persona |
| KnowledgeRetrievalMetric | 评估知识检索的有效性 | input, retrieval_context |
| Custom Metrics | 支持创建自定义评估指标以满足特定需求 | 根据具体实现而定 |

# RM-Gallery 评估能力概览

RM-Gallery 是一个全面的评估框架，提供了丰富的评估器（Evaluator）用于评估各种AI模型和智能体的性能。这些评估器按照功能领域进行组织，覆盖了从基础文本处理到复杂智能体能力的全方位评估需求。

## RM-Gallery 原始评估能力详解

以下是对RM-Gallery原始评估器能力的详细整理，与上述分类对齐：

| 能力领域 | 评估器名称 | 评估能力 | 所属模块 |
|---------|-----------|----------|---------|
| Agent Capabilities (智能体能力评估) | ActionMisalignmentGrader | 评估智能体行动是否与预期目标和任务对齐 | agent/action |
| | MemoryHallucinationGrader | 检测智能体在内存相关任务中产生的幻觉 | agent/memory |
| | MemoryOverSimplificationGrader | 识别智能体内存处理导致的信息过度简化情况 | agent/memory |
| | MemoryRetrievalFailureGrader | 评估智能体内存检索机制的有效性 | agent/memory |
| | PlanImpossibleActionGrader | 评估智能体制定的行动计划的可行性 | agent/plan |
| | ReflectionHallucinationGrader | 检测智能体在反思过程中出现的幻觉 | agent/reflection |
| | ReflectionOutcomeMisinterpretationGrader | 评估智能体在反思过程中对结果的误解 | agent/reflection |
| | ReflectionProgressMisjudgeGrader | 评估智能体在反思过程中对自己进度的误判 | agent/reflection |
| | ToolCallAccuracyGrader | 评估智能体工具参数提取和使用的准确性 | agent/tool |
| | ToolCallSuccessGrader | 衡量智能体工具执行的成功率 | agent/tool |
| | ToolParameterCheckGrader | 验证智能体使用的工具参数的正确性 | agent/tool |
| | ToolSelectionQualityGrader | 评估智能体工具选择的适当性 | agent/tool |
| Safety & Ethical Alignment (安全与伦理对齐) | DetoxificationGrader | 检测模型输出中的冒犯性或有害内容 | alignment/harmlessness |
| | HonestyGrader | 评估模型响应的诚实度和真实性 | alignment/harmlessness |
| | SafetyGrader | 评估模型对安全策略的遵守和对有害请求的拒绝能力 | alignment/harmlessness |
| Quality Assessment (质量评估) | BrainstormingGrader | 评估创意生成的质量 | alignment/helpfulness |
| | ChatGrader | 评估对话交互的质量 | alignment/helpfulness |
| | ClassificationGrader | 评估分类任务的性能 | alignment/helpfulness |
| | ClosedQAGrader | 评估封闭式问答响应的质量 | alignment/helpfulness |
| | CodeGrader | 评估代码问题解决的质量 | alignment/helpfulness |
| | FocusGrader | 评估执行操作期间注意力的维持 | alignment/helpfulness |
| | GenerationGrader | 评估内容生成的质量 | alignment/helpfulness |
| | MathGrader | 评估数学问题解决的质量 | alignment/helpfulness |
| | OpenQAGrader | 评估开放式问答的质量 | alignment/helpfulness |
| | PreciseIFGrader | 评估对指令的精确遵循 | alignment/helpfulness |
| | ReasoningGrader | 评估逻辑推理的质量 | alignment/helpfulness |
| | RewriteGrader | 评估文本重写任务的质量 | alignment/helpfulness |
| | RolePlayingGrader | 评估角色扮演场景中的表现 | alignment/helpfulness |
| | SummarizationGrader | 评估文本摘要的质量 | alignment/helpfulness |
| | TranslationGrader | 评估翻译质量 | alignment/helpfulness |
| | FactualityGrader | 评估模型响应的真实性和事实准确性 | alignment/honesty |
| Text Similarity & Matching (文本相似性与匹配) | StringMatchGrader | 支持多种算法的字符串匹配评估器，包括精确匹配、前缀匹配、后缀匹配、正则表达式匹配、子串匹配、包含全部、包含任意、词汇重叠、字符重叠等 | text |
| | SimilarityGrader | 支持多种算法的文本相似度评估器，包括BLEU、sentence_bleu、GLEU、CHRF、METEOR、ROUGE系列、F1分数、token F1、模糊匹配、编辑距离、余弦相似度、Jaccard相似度等 | text |
| | AccuracyGrader | 计算生成文本与参考文本之间的精确匹配准确率 | text |
| | NumberAccuracyGrader | 通过比较文本中的数字来检查数值计算准确性 | text |
| Code Evaluation (代码评估) | SyntaxCheckGrader | 使用抽象语法树检查代码语法以验证Python代码块 | code |
| | ExecutionVerificationGrader | 验证代码执行并检查输出正确性 | code |
| Math Evaluation (数学评估) | MathVerifyGrader | 使用数学验证库评估数学问题解决能力 | math |
| Format Compliance (格式规范) | ReasoningFormatGrader | 检查适当的思考和答案标签 | format |
| | ReasoningToolCallFormatGrader | 使用JSON验证验证工具调用格式 | format |
| | LengthPenaltyGrader | 对过短或过长的文本施加惩罚 | format |
| | NgramRepetitionPenaltyGrader | 计算支持中文的N-gram重复惩罚 | format |
| | PrivacyLeakageGrader | 检测生成内容中的隐私信息泄露 | format |
| | JsonMatchGrader | 递归地逐元素比较JSON结构 | format/json |
| | JsonValidatorGrader | 验证候选文本是否为有效的JSON | format/json |
| Multimodal Evaluation (多模态评估) | CustomCriteriaGrader | 用于自定义多模态评估的灵活框架 | multimodal |
| | ImageCoherenceGrader | 评估图像与文本上下文之间的一致性 | multimodal |
| | ImageEditingGrader | 评估图像编辑任务的质量 | multimodal |
| | ImageHelpfulnessGrader | 评估图像在理解文本方面的有用性 | multimodal |
| | ImageReferenceGrader | 检查图像是否在文本中得到适当引用 | multimodal |
| | TextToImageGrader | 评估文本到图像生成的质量 | multimodal |
| Custom & Composite Evaluation (自定义与复合评估) | CramoGrader | 实现组合奖励建模方法（CRAMO）框架 | composite |
| LLM-based Evaluation (基于LLM的评估) | HallucinationGrader | 检测没有上下文支持的幻觉或虚构信息 | llm_judge |
| | HarmfulnessGrader | 评估模型响应中的有害性 | llm_judge |
| | HelpfulnessGrader | 提供模型响应的整体有用性评估 | llm_judge |
| | InstructionAdherenceGrader | 评估对指定约束和指令的遵循程度 | llm_judge |
| | ReferenceAdherenceGrader | 检查模型输出与参考材料的一致性 | llm_judge |

## RM-Gallery 与其他平台评估器命名对照表

为了更好地统一评估器命名规范，以下表格展示了RM-Gallery评估器与其他主流评估平台相同或类似功能评估器的命名对照：

| RM-Gallery评估器 | 能力领域 | 功能描述 | OpenAI Evals对应项 | Azure AI Evaluation对应项 | LangSmith对应项 | COZE对应项 | Google ADK对应项 | Arize AI Phoenix对应项 | DeepEval对应项 |
|------------------|---------|----------|-------------------|--------------------------|----------------|------------|------------------|-----------------------|----------------|
| AccuracyGrader | Text Similarity & Matching | 计算生成文本与参考文本之间的精确匹配准确率 | Exact Match Grader | - | Exact Match Evaluator | - | - | Accuracy Evaluator | CorrectnessMetric |
| SimilarityGrader | Text Similarity & Matching | 统一的文本相似度评估器，支持F1分数(algorithm="f1_score")、ROUGE-L(algorithm="rougeL")、BLEU、METEOR等多种算法 | F1 Score Grader, ROUGE Score Grader | F1ScoreEvaluator, SimilarityEvaluator, RougeScoreEvaluator | Embedding Distance Evaluator | - | - | - | F1ScoreMetric, RAGAS Metrics |
| StringMatchGrader | Text Similarity & Matching | 字符串匹配评估 | - | - | Regex Evaluator | - | - | - | - |
| FactualityGrader | Safety & Ethical Alignment | 评估模型响应的真实性和事实准确性 | - | - | Factuality Evaluator | Factual Accuracy Evaluator | Factuality Evaluator | Factuality Evaluator | FactualityMetric (via GEval) |
| HallucinationGrader | LLM-based Evaluation | 检测没有上下文支持的幻觉或虚构信息 | - | - | Harmfulness Evaluator | Hallucination Evaluator | - | Hallucination Evaluator | HallucinationMetric |
| HelpfulnessGrader | LLM-based Evaluation | 提供模型响应的整体有用性评估 | - | HelpfulnessEvaluator | Helpfulness Evaluator | - | Helpfulness Evaluator | Helpfulness Evaluator | CompetencyMetric |
| SafetyGrader | Safety & Ethical Alignment | 评估模型对安全策略的遵守和对有害请求的拒绝能力 | Safety Evaluation Grader | ContentSafetyEvaluator, ViolenceEvaluator, SexualEvaluator, SelfHarmEvaluator, HateUnfairnessEvaluator | Harmfulness Evaluator | - | - | Safety Metrics Evaluator | ToxicityMetric, BiasMetric |
| DetoxificationGrader | Safety & Ethical Alignment | 检测模型输出中的冒犯性或有害内容 | - | - | Toxicity Evaluator | - | - | Toxicity Evaluator | ToxicityMetric |
| PrivacyLeakageGrader | Format Compliance | 检测生成内容中的隐私信息泄露 | - | - | - | - | - | - | - |
| JsonMatchGrader | Format Compliance | 递归地逐元素比较JSON结构 | - | - | Json Equality Evaluator | - | - | - | - |
| JsonValidatorGrader | Format Compliance | 验证候选文本是否为有效的JSON | - | - | Json Schema Evaluator | - | - | - | - |
| ReasoningFormatGrader | Format Compliance | 检查适当的思考和答案标签 | - | - | - | - | - | - | - |
| LengthPenaltyGrader | Format Compliance | 对过短或过长的文本施加惩罚 | - | - | - | - | - | - | ConcisenessMetric |
| NgramRepetitionPenaltyGrader | Format Compliance | 计算N-gram重复惩罚 | - | - | - | - | - | - | - |
| ToolCallAccuracyGrader | Agent Capabilities | 评估智能体工具参数提取和使用的准确性 | Function-based Grader | ToolCallAccuracyEvaluator | - | - | Tool Call Accuracy Evaluator | Tool Call Accuracy Evaluator | ToolCorrectnessMetric |
| ToolCallSuccessGrader | Agent Capabilities | 衡量智能体工具执行的成功率 | - | ToolCallAccuracyEvaluator | - | - | - | - | ToolCorrectnessMetric |
| ToolParameterCheckGrader | Agent Capabilities | 验证智能体使用的工具参数的正确性 | - | - | - | - | - | - | ToolCorrectnessMetric |
| ToolSelectionQualityGrader | Agent Capabilities | 评估智能体工具选择的适当性 | - | - | - | - | - | - | ToolCorrectnessMetric |
| MemoryHallucinationGrader | Agent Capabilities | 检测智能体在内存相关任务中产生的幻觉 | - | - | - | Context Consistency Evaluator | - | - | KnowledgeRetentionMetric |
| PlanImpossibleActionGrader | Agent Capabilities | 评估智能体制定的行动计划的可行性 | Reasoning Evaluation Grader | TaskAdherenceEvaluator | Trajectory Evaluator | - | Action Sequence Evaluator | - | TaskCompletionMetric |
| ReflectionHallucinationGrader | Agent Capabilities | 检测智能体在反思过程中出现的幻觉 | - | - | - | - | - | - | - |
| BrainstormingGrader | Quality Assessment | 评估创意生成的质量 | - | - | - | - | - | - | - |
| ChatGrader | Quality Assessment | 评估对话交互的质量 | - | - | - | - | - | - | ConversationRelevancyMetric |
| CodeGrader | Quality Assessment | 评估代码问题解决的质量 | Python Code Execution Grader | CodeVulnerabilityEvaluator | SQL Evaluator, Python Function Evaluator | - | - | - | - |
| MathGrader | Quality Assessment | 评估数学问题解决的质量 | - | - | - | - | - | - | - |
| SummarizationGrader | Quality Assessment | 评估文本摘要的质量 | - | - | - | - | Final Response Evaluator | - | SummarizationMetric |
| TranslationGrader | Quality Assessment | 评估翻译质量 | BLEU Score Grader, METEOR Score Grader | BleuScoreEvaluator, MeteorScoreEvaluator | - | - | - | - | - |
| ImageCoherenceGrader | Multimodal Evaluation | 评估图像与文本上下文之间的一致性 | - | - | - | Multimodal Evaluator | - | - | Custom Metrics |
| TextToImageGrader | Multimodal Evaluation | 评估文本到图像生成的质量 | - | - | - | Multimodal Evaluator | - | - | Custom Metrics |
| InstructionAdherenceGrader | LLM-based Evaluation | 评估对指定约束和指令的遵循程度 | LLM-as-Judge Grader | InstructionFollowingEvaluator | Labeled Criteria Evaluator | - | Instruction Following Evaluator | - | GEval |
| ReferenceAdherenceGrader | LLM-based Evaluation | 检查模型输出与参考材料的一致性 | - | - | Faithfulness Evaluator | - | - | Faithfulness Evaluator | FaithfulnessMetric |
| CramoGrader | Custom & Composite Evaluation | 实现组合奖励建模方法（CRAMO）框架 | - | - | Custom Evaluator | Quality Scoring Evaluator | - | Custom Evaluator | Custom Metrics |

注意：此对照表仅包含功能相似或相同的评估器，空白表示该平台没有明显对应的评估器或者功能差异较大。通过这个对照表，可以帮助我们更好地理解不同平台间的评估器命名习惯和功能划分，为RM-Gallery的命名规范提供参考。

## 待讨论项目
### common与alignment的重构
添加common子类放常见Grader，能力项与alignment重复，alignment是否需要移除

### 相同含义不同命名的参数归类展示

在分析了当前Grader的实现后，我们发现对于相同含义的概念，不同Grader采用了不同的参数命名方式。为了提高一致性并方便用户使用，建议统一以下参数命名：

#### 输入文本参数
- **输入文本**:
  - `text` (大多数Grader)
  - `input` (CodeExecutionGrader)
  - `input_text` (MathExpressionVerifyGrader)
  - `question`
  - `query`
  - `input_query`
  - `input_question`
  - `user_query`
  - `instruction`

- **生成文本/候选文本**:
  - `generated` (NumberAccuracyGrader, MathExpressionVerifyGrader)
  - `candidate` (StringMatchGrader)
  - `answer` (大多数LLM-based Grader, 如PrivacyLeakageGrader, CodeExecutionGrader)
  - `response` (SimilarityGrader, 在某些示例中)
  - `output` (在某些示例中)
  - `generated_text` (在某些示例中)
  - `prediction` (在某些示例中)
  - `generated_response` (在某些示例中)
  - `completion` (在某些示例中)
  - 建议统一为: `response`

- **参考文本/标准答案**:
  - `reference` (SimilarityGrader, NumberAccuracyGrader, MathExpressionVerifyGrader, StringMatchGrader)
  - `context` (StringMatchGrader)
  - `expected` (在某些示例中)
  - 建议统一为: `reference`

#### 工具相关参数
- **工具定义**:
  - `tool_definition` (ToolParameterCheckGrader)
  - `tool_definitions` (ToolParameterCheckGrader, 另一处)
  - 建议统一为: `tool_definitions`

- **工具调用**:
  - `generated_tool_call` (ToolParameterCheckGrader)
  - `tool_calls` (ToolParameterCheckGrader, 另一处)
  - 建议统一为: `tool_calls`

#### 算法/模式参数
- **匹配算法**:
  - `algorithm` (StringMatchGrader)
  - 建议保持: `algorithm`

#### 评估相关参数
- **评分标准**:
  - `rubrics` (LLMGrader)
  - 建议保持: `rubrics`

这种参数命名的统一将有助于：
1. 提高API的一致性
2. 降低用户学习成本
3. 简化文档编写
4. 减少因参数命名不一致导致的使用错误
