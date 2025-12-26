<div align="center">

<img src="./docs/images/logo.png" alt="Open-Judge Logo" width="500">

<br/>

<h3>
  <em>全面评估，质量驱动：提升应用效果</em>
</h3>

<p>
  🌟 <em>如果您觉得 OpenJudge 有帮助，请给我们一个 <b>Star</b>！</em> 🌟 
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://pypi.org/project/py-openjudge/)
[![PyPI](https://img.shields.io/badge/pypi-v0.2.0-blue?logo=pypi)](https://pypi.org/project/py-openjudge/)
[![Documentation](https://img.shields.io/badge/docs-online-blue?logo=readthedocs&logoColor=white)](https://modelscope.github.io/OpenJudge/)

[📖 文档](https://modelscope.github.io/OpenJudge/) | [🤝 贡献指南](./docs/community/contributing.md) | [English](./README.md)

</div>

---

## 📑 目录

- [核心特性](#-核心特性)
- [最新动态](#最新动态)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [集成](#-集成)
- [贡献](#-贡献)
- [引用](#-引用)

OpenJudge 是一个统一框架，旨在通过**全面评估**和**质量奖励**来提升应用效果。

> 💡 评估和奖励信号是应用的基石。**全面评估**能够系统分析不足之处以推动快速迭代，而**高质量**奖励则为高级优化和微调提供必要的基础。

OpenJudge 将评估指标和奖励信号统一为标准化的 **Grader** 接口，提供预构建的评分器、灵活的自定义能力以及无缝的框架集成。

---

## ✨ 核心特性

### 📦 系统化、质量保证的评分器库

访问 **50+ 生产就绪的评分器**，具有全面的分类体系，经过严格验证以确保可靠性能。

<table>
<tr>
<td width="33%" valign="top">

#### 🎯 通用

**关注点：** 语义质量、功能正确性、结构合规性

**核心评分器：**
- `Relevance` - 语义相关性评分
- `Similarity` - 文本相似度测量
- `Syntax Check` - 代码语法验证
- `JSON Match` - 结构合规性检查

</td>
<td width="33%" valign="top">

#### 🤖 智能体

**关注点：** 智能体生命周期、工具调用、记忆、计划可行性、轨迹质量

**核心评分器：**
- `Tool Selection` - 工具选择准确性
- `Memory` - 上下文保持能力
- `Plan` - 策略可行性
- `Trajectory` - 路径优化

</td>
<td width="33%" valign="top">

#### 🖼️ 多模态

**关注点：** 图文一致性、视觉生成质量、图像有用性

**核心评分器：**
- `Image Coherence` - 视觉-文本对齐
- `Text-to-Image` - 生成质量
- `Image Helpfulness` - 图像贡献度

</td>
</tr>
</table>

- 🌐 **多场景覆盖：** 广泛支持包括智能体、文本、代码、数学和多模态任务在内的多种领域。→ [探索支持的场景](./docs/built_in_graders/overview.md)
- 🔄 **全面的智能体评估：** 不仅评估最终结果，我们还评估整个生命周期——包括轨迹、记忆、反思和工具使用。→ [智能体生命周期评估](./docs/built_in_graders/agent_graders.md)
- ✅ **质量保证：** 每个评分器都配有基准数据集和 pytest 集成用于验证。→ [查看基准数据集](https://huggingface.co/datasets/agentscope-ai/OpenJudge)


### 🛠️ 灵活的评分器构建方法
选择适合您需求的构建方法：
* **自定义：** 轻松扩展或修改预定义的评分器以满足您的特定需求。👉 [自定义评分器开发指南](./docs/building_graders/create_custom_graders.md)
* **数据驱动的评分标准：** 有一些示例但没有明确规则？使用我们的工具根据您的数据自动生成白盒评估标准（Rubrics）。👉 [自动评分标准生成教程](./docs/building_graders/generate_graders_from_data.md)
* **训练评判模型（即将推出🚀）：** 对于大规模和专业化场景，我们正在开发训练专用评判模型的能力。SFT、Bradley-Terry 模型和强化学习工作流的支持即将推出，帮助您构建高性能、领域特定的评分器。


### 🔌 轻松集成（🚧 即将推出）

我们正在积极构建与主流可观测性平台和训练框架的无缝连接器。敬请期待！→ 查看 [集成](#-集成)

----
## 最新动态

- **2025-12-26** - 在 [PyPI](https://pypi.org/project/py-openjudge/) 上发布 OpenJudge v0.2.0 - **重大更新！** 此版本通过在奖励构建之上添加对多样化评估场景的强大支持，扩展了我们的核心能力。通过统一奖励和评估信号，OpenJudge v0.2.0 提供了一种更全面的方法来优化应用性能和卓越性。→ [v0.1.x 用户须知](#-v01x-用户须知)

- **2025-10-20** - [Auto-Rubric: Learning to Extract Generalizable Criteria for Reward Modeling](https://arxiv.org/abs/2510.17314) - 我们发布了一篇关于学习可泛化奖励标准以实现稳健建模的新论文。
- **2025-10-17** - [Taming the Judge: Deconflicting AI Feedback for Stable Reinforcement Learning](https://arxiv.org/abs/2510.15514) - 我们介绍了对齐评判反馈和提高强化学习稳定性的技术。
- **2025-07-09** - 在 [PyPI](https://pypi.org/project/rm-gallery/) 上发布 OpenJudge v0.1.0

---

## 📥 安装

```bash
pip install py-openjudge
```

> 💡 更多安装方法可在 [快速开始指南](./docs/get_started/quickstart.md) 中找到。

---

## 🚀 快速开始

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common.relevance import RelevanceGrader

async def main():
    # 1️⃣ 创建模型客户端
    model = OpenAIChatModel(model="qwen3-32b")

    # 2️⃣ 初始化评分器
    grader = RelevanceGrader(model=model)

    # 3️⃣ 准备数据
    data = {
        "query": "什么是机器学习？",
        "response": "机器学习是人工智能的一个子集，使计算机能够从数据中学习。",
    }

    # 4️⃣ 评估
    result = await grader.aevaluate(**data)

    print(f"分数: {result.score}")   # 分数: 5
    print(f"原因: {result.reason}")

if __name__ == "__main__":
    asyncio.run(main())
```

> 📚 完整的快速开始内容可在 [快速开始指南](./docs/get_started/quickstart.md) 中找到。

---

## 🔗 集成

无缝连接 OpenJudge 与主流可观测性和训练平台，更多集成即将推出：

| 类别 | 状态 | 平台 |
|:---------|:------:|:----------|
| **可观测性** | 🟡 进行中 | [LangSmith](https://smith.langchain.com/)、[LangFuse](https://langfuse.com/)、[Arize Phoenix](https://github.com/Arize-ai/phoenix) |
| **训练** | 🔵 计划中 | [verl](https://github.com/volcengine/verl)、[Trinity-RFT](https://github.com/modelscope/Trinity-RFT) |

> 💬 有您希望我们优先支持的框架吗？[提交 Issue](https://github.com/modelscope/OpenJudge/issues)！



---

## 🤝 贡献

我们欢迎您的贡献！我们希望让参与 OpenJudge 的贡献尽可能简单和透明。

> **🎨 添加新评分器** — 有领域特定的评估逻辑？与社区分享吧！  
> **🐛 报告 Bug** — 发现问题？通过 [提交 issue](https://github.com/modelscope/OpenJudge/issues) 帮助我们修复  
> **📝 改进文档** — 更清晰的解释或更好的示例总是受欢迎的  
> **💡 提议新功能** — 有新集成的想法？让我们讨论！

📖 查看完整的 [贡献指南](./docs/community/contributing.md) 了解编码标准和 PR 流程。

---

## 📦 v0.1.x 用户须知

> 包已从 `rm-gallery` 重命名为 `py-openjudge`。旧版本仍可通过 `pip install rm-gallery` 获取。源代码保存在 [`v0.1.6` 分支](https://github.com/modelscope/OpenJudge/tree/v0.1.6)。

---

## 📄 引用

如果您在研究中使用 OpenJudge，请引用：

```bibtex
@software{
  title  = {OpenJudge: A Unified Framework for Holistic Evaluation and Quality Rewards},
  author = {The OpenJudge Team},
  url    = {https://github.com/modelscope/OpenJudge},
  month  = {07},
  year   = {2025}
}
```

---

<div align="center">

**由 OpenJudge 团队用 ❤️ 打造**

[⭐ 给我们 Star](https://github.com/modelscope/OpenJudge) · [🐛 报告 Bug](https://github.com/modelscope/OpenJudge/issues) · [💡 提议功能](https://github.com/modelscope/OpenJudge/issues)

</div>

