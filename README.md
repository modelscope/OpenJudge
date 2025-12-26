<div align="center">

<img src="./docs/images/logo.png" alt="Open-Judge Logo" width="500">

<br/>

<h3>
  <em>Holistic Evaluation, Quality Rewards: Driving Application Excellence</em>
</h3>

<p>
  🌟 <em>If you find OpenJudge helpful, please give us a <b>Star</b>!</em> 🌟 
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://pypi.org/project/py-openjudge/)
[![PyPI](https://img.shields.io/badge/pypi-v0.2.0-blue?logo=pypi)](https://pypi.org/project/py-openjudge/)
[![Documentation](https://img.shields.io/badge/docs-online-blue?logo=readthedocs&logoColor=white)](https://modelscope.github.io/OpenJudge/)

[📖 Documentation](https://modelscope.github.io/OpenJudge/) | [🤝 Contributing](https://modelscope.github.io/OpenJudge/community/contributing/) | [中文](./README_zh.md)

</div>

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [News](#news)
- [Installation](#-installation)
- [Quickstart](#-quickstart)
- [Integrations](#-integrations)
- [Contributing](#-contributing)
- [Citation](#-citation)

OpenJudge is a unified framework designed to drive **LLM and Agent application excellence** through **Holistic Evaluation** and **Quality Rewards**.

> 💡 Evaluation and reward signals are the cornerstones of application excellence. **Holistic evaluation** enables the systematic analysis of shortcomings to drive rapid iteration, while **high-quality** rewards provide the essential foundation for advanced optimization and fine-tuning.

OpenJudge unifies evaluation metrics and reward signals into a single, standardized **Grader** interface, offering pre-built graders, flexible customization, and seamless framework integration.

---

## ✨ Key Features

### 📦 Systematic & Quality-Assured Grader Library

Access **50+ production-ready graders** featuring a comprehensive taxonomy, rigorously validated for reliable performance.

<table>
<tr>
<td width="33%" valign="top">

#### 🎯 General

**Focus:** Semantic quality, functional correctness, structural compliance

**Key Graders:**
- `Relevance` - Semantic relevance scoring
- `Similarity` - Text similarity measurement  
- `Syntax Check` - Code syntax validation
- `JSON Match` - Structure compliance

</td>
<td width="33%" valign="top">

#### 🤖 Agent

**Focus:** Agent lifecycle, tool calling, memory, plan feasibility, trajectory quality

**Key Graders:**
- `Tool Selection` - Tool choice accuracy
- `Memory` - Context preservation
- `Plan` - Strategy feasibility
- `Trajectory` - Path optimization

</td>
<td width="33%" valign="top">

#### 🖼️ Multimodal

**Focus:** Image-text coherence, visual generation quality, image helpfulness

**Key Graders:**
- `Image Coherence` - Visual-text alignment
- `Text-to-Image` - Generation quality
- `Image Helpfulness` - Image contribution

</td>
</tr>
</table>

- 🌐 **Multi-Scenario Coverage:** Extensive support for diverse domains including Agent, text, code, math, and multimodal tasks. 👉  [Explore Supported Scenarios](https://modelscope.github.io/OpenJudge/built_in_graders/overview/)
- 🔄 **Holistic Agent Evaluation:** Beyond final outcomes, we assess the entire lifecycle—including trajectories, Memory, Reflection, and Tool Use. 👉  [Agent Lifecycle Evaluation](https://modelscope.github.io/OpenJudge/built_in_graders/agent_graders/)
- ✅ **Quality Assurance:** Every grader comes with benchmark datasets and pytest integration for validation. 👉  [View Benchmark Datasets](https://huggingface.co/datasets/agentscope-ai/OpenJudge)


### 🛠️ Flexible Grader Building Methods
Choose the build method that fits your requirements:
* **Customization:** Easily extend or modify pre-defined graders to fit your specific needs.  👉 [Custom Grader Development Guide](https://modelscope.github.io/OpenJudge/building_graders/create_custom_graders/)
* **Data-Driven Rubrics:** Have a few examples but no clear rules? Use our tools to automatically generate white-box evaluation criteria (Rubrics) based on your data.👉 [Automatic Rubric Generation Tutorial](https://modelscope.github.io/OpenJudge/building_graders/generate_graders_from_data/)
* **Training Judge Models ( Coming Soon🚀):** For high-scale and specialized scenarios, we are developing the capability to train dedicated Judge models. Support for SFT, Bradley-Terry models, and Reinforcement Learning workflows is on the way to help you build high-performance, domain-specific graders.


### 🔌 Easy Integration (🚧 Coming Soon)

We're actively building seamless connectors for mainstream observability platforms and training frameworks. Stay tuned! → See [Integrations](#-integrations)

----
## News

- **2025-12-26** - Released OpenJudge v0.2.0 on [PyPI](https://pypi.org/project/py-openjudge/) - **Major Update!** This release expands our core capabilities by adding robust support for diverse evaluation scenarios on top of reward construction. By unifying reward and evaluation signals, OpenJudge v0.2.0 provides a more holistic approach to optimizing application performance and excellence. → [migration-guide](#migration-guide-v01x--v020)

- **2025-10-20** - [Auto-Rubric: Learning to Extract Generalizable Criteria for Reward Modeling](https://arxiv.org/abs/2510.17314) - We released a new paper on learning generalizable reward criteria for robust modeling.
- **2025-10-17** - [Taming the Judge: Deconflicting AI Feedback for Stable Reinforcement Learning](https://arxiv.org/abs/2510.15514) - We introduced techniques to align judge feedback and improve RL stability.
- **2025-07-09** - Released OpenJudge v0.1.0 on [PyPI](https://pypi.org/project/rm-gallery/)

---

## 📥 Installation

```bash
pip install py-openjudge
```

> 💡 More installation methods can be found in the [Quickstart Guide](https://modelscope.github.io/OpenJudge/get_started/quickstart/#installation).

---

## 🚀 Quickstart

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.common.relevance import RelevanceGrader

async def main():
    # 1️⃣ Create model client
    model = OpenAIChatModel(model="qwen3-32b")

    # 2️⃣ Initialize grader
    grader = RelevanceGrader(model=model)

    # 3️⃣ Prepare data
    data = {
        "query": "What is machine learning?",
        "response": "Machine learning is a subset of AI that enables computers to learn from data.",
    }

    # 4️⃣ Evaluate
    result = await grader.aevaluate(**data)

    print(f"Score: {result.score}")   # Score: 5
    print(f"Reason: {result.reason}")

if __name__ == "__main__":
    asyncio.run(main())
```

> 📚 Complete Quickstart can be found in the [Quickstart Guide](https://modelscope.github.io/OpenJudge/get_started/quickstart/).

---

## 🔗 Integrations

Seamlessly connect OpenJudge with mainstream observability and training platforms, with more integrations on the way:

| Category | Status | Platforms |
|:---------|:------:|:----------|
| **Observability** | 🟡 In Progress | [LangSmith](https://smith.langchain.com/), [LangFuse](https://langfuse.com/), [Arize Phoenix](https://github.com/Arize-ai/phoenix) |
| **Training** | 🔵 Planned | [verl](https://github.com/volcengine/verl), [Trinity-RFT](https://github.com/modelscope/Trinity-RFT) |

> 💬 Have a framework you'd like us to prioritize? [Open an Issue](https://github.com/modelscope/OpenJudge/issues)!



---

## 🤝 Contributing

We love your input! We want to make contributing to OpenJudge as easy and transparent as possible.

> **🎨 Adding New Graders** — Have domain-specific evaluation logic? Share it with the community!  
> **🐛 Reporting Bugs** — Found a glitch? Help us fix it by [opening an issue](https://github.com/modelscope/OpenJudge/issues)  
> **📝 Improving Docs** — Clearer explanations or better examples are always welcome  
> **💡 Proposing Features** — Have ideas for new integrations? Let's discuss!

📖 See full [Contributing Guidelines](https://modelscope.github.io/OpenJudge/community/contributing/) for coding standards and PR process.

---



### Migration Guide (v0.1.x → v0.2.0)
> OpenJudge was previously distributed as the legacy package `rm-gallery` (v0.1.x). Starting from v0.2.0, it is published as `py-openjudge` and the Python import namespace is `openjudge`.

**OpenJudge v0.2.0 is NOT backward compatible with v0.1.x.**  
If you are currently using v0.1.x, choose one of the following paths:

- **Stay on v0.1.x (legacy)**: keep using the old package

```bash
pip install rm-gallery
```

We preserved the source code of **v0.1.7 (the latest v0.1.x release)** in the [`v0.1.7-legacy` branch](https://github.com/modelscope/OpenJudge/tree/v0.1.7-legacy).

- **Migrate to v0.2.0 (recommended)**: follow the **[Installation](#-installation)** section above, then walk through **[Quickstart](#-quickstart)** (or the full [Quickstart Guide](https://modelscope.github.io/OpenJudge/get_started/quickstart/)) to update your imports / usage.

If you run into migration issues, please [open an issue](https://github.com/modelscope/OpenJudge/issues) with your minimal repro and current version.

---

## 📄 Citation

If you use OpenJudge in your research, please cite:

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

**Made with ❤️ by the OpenJudge Team**

[⭐ Star Us](https://github.com/modelscope/OpenJudge) · [🐛 Report Bug](https://github.com/modelscope/OpenJudge/issues) · [💡 Request Feature](https://github.com/modelscope/OpenJudge/issues)

</div>