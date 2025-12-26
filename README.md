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

[📖 Documentation](https://modelscope.github.io/OpenJudge/) | [🤝 Contributing](./docs/community/contributing.md) | [中文](./README_zh.md)

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

OpenJudge is a unified framework designed to drive application excellence through **Holistic Evaluation** and **Quality Rewards**.

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

- 🌐 **Multi-Scenario Coverage:** Extensive support for diverse domains including Agent, text, code, math, and multimodal tasks. → [Explore Supported Scenarios](./docs/built_in_graders/overview.md)
- 🔄 **Holistic Agent Evaluation:** Beyond final outcomes, we assess the entire lifecycle—including trajectories, Memory, Reflection, and Tool Use. → [Agent Lifecycle Evaluation](./docs/built_in_graders/agent_graders.md)
- ✅ **Quality Assurance:** Every grader comes with benchmark datasets and pytest integration for validation. → [View Benchmark Datasets](https://huggingface.co/datasets/agentscope-ai/OpenJudge)


### 🛠️ Flexible Grader Building Methods
Choose the build method that fits your requirements:
* **Customization:** Easily extend or modify pre-defined graders to fit your specific needs.  👉 [Custom Grader Development Guide](./docs/building_graders/create_custom_graders.md)
* **Data-Driven Rubrics:** Have a few examples but no clear rules? Use our tools to automatically generate white-box evaluation criteria (Rubrics) based on your data.👉 [Automatic Rubric Generation Tutorial](./docs/building_graders/generate_graders_from_data.md)
* **Training Judge Models ( Coming Soon🚀):** For high-scale and specialized scenarios, we are developing the capability to train dedicated Judge models. Support for SFT, Bradley-Terry models, and Reinforcement Learning workflows is on the way to help you build high-performance, domain-specific graders.


### 🔌 Easy Integration (🚧 Coming Soon)

We're actively building seamless connectors for mainstream observability platforms and training frameworks. Stay tuned! → See [Integrations](#-integrations)

----
## News

- **2025-12-26** - Released OpenJudge v0.2.0 on [PyPI](https://pypi.org/project/py-openjudge/) - **Major Update!** This release expands our core capabilities by adding robust support for diverse evaluation scenarios on top of reward construction. By unifying reward and evaluation signals, OpenJudge v0.2.0 provides a more holistic approach to optimizing application performance and excellence. → [For v0.1.x Users](#-for-v01x-users)

- **2025-10-20** - [Auto-Rubric: Learning to Extract Generalizable Criteria for Reward Modeling](https://arxiv.org/abs/2510.17314) - We released a new paper on learning generalizable reward criteria for robust modeling.
- **2025-10-17** - [Taming the Judge: Deconflicting AI Feedback for Stable Reinforcement Learning](https://arxiv.org/abs/2510.15514) - We introduced techniques to align judge feedback and improve RL stability.
- **2025-07-09** - Released OpenJudge v0.1.0 on [PyPI](https://pypi.org/project/rm-gallery/)

---

## 📥 Installation

```bash
pip install py-openjudge
```

> 💡 More installation methods can be found in the [Quickstart Guide](./docs/get_started/quickstart.md).

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

> 📚 Complete Quickstart can be found in the [Quickstart Guide](./docs/get_started/quickstart.md).

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

📖 See full [Contributing Guidelines](./docs/community/contributing.md) for coding standards and PR process.

---

## 📦 For v0.1.x Users

> Package renamed from `rm-gallery` → `py-openjudge`. Legacy version still available via `pip install rm-gallery`. Source code preserved in [`v0.1.6` branch](https://github.com/modelscope/OpenJudge/tree/v0.1.6).

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