<div align="center">

<img src="./docs/images/logo.png" alt="Open-Judge Logo" width="500">

<br/>

<h3>
  <em>Holistic Evaluation, Quality Rewards: Driving Application Excellence</em>
</h3>

<p>
  🌟 <em>If you find OpenJudge helpful, please give us a <b>Star</b>!</em> 🌟 
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white)](https://pypi.org/project/open_judge/)
[![PyPI](https://img.shields.io/badge/pypi-v0.2.0-blue?logo=pypi)](https://pypi.org/project/py-openjudge/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?logo=apache)](./LICENSE)
[![Documentation](https://img.shields.io/badge/docs-online-blue?logo=readthedocs&logoColor=white)](https://modelscope.github.io/OpenJudge/)

[📖 Documentation](https://modelscope.github.io/OpenJudge/) | [🤝 Contributing](./docs/community/contributing.md) | [🇨🇳 中文](./README_zh.md)

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

OpenJudge unifies these signals into a single, standardized **Grader** interface, offering pre-built graders, flexible customization, and seamless framework integration.

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

**Focus:** Agent lifecycle, tool calling, memory, plan feasibility

**Key Graders:**
- `Tool Selection` - Tool choice accuracy
- `Memory` - Context preservation
- `Plan` - Strategy feasibility
- `Trajectory` - Path optimization

</td>
<td width="33%" valign="top">

#### 🖼️ Multimodal

**Focus:** Image-text coherence, visual generation quality

**Key Graders:**
- `Image Coherence` - Visual-text alignment
- `Text-to-Image` - Generation quality

</td>
</tr>
</table>

<details>
<summary><b>🔍 Learn More About Graders</b></summary>

- 🌐 **Multi-Scenario Coverage:** Extensive support for diverse domains including Agent, text, code, math, and multimodal tasks. → [Explore Supported Scenarios](./docs/built_in_graders/overview.md)
- 🔄 **Holistic Agent Evaluation:** Beyond final outcomes, we assess the entire lifecycle—including trajectories, Memory, Reflection, and Tool Use. → [Agent Lifecycle Evaluation](./docs/built_in_graders/agent_graders.md)
- ✅ **Quality Assurance:** Every grader comes with benchmark datasets and pytest integration for validation. → [View Benchmark Datasets](https://huggingface.co/datasets/agentscope-ai/OpenJudge)

</details>

### 🛠️ Flexible Grader Building Methods

Choose the build method that fits your requirements:

| Method | Description | Guide |
|:-------|:------------|:-----:|
| 🎨 **Customization** | Extend or modify pre-defined graders for your specific needs | [📘 Guide](./docs/building_graders/create_custom_graders.md) |
| 📊 **Data-Driven Rubrics** | Auto-generate evaluation criteria from your examples | [📘 Guide](./docs/building_graders/generate_graders_from_data.md) |
| 🧠 **Trainable Models** | Train dedicated Judge models with SFT, Bradley-Terry, RL | 🚀 Coming Soon |

### 🔌 Easy Integration

Seamlessly connect with mainstream observability platforms and training frameworks. → See [Integrations](#-integrations)


## News

- **2025-12-26** - Released OpenJudge v0.2.0 on [PyPI](https://pypi.org/project/py-openjudge/) - **Major Update!** This release expands our core capabilities by adding robust support for diverse evaluation scenarios on top of reward construction. By unifying reward and evaluation signals, OpenJudge v0.2.0 provides a more holistic approach to optimizing application performance and excellence.

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

> 📚 Complete Quickstart can be found in the [documentation](./docs/get_started/quickstart.md).

---

## 🔗 Integrations

We are committed to supporting the most critical stages of the AI lifecycle:

| Category | Status | Platforms |
|:---------|:------:|:----------|
| 🔭 **Observability** | 🟡 In Progress | LangSmith, LangFuse, Arize Phoenix |
| 🏋️ **Training** | 🔵 Planned | RLHF, Agent Training, SFT |

> 💬 Have a framework you'd like us to prioritize? [Open an Issue](https://github.com/modelscope/OpenJudge/issues)!



---

## 🤝 Contributing

We love your input! We want to make contributing to OpenJudge as easy and transparent as possible.

<table>
<tr>
<td align="center" width="25%">

**🎨 Add Graders**

Share domain-specific evaluation logic

</td>
<td align="center" width="25%">

**🐛 Report Bugs**

Help us fix issues

</td>
<td align="center" width="25%">

**📝 Improve Docs**

Better explanations welcome

</td>
<td align="center" width="25%">

**💡 Propose Features**

New ideas? Let's discuss!

</td>
</tr>
</table>

### Quick Start

<table>
<tr>
<td align="center">

**1️⃣ Fork**

Clone & branch

</td>
<td align="center">

**2️⃣ Install**

```bash
pip install -e ".[dev]"
```

</td>
<td align="center">

**3️⃣ Test**

```bash
pytest tests/
```

</td>
<td align="center">

**4️⃣ Submit**

Open a PR!

</td>
</tr>
</table>

> 📖 See full [Contributing Guidelines](./docs/community/contributing.md) for coding standards and PR process.

---

## 📄 Citation

If you use OpenJudge in your research, please cite:

```bibtex
@software{openjudge2025,
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
