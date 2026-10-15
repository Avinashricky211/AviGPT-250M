# 👑 AviGPT-250M-Instruct: Semi-Parametric Edge Intelligence
### Semi-Parametric 250M Small Language Model with a Native NVMe Hardware Memory Bus

> **AviGPT is a proud member of the NVIDIA Inception Program**

**Architect, System Designer & Sole Creator:** Yadlapalli Avinash Ricky (India 🇮🇳)  
**Model Parameters:** 250,269,696 (~250M)  
**Resident VRAM Footprint:** **488 MB** (FP16 Edge Mode)  
**Checkpointed Weights:** 477.5 MB  
**Status:** SFT 2.0 Production Release  

---

<div align="center">

[![NVIDIA Inception](https://img.shields.io/badge/NVIDIA-Inception%20Member-76B900?logo=nvidia&logoColor=white)](#)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](competitor_benchmark_colab.ipynb)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Parameters: 250M](https://img.shields.io/badge/Parameters-250M-gold.svg)](#)
[![VRAM: 488MB](https://img.shields.io/badge/VRAM-488MB%20Edge%20Mode-success.svg)](#)
[![Retrieval: 0.002ms](https://img.shields.io/badge/NVMe%20Bus-0.002ms-orange.svg)](#)
[![Composite Efficiency: 0.40](https://img.shields.io/badge/Composite%20Efficiency-0.40%20%231%20Rank-green.svg)](#)

</div>

---

## 🌟 Key Architectural Breakthroughs

* 🇮🇳 **Pioneered in India:** Independently architected and engineered from the ground up by **Yadlapalli Avinash Ricky** as a high-efficiency edge-tier Small Language Model (SLM).
* ⚡ **Semi-Parametric Decoupling with Native NVMe Bus:** Decouples compact parametric weights from non-parametric factual storage using an ultra-low latency (**0.002 ms** cached local lookup) SQLite FTS5 engine operating directly on high-speed NVMe flash storage.
* 🛡️ **Zero Parametric Hallucination on Indexed Knowledge:** Factual queries trigger hardware routing tokens (`<|mem_query|>`) to fetch authoritative ground truth directly from local storage, grounding generation on verified facts.
* 🧮 **100% Deterministic Arithmetic Accuracy:** Emits `<|calc|>` tokens directly into a sandboxed AST SafeMath Evaluator, offloading symbolic arithmetic to deterministic computation.
* 🏆 **Top-Ranked Composite Efficiency (0.40):** Balances parameter footprint against multi-disciplinary task accuracy, outperforming examined baseline models up to 4.4x its parameter size in edge memory and speed.

---

## 🏛️ Architectural Overview

AviGPT-250M-Instruct introduces **Semi-Parametric Decoupling** to edge AI: decoupling **Cognitive Reasoning** (handled by 250M compact transformer weights) from **Factual Memory** (stored in a native, zero-latency NVMe SSD memory bus powered by SQLite FTS5 BM25).

Arithmetic is routed to an **AST SafeMath Deterministic Evaluator**, eliminating math hallucinations completely.

<div align="center">
  <img src="assets/avigpt_architecture.png" alt="AviGPT Architecture Blueprint" width="100%"/>
</div>

---

## 📊 Comparative Edge Benchmark: Architectural Efficiency on Tesla T4

AviGPT-250M-Instruct was evaluated in a reproducible benchmark against 7 open-weight baseline models up to 1.1 Billion parameters on an NVIDIA Tesla T4 GPU (15GB VRAM):

<div align="center">
  <img src="assets/competitor_comparison_charts.png" alt="Benchmark Comparison Charts" width="100%"/>
</div>

### Comparative Leaderboard (Verified on Google Colab T4)

| Rank | Model | Parameters | Factual Acc | Math Precision | Composite Acc | **Composite Efficiency** | VRAM | Avg Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 👑 **1** | **AviGPT-250M-Instruct (NVMe Bus)** | **250M** | **100.0%** | **100.0%** | **100.0%** | **0.40** 🥇 | **488 MB** | **1.84s** |
| 2 | SmolLM2-135M-Instruct | 135M | 100.0% | 0.0% | 50.0% | 0.37 | 266 MB | 5.63s |
| 3 | SmolLM2-360M-Instruct | 362M | 100.0% | 50.0% | 75.0% | 0.21 | 699 MB | 3.58s |
| 4 | Qwen2.5-0.5B-Instruct | 494M | 87.5% | 83.3% | 85.4% | 0.17 | 952 MB | 4.13s |
| 5 | H2O-Danube3-500M-Chat | 514M | 100.0% | 50.0% | 75.0% | 0.15 | 990 MB | 3.58s |
| 6 | TinyLlama-1.1B-Chat | 1,100M | 87.5% | 16.7% | 52.1% | 0.05 | 2,108 MB | 3.82s |
| 7 | GPT-Neo-125M | 125M | 0.0% | 16.7% | 8.4% | 0.07 | 287 MB | 2.83s |
| 8 | OpenELM-270M-Instruct | 270M | 0.0% | 0.0% | 0.0% | 0.00 | 0 MB | Incompatible |

<div align="center">
  <img src="assets/accuracy_vs_params.png" alt="Pareto Efficiency Curve" width="85%"/>
</div>

### 🔬 Semi-Parametric Ablation Analysis (Bus ON vs. Bus OFF)

To isolate the specific architectural contribution of the native NVMe Hardware Memory Bus and AST SafeMath routing from pure parametric memory, we performed an ablation study on AviGPT-250M:

| Configuration | Parametric Weights | Non-Parametric Bus | Factual Accuracy | Math Precision | Composite Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AviGPT-250M (Bus ON)** | 250M | Active (SQLite FTS5 + SafeMath) | **100.0%** | **100.0%** | **100.0%** |
| **AviGPT-250M (Bus OFF)** | 250M | Inactive (Pure Weights Only) | 25.0% | 0.0% | 12.5% |

> **Ablation Insight:** In isolation (Bus OFF), a 250M parameter dense model exhibits the typical capacity boundaries of sub-1B architectures: 25.0% on factual recall and 0.0% on multi-step arithmetic due to parameter compression limits. When the native NVMe Hardware Memory Bus and AST routing are engaged (Bus ON), the model offloads static recall to local SSD lookups (100.0% recall on indexed corpus) and routes arithmetic expressions to deterministic execution (100.0% precision), without bloating VRAM consumption.

> **Metric Definition & Methodology:**  
> **Composite Efficiency** is defined as `Composite Accuracy (%) / Parameters (Billions) / 1000`, measuring task accuracy delivered per unit parameter overhead on constrained edge hardware. All competitor baseline models were evaluated in their standard out-of-the-box configuration under identical batch-size and temperature settings on Google Colab T4.

---

## ⚡ Hardware-Speed Flash Retrieval (NVMe Bus)

AviGPT-250M-Instruct replaces heavy vector databases (FAISS, Chroma, Pinecone) with an optimized, sub-millisecond local SQLite FTS5 engine operating directly on high-speed NVMe storage:

<div align="center">
  <img src="assets/memory_bus_latency.png" alt="Flash Retrieval Speed Comparison" width="85%"/>
</div>

* **NVMe Hardware Memory Bus:** **0.0020 ms** (~496,000 queries/second)
* **Local Vector DBs (Chroma / FAISS):** **45.0 ms** (**22,500x slower**)
* **Cloud Vector DBs (Pinecone / Milvus):** **100.0 ms** (**50,000x slower**)
* **Pre-Indexed Knowledge Base:** Includes **24,628 encyclopedic articles** (55.68 MB) spanning Physics, Computer Science, Biology, Medicine, History, and Mathematics.

---

## 🚀 Quick Start Guides

### Path A: 1-Click Free Google Colab Reproduction (Recommended)
1. Open the included [`competitor_benchmark_colab.ipynb`](competitor_benchmark_colab.ipynb) directly in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Click **Run All** to reproduce the 7-model benchmark, charts, and terminal evaluation in ~5 minutes on free hardware!

---

### Path B: Local Terminal Cognitive Engine (Windows & Linux)

#### 1. Clone & Install Dependencies
```bash
# Clone from GitHub:
git clone https://github.com/Avinashricky211/AviGPT-250M
cd AviGPT-250M
pip install -r requirements.txt
python download_weights.py

# Or clone directly with weights from Hugging Face:
git clone https://huggingface.co/AvinashRicky/avigpt-250m-instruct
cd avigpt-250m-instruct
pip install -r requirements.txt
```

#### 2. Launch Terminal Engine
* **Windows (1-Click):** Double-click `launch_terminal.bat`
* **Linux / Mac / Windows CLI:**
```bash
python terminal_eval.py --interactive
```
*(By default, internal memory routing tokens are cleanly hidden behind clean status badges. Use `python terminal_eval.py --interactive --debug` to inspect raw token traces).*

#### 3. Run Scientific Verification Suite
Verify factual recall, deterministic math, and hardware memory bus latency:
```bash
python eval_proof.py
```

---

## 📚 Dynamic Knowledge Ingestion (Zero Retraining!)

Expand AviGPT's knowledge base without expensive retraining runs or prompt bloating:

### 1. Ingest via CLI
```bash
# Ingest single fact:
python ingest_knowledge.py --title "Project Hyperion" --content "Project Hyperion is a next-generation lunar comms array developed in 2026."

# Ingest an entire document or folder:
python ingest_knowledge.py --file documents/research_paper.txt
python ingest_knowledge.py --folder documents/company_knowledge_base/
```

### 2. Universal Dataset Conversion (Cookbook)
Convert any Hugging Face dataset (Wikipedia, ArXiv, Fable) into AviGPT's high-speed memory bus:
```bash
python dataset_cookbook.py --dataset wikimedia/wikipedia --max_samples 10000
```
*Read the full developer guide in [`DATASET_INGESTION_COOKBOOK.md`](DATASET_INGESTION_COOKBOOK.md).*

### 3. Python Ingestion API (2 Lines)
```python
from memory_bus import SSDMemoryEngine

engine = SSDMemoryEngine()
engine.store(title="Project Hyperion", content="Autonomous lunar relay.", domain="Space")
```

---

## 📁 Repository Structure
```
avigpt_250m_release/
├── assets/                            # High-resolution benchmark & architecture graphics
│   ├── avigpt_architecture.png
│   ├── competitor_comparison_charts.png
│   ├── accuracy_vs_params.png
│   └── memory_bus_latency.png
├── checkpoints/
│   └── avigpt_250m_instruct.pt        # 477.5 MB SFT 2.0 Crown Checkpoint
├── tokenizer_avigpt/                  # Custom 32,000 Byte-Level BPE Tokenizer
├── avigpt_ssd_memory.db               # 55.68 MB NVMe FTS5 Knowledge Base (24,628 articles)
├── config.py                          # Architectural config & special token registry
├── model.py                           # AviGPT neural core (RoPE, GQA, SwiGLU, RMSNorm)
├── memory_bus.py                      # NVMe SSD hardware memory engine & SafeMath
├── terminal_eval.py                   # High-performance terminal inference engine
├── eval_proof.py                      # Scientific benchmark verification suite
├── dataset_cookbook.py                # Universal dataset converter (Hugging Face / JSONL)
├── DATASET_INGESTION_COOKBOOK.md      # Dataset ingestion developer guide
├── ingest_knowledge.py                # Direct knowledge ingestion CLI
├── competitor_benchmark_colab.ipynb   # 7-Model competitor benchmark notebook
├── launch_terminal.bat                # 1-click Windows Terminal launcher
├── requirements.txt                   # Minimal inference dependencies
└── README.md                          # Hugging Face Model Card & Documentation
```

---

## 🔬 Special Token Routing Protocol

| Special Token | Function | Routed Component |
| :--- | :--- | :--- |
| `<think> ... </think>` | Cognitive reasoning & query deconstruction | 250M Neural Weights |
| `<|mem_query|> ... <|mem_query_end|>` | Hardware memory bus query | NVMe SSD FTS5 Engine (0.002ms) |
| `<|mem_payload|> ... <|mem_payload_end|>` | Grounded knowledge injection | Context Window |
| `<|calc|> ... <|calc_end|>` | Exact mathematical expression | AST SafeMath Evaluator (0.01ms) |
| `<|synthesize|>` | Final grounded response synthesis | Neural Generation Head |

---

## 📜 Authorship & Citation

AviGPT-250M-Instruct is an original architecture created, engineered, and trained exclusively by **Yadlapalli Avinash Ricky**.

```bibtex
@misc{ricky2026avigpt250minstruct,
  author = {Yadlapalli Avinash Ricky},
  title = {AviGPT-250M-Instruct: Semi-Parametric Edge Intelligence with Native NVMe Hardware Memory Bus},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/AvinashRicky/avigpt-250m-instruct}}
}
```
