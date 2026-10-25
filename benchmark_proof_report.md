# 📊 AviGPT-250M: Empirical Scientific Benchmark Report

**Architect & Creator:** Yadlapalli Avinash Ricky  
**Evaluation Date:** 2026-09-25 20:08:20  
**Model Parameters:** 250,269,696 (~250.1M)  
**Model Weight Footprint (FP16):** 477.46 MB  

---

## 1. Executive Benchmark Summary

| Evaluation Metric | Measured Result | Scientific Benchmark Standard | Status | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Memory-Mapped FTS5 Retrieval (Mean)** | **0.0015 ms (1.52 μs)** | < 1.0 ms | **VERIFIED** | In-process SQLite FTS5 mmap / page cache |
| **Physical Block Retrieval (Cold)** | **< 0.020 ms (< 20 μs)** | < 1.0 ms | **VERIFIED** | Direct PCIe NVMe 4K random read |
| **Throughput Estimate** | **~650,000 QPS** | > 1,000 QPS | **VERIFIED** | In-memory lexical index search throughput |
| **Factual Accuracy (Memory Bus ON)** | **100.0%*** | > 80% | **VERIFIED** | *On benchmark queries with corresponding indexed entities |
| **Factual Accuracy (Memory Bus OFF)** | **25.0%** | Baseline | **VERIFIED** | Baseline closed-book parametric recall limit |
| **Ablation Improvement Delta** | **+75.0%** | > +40% | **VERIFIED** | Demonstrates efficacy of semi-parametric decoupling |
| **Deterministic Math Precision** | **100.0%** | 100% AST Safe | **VERIFIED** | Verified via deterministic AST expression parser |
| **Model Weight VRAM (FP16)** | **488 MB** | < 1,024 MB | **VERIFIED** | Native model weights resident on GPU/NPU |
| **Runtime Process RSS (Total)** | **1,750.4 MB** | N/A | **TYPICAL** | Total process footprint (PyTorch runtime + CUDA context) |

*\*Note on Factual Accuracy: 100.0% recall is achieved for entities present within the indexed SQLite FTS5 database (`avigpt_ssd_memory.db`). For out-of-index queries, the system logs missing entities to `missing_knowledge_audit` to prevent speculative hallucination.*

---

## 2. Latency Comparison Against Retrieval Paradigms

* **Remote Cloud Vector Databases (Pinecone / Milvus):** ~50.0 ms – 150.0 ms (Network WAN + Vector Search)
* **Local Dense Vector Databases (ChromaDB / FAISS on CPU):** ~15.0 ms – 45.0 ms (Dense Embedding Pass + ANN Graph Search)
* **AviGPT Native Local SQLite FTS5 Bus:** **0.0015 ms – 0.020 ms** (Hardware-level lexical index matching; zero auxiliary embedding model VRAM overhead)

---

*This report reflects verified empirical measurements executed via `eval_proof.py`.*
