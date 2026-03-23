# Vector Database Benchmark

This repository contains a comprehensive benchmark tool designed to evaluate the quality and performance of various vector search engines. It measures insertion speed, search latency (diverse, sequential, filtered, and bulk), recall accuracy, and memory usage using the **SIFT1M** dataset.

## Supported Engines

The benchmark supports both widely available open-source engines and specialized custom implementations:

### Standard Engines
- **ChromaDB**: Popular open-source vector database.
- **LanceDB**: Serverless vector database based on the Lance data format.
- **Qdrant**: High-performance vector search engine with an advanced filtering system.
- **FAISS**: Efficient similarity search and clustering of dense vectors.
- **USearch**: A smaller, faster HNSW implementation.

### Custom Engines (Internal/Private)
- **MeMo (`pymemo`)**: A custom embedded database, optimized for local vector storage and retrieval.
- **mSEARCH**: A modular HNSW vector search implementation.

> [!IMPORTANT]
> The custom engines (**MeMo** and **mSEARCH**) are in-development components. They are NOT included in the public repository and are available only upon request.

## Setup and Usage

### Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install -r vector_bench_requirements.txt
```

### Running the Benchmark
You can run the benchmark using the main script. By default, it indices 100,000 vectors.

```bash
python vector_bench.py --elements 100000
```

### Loading Custom Engines
To enable the custom engines, place their respective libraries/modules in a local directory named `pymemo` or `msearch` at the root of the benchmark directory:

```text
.
├── vector_bench.py
├── numerable/      <-- Place pymemo_embedded and msearch libraries here
└── sift/           <-- Place SIFT dataset files here (.fvecs, .ivecs)
```

The script will automatically detect and include these engines in the benchmark if found.

## Benchmark Results (Reference)

Below is a baseline performance comparison conducted with **10,000 vectors** and 100 queries.

| Database | Operation | Time (ms) | Ops/sec | Recall@K (%) | p95 (ms) | Total Mem (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ChromaDB** | Insert/Index | 2045.51 | 4,888.77 | - | - | 348.7 |
| ChromaDB | Search (Scenario 1: 100 diverse) | 109.86 | 910.29 | 100.0% | 0.99 | |
| ChromaDB | Search (Scenario 2: Sequential×100) | 83.72 | 1,194.51 | 100.0% | 0.99 | |
| ChromaDB | Search (Scenario 3: Filtered×100) | 507.03 | 197.23 | 100.0% | 5.40 | |
| ChromaDB | Bulk Search (Scenario 4: 100×batch) | 315.79 | 3,166.62 | 100.0% | 0.33 | |
| **LanceDB** | Insert/Index | 361.89 | 27,632.38 | - | - | 318.7 |
| LanceDB | Search (Scenario 1: 100 diverse) | 456.99 | 218.82 | 98.9% | 5.30 | |
| LanceDB | Search (Scenario 2: Sequential×100) | 383.82 | 260.54 | 100.0% | 4.39 | |
| LanceDB | Search (Scenario 3: Filtered×100) | 506.71 | 197.35 | 99.7% | 5.69 | |
| LanceDB | Bulk Search (Scenario 4: 100×batch) | 3,533.12 | 283.04 | 98.9% | 3.60 | |
| **Qdrant** | Insert/Index | 876.3 | 11,411.63 | - | - | 324.6 |
| Qdrant | Search (Scenario 1: 100 diverse) | 615.94 | 162.35 | 100.0% | 6.49 | |
| Qdrant | Search (Scenario 2: Sequential×100) | 619.96 | 161.3 | 100.0% | 6.77 | |
| Qdrant | Search (Scenario 3: Filtered×100) | 7,408.33 | 13.5 | 100.0% | 80.58 | |
| Qdrant | Bulk Search (Scenario 4: 100×batch) | 6,492.7 | 154.02 | 100.0% | 6.92 | |
| **MeMo** | Insert/Index | 83.59 | 119,637.12 | - | - | 270.8 |
| MeMo | Search (Scenario 1: 100 diverse) | 11.61 | 8,614.82 | 99.7% | 0.14 | |
| MeMo | Search (Scenario 2: Sequential×100) | 5.13 | 19,493.56 | 100.0% | 0.07 | |
| MeMo | Search (Scenario 3: Filtered×100) | 112.21 | 891.22 | 99.7% | 1.37 | |
| MeMo | Bulk Search (Scenario 4: 100×batch) | 14.92 | 67,036.71 | 99.7% | 0.02 | |
| **FAISS** | Insert/Index | 126.96 | 78,762.36 | - | - | 262.7 |
| FAISS | Search (Scenario 1: 100 diverse) | 15.49 | 6,454.07 | 100.0% | 0.22 | |
| FAISS | Search (Scenario 2: Sequential×100) | 8.25 | 12,122.83 | 100.0% | 0.12 | |
| FAISS | Search (Scenario 3: Filtered×100) | 15.34 | 6,519.41 | 100.0% | 0.18 | |
| FAISS | Bulk Search (Scenario 4: 100×batch) | 21.52 | 46,473.8 | 100.0% | 0.03 | |
| **USearch** | Insert/Index | 249.27 | 40,117.95 | - | - | 262.8 |
| USearch | Search (Scenario 1: 100 diverse) | 15.66 | 6,386.67 | 100.0% | 0.25 | |
| USearch | Search (Scenario 2: Sequential×100) | 9.69 | 10,321.3 | 100.0% | 0.15 | |
| USearch | Search (Scenario 3: Filtered×100) | 741.74 | 134.82 | 99.9% | 9.21 | |
| USearch | Bulk Search (Scenario 4: 100×batch) | 36.58 | 27,340.56 | 100.0% | 0.05 | |
| **mSEARCH** | Insert/Index | 130.31 | 76,738.43 | - | - | 264.9 |
| mSEARCH | Search (Scenario 1: 100 diverse) | 9.6 | 10,421.01 | 99.8% | 0.14 | |
| mSEARCH | Search (Scenario 2: Sequential×100) | 5.41 | 18,481.21 | 100.0% | 0.06 | |
| mSEARCH | Search (Scenario 3: Filtered×100) | 4.25 | 23,508.39 | 99.9% | 0.06 | |
| mSEARCH | Bulk Search (Scenario 4: 100×batch) | 35.68 | 28,027.14 | 99.8% | 0.04 | |

## Methodology
- **Dataset**: SIFT1M (128 dimensions). 
    - Download link used in this benchmark: [SIFT1M (sift.tar.gz)](ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz)
- **Isolation**: Each engine is executed in a fresh subprocess to ensure accurate memory (RSS) and resource tracking.
- **Recall Calculation**: Calculated dynamically against the current subset of the dataset (Recalculated Ground Truth).
- **Latency**: Measured for single queries, sequential repeats, filtered searches, and bulk batches.
