import warnings
import sys
import os

# Suppress annoying library warnings BEFORE any other imports
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*SwigPy.*")
warnings.filterwarnings("ignore", message=".*swigvarlink.*")
warnings.filterwarnings("ignore", module="pydantic")
warnings.filterwarnings("ignore", module="marqo")

# Force silence for environment
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import multiprocessing as mp
import threading
import time
import numpy as np
import pandas as pd
import psutil
if sys.platform == "win32":
    import msvcrt

# Force silence for Swig internal types which trigger before filter takes effect in some envs
if not sys.warnoptions:
    os.environ["PYTHONWARNINGS"] = "ignore"

# --- Add Local Paths for Custom Engines ---
# Custom/private engine libraries are expected directly in the ./numerable/ folder
abs_numerable = os.path.abspath("./numerable")
if os.path.isdir(abs_numerable) and abs_numerable not in sys.path:
    sys.path.insert(0, abs_numerable)

from tabulate import tabulate
from sklearn.metrics.pairwise import euclidean_distances

# --- IMPORT ENGINES ---
# Standard Engines
from engines.chroma_engine import ChromaEngine
from engines.lance_engine import LanceEngine
from engines.qdrant_engine import QdrantEngine
from engines.usearch_engine import USearchEngine
from engines.faiss_engine import FaissEngine

# Optional Custom Engines (Available upon request)
try:
    from engines.memo_engine import MeMoEngine
    _HAS_MEMO = True
except ImportError:
    _HAS_MEMO = False

try:
    from engines.msearch_engine import MSearchEngine
    _HAS_MSEARCH = True
except ImportError:
    _HAS_MSEARCH = False

# --- CONFIGURATION ---
# The benchmark looks for the SIFT dataset in a local 'sift' folder
SIFT_DIR = os.path.abspath(os.path.join(os.getcwd(), "sift"))
DIMENSION = 128
DEFAULT_DOCS = 100000
TOP_K = 10
NUM_QUERIES = 100
BATCH_SIZE = 100

# --- DATA LOADING UTILS ---
def load_fvecs(filename):
    """Read .fvecs file into numpy array."""
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        vecs = np.fromfile(f, dtype=np.float32)
        return vecs.reshape(-1, dim + 1)[:, 1:]

def load_ivecs(filename):
    """Read .ivecs file into numpy array."""
    with open(filename, 'rb') as f:
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        vecs = np.fromfile(f, dtype=np.int32)
        return vecs.reshape(-1, dim + 1)[:, 1:]

# --- ISOLATED BENCHMARK WORKER ---
def run_engine_benchmark(engine_class, engine_args, texts, vectors, metadatas, query_vectors, all_ground_truths, all_filtered_ground_truths, top_k, result_queue):
    """Function to be run in a separate process for complete isolation."""
    engine = engine_class(*engine_args)
    results = []
    
    def record(op, start_time, iterations=1, latencies=None, avg_recall=None):
        duration_ms = (time.perf_counter() - start_time) * 1000
        p95 = np.percentile(latencies, 95) * 1000 if latencies else 0
        results.append({
            "Operation": op,
            "Time (ms)": round(duration_ms, 2),
            "Ops/sec": round(iterations / (duration_ms / 1000), 2) if duration_ms > 0 else 0,
            "Recall@K (%)": f"{avg_recall:.1f}%" if avg_recall is not None else "-",
            "p95 (ms)": f"{p95:.2f}" if p95 > 0 else "-"
        })

    def recall_at_k(found_ids, ground_truth_set):
        found = set(int(i) for i in found_ids)
        return (len(found & ground_truth_set) / len(ground_truth_set) * 100) if ground_truth_set else 0

    try:
        engine.initialize()

        # 1. Insert
        start = time.perf_counter()
        engine.insert(texts, vectors, metadatas)
        record("Insert/Index", start, len(vectors))

        # 2. Diverse Search
        num_queries = len(query_vectors)
        latencies, recalls = [], []
        start_search = time.perf_counter()
        for qi, qv in enumerate(query_vectors):
            s = time.perf_counter()
            res_ids = engine.search(qv, top_k)
            latencies.append(time.perf_counter() - s)
            recalls.append(recall_at_k(res_ids, all_ground_truths[qi]))
        record(f"Search (Scenario 1: {num_queries} diverse)", start_search, num_queries, latencies, np.mean(recalls))

        # 3. Sequential Search
        qv0, gt0 = query_vectors[0], all_ground_truths[0]
        latencies = []
        start_search = time.perf_counter()
        for _ in range(100):
            s = time.perf_counter()
            res_ids = engine.search(qv0, top_k)
            latencies.append(time.perf_counter() - s)
        record("Search (Scenario 2: Sequential×100)", start_search, 100, latencies, recall_at_k(res_ids, gt0))

        # 4. Filtered Search
        latencies, filt_recalls = [], []
        start_search = time.perf_counter()
        for qi, qv in enumerate(query_vectors):
            s = time.perf_counter()
            filt_ids = engine.search(qv, top_k, filter_dict={"category": "special"})
            latencies.append(time.perf_counter() - s)
            filt_recalls.append(recall_at_k(filt_ids, all_filtered_ground_truths[qi]))
        record(f"Search (Scenario 3: Filtered×{num_queries})", start_search, num_queries, latencies, np.mean(filt_recalls))

        # 5. Bulk Search
        batch_size = 100
        batch_qv = query_vectors[:batch_size]
        batch_gts = all_ground_truths[:batch_size]
        latencies = []
        start_bulk = time.perf_counter()
        bulk_results = None
        for _ in range(10):
            s = time.perf_counter()
            bulk_results = engine.search_batch(batch_qv, top_k)
            latencies.append((time.perf_counter() - s) / batch_size)
        
        batch_recalls = [recall_at_k(res_ids, batch_gts[ri]) for ri, res_ids in enumerate(bulk_results)] if bulk_results else []
        record(f"Bulk Search (Scenario 4: {batch_size}×batch)", start_bulk, 10 * batch_size, latencies, np.mean(batch_recalls) if batch_recalls else 0)

        engine.cleanup()
        result_queue.put(results)
    except Exception as e:
        result_queue.put(f"Error: {e}")

def monitor_memory(pid, peak_rss):
    """Monitor peak RSS of a specific process."""
    try:
        p = psutil.Process(pid)
        while p.is_running():
            try:
                rss = p.memory_info().rss / (1024 * 1024)
                if rss > peak_rss[0]:
                    peak_rss[0] = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(0.05)
    except psutil.NoSuchProcess:
        pass

# --- MAIN BENCHMARK FUNCTION ---

def run_benchmark():
    parser = argparse.ArgumentParser(description="Vector Database Quality & Performance Benchmark")
    parser.add_argument("--elements", type=int, default=DEFAULT_DOCS, help=f"Number of vectors to index (default: {DEFAULT_DOCS})")
    args = parser.parse_args()

    num_docs = args.elements
    results = []

    print(f"Loading SIFT1M dataset from {SIFT_DIR}...")
    base_file = os.path.join(SIFT_DIR, "sift_base.fvecs")
    query_file = os.path.join(SIFT_DIR, "sift_query.fvecs")
    gt_file = os.path.join(SIFT_DIR, "sift_groundtruth.ivecs")

    if not all(os.path.exists(f) for f in [base_file, query_file, gt_file]):
        print(f"Error: SIFT files not found in {SIFT_DIR}")
        return

    # SIFT1M Loading
    vectors = load_fvecs(base_file).astype(np.float32)
    query_vectors = load_fvecs(query_file).astype(np.float32)

    # 1. Flexible Subsetting: Pick a subset of vectors for speed
    print(f"Subsetting dataset to {num_docs} vectors and {NUM_QUERIES} queries...")
    vectors = vectors[:num_docs]
    query_vectors = query_vectors[:NUM_QUERIES]

    # Create fake text/metadata for engines that require them
    print(f"Preparing {len(vectors)} documents...")
    texts = [f"SIFT Document {i}" for i in range(len(vectors))]
    metadatas = []
    for i in range(len(vectors)):
        metadatas.append({
            "idx": i,
            "category": "special" if i % 10 == 0 else "general",
            "topic": "SIFT"
        })

    # 2. Dynamic Ground Truth Calculation
    # Since we are using a subset, we CANNOT use the pre-baked .ivecs file.
    # We recalculate Ground Truth in Python for the current subset.
    print(f"Calculating Top-{TOP_K} Ground Truth for {NUM_QUERIES} queries...")
    
    # Standard Search GT
    all_dists_std = euclidean_distances(query_vectors, vectors)
    all_ground_truths = []
    for dists in all_dists_std:
        top_idx = np.argsort(dists)[:TOP_K]
        all_ground_truths.append(set(int(i) for i in top_idx))

    # Filtered Search GT
    print("Calculating Filtered Ground Truth...")
    special_indices = [i for i, m in enumerate(metadatas) if m['category'] == "special"]
    special_vectors = vectors[special_indices]
    
    all_dists_filt = euclidean_distances(query_vectors, special_vectors)
    all_filtered_ground_truths = []
    for dists in all_dists_filt:
        top_idx = np.argsort(dists)[:TOP_K]
        # Map back to original IDs
        fgt = set(int(metadatas[special_indices[j]]['idx']) for j in top_idx)
        all_filtered_ground_truths.append(fgt)
    engine_configs = [
        (ChromaEngine, (DIMENSION,), "ChromaDB"),
        (LanceEngine, (DIMENSION,), "LanceDB"),
        (QdrantEngine, (DIMENSION,), "Qdrant"),   
    ]

    if _HAS_MEMO:
        engine_configs.append((MeMoEngine, (DIMENSION,), "MeMo"))
         
    engine_configs.extend([
        (FaissEngine, (DIMENSION,), "FAISS"),
        (USearchEngine, (DIMENSION,), "USearch"),
    ])

    if _HAS_MSEARCH:
        engine_configs.append((MSearchEngine, (DIMENSION,), "mSEARCH"))

    for engine_class, engine_args, engine_name in engine_configs:
        print(f"\nBenchmarking {engine_name}")

        result_queue = mp.Queue()
        p = mp.Process(target=run_engine_benchmark, args=(
            engine_class, engine_args, texts, vectors, metadatas,
            query_vectors, all_ground_truths, all_filtered_ground_truths,
            TOP_K, result_queue
        ))

        peak_rss = [0.0]
        p.start()

        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_memory, args=(p.pid, peak_rss))
        monitor_thread.start()

        p.join()
        monitor_thread.join()

        try:
            engine_results = result_queue.get(timeout=1)
            if isinstance(engine_results, str):
                print(f"  {engine_results}")
                continue

            for res in engine_results:
                res["Database"] = engine_name
                # Append peak memory only to the first operation record (Insertion)
                if res["Operation"] == "Insert/Index":
                    res["Total Mem (MB)"] = round(peak_rss[0], 1)
                else:
                    res["Total Mem (MB)"] = ""
                results.append(res)
        except Exception as e:
            print(f"  Failed to get results for {engine_name}: {e}")

    # --- FINAL SUMMARY ---
    df = pd.DataFrame(results)
    # Reorder columns: Database on the left
    cols = ["Database", "Operation", "Time (ms)", "Ops/sec", "Recall@K (%)", "p95 (ms)", "Total Mem (MB)"]
    df = df[cols]
    
    print("\n" + "="*100)
    print("                    VECTOR DATABASE QUALITY & PERFORMANCE BENCHMARK")
    print("="*100)
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
    print(f"\nRecall@{TOP_K}: |ANN ∩ GT| / K, averaged over {NUM_QUERIES} queries (ann-benchmarks methodology)")
    print("Ground truth: Official SIFT10K L2 Top-K" if gt_file else "")
    print("Isolation: Each engine runs in a fresh subprocess to ensure clean memory metrics")
    print(f"Total Mem: Peak Resident Set Size (RSS) observed for the engine process")
    print(f"Dataset: SIFT1M ({len(vectors)} docs, {DIMENSION} dimensions, K={TOP_K})")

if __name__ == "__main__":
    run_benchmark()
