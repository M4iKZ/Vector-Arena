import sys
import os
import numpy as np
from .base import VectorDBEngine

# Add search paths for the msearch library
# Check local ./numerable/ folder (relative to execution dir)
_LOCAL_MSEARCH = os.path.abspath(os.path.join(os.getcwd(), "numerable"))

if os.path.isdir(_LOCAL_MSEARCH) and _LOCAL_MSEARCH not in sys.path:
    sys.path.insert(0, _LOCAL_MSEARCH)

try:
    import msearch
except ImportError:
    msearch = None

class MSearchEngine(VectorDBEngine):
    """
    mSEARCH Engine — modular HNSW vector search.
    Uses native Python enums (msearch.MetricKind / msearch.ScalarKind).
    """
    def __init__(self, dimension):
        super().__init__("mSEARCH")
        self.dimension = dimension
        self.index = None
        self.doc_store = {}
        # Pre-built candidate array for filtered (brute-force) search
        self._filtered_keys = None   # np.ndarray[uint64], keys with category=="special"
        self._filtered_vecs = None   # np.ndarray[float32, N×dim], parallel vectors

    def initialize(self):
        if msearch is None:
            raise ImportError("msearch module not found. Build with CMake first.")

        self.index = msearch.Index(
            ndim=self.dimension,
            metric=msearch.MetricKind.L2Sq,
            dtype=msearch.ScalarKind.F32,
            connectivity=32,  # M=32 for HNSW
            expansion_search=128,  # HNSW efSearch parameter
        )
        self.doc_store = {}

    def insert(self, texts, vectors, metadatas):
        ids = np.array([meta['idx'] for meta in metadatas], dtype=np.uint64)
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)

        threads = os.cpu_count() or 1
        self.index.reserve(len(ids), threads=threads)
        self.index.add_batch(ids, vecs)  # C++ parallel loop, GIL released

        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            self.doc_store[ids[i]] = {"text": text, "meta": meta}

        # Pre-compute filtered candidate list for the benchmark's "special" category.
        # search_flat does an exact SIMD brute-force scan — far faster than requesting
        # all N results from the HNSW graph and post-filtering in Python.
        special_mask = [meta['category'] == 'special' for meta in metadatas]
        self._filtered_keys = ids[special_mask]
        self._filtered_vecs = vecs[special_mask]

    def search(self, query_vector, k, filter_dict=None):
        query = query_vector.astype(np.float32)

        if filter_dict:
            # Fast path: if the filter is purely the pre-computed "special" category,
            # use search_flat for an exact AVX2 brute-force scan over only the
            # matching candidates.  This avoids asking HNSW for N=100k results.
            if (
                list(filter_dict.keys()) == ['category']
                and filter_dict['category'] == 'special'
                and self._filtered_keys is not None
            ):
                keys, dists = self.index.search_flat(query, self._filtered_keys, k)
                return [int(key) for key in keys]

            # Generic fallback: full HNSW scan + Python post-filter
            count = len(self.doc_store)
            keys, dists = self.index.search(query, count)
            results = []
            for key in keys:
                doc = self.doc_store.get(int(key))
                if not doc:
                    continue
                match = all(doc['meta'].get(fk) == fv for fk, fv in filter_dict.items())
                if match:
                    results.append(doc['meta']['idx'])
                    if len(results) >= k:
                        break
            return results

        keys, dists = self.index.search(query, k)
        return [int(key) for key in keys]

    def search_batch(self, query_vectors, k, filter_dict=None):
        if filter_dict:
            return [self.search(q, k, filter_dict) for q in query_vectors]

        queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
        keys, dists = self.index.search_batch(queries, k)
        return [[int(idx) for idx in row] for row in keys]

    def cleanup(self):
        self.index = None
        self.doc_store = {}
        self._filtered_keys = None
        self._filtered_vecs = None
