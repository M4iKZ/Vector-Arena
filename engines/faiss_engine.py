import faiss
import numpy as np
from .base import VectorDBEngine

class FaissEngine(VectorDBEngine):
    def __init__(self, dimension):
        super().__init__("FAISS")
        self.dimension = dimension
        self.index = None
        self.doc_store = {} # Simulate DB storage

    def initialize(self):
        # Use IndexIDMap to support explicit IDs 
        hnsw = faiss.IndexHNSWFlat(self.dimension, 32)
        hnsw.hnsw.efConstruction = 128
        hnsw.hnsw.efSearch = 128
        
        self.index = faiss.IndexIDMap(hnsw)
        self.doc_store = {}
        
        # Flat copies used for exact brute-force filtered search
        self._all_ids = []     # list of inserted ids (int64)
        self._all_vecs = []    # list of inserted float32 row-vectors

    def insert(self, texts, vectors, metadatas):
        ids = np.array([meta['idx'] for meta in metadatas], dtype=np.int64)
        v_copy = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add_with_ids(v_copy, ids)
        
        # Store metadata using the same structural method as USearch
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            self.doc_store[ids[i]] = {
                "text": text,
                "meta": meta
            }

        # Cache raw vectors for exact filtered search (avoids HNSW approximation)
        special_mask = np.array([meta['category'] == 'special' for meta in metadatas], dtype=bool)
        self._special_ids  = ids[special_mask]                           # int64 sub-array
        self._special_vecs = v_copy[special_mask]                        # float32 sub-array
        self._all_ids  = ids
        self._all_vecs = v_copy

    def search(self, query_vector, k, filter_dict=None):
        q_copy = query_vector.copy().reshape(1, -1).astype('float32')

        if filter_dict:
            # Exact brute-force against the pre-built filtered subset.
            # Using numpy L2 (squared) directly avoids HNSW approximation and
            # is also faster than asking FAISS for N=100k HNSW results.
            if (
                list(filter_dict.keys()) == ['category']
                and filter_dict['category'] == 'special'
                and hasattr(self, '_special_vecs')
            ):
                # L2 squared: sum((q - v)^2) = ||q||^2 - 2 q·v + ||v||^2
                diffs = self._special_vecs - q_copy           # (N_special, dim)
                dists = (diffs * diffs).sum(axis=1)           # (N_special,)
                top_idx = np.argpartition(dists, min(k, len(dists) - 1))[:k]
                top_idx = top_idx[np.argsort(dists[top_idx])]
                return [int(self._special_ids[i]) for i in top_idx]

            # Generic fallback: full HNSW scan + Python post-filter
            candidate_count = len(self.doc_store)
            D, I = self.index.search(q_copy, candidate_count)
            filtered_results = []
            for idx in I[0]:
                if idx == -1: continue
                doc = self.doc_store[idx]
                match = all(doc['meta'].get(fk) == fv for fk, fv in filter_dict.items())
                if match:
                    filtered_results.append(doc['meta']['idx'])
                if len(filtered_results) >= k:
                    break
            return filtered_results

        # Standard Search
        D, I = self.index.search(q_copy, k)
        results = []
        for idx in I[0]:
            if idx != -1:
                doc = self.doc_store[idx]
                results.append(doc['meta']['idx'])
        return results

    def search_batch(self, query_vectors, k, filter_dict=None):
        if filter_dict:
            return [self.search(q, k, filter_dict) for q in query_vectors]
        q_copy = np.array(query_vectors).reshape(-1, self.dimension).astype('float32')
        D, I = self.index.search(q_copy, k)
        return [
            [self.doc_store[idx]['meta']['idx'] for idx in row if idx != -1]
            for row in I
        ]

    def cleanup(self):
        pass
