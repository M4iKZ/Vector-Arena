import numpy as np
import time
from .base import VectorDBEngine

try:
    from usearch.index import Index
except ImportError:
    Index = None

class USearchEngine(VectorDBEngine):
    """
    USearch Baseline Engine with Metadata Emulation.
    To make the baseline fair, we store metadata in a Python dict,
    similar to the Faiss implementation.
    """
    def __init__(self, dimension):
        super().__init__("USearch")
        self.dimension = dimension
        self.index = None
        self.doc_store = {} # Simulated Metadata Storage

    def initialize(self):
        if not Index:
            raise ImportError("USearch package not found. Install with 'pip install usearch'")
        
        self.index = Index(
            ndim=self.dimension,
            metric="l2sq",
            dtype="f32",
            connectivity=32,  # M=32 for HNSW
            expansion_search=128,  # HNSW efSearch parameter
        )
        self.doc_store = {}

    def insert(self, texts, vectors, metadatas):
        ids = np.array([meta['idx'] for meta in metadatas], dtype=np.uint64)
        v_copy = np.ascontiguousarray(vectors, dtype=np.float32)
        
        # Index the vectors
        self.index.add(ids, v_copy)
        
        # Store metadata (Simulated DB)
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            self.doc_store[ids[i]] = {
                "text": text,
                "meta": meta
            }

    def search(self, query_vector, k, filter_dict=None):
        # Ensure contiguous float32 for maximum precision/speed
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        
        if filter_dict:
            # Full scan for filtered baseline
            count = len(self.doc_store)
            matches = self.index.search(query, count)
            found_keys = matches.keys.flatten()
            
            results = []
            for key in found_keys:
                doc = self.doc_store.get(int(key))
                if not doc: continue
                match = all(doc['meta'].get(fk) == fv for fk, fv in filter_dict.items())
                if match:
                    results.append(doc['meta']['idx'])
                    if len(results) >= k: break
            return results
        
        # Standard Search with very high expansion for 100% recall
        # Some USearch versions use expansion_search in constructor, 
        # others allow it in search call. We'll set it very high in constructor.
        matches = self.index.search(query, k)
        return matches.keys.flatten().tolist()

    def search_batch(self, query_vectors, k, filter_dict=None):
        if filter_dict:
            return [self.search(q, k, filter_dict) for q in query_vectors]
        query = np.array(query_vectors).reshape(-1, self.dimension)
        matches = self.index.search(query, k)
        return [matches.keys[i].tolist() for i in range(len(query_vectors))]

    def cleanup(self):
        self.index = None
        self.doc_store = {}
