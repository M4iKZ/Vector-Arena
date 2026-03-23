import os
import sys
import time
import numpy as np
from .base import VectorDBEngine

# MeMo Import setup
# Add search paths for the MeMo library (pymemo)
# Check local ./numerable/ folder (relative to execution dir)
_LOCAL_MEMO = os.path.abspath(os.path.join(os.getcwd(), "numerable"))

if os.path.isdir(_LOCAL_MEMO) and _LOCAL_MEMO not in sys.path:
    sys.path.insert(0, _LOCAL_MEMO)
    # Also check for Release/Debug subfolders
    for sub in ["Release", "Debug"]:
        sp = os.path.join(_LOCAL_MEMO, sub)
        if os.path.isdir(sp):
            sys.path.insert(0, sp)

try:
    import pymemo_embedded as memo
except ImportError:
    memo = None

class MeMoEngine(VectorDBEngine):
    def __init__(self, dimension):
        super().__init__("MeMo")
        self.dimension = dimension
        self.engine = None
        self.coll = "bench.vec_bench"

    def initialize(self):
        if not memo:
            raise ImportError("MeMo package not found")
        self.engine = memo.create_engine()
        self.engine.initialize()
        
        # Performance tuning: Disable global auto-indexing (which indexes EVERYTHING)
        # and manually create only the indexes we need.
        self.engine.set_auto_indexing_enabled(False)
        self.engine.create_database("bench")
        self.engine.create_collection(self.coll)
        #self.engine.create_index(self.coll, "category")
        
        # Using the NEW typed Configuration
        config = memo.VectorIndexConfig()
        config.metric = memo.VectorMetric.L2SQ
        config.connectivity = 32  # M=32 for HNSW
        config.expansion_search = 128  # HNSW efSearch parameter
        
        self.engine.create_vector_index(self.coll, "vec", self.dimension, config)

    def insert(self, texts, vectors, metadatas):
        # Optimized ID extraction (single pass)
        ids = np.fromiter((m['idx'] for m in metadatas), dtype=np.uint64, count=len(metadatas))
        
        # FAST LANE: Enrich the existing metadata list in-place 
        # instead of recreating 100k dictionaries from scratch.
        for m, t in zip(metadatas, texts):
            m['text'] = t
        
        # Bulk API handles parallel indexing internally
        self.engine.insert_vectors(self.coll, "vec", ids, vectors, metadatas)

    def search(self, query_vector, k, filter_dict=None):
        if filter_dict:
            # INTEGRATED PRE-FILTERING (The "New System" Bridge)
            category = filter_dict.get("category")
            filter_q = memo.create_query().where("category", category)
            return self.engine.find_vector_ids(self.coll, "vec", query_vector, k, filter_q)
        else:
            return self.engine.find_vector_ids(self.coll, "vec", query_vector, k)

    def search_batch(self, query_vectors, k, filter_dict=None):
        """
        Parallel Batch Search (Native C++ Implementation)
        - GIL is released for the entire search.
        - Uses global MeMo thread pool for query parallelization.
        - Integrated pre-filtering (if applicable).
        """
        query_matrix = np.array(query_vectors, dtype=np.float32)

        if filter_dict:
            category = filter_dict.get("category")
            filter_q = memo.create_query().where("category", category)
            results = self.engine.find_vector_ids_batch(self.coll, "vec", query_matrix, k, filter_q)
        else:
            results = self.engine.find_vector_ids_batch(self.coll, "vec", query_matrix, k)
        
        # results is a list of numpy arrays; convert each to list for benchmark compatibility
        return [res.tolist() if hasattr(res, 'tolist') else res for res in results]

    def cleanup(self):
        if self.engine:
            self.engine.shutdown()
