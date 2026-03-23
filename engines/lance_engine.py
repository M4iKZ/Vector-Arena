import lancedb
import pandas as pd
import shutil
import os
import time
import gc
from .base import VectorDBEngine

class LanceEngine(VectorDBEngine):
    def __init__(self, dimension):
        super().__init__("LanceDB")
        self.dimension = dimension
        self.db = None
        self.table = None
        self.db_path = os.path.join(os.getcwd(), f"lancedb_bench_{os.getpid()}")

    def initialize(self):
        self._release()
        self._remove_db_path()
        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)

    def _release(self):
        """Release Lance file handles before any filesystem operation."""
        if self.table is not None:
            self.table = None
        if self.db is not None:
            self.db = None
        gc.collect()
        # Give Lance's Rust async runtime time to flush and close file handles.
        # This is required on Windows where open handles block renames/deletes.
        time.sleep(0.5)

    def _remove_db_path(self):
        if not os.path.exists(self.db_path):
            return
        # Retry loop: Windows may still hold handles briefly after gc.collect().
        for attempt in range(5):
            try:
                shutil.rmtree(self.db_path)
                return
            except OSError:
                if attempt < 4:
                    time.sleep(0.3 * (attempt + 1))
        shutil.rmtree(self.db_path, ignore_errors=True)

    def insert(self, texts, vectors, metadatas):
        # Convert to Pandas for more efficient ingestion
        df = pd.DataFrame({
            "vector": list(vectors),
            "idx": [m['idx'] for m in metadatas],
            "text": texts,
            "category": [m['category'] for m in metadatas]
        })
        
        self.table = self.db.create_table("bench", data=df)
        self.table.create_scalar_index("category", index_type="BITMAP")
        
        self.table.create_index(
            index_type="IVF_HNSW_SQ",
            m=32,
            ef_construction=128,
            metric="l2",
            vector_column_name="vector",
            replace=True
        )
        # Ensure writes are flushed before any search/cleanup
        gc.collect()


    def search(self, query_vector, k, filter_dict=None):
        q = self.table.search(query_vector).limit(k).ef(128)
        if filter_dict:
            filter_str = " and ".join([f"{k}='{v}'" for k, v in filter_dict.items()])
            q = q.where(filter_str)
        res = q.to_pandas()
        return res['idx'].tolist()

    def cleanup(self):
        self._release()
        self._remove_db_path()
