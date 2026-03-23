import time
import psutil
import numpy as np

class VectorDBEngine:
    def __init__(self, name):
        self.name = name
        self.process = psutil.Process()

    def get_memory_mb(self):
        return self.process.memory_info().rss / (1024 * 1024)

    def initialize(self):
        """Setup the database, collections, etc."""
        pass

    def insert(self, texts, vectors, metadatas):
        """Insert records into the database."""
        pass

    def search(self, query_vector, k, filter_dict=None):
        """Search the database. Should return a list of indices (integers)."""
        pass

    def search_batch(self, query_vectors, k, filter_dict=None):
        """Batch search. Default: sequential loop over search(). Override for native batch support."""
        return [self.search(q, k, filter_dict) for q in query_vectors]

    def cleanup(self):
        """Cleanup resources, delete temp files, etc."""
        pass
