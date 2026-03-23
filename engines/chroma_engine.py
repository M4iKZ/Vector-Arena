import chromadb
from .base import VectorDBEngine

class ChromaEngine(VectorDBEngine):
    def __init__(self, dimension):
        super().__init__("ChromaDB")
        self.dimension = dimension
        self.client = None
        self.collection = None

    def initialize(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="bench",
            metadata={
                "hnsw:space": "l2",
                "hnsw:M": 32,
                "hnsw:construction_ef": 128,
                "hnsw:search_ef": 128
            }
        )

    def insert(self, texts, vectors, metadatas):
        # ChromaDB has a limit on batch size (approx 5400)
        chunk_size = 5000
        for i in range(0, len(texts), chunk_size):
            end = min(i + chunk_size, len(texts))
            self.collection.add(
                embeddings=vectors[i:end].tolist(),
                ids=[str(m['idx']) for m in metadatas[i:end]],
                metadatas=metadatas[i:end],
                documents=texts[i:end]
            )

    def search(self, query_vector, k, filter_dict=None):
        where = filter_dict if filter_dict else None
        res = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k,
            where=where
        )
        return [m['idx'] for m in res['metadatas'][0]]

    def search_batch(self, query_vectors, k, filter_dict=None):
        where = filter_dict if filter_dict else None
        res = self.collection.query(
            query_embeddings=[q.tolist() for q in query_vectors],
            n_results=k,
            where=where
        )
        return [[m['idx'] for m in row] for row in res['metadatas']]

    def cleanup(self):
        pass
