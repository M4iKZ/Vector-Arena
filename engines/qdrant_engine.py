import warnings
import qdrant_client
from qdrant_client.http import models as qmodels
from .base import VectorDBEngine

class QdrantEngine(VectorDBEngine):
    def __init__(self, dimension):
        super().__init__("Qdrant")
        self.dimension = dimension
        self.client = None

    def initialize(self):
        self.client = qdrant_client.QdrantClient(":memory:")
        self.client.create_collection(
            "bench",
            vectors_config=qmodels.VectorParams(size=self.dimension, 
                                                distance=qmodels.Distance.EUCLID,
                                                hnsw_config=qmodels.HnswConfigDiff(
                                                    m=32,             # M=32 for HNSW
                                                    ef_construct=128, # HNSW efConstruction parameter
                                                    full_scan_threshold=10000
                                                ))
        )

    def insert(self, texts, vectors, metadatas):
        points = []
        for i, (vec, meta) in enumerate(zip(vectors, metadatas)):
            points.append(qmodels.PointStruct(
                id=meta['idx'],
                vector=vec.tolist(),
                payload=meta
            ))
        self.client.upsert("bench", points=points)
        # Payload indexes only work in server mode; suppress the warning in local/:memory: mode
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.client.create_payload_index("bench", field_name="category", field_schema=qmodels.PayloadSchemaType.KEYWORD)

    def search(self, query_vector, k, filter_dict=None):
        query_filter = None
        if filter_dict:
            must = []
            for key, val in filter_dict.items():
                must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=val)))
            query_filter = qmodels.Filter(must=must)

        response = self.client.query_points(
            "bench",
            query=query_vector.tolist(),
            limit=k,
            query_filter=query_filter,
            with_payload=True
        )
        res = response.points if response is not None else []
        return [p.payload['idx'] for p in res if p.payload]

    def search_batch(self, query_vectors, k, filter_dict=None):
        query_filter = None
        if filter_dict:
            must = []
            for key, val in filter_dict.items():
                must.append(qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=val)))
            query_filter = qmodels.Filter(must=must)

        requests = [
            qmodels.QueryRequest(
                query=qv.tolist(),
                limit=k,
                filter=query_filter,
                with_payload=True
            )
            for qv in query_vectors
        ]
        # query_batch_points returns a list of results, each having a 'points' attribute
        responses = self.client.query_batch_points("bench", requests=requests) or []
        return [[p.payload['idx'] for p in resp.points if p.payload] for resp in responses]

    def cleanup(self):
        pass
