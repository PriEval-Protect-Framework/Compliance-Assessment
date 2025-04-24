import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

class QdrantEmbedding:

    def __init__(self):
        self.qdrant = QdrantClient("localhost", port=6333)
        self.model = SentenceTransformer('BAAI/bge-base-en-v1.5')


    def embed_gdpr(self):
        json_path = 'data/gdpr/gdpr_articles_recitals.jsonl'
        data = [json.loads(line) for line in open(json_path, 'r', encoding='utf-8')]

        texts = [entry['output-text'] for entry in data]
        titles = [entry['input-text'] for entry in data]
        vectors = self.model.encode(texts, convert_to_numpy=True)


        collection_name = "gdpr_articles"
        if not self.qdrant.collection_exists(collection_name):
            
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE)
            )

            self.qdrant.upload_points(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec.tolist(),
                        payload={"article": title, "text": text}
                    )
                    for vec, title, text in zip(vectors, titles, texts)
                ]
            )
            print(f"Uploaded {len(texts)} GDPR articles to Qdrant.")


    def search_Qdrant(self):

        query = "What are the principles for lawful data processing?"
        query_vector = self.model.encode(query).tolist()

        search_results = self.qdrant.query_points(
            collection_name="gdpr_articles",
            query=query_vector,
            limit=3,
            with_payload=True
        ).points

        for point in search_results:
            score = point.score
            payload = point.payload
            print(f"Score: {score}, Article: {payload['article']}, Text: {payload['text']}")
            print("-" * 80)

    def delete_collection(self, collection_name="gdpr_articles"):
        
        if self.qdrant.collection_exists(collection_name):
            self.qdrant.delete_collection(collection_name)
            print(f"Deleted collection: {collection_name}")
        else:
            print(f"Collection {collection_name} does not exist.")


if __name__ == "__main__":

    QdrantEmbedding = QdrantEmbedding()
    QdrantEmbedding.embed_gdpr()
    QdrantEmbedding.search_Qdrant()
    # QdrantEmbedding.delete_collection()