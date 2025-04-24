import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import re

class Qdrant:

    def __init__(self):
        self.qdrant = QdrantClient("localhost", port=6333)
        self.model = SentenceTransformer('BAAI/bge-base-en-v1.5')


    def split_numbered_sections(self, text):
        pattern = re.compile(r"\((\d+)\)\s+")
        splits = pattern.split(text)
        chunks = []

        for i in range(1, len(splits), 2):
            section_number = splits[i]
            section_text = splits[i + 1].strip()
            chunks.append((section_number, section_text))

        return chunks

    def embed_gdpr(self):
        json_path = 'data/gdpr/gdpr_articles_recitals.jsonl'
        data = [json.loads(line) for line in open(json_path, 'r', encoding='utf-8')]

        chunk_texts = []
        chunk_titles = []

        for entry in data:
            title = entry['input-text']
            full_text = entry['output-text']

            if re.search(r"\(\d+\)", full_text):
                chunks = self.split_numbered_sections(full_text)
                for section_number, section_text in chunks:
                    chunk_titles.append(f"{title} - ({section_number})")
                    chunk_texts.append(section_text)
            else:
                chunk_titles.append(title)
                chunk_texts.append(full_text.strip())

        vectors = self.model.encode(chunk_texts, convert_to_numpy=True)

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
                for vec, title, text in zip(vectors, chunk_titles, chunk_texts)
            ]
        )
        print(f"Uploaded {len(chunk_texts)} GDPR chunks to Qdrant.")



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

    QdrantEmbedding = Qdrant()
    QdrantEmbedding.embed_gdpr()
    QdrantEmbedding.search_Qdrant()
    # QdrantEmbedding.delete_collection()