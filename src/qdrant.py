import json
import os
import re
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

class Qdrant:

    def __init__(self):
        # Use env variables if available (supporting docker + local)
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))

        # Strip protocol if needed (e.g. http://qdrant)
        host = host.replace("http://", "").replace("https://", "").split(":")[0]

        self.qdrant = QdrantClient(host=host, port=port)
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

    def embed_gdpr(self, skip_existing=True):
        json_path = "../data/gdpr/gdpr_articles_recitals.jsonl"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"GDPR source file not found at {json_path}")

        data = [json.loads(line) for line in open(json_path, 'r', encoding='utf-8')]

        chunk_texts = []
        chunk_titles = []
        chunk_ids = []

        for entry in data:
            title = entry['input-text']
            full_text = entry['output-text']

            if re.search(r"\(\d+\)", full_text):
                chunks = self.split_numbered_sections(full_text)
                for section_number, section_text in chunks:
                    chunk_title = f"{title} - ({section_number})"
                    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk_title}:{section_text}"))
                    chunk_titles.append(chunk_title)
                    chunk_texts.append(section_text)
                    chunk_ids.append(chunk_id)
            else:
                chunk_title = title
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk_title}:{full_text.strip()}"))
                chunk_titles.append(chunk_title)
                chunk_texts.append(full_text.strip())
                chunk_ids.append(chunk_id)

        if skip_existing and self.qdrant.collection_exists("gdpr_articles"):
            existing_ids = set()
            for chunk_id in chunk_ids:
                if self.qdrant.retrieve(collection_name="gdpr_articles", ids=[chunk_id]):
                    existing_ids.add(chunk_id)

            if existing_ids:
                new_chunk_texts = []
                new_chunk_titles = []
                new_chunk_ids = []
                for i, chunk_id in enumerate(chunk_ids):
                    if chunk_id not in existing_ids:
                        new_chunk_texts.append(chunk_texts[i])
                        new_chunk_titles.append(chunk_titles[i])
                        new_chunk_ids.append(chunk_ids[i])

                chunk_texts = new_chunk_texts
                chunk_titles = new_chunk_titles
                chunk_ids = new_chunk_ids

                print(f"Skipping {len(existing_ids)} existing chunks.")
                if not chunk_texts:
                    print("All chunks already exist in the database.")
                    return

        vectors = self.model.encode(chunk_texts, convert_to_numpy=True)

        collection_name = "gdpr_articles"
        if not self.qdrant.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}")
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE)
            )

        self.qdrant.upload_points(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector=vec.tolist(),
                    payload={"article": title, "text": text}
                )
                for vec, title, text, chunk_id in zip(vectors, chunk_titles, chunk_texts, chunk_ids)
            ]
        )
        print(f"Uploaded {len(chunk_texts)} GDPR chunks to Qdrant.")

    def search_Qdrant(self, query):
        query_vector = self.model.encode(query).tolist()

        search_results = self.qdrant.query_points(
            collection_name="gdpr_articles",
            query=query_vector,
            limit=1,
            with_payload=True
        ).points

        for point in search_results:
            score = point.score
            payload = point.payload
            print("-" * 80)
            print("TOP MATCH:")
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

    query = "What are the principles for lawful data processing?"
    QdrantEmbedding.search_Qdrant(query)
