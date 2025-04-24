import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

json_path = './data/gdpr_articles_recitals.jsonl'
data = [json.loads(line) for line in open(json_path, 'r', encoding='utf-8')]

model = SentenceTransformer('BAAI/bge-base-en-v1.5')
texts = [entry['output-text'] for entry in data]
titles = [entry['input-text'] for entry in data]
vectors = model.encode(texts, convert_to_numpy=True)

qdrant = QdrantClient("localhost", port=6333)

collection_name = "gdpr_articles"
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE)
)

qdrant.upload_points(
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
