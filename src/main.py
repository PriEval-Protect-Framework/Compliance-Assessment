from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from llm_report import LLMReport
from regulation_compare import RegulationCompare
import shutil
import os
import uuid
from qdrant import Qdrant

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
REPORT_PATH = os.getenv("REPORT_PATH", "../report/final_report.txt")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.on_event("startup")
# def startup_event():
#     qdrant_instance = Qdrant()
#     qdrant_instance.embed_gdpr()


@app.post("/gdpr/evaluate")
async def evaluate_gdpr_policy(policy: UploadFile = File(...)):
    qdrant_instance = Qdrant()
    qdrant_instance.embed_gdpr()


    print(f"[API] Received policy upload: {policy.filename}")

    file_id = str(uuid.uuid4())
    file_path = f"../uploads/{file_id}_{policy.filename}"
    os.makedirs("../uploads", exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(policy.file, buffer)

    article_weights = {
        "Article 5": 0.15, "Article 6": 0.10, "Article 7": 0.05,
        "Article 12": 0.05, "Article 13": 0.025, "Article 14": 0.025,
        "Article 15": 0.025, "Article 16": 0.025, "Article 17": 0.025,
        "Article 18": 0.025, "Article 19": 0.025, "Article 20": 0.025,
        "Article 21": 0.025, "Article 22": 0.025, "Article 24": 0.05,
        "Article 25": 0.10, "Article 30": 0.05, "Article 32": 0.10,
        "Article 33": 0.025, "Article 34": 0.025, "Article 35": 0.05,
        "Article 37": 0.025, "Article 38": 0.015, "Article 39": 0.01,
    }

    regulation = RegulationCompare("GDPR")
    
    print("[API] Constructing article dictionary")
    articles_dict = regulation.construct_article_dict()
    print(f"[API] Article dictionary constructed with {len(articles_dict)} articles")
    
    print("[API] Loading PDF content")
    text = regulation.load_pdf(file_path)
    with open("../report/pdfs.txt", "w") as f:
        f.write(policy.filename+ "\n")
        f.write(text)
        f.write("-" * 80 + "\n")
    print("[API] Saved extracted text to ../report/pdfs.txt")

    try:
        print("[API] Starting structure-aware chunking")
        text_chunks = regulation.structure_aware_chunker(text)
        print(f"[API] Chunking complete, created {len(text_chunks)} chunks")
    except Exception as e:
        print(f"[API] Error during chunking: {e}")
        import traceback
        traceback.print_exc()
        return {"error": "Chunking failed. Please check the input file, it must be a hospital regulation."}

    with open("../report/chunks.txt", "w", encoding="utf-8") as f:
        f.write("*" * 80 + "\n")
        f.write(policy.filename + "\n")
        for chunk in text_chunks:
            f.write(chunk + "\n")
            f.write("-" * 80 + "\n")
    print("[API] Saved chunks to ../report/chunks.txt")

    print("[API] Generating embeddings")
    embeddings = regulation.generate_embeddings_batch(text_chunks, batch_size=32)
    print(f"[API] Embeddings generated, shape: {len(embeddings)} vectors")
    
    chunk_map = {i: chunk for i, chunk in enumerate(text_chunks)}
    
    print("[API] Calculating similarity scores")
    score = regulation.total_similarity_score(embeddings, articles_dict, chunk_map, policy.filename, article_weights)
    print(f"[API] Score calculation complete: {score}")

    llm_report = LLMReport(
        model_url=OLLAMA_HOST,
        model_name=MODEL_NAME,
        report_path=REPORT_PATH
    )
    llm_report=llm_report.generate_report()

    return {
        "filename": policy.filename,
        "score": score,
        "llm_report": llm_report
    }