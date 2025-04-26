from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from llm_report import LLMReport
from regulation_compare import RegulationCompare
import shutil
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/gdpr/evaluate")
async def evaluate_gdpr_policy(policy: UploadFile = File(...)):

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
    articles_dict = regulation.construct_article_dict()
    text = regulation.load_pdf(file_path)
    with open("../report/pdfs.txt", "w") as f:
        f.write(policy.filename+ "\n")
        f.write(text)
        f.write("-" * 80 + "\n")

    try:
        text_chunks = regulation.structure_aware_chunker(text)
    except Exception as e:
        print(f"Error during chunking: {e}")
        return {"error": "Chunking failed. Please check the input file, it must be a hospital regulation."}

    with open("../report/chunks.txt", "w") as f:
        f.write("*" * 80 + "\n")
        f.write(policy.filename+ "\n")
        for chunk in text_chunks:
            f.write(chunk + "\n")
            f.write("-" * 80 + "\n")

    embeddings = regulation.generate_embeddings_batch(text_chunks, batch_size=32)
    chunk_map = {i: chunk for i, chunk in enumerate(text_chunks)}
    score = regulation.total_similarity_score(embeddings, articles_dict, chunk_map, policy.filename, article_weights)

    llm_report = LLMReport(
        model_url="http://localhost:11434",
        model_name="llama3.2:3b",
        report_path="../report/final_report.txt"
    )
    llm_report=llm_report.generate_report()

    return {
        "filename": policy.filename,
        "score": score,
        "llm_report": llm_report
    }
