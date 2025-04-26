from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

class RegulationCompare:

    def __init__(self, regulation):
        self.regulation = regulation
        self.qdrant = QdrantClient("localhost", port=6333)
        self.model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        self.collection_name = "gdpr_articles"


    def load_pdf(self, pdf_path):
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
        page_texts = []
        for page in documents:

            text_lines = page.page_content.split('\n')
            cleaned_lines = []

            for line in text_lines:
                line = line.strip()
                    
                if line:
                    cleaned_lines.append(line)
                    
            page_texts.append("\n".join(cleaned_lines))


        raw_text = " ".join(page_texts)

        return raw_text
    

    def structure_aware_chunker(self, text, min_length=100, max_length=600):

        if not isinstance(text, str):
            raise ValueError("Expected `text` to be a string, got: " + str(type(text)))

        # Combine all patterns using lookahead so the split point is preserved
        section_pattern = re.compile(
            r"(?:\n|^)\s*(?=("
            r"\d{1,2}(\.\d{1,2}){0,2}[\.\)]"         
            r"|[A-Z]{1,4}[\.\)]"                   
            r"|[a-z]{1,2}[\.\)]"                     
            r"))"
        )


        raw_chunks = section_pattern.split(text)

        final_chunks = []

        for chunk in raw_chunks:
            if not chunk or not chunk.strip() or not isinstance(chunk,str):
                continue
            paragraphs = re.split(r'\n{2,}', chunk.strip())
            buffer = ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                if len(buffer) + len(para) < max_length:
                    buffer += " " + para
                else:
                    if buffer:
                        final_chunks.append(buffer.strip())
                    buffer = para

            if buffer:
                final_chunks.append(buffer.strip())

        final_chunks =final_chunks[1:]  

        return [c for c in final_chunks if len(c) >= min_length]

    
    def process_chunk(self, batch):
        return self.model.encode(batch, normalize_embeddings=True)

    def generate_embeddings_batch(self, generator, batch_size):
        return self.model.encode(generator, batch_size=batch_size, normalize_embeddings=True)


    def construct_article_dict(self):

        articles_dict = {} # {"article_name": {"embedding": np.array, "text": str}}
        points, _  = regulation.qdrant.scroll(collection_name=regulation.collection_name, with_payload=True,with_vectors=True, limit=10000)
        
        for point in points:
            article_name = point.payload["article"]
            article_text = point.payload["text"]
            article_vector = np.array(point.vector)

            core_article_name = re.search(r"Article\s+\d+", article_name)
            if core_article_name:
                key = core_article_name.group()
            else:
                continue 

            if key not in articles_dict:
                articles_dict[key] = []

            articles_dict[key].append({
                "embedding": article_vector,
                "text": article_text,
                "uuid": point.id
            })

        return articles_dict
    
    def cosine_sim(self, a, b):
        return float(cosine_similarity([a], [b])[0][0])

    def total_similarity_score(self, policy_chunk_embeddings, articles, pdf_chunks_text, pdf_name, article_weights={}, similarity_threshold=0.5):
        """
        Assess GDPR compliance by evaluating how well each GDPR article is covered by hospital policy chunks.

        Args:
            policy_chunk_embeddings (List[np.array]): List of chunk embeddings
            articles (dict): {article_name: {"embedding": np.array, "text": str}}
            policy_chunk_texts (List[str]): List of chunk texts
            similarity_threshold (float): Threshold above which an article is considered "covered"
            article_weights (dict): Optional weights for GDPR articles (should sum to 1)

        Returns:
            dict: {
                "weighted_score": float,
                "coverage_score": float,
                "article_scores": dict,
                "missing_articles": list,
                "report_path": str
            }
        """
    

        os.makedirs("test", exist_ok=True)
        report_path = "test/final_report.txt"
        with open(report_path, "w") as f:
            f.write("GDPR Compliance Report (Article → Chunk Evaluation)\n")
            f.write(pdf_name + "\n")
            f.write("=" * 80 + "\n")

        article_scores = {}
        missing_articles = []

        for article_name, article_entries in articles.items():
            best_score = 0.0
            best_text = ""
            best_chunk_idx = -1

            for article_entry in article_entries:
                article_embedding = article_entry["embedding"]
                for idx, chunk_vec in enumerate(policy_chunk_embeddings):
                    score = self.cosine_sim(chunk_vec, article_embedding)
                    if score > best_score:
                        best_score = score
                        best_chunk_idx = idx
                        best_text = article_entry["text"]

            article_scores[article_name] = best_score

            with open(report_path, "a") as f:
                f.write(f"{article_name}:\n")
                f.write(f"Best Matching Chunk ID: {best_chunk_idx + 1 if best_chunk_idx >= 0 else 'N/A'}\n")
                f.write(f"Best Matching Chunk Text: {pdf_chunks_text[best_chunk_idx][:250].strip()}...\n")
                f.write(f"Best Similarity Score: {best_score:.4f}\n")
                f.write(f"Article Preview: {best_text[:250].strip()}...\n")
                f.write("-" * 80 + "\n")

            if best_score < similarity_threshold:
                missing_articles.append(article_name)

        # Weighted score calculation
        weighted_score = sum(article_scores.get(a, 0.0) * w for a, w in article_weights.items())
        penalty = min(0.5, 0.05 * len(missing_articles))  # 5% penalty per missing article, capped at 90%
        final_score = round(weighted_score * (1 - penalty) * 100, 2)

        # Coverage score (simple heuristic)
        fully = sum(1 for score in article_scores.values() if score > 0.7)
        partial = sum(1 for score in article_scores.values() if 0.5 < score <= 0.7)
        none = len(article_scores) - fully - partial
        coverage_score = round((fully + 0.5 * partial) / len(article_scores) * 100, 2)

        with open(report_path, "a") as f:
            f.write("\nSummary:\n")
            f.write(f"Weighted Compliance Score: {final_score}/100\n")
            f.write(f"Coverage-Based Score: {coverage_score}/100\n")
            f.write(f"Fully Covered Articles (>0.7): {fully}\n")
            f.write(f"Partially Covered Articles (0.5–0.7): {partial}\n")
            f.write(f"Not Covered Articles (≤0.5): {none}\n")
            if missing_articles:
                f.write(f"Articles Below Threshold ({similarity_threshold}): {', '.join(missing_articles)}\n")
            f.write("=" * 80 + "\n")

        return {
            "weighted_score": final_score,
            "coverage_score": coverage_score,
            "article_scores": article_scores,
            "missing_articles": missing_articles,
            "report_path": report_path
        }
    

if __name__=="__main__":

    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    article_weights = {
        "Article 5": 0.15,
        "Article 6": 0.10,
        "Article 7": 0.05,
        "Article 12": 0.05,
        "Article 13": 0.025,
        "Article 14": 0.025,
        "Article 15": 0.025,
        "Article 16": 0.025,
        "Article 17": 0.025,
        "Article 18": 0.025,
        "Article 19": 0.025,
        "Article 20": 0.025,
        "Article 21": 0.025,
        "Article 22": 0.025,
        "Article 24": 0.05,
        "Article 25": 0.10,
        "Article 30": 0.05,
        "Article 32": 0.10,
        "Article 33": 0.025,
        "Article 34": 0.025,
        "Article 35": 0.05,
        "Article 37": 0.025,
        "Article 38": 0.015,
        "Article 39": 0.01,
    }


    regulation = RegulationCompare("GDPR")
    # pdf_paths = ["data/policies/NHG-Privacy-Policy-2021.pdf", 
    #             "data/policies/HIPAA-Privacy-Policy-dtd-1.1.22-1.pdf",
    #             "data/policies/AHA-Privacy-Policy-2021.pdf",
    #             "data/policies/Data-Protection-Policy-2021.pdf",
    #             "data/policies/Gertrudes-Hospital-Healthcare-Privacy-Policy.pdf",
    #             "data/policies/SHIP-DPO-MANUAL.pdf"
    #         ]
    
    articles_dict = regulation.construct_article_dict()
    pdf_name = pdf_path.split("/")[-1]
    text = regulation.load_pdf(pdf_path)
    with open("test/pdfs.txt", "w") as f:
        f.write(pdf_name+ "\n")
        f.write(text)
        f.write("-" * 80 + "\n")
    text_chunks = regulation.structure_aware_chunker(text)
    with open("test/chunks.txt", "w") as f:
        f.write("*" * 80 + "\n")
        f.write(pdf_name+ "\n")
        for chunk in text_chunks:
            f.write(chunk + "\n")
            f.write("-" * 80 + "\n")
    if not isinstance(text_chunks, list):
        text_chunks = list(text_chunks)
    embeddings = regulation.generate_embeddings_batch(text_chunks, batch_size=32)
    pdf_chunks_text = {i: text for i, text in enumerate(text_chunks)}
    similarity_score = regulation.total_similarity_score(embeddings, articles_dict, pdf_chunks_text, pdf_name, article_weights)
    # print(f"Total similarity score: {similarity_score} for {pdf_name}")



    


