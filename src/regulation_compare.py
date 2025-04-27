from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RegulationCompare:

    def __init__(self, regulation):
        print(f"[RegComp] Initializing RegulationCompare with regulation: {regulation}")
        self.regulation = regulation
        try:
            print("[RegComp] Connecting to Qdrant at localhost:6333")
            self.qdrant = QdrantClient("localhost", port=6333)
            print("[RegComp] Connected to Qdrant successfully")
        except Exception as e:
            print(f"[RegComp] Failed to connect to Qdrant: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        try:
            print("[RegComp] Loading SentenceTransformer model: BAAI/bge-base-en-v1.5")
            self.model = SentenceTransformer('BAAI/bge-base-en-v1.5')
            print("[RegComp] SentenceTransformer model loaded successfully")
        except Exception as e:
            print(f"[RegComp] Failed to load SentenceTransformer model: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        self.collection_name = "gdpr_articles" if self.regulation == "GDPR" else "hipaa_articles"
        print(f"[RegComp] Using collection name: {self.collection_name}")

    def load_pdf(self, pdf_path):
        print(f"[RegComp] Loading PDF from path: {pdf_path}")
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            print(f"[RegComp] PDF loaded, found {len(documents)} pages")
            
            page_texts = []
            for i, page in enumerate(documents):
                text_lines = page.page_content.split('\n')
                cleaned_lines = []

                for line in text_lines:
                    line = line.strip()
                        
                    if line:
                        cleaned_lines.append(line)
                        
                page_texts.append("\n".join(cleaned_lines))

            raw_text = " ".join(page_texts)
            print(f"[RegComp] PDF text extraction complete, total length: {len(raw_text)} characters")
            return raw_text
        except Exception as e:
            print(f"[RegComp] Error loading PDF: {e}")
            import traceback
            traceback.print_exc()
            raise

    def structure_aware_chunker(self, text, min_length=100, max_length=600):
        print(f"[RegComp] Starting structure-aware chunking with min_length={min_length}, max_length={max_length}")
        
        if not isinstance(text, str):
            error_msg = f"Expected `text` to be a string, got: {type(text)}"
            print(f"[RegComp] ERROR: {error_msg}")
            raise ValueError(error_msg)

        print(f"[RegComp] Input text length: {len(text)} characters")
        
        # Combine all patterns using lookahead so the split point is preserved
        section_pattern = re.compile(
            r"(?:\n|^)\s*(?=("
            r"\d{1,2}(\.\d{1,2}){0,2}[\.\)]"         
            r"|[A-Z]{1,4}[\.\)]"                   
            r"|[a-z]{1,2}[\.\)]"                     
            r"))"
        )

        print("[RegComp] Splitting text based on section patterns")
        raw_chunks = section_pattern.split(text)
        print(f"[RegComp] Initial split resulted in {len(raw_chunks)} raw chunks")

        final_chunks = []

        for i, chunk in enumerate(raw_chunks):
            if not chunk or not chunk.strip() or not isinstance(chunk, str):
                print(f"[RegComp] Skipping invalid chunk at index {i}")
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

        if len(final_chunks) > 0:
            final_chunks = final_chunks[1:]  # Skip the first chunk which is usually just an intro or blank
            print(f"[RegComp] After processing, total chunks: {len(final_chunks)}")
        else:
            print("[RegComp] WARNING: No chunks were created")

        result = [c for c in final_chunks if len(c) >= min_length]
        print(f"[RegComp] Final chunks after length filtering: {len(result)}")
        
        return result

    def process_chunk(self, batch):
        print(f"[RegComp] Processing batch of size {len(batch)}")
        try:
            result = self.model.encode(batch, normalize_embeddings=True)
            print(f"[RegComp] Batch processed successfully")
            return result
        except Exception as e:
            print(f"[RegComp] Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            raise

    def generate_embeddings_batch(self, generator, batch_size):
        print(f"[RegComp] Generating embeddings with batch size {batch_size} for {len(generator)} items")
        try:
            result = self.model.encode(generator, batch_size=batch_size, normalize_embeddings=True)
            print(f"[RegComp] Embeddings generated successfully")
            return result
        except Exception as e:
            print(f"[RegComp] Error generating embeddings: {e}")
            import traceback
            traceback.print_exc()
            raise

    def construct_article_dict(self):
        print(f"[RegComp] Constructing article dictionary from {self.collection_name}")
        articles_dict = {}  # {"article_name": {"embedding": np.array, "text": str}}
        
        try:
            print("[RegComp] Scrolling through collection points")
            points, _ = self.qdrant.scroll(collection_name=self.collection_name, with_payload=True, with_vectors=True, limit=10000)
            print(f"[RegComp] Retrieved {len(points)} points from collection")
            
            for i, point in enumerate(points):
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

            print(f"[RegComp] Article dictionary constructed with {len(articles_dict)} unique articles")
            return articles_dict
            
        except Exception as e:
            print(f"[RegComp] Error constructing article dictionary: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def cosine_sim(self, a, b):
        try:
            result = float(cosine_similarity([a], [b])[0][0])
            return result
        except Exception as e:
            print(f"[RegComp] Error calculating cosine similarity: {e}")
            import traceback
            traceback.print_exc()
            raise

    def total_similarity_score(self, policy_chunk_embeddings, articles, pdf_chunks_text, pdf_name, article_weights={}, similarity_threshold=0.5):
        print(f"[RegComp] Calculating similarity scores for {len(policy_chunk_embeddings)} chunks against {len(articles)} articles")
        print(f"[RegComp] Using similarity threshold of {similarity_threshold}")

        try:
            os.makedirs("report", exist_ok=True)
            report_path = "report/final_report.txt"
            print(f"[RegComp] Writing report to {report_path}")
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("GDPR Compliance Report (Article → Chunk Evaluation)\n")
                f.write(pdf_name + "\n")
                f.write("=" * 80 + "\n")

            article_scores = {}
            missing_articles = []

            for article_name, article_entries in articles.items():
                print(f"[RegComp] Processing {article_name} with {len(article_entries)} entries")
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
                print(f"[RegComp] {article_name}: Best score = {best_score:.4f}, Best chunk = {best_chunk_idx + 1 if best_chunk_idx >= 0 else 'N/A'}")

                with open(report_path, "a", encoding="utf-8") as f:
                    f.write(f"{article_name}:\n")
                    f.write(f"Best Matching Chunk ID: {best_chunk_idx + 1 if best_chunk_idx >= 0 else 'N/A'}\n")
                    f.write(f"Best Matching Chunk Text: {pdf_chunks_text[best_chunk_idx][:250].strip()}...\n")
                    f.write(f"Best Similarity Score: {best_score:.4f}\n")
                    f.write(f"Article Preview: {best_text[:250].strip()}...\n")
                    f.write("-" * 80 + "\n")

                if best_score < similarity_threshold:
                    missing_articles.append(article_name)
                    print(f"[RegComp] {article_name} added to missing articles list (score: {best_score:.4f})")

            # Weighted score calculation
            weighted_score = sum(article_scores.get(a, 0.0) * w for a, w in article_weights.items())
            penalty = min(0.5, 0.05 * len(missing_articles))  # 5% penalty per missing article, capped at 90%
            final_score = round(weighted_score * (1 - penalty) * 100, 2)
            print(f"[RegComp] Weighted score: {weighted_score:.4f}, Penalty: {penalty:.4f}, Final score: {final_score}")

            # Coverage score (simple heuristic)
            fully = sum(1 for score in article_scores.values() if score > 0.7)
            partial = sum(1 for score in article_scores.values() if 0.5 < score <= 0.7)
            none = len(article_scores) - fully - partial
            coverage_score = round((fully + 0.5 * partial) / len(article_scores) * 100, 2)
            print(f"[RegComp] Coverage score: {coverage_score}, Fully covered: {fully}, Partially: {partial}, None: {none}")

            with open(report_path, "a", encoding="utf-8") as f:
                f.write("\nSummary:\n")
                f.write(f"Weighted Compliance Score: {final_score}/100\n")
                f.write(f"Coverage-Based Score: {coverage_score}/100\n")
                f.write(f"Fully Covered Articles (>0.7): {fully}\n")
                f.write(f"Partially Covered Articles (0.5–0.7): {partial}\n")
                f.write(f"Not Covered Articles (≤0.5): {none}\n")
                if missing_articles:
                    f.write(f"Articles Below Threshold ({similarity_threshold}): {', '.join(missing_articles)}\n")
                f.write("=" * 80 + "\n")

            print("[RegComp] Score calculation complete")
            return {
                "weighted_score": final_score,
                "coverage_score": coverage_score,
                "fully_covered": fully,
                "partially_covered": partial,
                "not_covered": none,
                "missing_articles": missing_articles,
            }
        except Exception as e:
            print(f"[RegComp] Error calculating similarity scores: {e}")
            import traceback
            traceback.print_exc()
            raise