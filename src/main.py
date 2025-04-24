from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
import re
import os

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

    
    def total_similarity_score(self, embeddings, similarity_threshold=0.7, article_weights=None):
        """
        Calculate similarity score between embeddings and GDPR articles.
        
        Args:
            embeddings: List of embedding vectors to compare against GDPR articles
            similarity_threshold: Minimum similarity score to consider a match (default: 0.7)
            article_weights: Dictionary mapping article names to their weights (optional)
            
        Returns:
            float: Compliance score (0-100)
        """
        
        # THIS WILL LATER BE FURTHER TUNED
        if article_weights is None:
            article_weights = {
                "Article 5": 0.2,  # Principles
                "Article 6": 0.15, # Lawfulness
                "Article 9": 0.1,  # Special categories
                "Article 12": 0.05, # Transparent information
                "Article 13": 0.15, # Information to be provided
                "Article 17": 0.05, # Right to erasure
                "Article 25": 0.1,  # Data protection by design
                "Article 32": 0.1,  # Security of processing
                "Article 35": 0.05, # Data protection impact assessment
                "Article 44": 0.05, # Transfer of personal data
                # ... weights should sum to 1
            }
        
        article_scores = {}
        
        os.makedirs("test", exist_ok=True)
        
        with open("test/report.txt", "w") as f:
            f.write("GDPR Compliance Report\n")
            f.write("=" * 80 + "\n")
        
        # Process each policy chunk
        for idx, query_vector in enumerate(embeddings):

            search_results = self.qdrant.query_points(
                collection_name="gdpr_articles",
                query=query_vector,
                limit=1,  # top 1 match
                with_payload=True
            ).points
            
            if not search_results:
                continue
            
            top_point = search_results[0]
            score = top_point.score
            payload = top_point.payload
            full_article_name = payload.get("article", "Unknown Article")
            
            article_match = re.search(r"(Article\s+\d+)", full_article_name)
            recital_match = re.search(r"recital-(\d+)", full_article_name)
            
            if article_match:
                article = article_match.group(1)
            elif recital_match:
                article = f"Recital {recital_match.group(1)}" 
            else:
                article = full_article_name
            
            if score >= similarity_threshold:
                if article not in article_scores:
                    article_scores[article] = 0.001
                    
                article_scores[article] = max(article_scores[article], score)
                
                with open("test/report.txt", "a") as f:
                    f.write(f"Chunk {idx + 1}:\n")
                    f.write(f"Matched Article: {full_article_name}\n")
                    f.write(f"Normalized Name: {article}\n")
                    f.write(f"Text: {payload.get('text', '')[:200]}...\n")  
                    f.write(f"Similarity Score: {score:.4f}\n")
                    f.write("-" * 80 + "\n")
        
        for article in article_weights:
            if article not in article_scores:
                article_scores[article] = 0.001
        
        weighted_score = 0.0
        missing_articles = []
        
        for article, weight in article_weights.items():
            score = article_scores.get(article, 0.0)
            weighted_score += score * weight
            if score < similarity_threshold:
                missing_articles.append(article)
        
        # Apply penalty for missing critical articles (10% per missing article)
        penalty = min(0.9, 0.1 * len(missing_articles))  # Cap penalty at 90%
        weighted_final_score = max(0.0, weighted_score * (1 - penalty))
        weighted_final_score = round(weighted_final_score * 100, 2)
        
        fully_covered = 0
        partially_covered = 0
        not_covered = 0
        
        for article in article_weights:
            max_score = article_scores.get(article, 0.0)
            if max_score > 0.8:  # Strong match
                fully_covered += 1
            elif max_score > 0.5:  # Partial match
                partially_covered += 1
            else:
                not_covered += 1
        
        total_articles = len(article_weights)
        
        if total_articles > 0:
            compliance_score = (fully_covered + 0.5 * partially_covered) / total_articles
            compliance_score = round(compliance_score * 100, 2)
        else:
            compliance_score = 0.0
        
        with open("test/report.txt", "a") as f:
            f.write("\nSummary:\n")
            f.write(f"Weighted Compliance Score: {weighted_final_score}/100\n")
            f.write(f"Coverage-Based Compliance Score: {compliance_score}/100\n")
            f.write(f"Coverage Metrics:\n")
            f.write(f"Fully Covered Articles (>0.8): {fully_covered}\n")
            f.write(f"Partially Covered Articles (0.5–0.8): {partially_covered}\n")
            f.write(f"Not Covered Articles (≤0.5): {not_covered}\n")
            f.write("Article Coverage:\n")
            
            for article in sorted(article_weights.keys()):
                score = article_scores.get(article, 0.0)
                f.write(f"{article}: {score:.4f}\n")
                
            for article, score in sorted(article_scores.items()):
                if article not in article_weights and score >= similarity_threshold:
                    f.write(f"{article} (not weighted): {score:.4f}\n")
                    
            if missing_articles:
                f.write(f"Missing or Weak Coverage (below {similarity_threshold}): {', '.join(missing_articles)}\n")
            f.write("=" * 80 + "\n")
        
        return compliance_score

    

if __name__=="__main__":

    regulation = RegulationCompare("GDPR")
    pdf_paths = ["data/policies/NHG-Privacy-Policy-2021.pdf", 
                "data/policies/HIPAA-Privacy-Policy-dtd-1.1.22-1.pdf",
                "data/policies/AHA-Privacy-Policy-2021.pdf",
                "data/policies/Data-Protection-Policy-2021.pdf",
                "data/policies/Gertrudes-Hospital-Healthcare-Privacy-Policy.pdf",
                "data/policies/SHIP-DPO-MANUAL.pdf"
            ]
    
    for pdf_path in pdf_paths:
        pdf_name = pdf_path.split("/")[-1]
        text = regulation.load_pdf(pdf_path)

        with open("test/pdfs.txt", "a") as f:
            f.write(pdf_name+ "\n")
            f.write(text)
            f.write("-" * 80 + "\n")

        text_chunks = regulation.structure_aware_chunker(text)

        with open("test/chunks.txt", "a") as f:
            f.write("*" * 80 + "\n")
            f.write(pdf_name+ "\n")
            for chunk in text_chunks:
                f.write(chunk + "\n")
                f.write("-" * 80 + "\n")

        if not isinstance(text_chunks, list):
            text_chunks = list(text_chunks)

        embeddings = regulation.generate_embeddings_batch(text_chunks, batch_size=32)

        similarity_score = regulation.total_similarity_score(embeddings)

        print(f"Total similarity score: {similarity_score} for {pdf_name}")



    


