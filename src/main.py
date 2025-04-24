from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

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

        return [c for c in final_chunks if len(c) >= min_length]

    
    def process_chunk(self, batch):
        return self.model.encode(batch, normalize_embeddings=True)

    def generate_embeddings_batch(self, generator, batch_size):
        return self.model.encode(generator, batch_size=batch_size, normalize_embeddings=True)

    
    def total_similarity_score(self, embeddings):
        final_score = 0
        for query_vector in embeddings:
            search_results = self.qdrant.query_points(
                collection_name="gdpr_articles",
                query=query_vector,
                with_payload=True
            ).points

            for point in search_results:
                score = point.score
                payload = point.payload
                with open("test/report.txt", "a") as f:
                    f.write(f"Article: {payload['article']}\n")
                    f.write(f"Text: {payload['text']}\n")
                    f.write(f"Score: {score}\n")
                    f.write("-" * 80 + "\n")
                final_score+=score

        info = self.qdrant.get_collection(collection_name=self.collection_name)
        num_embeddings = info.points_count

        return round(final_score/num_embeddings, 4)


    

if __name__=="__main__":

    regulation = RegulationCompare("GDPR")
    pdf_path = "data/policies/SHIP-DPO-MANUAL.pdf"
    text = regulation.load_pdf(pdf_path)

    with open("test/all.txt", "w") as f:
        f.write(text)

    text_chunks = regulation.structure_aware_chunker(text)

    with open("test/chunks.txt", "w") as f:
        for chunk in text_chunks:
            f.write(chunk + "\n")
            f.write("-" * 80 + "\n")

    if not isinstance(text_chunks, list):
        text_chunks = list(text_chunks)

    embeddings = regulation.generate_embeddings_batch(text_chunks, batch_size=32)

    similarity_score = regulation.total_similarity_score(embeddings)

    print(f"Total similarity score: {similarity_score}")



    


