import sys
from llm_report import LLMReport
from regulation_compare import RegulationCompare



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
    
    articles_dict = regulation.construct_article_dict()

    pdf_name = pdf_path.split("/")[-1]

    text = regulation.load_pdf(pdf_path)

    with open("report/pdfs.txt", "w") as f:
        f.write(pdf_name+ "\n")
        f.write(text)
        f.write("-" * 80 + "\n")

    text_chunks = regulation.structure_aware_chunker(text)

    with open("report/chunks.txt", "w") as f:
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

    llm_report = LLMReport(
        model_url="http://localhost:11434",
        model_name="llama3.2:3b",
        report_path="report/final_report.txt"
    )

    llm_report.generate_report()



    


