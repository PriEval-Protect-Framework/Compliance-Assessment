# PriEval-Protect: GDPR Compliance Assessment Module

This repository contains the **Compliance Assessment** module of the [PriEval-Protect Framework](https://github.com/PriEval-Protect-Framework), a unified system for evaluating and enhancing data privacy in healthcare environments.

## Overview

This module performs **automated GDPR compliance evaluation** of hospital policy documents using a hybrid **Retrieval-Augmented Generation (RAG)** pipeline combined with a **fine-tuned Large Language Model (LLM)**. It enables:

- Extraction and chunking of policy text
- Semantic similarity matching with GDPR articles (via Qdrant)
- Article-level compliance scoring
- Penalty handling for omissions or partial fulfillment
- LLM-based textual report generation and recommendations

The goal is to make legal compliance **explainable**, **quantifiable**, and **actionable**.

---

## Key Features

- **Structure-aware policy parsing**  
- **BAAI/bge-base-en-v1.5** embeddings for legal text similarity  
- **Weighted GDPR scoring** with penalty adjustment  
- **Fine-tuned LLM (Saul-7B)** for compliance classification and reporting  
- REST API output for frontend dashboard integration  
- Ready-to-use FastAPI microservice for local or cloud deployment

---

## Technologies

- Python, FastAPI, Qdrant
- SentenceTransformers, Hugging Face Transformers
- Saul-7B-Instruct (fine-tuned on GDPR)
- Docker-ready deployment

---

## Repository Structure

```bash
Compliance-Assessment/
├── data/                          # Raw data and preprocessed datasets
│   └── finetuning/
│       ├── cleaning/              # Notebooks for data cleaning and preparation
│       │   ├── articles_datasets_cleaning.ipynb
│       │   ├── compliance_datasets_cleaning.ipynb
│       ├── data/                 # Cleaned data used for fine-tuning
│       └── FinetunedModel.ipynb  # Notebook for LLM fine-tuning
│
├── prompts/
│   └── prompts.yaml              # Instruction prompts used for model evaluation
│
├── report/
│   └── src/                      # Main source code for GDPR compliance assessment
│       ├── __init__.py
│       ├── main.py               # FastAPI app entrypoint
│       ├── qdrant.py             # Qdrant-based semantic retrieval logic
│       ├── regulation_compare.py # Regulation-to-policy semantic matching
│       └── llm_report.py         # LLM-based compliance justification and output
│
├── Dockerfile                    # Container configuration for deployment
├── docker-compose.yml            # Optional multi-service orchestration
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
````

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/PriEval-Protect-Framework/Compliance-Assessment.git
cd Compliance-Assessment

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI app
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000/docs`

---

## Fine-Tuning Details

The underlying model, **Saul-7B-Instruct**, was **fine-tuned on over 11,000 GDPR-labeled compliance statements**, including:

* Article references
* Policy fragments
* Violation summaries

We used LoRA adapters and 4-bit quantization to optimize resource use while maintaining performance.

---

## Outputs

* GDPR compliance score (0–100)
* Article coverage breakdown (with weights)
* LLM-generated explanation and article citations
* Highlighted risk factors and improvement suggestions

---

## Related Repositories

* [Privacy-Engine](https://github.com/PriEval-Protect-Framework/Privacy-Metrics): Technical privacy evaluation
* [Dashboard-UI](https://github.com/PriEval-Protect-Framework/Dashboard-UI): Frontend for visualization

---

## Authors

Developed by **Ilef Chebil** and **Asma ElHadj**
Supervised by \[EFREI Paris] and \[INSAT Tunisia]

---

## License

This project is licensed under the [MIT License](LICENSE).
