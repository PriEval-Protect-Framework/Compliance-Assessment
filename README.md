# PriEval-Protect – GDPR Compliance Checker

A backend system for evaluating hospital privacy policies against GDPR compliance standards using document embeddings and local LLM summarization. Includes a vector-based article matching engine and report generation.

---

## Setup Instructions

### 1. Install Ollama (for local LLM inference)

#### On Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### On macOS:
Download and install from:  
[https://ollama.com/download/Ollama-darwin.zip](https://ollama.com/download/Ollama-darwin.zip)

#### On Windows:
Download and install from:  
[https://ollama.com/download/OllamaSetup.exe](https://ollama.com/download/OllamaSetup.exe)

---

### 2. Install Python Dependencies

Create a virtual environment and install all required libraries:

```bash
pip install -r requirements.txt
```

---

### 3. Start the Ollama Server

Make sure the Ollama background server is running:

```bash
ollama serve
```

---

### 4. Load a Local Model

Load the model you want to use. For example:

```bash
ollama run llama3.2:3b
```

This will download the model if it’s not already present.

---

### 5. Start Qdrant Vector Store (for article embedding search)

Launch the Qdrant server using Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

This stores and retrieves GDPR article embeddings for semantic similarity analysis.

---

### 6. Run the GDPR Compliance API

Navigate to the `src/` directory and launch the FastAPI server:

```bash
cd src/
python -m uvicorn main:app --reload --port 8001
```

This will expose an API endpoint at `http://127.0.0.1:8001/gdpr/evaluate`.

---

### 7. Test the API

Send a `POST` request to the following endpoint with a PDF file:

```
http://127.0.0.1:8001/gdpr/evaluate
```

Use a tool like **Postman** or **cURL**. The PDF file must be attached under the key `policy` as form-data.

#### Example using `curl`:
```bash
curl -X POST http://127.0.0.1:8001/gdpr/evaluate \
  -F 'policy=@/path/to/your/hospital_policy.pdf'
```

The API will:
- Parse and chunk the document
- Compare each chunk with GDPR articles
- Compute an overall compliance score
- Generate a summary report using a local LLM

---