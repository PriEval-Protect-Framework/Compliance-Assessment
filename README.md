### 1. Install Ollama

#### Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### macOS:
[Download Ollama for macOS](https://ollama.com/download/Ollama-darwin.zip)

#### Windows:
[Download Ollama for Windows](https://ollama.com/download/OllamaSetup.exe)

---

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Start the Ollama Server

```bash
ollama serve
```

---

### 4. Load a Local Model

For example, to load the LLaMA 3.2 3B model:

```bash
ollama run llama3.2:3b
```

Make sure the model is downloaded and running locally.

---

### 5. Start Qdrant Vector Store (Docker)

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---

### 6. Run the GDPR Compliance Checker

Run the evaluation script by providing the path to a hospital privacy policy PDF:

```bash
python3 main.py data/policies/<file-name>.pdf
```

This will:
- Load and chunk the policy
- Compute GDPR article similarity
- Generate a compliance report
- Summarize the report using an LLM

---
