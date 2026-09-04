# OmniRAG: Complete Installation, Setup & Developer Onboarding Guide

This guide provides step-by-step instructions to configure, run, test, and code on the **OmniRAG Enterprise Platform** on any new computer (Windows, macOS, or Linux).

---

## Table of Contents
1. [System Prerequisites](#1-system-prerequisites)
2. [Quick Setup in 5 Minutes](#2-quick-setup-in-5-minutes)
3. [Step-by-Step Installation (From Scratch)](#3-step-by-step-installation-from-scratch)
   - [Step 1: Clone or Copy the Repository](#step-1-clone-or-copy-the-repository)
   - [Step 2: Create a Dedicated Virtual Environment](#step-2-create-a-dedicated-virtual-environment)
   - [Step 3: Upgrade pip and Install Dependencies](#step-3-upgrade-pip-and-install-dependencies)
   - [Step 4: Configure Environment Variables (.env)](#step-4-configure-environment-variables-env)
   - [Step 5: Database Setup (PostgreSQL vs. SQLite Fallback)](#step-5-database-setup-postgresql-vs-sqlite-fallback)
   - [Step 6: Generate Sample Documents](#step-6-generate-sample-documents)
   - [Step 7: Ingest and Index Documents](#step-7-ingest-and-index-documents)
   - [Step 8: Run the Automated Verification Suite](#step-8-run-the-automated-verification-suite)
   - [Step 9: Launch the Server & Access the UI](#step-9-launch-the-server--access-the-ui)
4. [How to Code & Extend the Project](#4-how-to-code--extend-the-project)
   - [Architecture & Directory Map](#architecture--directory-map)
   - [Adding a New Chunking Strategy](#adding-a-new-chunking-strategy)
   - [Adding a New LLM Provider (Ollama, Claude, DeepSeek)](#adding-a-new-llm-provider-ollama-claude-deepseek)
   - [Tuning Hybrid RAG & BM25 Scoring](#tuning-hybrid-rag--bm25-scoring)
   - [Adding New Unit Tests](#adding-new-unit-tests)
   - [Customizing the Web UI](#customizing-the-web-ui)
5. [Testing via Swagger Docs & Postman](#5-testing-via-swagger-docs--postman)
6. [Troubleshooting & Common Gotchas](#6-troubleshooting--common-gotchas)

---

## 1. System Prerequisites

Before starting, ensure the target computer has the following:

| Requirement | Minimum Version | Recommended | Notes |
|---|---|---|---|
| **Operating System** | Windows 10/11, macOS 12+, Ubuntu 20.04+ | Any modern OS | Cross-platform Python codebase |
| **Python** | Python 3.10 | **Python 3.11** | Recommended for fastest execution & stable C-extensions |
| **Git** | 2.30+ | Latest | For cloning & version control |
| **Google Gemini API Key** | Optional | Recommended | Free at [aistudio.google.com](https://aistudio.google.com/) *(system has offline local fallback)* |
| **PostgreSQL** | 14+ | Optional | **Not required** — automatic SQLite fallback is built-in |
| **RAM** | 4 GB | 8 GB+ | Needed for parsing large PDFs/Excels and running local Qdrant |
| **Disk Space** | 1 GB | 2 GB+ | For virtualenv, Qdrant vectors, and sample data |

---

## 2. Quick Setup in 5 Minutes

For developers who already have Python 3.10+ installed, run these commands in your terminal:

```bash
# 1. Clone repository
git clone <repo-url>
cd RAG_FINAL_PROJECT

# 2. Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create environment file
# Windows:
Copy-Item .env.example .env
# macOS/Linux:
# cp .env.example .env

# 5. Add your Gemini API key to .env (or test with offline mode)
# Edit .env and set: GEMINI_API_KEY=your_key_here

# 6. Generate test samples and index them
python create_samples.py
python index.py --all-samples

# 7. Verify all 10 tests pass
python -m unittest tests/test_all.py

# 8. Start the application
python main.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser!

---

## 3. Step-by-Step Installation (From Scratch)

### Step 1: Clone or Copy the Repository
If copying via USB, cloud drive, or git:
```bash
git clone <repo-url>
cd RAG_FINAL_PROJECT
```
Ensure you are in the directory containing `main.py`, `requirements.txt`, and `README.md`.

---

### Step 2: Create a Dedicated Virtual Environment

#### Option A: Using Python's Built-in `venv` (Recommended)

**On Windows (PowerShell or Command Prompt):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```
> *Note for Windows PowerShell:* If you see `Execution_Policies` restriction error, run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\activate
> ```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Option B: Using Anaconda / Miniconda
```bash
conda create -n omnirag python=3.11 -y
conda activate omnirag
```

---

### Step 3: Upgrade pip and Install Dependencies
With your virtual environment activated:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### What is installed:
- **FastAPI & Uvicorn**: High-performance asynchronous REST API framework.
- **Qdrant-Client**: Local embedded HNSW vector search database.
- **Rank-BM25 & Scikit-Learn**: Lexical sparse scoring and mathematical fusion algorithms.
- **Google-GenAI & Google-GenerativeAI**: Gemini Flash and Gemini Embeddings.
- **Document Parsers**: `pypdf` (PDF), `python-docx` (Word), `openpyxl` & `pandas` (Excel/CSV), `reportlab`.
- **SQLAlchemy & Psycopg2-Binary**: Dual-mode database ORM (PostgreSQL + SQLite).
- **NetworkX**: Directed entity-relationship graph network traversal.

---

### Step 4: Configure Environment Variables (`.env`)

Copy the configuration template:
```bash
# Windows:
Copy-Item .env.example .env

# macOS / Linux:
cp .env.example .env
```

Open `.env` in any text editor (VS Code, Notepad, vim) and configure:

```env
# 1. Google Gemini API (Required for Gemini synthesis & embeddings)
# Get a free API key at: https://aistudio.google.com/
GEMINI_API_KEY=AIzaSy...your_actual_gemini_key
GEMINI_MODEL=gemini-flash-latest

# 2. Database (PostgreSQL vs SQLite)
# If using PostgreSQL:
POSTGRES_USER=postgres
POSTGRES_PASSWORD=sa
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag
# Or leave default for automatic SQLite fallback at ./data/rag_app.db:
# DATABASE_URL=sqlite:///./data/rag_app.db

# 3. Qdrant Vector DB (Embedded local storage by default)
QDRANT_PATH=./data/qdrant_storage

# 4. Default RAG Paradigm & Top-K
DEFAULT_RAG_TYPE=hybrid
DEFAULT_TOP_K=4
DEFAULT_LANGUAGE=en
```

> **Offline Mode Tip:** If you do not have a Gemini API key or internet access, set:
> ```env
> DEFAULT_EMBEDDING_STRATEGY=local
> ```
> The system will use deterministic 768-dimensional local hash embeddings and local mock generation!

---

### Step 5: Database Setup (PostgreSQL vs. SQLite Fallback)

OmniRAG features a **Dual-Database Architecture**:

#### Option A: Zero-Config SQLite (Default Out-of-the-Box)
You do **not** need to install or start anything! If PostgreSQL is not detected, OmniRAG automatically initializes a high-performance SQLite database at `./data/rag_app.db` on first boot.

#### Option B: Production PostgreSQL (Optional)
If you have PostgreSQL installed and running locally:
1. Ensure the PostgreSQL service is active.
2. Run the automated database setup script:
   ```bash
   python setup_postgres.py
   ```
   This connects to PostgreSQL, creates the `rag` database if missing, and initializes all tables (`documents`, `document_chunks`, `graph_entities`, `graph_relations`, `chat_sessions`, `chat_messages`, `query_logs`).

---

### Step 6: Generate Sample Documents
Create sample test files in all 5 supported formats (**PDF, CSV, XLSX, DOCX, TXT**):
```bash
python create_samples.py
```
This generates:
- `./data/samples/executive_summary.pdf` (Corporate overview & strategy)
- `./data/samples/products_catalog.csv` (Product inventory with IDs & pricing)
- `./data/samples/financial_report.xlsx` (Quarterly revenue & operational costs)
- `./data/samples/system_specifications.docx` (Architecture specs & requirements)
- `./data/samples/ai_rag_overview.txt` (Technical RAG concepts & explanations)

---

### Step 7: Ingest and Index Documents
Index all 5 sample files into Qdrant vector store and the relational database:
```bash
python index.py --all-samples
```

To index your own custom documents:
```bash
# Index a PDF using semantic sentence chunking and Gemini embeddings:
python index.py --file "C:/path/to/document.pdf" --chunker semantic --embedder gemini

# Index an Excel sheet with structured row serialization:
python index.py --file "C:/path/to/data.xlsx" --chunker structured
```

---

### Step 8: Run the Automated Verification Suite
Verify that every subsystem works on the new machine:
```bash
python -m unittest tests/test_all.py
```
**Expected Output:**
```
Ran 10 tests in ~65s
OK
```
All 10 tests should pass:
1. `test_01_parsers_all_formats` (PDF, CSV, XLSX, DOCX, TXT extraction)
2. `test_02_chunking_strategies` (Recursive, Semantic, Structured chunkers)
3. `test_03_embeddings_and_vector_store` (768-d vector search in Qdrant)
4. `test_04_scoring_and_rrf` (BM25 lexical ranking and RRF $k=60$)
5. `test_05_three_rag_types` (Simple, Hybrid, Graph RAG)
6. `test_06_multilingual_and_code_mode` (Hindi synthesis & Python code generation)
7. `test_07_evaluations_and_corruption` (Faithfulness, relevancy, and stress test)
8. `test_08_semantic_cache` (Sub-10ms cache hit detection)
9. `test_09_api_endpoints` (FastAPI REST endpoints)
10. `test_10_conversational_rag_with_memory` (Multi-turn conversational memory & coreference)

---

### Step 9: Launch the Server & Access the UI

Start the application:
```bash
python main.py
```
Or with development hot-reloading:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Access Links:
- **Interactive Glassmorphic Web Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc Specification**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Raw OpenAPI Schema JSON**: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

---

## 4. How to Code & Extend the Project

If you are developing new features, modifying algorithms, or customizing UI on the new machine, use this section as your blueprint.

### Architecture & Directory Map

```
RAG_FINAL_PROJECT/
├── main.py                     # FastAPI REST API & Glassmorphism Web Dashboard (HTML/CSS/JS)
├── index.py                    # Multi-format document parser & indexing CLI/module
├── database.py                 # SQLAlchemy relational schema & SQLite/PostgreSQL fallback logic
├── setup_postgres.py           # Automated PostgreSQL DB creator and table initializer
├── create_samples.py           # Synthetic dataset generator for test documents
├── OmniRAG.postman_collection.json # Postman collection for 1-click import
├── OmniRAG_Complete_User_and_API_Guide.docx # Word documentation manual
│
├── rag/                        # Core Retrieval-Augmented Generation Engine
│   ├── chunking.py             # Recursive, Semantic Sentence, and Structured chunkers
│   ├── embeddings.py           # Gemini 768-d & Local Fallback hashing embedder
│   ├── vector_store.py         # Qdrant client, collections, and HNSW cosine search
│   ├── scoring.py              # BM25Okapi, Reciprocal Rank Fusion (RRF), Cross-Reranker
│   ├── simple_rag.py           # Simple RAG pipeline (Dense vector retrieval)
│   ├── hybrid_rag.py           # Hybrid RAG pipeline (Dense Qdrant + Sparse BM25 + RRF)
│   ├── graph_rag.py            # Graph RAG pipeline (Entity-Relation Extraction + NetworkX)
│   └── memory.py               # Conversational Session Memory & Pronoun Coreference Resolver
│
├── llm/                        # Language Model Layer & Multi-Engine Adapters
│   ├── prompts.py              # English, Hindi, Python coding, and KG extraction prompts
│   ├── gemini_client.py        # Gemini API wrapper with token & latency telemetry
│   └── engine_factory.py       # Engine factory for Gemini, vLLM, and SGLang
│
├── optimization/               # Performance & Speed Optimization Suite
│   ├── cache.py                # Semantic Vector Response Cache (<10ms)
│   ├── query_rewriter.py       # Query expansion & sub-query decomposition
│   ├── hyde.py                 # Hypothetical Document Embeddings
│   └── compression.py          # Context compression and sentence pruning
│
├── evaluation/                 # Diagnostics & Multi-Level Evaluation Lab
│   ├── prompt_eval.py          # Prompt clarity, token efficiency, injection guard
│   ├── response_eval.py        # Faithfulness, hallucination rate, answer relevancy
│   ├── model_eval.py           # Cross-model consistency & stability benchmark
│   ├── corruption.py           # Corruption stress testing (OCR typos, token omission, distractors)
│   ├── latency_profiler.py     # Latency breakdown (TTFT, TPS, P95 metrics)
│   └── evaluator.py            # Unified evaluation orchestrator
│
└── tests/
    └── test_all.py             # 10 automated unit & integration tests
```

---

### Adding a New Chunking Strategy
Open [`rag/chunking.py`](file:///c:/Users/deepa/OneDrive/Desktop/AI%20ML%20DATASET/RAG_FINAL_PROJECT/rag/chunking.py):
1. Create a subclass inheriting from `BaseChunker`:
   ```python
   class CustomRegexChunker(BaseChunker):
       def chunk(self, text: str, metadata: dict = None) -> List[Dict[str, Any]]:
           # Implement your custom splitting logic
           ...
   ```
2. Register it in `get_chunker(strategy_name, ...)`:
   ```python
   elif strategy_name == "custom_regex":
       return CustomRegexChunker(...)
   ```

---

### Adding a New LLM Provider (Ollama, Claude, DeepSeek)
Open [`llm/engine_factory.py`](file:///c:/Users/deepa/OneDrive/Desktop/AI%20ML%20DATASET/RAG_FINAL_PROJECT/llm/engine_factory.py):
1. Create a new engine adapter class:
   ```python
   class OllamaEngine(BaseLLMEngine):
       def __init__(self, base_url="http://localhost:11434", model="llama3.1"):
           self.base_url = base_url
           self.model = model

       def generate(self, prompt: str, system_prompt: str = None, **kwargs):
           # Call Ollama REST API via httpx or requests
           ...
           return answer, telemetry_dict
   ```
2. Register your engine in `get_llm_engine(backend_name)`:
   ```python
   elif backend_name == "ollama":
       return OllamaEngine(...)
   ```

---

### Tuning Hybrid RAG & BM25 Scoring
Open [`rag/scoring.py`](file:///c:/Users/deepa/OneDrive/Desktop/AI%20ML%20DATASET/RAG_FINAL_PROJECT/rag/scoring.py):
- **Adjust RRF Constant $k$**: Modifying $k$ in `reciprocal_rank_fusion(dense_results, sparse_results, k=60)` changes how aggressively top ranks are rewarded vs lower ranks.
- **Tune BM25 Parameters**: Adjust $k_1$ (term frequency saturation) and $b$ (document length normalization) in `BM25Scorer`.

---

### Adding New Unit Tests
Open [`tests/test_all.py`](file:///c:/Users/deepa/OneDrive/Desktop/AI%20ML%20DATASET/RAG_FINAL_PROJECT/tests/test_all.py):
Add a new method prefixed with `test_`:
```python
def test_11_my_new_feature(self):
    """Test custom pipeline logic."""
    ...
    self.assertTrue(...)
```
Run the test suite:
```bash
python -m unittest tests.test_all.TestRAGSystem.test_11_my_new_feature
```

---

### Customizing the Web UI
Open [`main.py`](file:///c:/Users/deepa/OneDrive/Desktop/AI%20ML%20DATASET/RAG_FINAL_PROJECT/main.py):
- The HTML, CSS, and Vanilla JavaScript for the Glassmorphism Web Dashboard are located inside the `index_page()` function starting around line 355.
- CSS classes follow a modern glassmorphism design system (`.glass-card`, `.nav-item`, `.tab-content`, `.btn-primary`).
- JavaScript client methods (`executeQuery()`, `uploadDocument()`, `loadKnowledgeGraph()`, `runCorruptionTest()`) interface asynchronously via `fetch()` with the FastAPI `/api/*` endpoints.

---

## 5. Testing via Swagger Docs & Postman

### Swagger UI (`/docs`)
1. Ensure the server is running (`python main.py`).
2. Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.
3. Click on any endpoint (e.g. `POST /api/query`) $\to$ **Try it out** $\to$ customize JSON $\to$ **Execute**.

### Postman Application
1. Open Postman.
2. Click **Import** in the top left.
3. Drag and drop the bundled file: **`OmniRAG.postman_collection.json`** (in the root directory).
4. All 15+ endpoints are organized into 6 folders with pre-configured variables:
   - `{{base_url}}`: `http://127.0.0.1:8000`
   - `{{session_id}}`: `postman_demo_session`
5. Select any request and click **Send**.

---

## 6. Troubleshooting & Common Gotchas

### 1. `Execution_Policies` error on Windows PowerShell
**Symptom:** Running `.\venv\Scripts\activate` fails with `cannot be loaded because running scripts is disabled on this system`.
**Solution:** Run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate
```

### 2. Port 8000 is already in use
**Symptom:** `OSError: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)`
**Solution:**
- Run on a different port:
  ```bash
  uvicorn main:app --port 8080 --reload
  ```
- Or terminate the existing process using port 8000:
  - *Windows:*
    ```powershell
    netstat -ano | findstr :8000
    taskkill /PID <PID_NUMBER> /F
    ```
  - *Linux / macOS:*
    ```bash
    lsof -i :8000
    kill -9 <PID_NUMBER>
    ```

### 3. Qdrant Storage Locking Error
**Symptom:** `qdrant_client.exceptions.UnexpectedResponse: Storage directory ./data/qdrant_storage is already accessed by another instance of Qdrant client`.
**Solution:** Local embedded Qdrant accesses the directory directly on disk. Only **one** Python process can write to `./data/qdrant_storage` simultaneously.
- Make sure no duplicate `python main.py` or indexer scripts are running in the background.
- If stuck, terminate background Python processes or delete `./data/qdrant_storage/.lock` if present.

### 4. Gemini API Rate Limits or Quota Exceeded
**Symptom:** `ResourceExhausted 429 quota exceeded for quota metric 'Queries'`.
**Solution:**
- Change `GEMINI_MODEL` in `.env` to `gemini-flash-latest` (which has higher rate limits).
- Or switch embeddings to local mode in `.env`:
  ```env
  DEFAULT_EMBEDDING_STRATEGY=local
  ```

### 5. PostgreSQL Connection Error
**Symptom:** `[Database Warning] PostgreSQL connection failed. Falling back to local SQLite`.
**Explanation:** This is an **intentional safety feature**! If your target computer doesn't have PostgreSQL installed or credentials differ, OmniRAG automatically boots with SQLite at `./data/rag_app.db` without crashing.
- To use PostgreSQL: Ensure PostgreSQL service is started and verify credentials in `.env`, then run `python setup_postgres.py`.
- To use SQLite: Do nothing! The system is 100% operational with SQLite.

### 6. ModuleNotFoundError: No module named 'rag'
**Symptom:** When running scripts inside subdirectories (e.g. `python rag/hybrid_rag.py`), Python throws `ModuleNotFoundError`.
**Solution:** Always execute Python commands from the **root directory** of the project (`RAG_FINAL_PROJECT/`), or run using module syntax:
```bash
python -m unittest tests/test_all.py
```

---

## 7. Summary Checklist for Moving to a New Machine

| Step | Action | Status |
|---|---|---|
| 1 | Clone or copy `RAG_FINAL_PROJECT` folder | [ ] |
| 2 | Create virtual environment (`python -m venv venv`) | [ ] |
| 3 | Activate virtual environment (`.\venv\Scripts\activate` or `source venv/bin/activate`) | [ ] |
| 4 | Install requirements (`pip install -r requirements.txt`) | [ ] |
| 5 | Copy `.env.example` to `.env` and set `GEMINI_API_KEY` | [ ] |
| 6 | Generate sample files (`python create_samples.py`) | [ ] |
| 7 | Ingest documents (`python index.py --all-samples`) | [ ] |
| 8 | Run test suite (`python -m unittest tests/test_all.py`) | [ ] |
| 9 | Start server (`python main.py`) & open `http://127.0.0.1:8000` | [ ] |
| 10 | Import `OmniRAG.postman_collection.json` into Postman | [ ] |

You are now fully configured and ready to code, test, and deploy OmniRAG on any computer!
