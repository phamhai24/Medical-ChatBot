# Medical RAG Chatbot

> **Portfolio Project** — Retrieval-Augmented Generation cho chatbot y tế tiếng Việt. Dự án chuẩn production với full evaluation framework, Docker containerization, và automated CI/CD.

[![CI](https://github.com/YOUR_USERNAME/medical-rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/medical-rag-chatbot/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Demo

```
User: Triệu chứng bệnh tiểu đường type 2 là gì?

Assistant: Bệnh tiểu đường type 2 có các triệu chứng thường gặp:
- Khát nhiều nước (polydipsia)
- Đi tiểu thường xuyên (polyuria)
- Mệt mỏi, tăng cảm giác đói
- Nhìn mờ
- Vết thương lâu lành

⚠️ Lưu ý: Thông tin này chỉ mang tính tham khảo. Hãy tham khảo ý kiến bác sĩ.
```

## Features

| Module | Chi tiết |
|---|---|
| **RAG Pipeline** | Retrieval-Augmented Generation với ChromaDB vector store |
| **Embedder** | `paraphrase-multilingual-MiniLM-L12-v2` — hỗ trợ tiếng Việt |
| **Generator** | Local (Qwen2.5-7B-Instruct) hoặc API (Groq/OpenAI) |
| **Retrieval** | Vector search + BM25 hybrid + MMR (Max Marginal Relevance) |
| **Evaluation** | Hit Rate, MRR, NDCG, RAGAS, LLM-as-Judge |
| **API** | FastAPI với versioned endpoints, Pydantic schemas |
| **CI/CD** | GitHub Actions — lint, test, eval, Docker build |
| **Container** | Multi-stage Dockerfile, Docker Compose |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Streamlit UI                          │
│               (http://localhost:8501)                        │
└──────────────────────┬────────────────────────────────────────┘
                       │ HTTP / Streaming
┌──────────────────────▼────────────────────────────────────┐
│                   FastAPI Backend                            │
│  /api/v1/chat/ask  ·  /api/v1/chat/stream                 │
│  /api/v1/chat/history  ·  /api/v1/admin/ingest             │
│  /api/v1/admin/reindex  ·  /health  ·  /metrics             │
└──────┬──────────────────┬──────────────────┬───────────────┘
       │                  │                  │
┌──────▼──────┐  ┌───────▼──────┐  ┌──────▼──────┐
│  ChromaDB   │  │  Redis Cache │  │ Prometheus   │
│ (vector DB) │  │ (sessions)   │  │  (metrics)   │
└─────────────┘  └──────────────┘  └──────────────┘
```

```
RAG Pipeline Flow:
User Query → Embed Query → ChromaDB Retrieval → Hybrid Rerank (BM25 + vector)
           → Build Context → LLM Generation → Response + Sources
```

## Quick Start

### Docker (Recommended)

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/medical-rag-chatbot.git
cd medical-rag-chatbot

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start all services
docker compose up -d

# API: http://localhost:8000/docs
# UI:   http://localhost:8501
```

### Local Development

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env

# 4. Ingest data
python -m src.rag.ingest --config config/rag_config.yaml --rebuild

# 5. Start API
python -m src.api.main

# 6. (Separate terminal) Start UI
streamlit run src/web/app.py
```

### CLI Usage

```bash
# Ingest data
python -m src.cli.main ingest --rebuild

# Run evaluation
python -m src.cli.main eval --output reports --format html --format csv

# Start server
python -m src.cli.main serve --port 8000

# Check stats
python -m src.cli.main stats
```

## Evaluation

Evaluation framework bao gồm:

| Loại | Metrics |
|---|---|
| **Retrieval** | Hit Rate, MRR, NDCG@k, Precision@k, Recall@k |
| **Generation** | Faithfulness, Answer Relevance, Context Precision/Recall |
| **LLM-as-Judge** | Accuracy, Completeness, Clarity, Safety, Hallucination |

Chạy evaluation:

```bash
# Basic evaluation (20 benchmark questions)
python scripts/run_eval.py --output reports/

# With LLM-as-Judge
python scripts/run_eval.py --output reports/ --llm-judge --use-api-judge
```

Output: `reports/eval_report_YYYYMMDD_HHMM.html` (HTML report + CSV)

## Configuration

Cấu hình qua `.env`:

```env
# API Server
API_HOST=0.0.0.0
API_PORT=8000

# Redis (optional)
REDIS_ENABLED=false
REDIS_HOST=localhost
REDIS_PORT=6379

# Embedding
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cpu  # or cuda

# Vector Store
VECTOR_STORE_TYPE=chroma
VECTOR_STORE_PATH=data/vectorstore
RETRIEVAL_TOP_K=5

# Generation (local)
GENERATOR_MODE=local
GENERATOR_MODEL=Qwen/Qwen2.5-7B-Instruct
GENERATOR_TEMPERATURE=0.3

# Generation (API - Groq)
GENERATOR_MODE=api
GENERATOR_API_PROVIDER=groq
GENERATOR_API_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=your_key_here

# Evaluation
RAGAS_ENABLED=false

# Logging
LOG_LEVEL=INFO
LOG_JSON=false
```

## API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/v1/chat/ask` | Hỏi câu hỏi, nhận JSON response |
| `POST` | `/api/v1/chat/stream` | Streaming response |
| `GET` | `/api/v1/chat/history/{session_id}` | Lấy lịch sử chat |
| `POST` | `/api/v1/chat/history/new` | Tạo session ID mới |
| `DELETE` | `/api/v1/chat/history/{session_id}` | Xóa lịch sử chat |
| `POST` | `/api/v1/admin/ingest` | Trigger data ingestion |
| `POST` | `/api/v1/admin/reindex` | Rebuild vector index |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |

Swagger docs: `http://localhost:8000/docs`

## Project Structure

```
Chatbot Y tế/
├── src/
│   ├── api/              # FastAPI (routes, schemas, deps)
│   │   ├── main.py      # App entry + middleware
│   │   ├── schemas.py   # Pydantic models
│   │   ├── deps.py      # Dependency injection
│   │   └── routes/      # /chat, /admin, /health
│   ├── rag/              # RAG pipeline modules
│   │   ├── pipeline.py  # Orchestrator
│   │   ├── embedder.py  # Sentence-transformers
│   │   ├── retriever.py # Retrieval logic
│   │   ├── generator.py # LLM (local)
│   │   ├── api_generator.py # LLM (API)
│   │   ├── vector_store.py # ChromaDB/FAISS
│   │   ├── chunker.py  # Text splitting
│   │   ├── reranker.py # Cross-encoder reranking
│   │   └── hybrid_search.py # BM25 + vector fusion
│   ├── ingestion/        # Data ingestion pipeline
│   │   ├── pipeline.py  # Orchestrator
│   │   └── loaders/    # JSON loader
│   ├── eval/            # Evaluation framework
│   │   ├── evaluator.py # Main orchestrator
│   │   ├── metrics/    # Retrieval + generation metrics
│   │   ├── benchmarks/ # Medical Q&A dataset
│   │   └── reporters/   # HTML/CSV reports
│   │       └── report.py
│   ├── core/            # Foundation modules
│   │   ├── config.py   # Pydantic Settings
│   │   ├── logging.py  # Loguru setup
│   │   ├── exceptions.py # Custom exceptions
│   │   ├── metrics.py  # Prometheus metrics
│   │   └── redis_client.py # Redis session/cache
│   ├── web/             # Streamlit UI
│   │   └── app.py      # Refactored Streamlit app
│   ├── utils/           # Utilities
│   └── cli/             # CLI tools
├── tests/               # pytest tests
│   ├── unit/           # Unit tests
│   └── integration/    # API tests
├── scripts/             # CLI scripts
│   ├── run_eval.py     # Evaluation runner
│   └── ingest_data.py  # Ingestion runner
├── config/
│   └── rag_config.yaml # RAG configuration
├── .github/workflows/   # CI/CD pipelines
│   ├── ci.yml         # Lint, test, docker
│   ├── eval.yml       # Automated evaluation
│   └── release.yml    # Docker release
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **Backend** | FastAPI + Pydantic v2 | 0.109+ / 2.4+ |
| **Embedder** | sentence-transformers | 2.2+ |
| **Vector DB** | ChromaDB | 0.4+ |
| **Generator** | Qwen2.5-7B-Instruct / Groq API | - |
| **Cache** | Redis | 7+ |
| **Metrics** | prometheus-client | 0.17+ |
| **Logging** | Loguru | 3.8+ |
| **UI** | Streamlit | 1.29+ |
| **Tests** | pytest + pytest-asyncio | 7.4+ / 0.21+ |
| **Container** | Docker + Compose | 24+ |

## Contributing

1. Fork và create a feature branch
2. Run tests: `pytest tests/ -v`
3. Ensure linting passes: `ruff check src/ tests/`
4. Submit a pull request

## License

MIT License — xem [LICENSE](LICENSE).
