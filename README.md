# Medical RAG Chatbot

> Chatbot hỏi đáp y tế tiếng Việt dựa trên **Retrieval-Augmented Generation**: truy xuất lai (vector + BM25), rerank bằng cross-encoder, trả lời có trích dẫn nguồn, kèm bộ đánh giá độc lập.

[![CI](https://github.com/phamhai24/Medical-ChatBot/actions/workflows/ci.yml/badge.svg)](https://github.com/phamhai24/Medical-ChatBot/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> ⚠️ Thông tin do chatbot cung cấp chỉ mang tính tham khảo, không thay thế chẩn đoán hay chỉ định của bác sĩ.

## Điểm nổi bật

| | |
|---|---|
| **Truy xuất lai** | Vector search (ChromaDB) + BM25 trên toàn bộ corpus (`bm25s`), gộp bằng Reciprocal Rank Fusion |
| **Rerank** | Cross-encoder `BAAI/bge-reranker-v2-m3` chấm lại 20 ứng viên, giữ top 5 |
| **Embedding** | `BAAI/bge-m3` (1024 chiều, đa ngôn ngữ, có tiếng Việt), chạy fp16 trên GPU |
| **Trích dẫn** | Câu trả lời đánh số `[1]`, `[2]`… trỏ đúng đoạn nguồn; nguồn không liên quan bị lọc bỏ; phát hiện khi mô hình từ chối trả lời |
| **Streaming** | `/api/v1/chat/stream` trả NDJSON (từng đoạn chữ, rồi danh sách nguồn) |
| **Đánh giá** | Benchmark độc lập 60 câu (có 12 câu đối kháng), Hit Rate / MRR / NDCG, LLM-as-Judge |
| **Triển khai** | Docker Compose (API + Redis + React/nginx), có GPU |

## Kết quả

Đo trên benchmark độc lập (`backend/data/eval/independent_benchmark_v1.json`): 48 câu trong phạm vi + 12 câu đối kháng (hỏi ngoài phạm vi, hỏi liều thuốc/tiên lượng cá nhân…).

| Chỉ số | Giá trị |
|---|---|
| NDCG@5 | 83.9% |
| Hit Rate@5 | 80.0% |
| MRR | 0.80 |
| LLM-as-Judge (tổng thể) | 4.85 / 5 |
| Câu đối kháng được từ chối đúng | 12 / 12 |
| Độ trễ mỗi câu hỏi (chạy trực tiếp, sau warm-up, RTX 3050 Laptop 4GB) | ~3.5–5.6 s |

Độ trễ giảm từ ~126 s xuống còn vài giây nhờ chạy embedder và reranker ở fp16 (hai model fp32 cộng lại tràn 4GB VRAM, khiến Windows đẩy sang RAM dùng chung) và giới hạn rerank ở 20 ứng viên. Cách xây dựng benchmark và phân tích lỗi (đo trên phiên bản trước khi chuyển sang bge-m3): [`backend/reports/Independent_Benchmark_Report_v1_20260916.md`](backend/reports/Independent_Benchmark_Report_v1_20260916.md).

## Kiến trúc

```
                ┌──────────────────────────────────────────┐
                │  React + Vite + TypeScript (nginx :3000) │
                └────────────────────┬─────────────────────┘
                                     │ /api/v1/chat/ask · /api/v1/chat/stream (NDJSON)
                ┌────────────────────▼─────────────────────┐
                │              FastAPI (:8000)             │
                └───┬──────────────┬───────────────┬───────┘
                    │              │               │
             ┌──────▼─────┐ ┌──────▼──────┐ ┌──────▼──────┐
             │  ChromaDB  │ │ BM25 (bm25s)│ │ Redis (tùy  │
             │  + bge-m3  │ │ toàn corpus │ │ chọn)       │
             └────────────┘ └─────────────┘ └─────────────┘
```

Luồng xử lý một câu hỏi:

```
Câu hỏi ─► bge-m3 embed ─► Chroma top-40 ─┐
        └─► BM25 top-40 ──────────────────┴─► RRF ─► bge-reranker (20 → 5)
        ─► ngữ cảnh đánh số [1..5] ─► LLM (gpt-4o-mini) ─► câu trả lời + trích dẫn + nguồn
```

Dữ liệu: 16.506 cặp hỏi–đáp y tế tiếng Việt, cắt thành ~289.000 đoạn (gom theo đoạn văn, 1200 ký tự, chồng lấn 100, không cắt giữa câu).

## Chạy bằng Docker

**Yêu cầu:** Docker Desktop có hỗ trợ GPU NVIDIA (WSL2 trên Windows), khoảng 20 GB trống.

**1. Tải sẵn model về máy.** Container đọc cache HuggingFace của máy ở chế độ chỉ đọc và offline, nên model phải có sẵn (~4.4 GB, chỉ tải một lần):

```bash
pip install "sentence-transformers>=2.2,<3"
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

**2. Cấu hình.**

```bash
cp backend/.env.example backend/.env
# Điền OPENAI_API_KEY (hoặc đổi API_GENERATOR_PROVIDER sang groq/anthropic và điền key tương ứng)
```

**3. Chuẩn bị dữ liệu.** Corpus không nằm trong repo. Đặt file đã xử lý tại `backend/data/processed/data.json` (sinh bởi `backend/data/CrawlData.ipynb`), rồi nạp thư mục `data/` vào named volume của Docker:

```bash
docker volume create medical_rag_chatbot_data
docker run --rm -v "$PWD/backend/data:/from:ro" -v medical_rag_chatbot_data:/to \
  alpine sh -c "cp -a /from/. /to/ && chown -R 1000:1000 /to"
```

Dữ liệu nằm trong named volume thay vì mount thư mục Windows, vì đọc qua bind-mount làm bước khởi động chậm thêm hàng trăm giây.

**4. Khởi động và ingest.**

```bash
docker compose up -d --build
make docker-ingest          # lần đầu: chunk + embed + dựng BM25, ghi thẳng vào volume (khá lâu)
```

`make docker-ingest` gọi `backend/scripts/docker_ingest.ps1` (PowerShell). Trên máy không có PowerShell, chạy tương đương:

```bash
docker compose exec api python scripts/ingest_data.py --config config/rag_config.yaml --rebuild
docker compose exec api python scripts/build_bm25_index.py
docker compose restart api
```

| Dịch vụ | Địa chỉ |
|---|---|
| Giao diện | http://localhost:3000 |
| API + Swagger | http://localhost:8000/docs |

Xem log: `docker compose logs -f api`. Backend sẵn sàng khi thấy dòng `LIVE and READY` (warm-up khoảng 1 phút).

> **Máy không có GPU:** xóa khối `deploy.resources` của service `api` trong `docker-compose.yml` và đặt `EMBEDDING_DEVICE=cpu` trong `backend/.env`. Rerank trên CPU sẽ chậm hơn đáng kể.

**Dọn dẹp sau khi build lại.** Mỗi lần `--build` để lại build cache và image cũ:

```bash
docker builder prune -f
docker image prune -f
```

## Chạy trực tiếp (không Docker)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# GPU: cài torch bản CUDA trước (bỏ qua nếu chạy CPU)
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-dev.txt

cp .env.example .env
python -m src.rag.ingest --config config/rag_config.yaml --rebuild
python scripts/build_bm25_index.py      # chạy lại sau mỗi lần ingest
python -m src.api.main                  # http://localhost:8000
```

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173
```

Hoặc dùng `make be` và `make fe` ở hai terminal.

> Chạy trực tiếp dùng dữ liệu trong `backend/data/` trên máy; Docker dùng volume `medical_rag_chatbot_data`. Hai nơi độc lập với nhau: ingest ở đâu thì chỉ nơi đó có dữ liệu mới.

## Đánh giá

```bash
cd backend
python scripts/run_independent_eval.py --output reports/            # đầy đủ, có LLM judge
python scripts/run_independent_eval.py --no-judge --limit 10        # chạy nhanh
python scripts/run_eval.py --output reports/                        # benchmark lấy từ corpus
```

Kết quả (HTML, CSV, JSON thô) được ghi vào `backend/reports/` và không đưa lên git; chỉ các báo cáo phân tích `.md` được giữ lại.

## API

| Method | Endpoint | Mô tả |
|---|---|---|
| `POST` | `/api/v1/chat/ask` | Hỏi, nhận JSON (câu trả lời + nguồn) |
| `POST` | `/api/v1/chat/stream` | Hỏi, nhận NDJSON streaming |
| `POST` | `/api/v1/chat/history/new` | Tạo phiên chat mới |
| `GET` | `/api/v1/chat/history/{session_id}` | Lấy lịch sử phiên |
| `DELETE` | `/api/v1/chat/history/{session_id}` | Xóa lịch sử phiên |
| `POST` | `/api/v1/admin/ingest` | Chạy ingest (header `X-Admin-Key` nếu có đặt `ADMIN_API_KEY`) |
| `POST` | `/api/v1/admin/reindex` | Dựng lại index |
| `GET` | `/health` | Trạng thái + số tài liệu |
| `GET` | `/stats` | Cấu hình đang chạy |
| `GET` | `/metrics` | Prometheus metrics |

## Cấu hình chính

Toàn bộ cấu hình nằm trong `backend/.env` (mẫu: [`backend/.env.example`](backend/.env.example)). `config/rag_config.yaml` chỉ còn dùng cho system prompt.

| Biến | Giá trị mẫu | Ý nghĩa |
|---|---|---|
| `API_GENERATOR_PROVIDER` / `API_GENERATOR_MODEL` | `openai` / `gpt-4o-mini` | LLM sinh câu trả lời |
| `OPENAI_API_KEY`, `GROQ_API_KEY` | — | Key theo provider |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Đổi model thì **bắt buộc** ingest lại (khác số chiều vector) |
| `EMBEDDING_DEVICE` | `cpu` | `cuda` hoặc `cpu` (Docker Compose tự đặt `cuda`) |
| `VECTOR_SEARCH_TYPE` | `hybrid` | `hybrid` (vector + BM25) hoặc `similarity` |
| `RETRIEVAL_FETCH_K` | `40` | Số ứng viên lấy từ mỗi nhánh vector/BM25 |
| `RETRIEVAL_RERANK_ENABLED` | `true` | Bật cross-encoder rerank |
| `RETRIEVAL_RERANK_FETCH_K` | `20` | Số ứng viên đưa vào rerank |
| `RETRIEVAL_TOP_K` | `5` | Số đoạn đưa vào ngữ cảnh |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1200` / `100` | Chỉ có hiệu lực ở lần ingest tiếp theo |
| `REDIS_ENABLED` | `false` | Lưu phiên chat và cache |
| `ADMIN_API_KEY` | trống | Bảo vệ các endpoint `/api/v1/admin/*` |

## Cấu trúc thư mục

```
├── backend/                  FastAPI + RAG pipeline
│   ├── src/
│   │   ├── api/              App, routes (chat, session, admin, health), schemas
│   │   ├── rag/              pipeline, embedder, retriever, bm25_index, reranker,
│   │   │                     chunker, vector_store, api_generator, ingest
│   │   ├── eval/             Metrics, LLM judge, reporters
│   │   ├── core/             Settings, logging, metrics, Redis
│   │   └── utils/            Config loader, answer_signals (phát hiện từ chối)
│   ├── scripts/              ingest_data, build_bm25_index, run_independent_eval,
│   │                         run_eval, docker_ingest.ps1
│   ├── tests/                unit/, integration/
│   ├── data/                 eval/ (benchmark), processed/ (corpus, không có trên git)
│   ├── reports/              Báo cáo đánh giá (.md)
│   ├── requirements.txt      Thư viện runtime (cài vào Docker image)
│   └── requirements-dev.txt  + test, lint, notebook
├── frontend/                 React 18 + Vite 5 + TypeScript + Tailwind, nginx
├── docker-compose.yml        api + redis + web
└── Makefile                  be, fe, install, test-unit, docker-ingest, …
```

## Kiểm thử

```bash
cd backend
pytest tests/unit -q
ruff check src tests
```

## Hạn chế đã biết

- BM25 tách từ theo khoảng trắng, chưa nhận diện từ ghép tiếng Việt.
- Chế độ sinh câu trả lời cục bộ (`GENERATOR_MODE=local`, Qwen2.5-7B) cần thêm thư viện không có trong `requirements.txt` và không được kiểm thử thường xuyên.
- Cấu hình mặc định được tinh chỉnh cho GPU 4 GB VRAM.

## License

MIT — xem [LICENSE](LICENSE).
