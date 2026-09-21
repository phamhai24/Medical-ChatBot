.PHONY: help install test lint eval docker-build docker-up docker-down ingest clean be fe

help:
	@echo "Medical RAG Chatbot - Available commands:"
	@echo "  make be           Start backend (API server)"
	@echo "  make fe           Start frontend (dev server)"
	@echo "  make install       Install backend dependencies"
	@echo "  make test         Run backend tests"
	@echo "  make lint         Run linting"
	@echo "  make ingest       Ingest data into vector store"
	@echo "  make eval         Run evaluation"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-up    Start all services with Docker"
	@echo "  make docker-down  Stop Docker services"
	@echo "  make clean        Clean cache and generated files"

be:
	cd backend && python -m src.api.main

fe:
	cd frontend && npm run dev

install:
	cd backend && pip install -r requirements.txt

test:
	cd backend && pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	cd backend && pytest tests/unit/ -v

test-integration:
	cd backend && pytest tests/integration/ -v

lint:
	cd backend && ruff check src/ tests/ --fix
	cd backend && mypy src/ --ignore-missing-imports || true

ingest:
	cd backend && python scripts/ingest_data.py --rebuild

ingest-no-rebuild:
	cd backend && python scripts/ingest_data.py

eval:
	cd backend && python scripts/run_eval.py --output reports/ --format html --format csv

eval-llm:
	cd backend && python scripts/run_eval.py --output reports/ --llm-judge --use-api-judge

docker-build:
	docker build -t medical-rag-chatbot:latest ./backend

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

clean:
	rm -rf backend/__pycache__ backend/.pytest_cache .mypy_cache
	find backend -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find backend -type f -name "*.pyc" -delete
	rm -rf backend/htmlcov/ backend/.coverage
