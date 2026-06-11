.PHONY: help install test lint eval docker-build docker-up docker-down ingest clean

help:
	@echo "Medical RAG Chatbot - Available commands:"
	@echo "  make install       Install dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linting"
	@echo "  make ingest       Ingest data into vector store"
	@echo "  make eval         Run evaluation"
	@echo "  make serve        Start API server"
	@echo "  make ui           Start Streamlit UI"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-up    Start all services with Docker"
	@echo "  make docker-down  Stop Docker services"
	@echo "  make clean        Clean cache and generated files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check src/ tests/ --fix
	mypy src/ --ignore-missing-imports || true

ingest:
	python scripts/ingest_data.py --rebuild

ingest-no-rebuild:
	python scripts/ingest_data.py

eval:
	python scripts/run_eval.py --output reports/ --format html --format csv

eval-llm:
	python scripts/run_eval.py --output reports/ --llm-judge --use-api-judge

serve:
	python -m src.api.main

ui:
	streamlit run src/web/app.py

docker-build:
	docker build -t medical-rag-chatbot:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage reports/
