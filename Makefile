# Makefile para automação de tarefas de qualidade

.PHONY: install test lint format format-check typecheck qa qa-strict qa-fix qa-strict-fix clean help run

help:  ## Mostra esta ajuda
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Instala dependências de desenvolvimento
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:  ## Executa todos os testes
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

lint:  ## Verifica estilo do código
	python -m flake8 src/

format:  ## Formata código automaticamente
	python -m black src/ tests/

format-check:  ## Verifica formatação sem alterar
	python -m black src/ tests/ --check

typecheck:  ## Verifica tipos
	python -m mypy src/

qa: lint format-check test  ## Executa todas as verificações de qualidade
qa-strict: lint format-check typecheck test  ## Executa QA incluindo verificação de tipos
qa-fix: lint format test  ## Formata automaticamente e executa QA
qa-strict-fix: lint format typecheck test  ## Formata automaticamente e executa QA completo

clean:  ## Remove arquivos temporários
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

run:  ## Executa o programa principal
	python src/main.py