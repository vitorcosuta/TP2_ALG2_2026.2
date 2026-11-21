# Comandos de desenvolvimento. Uso: make <alvo>   (make help lista os alvos)

VENV := .venv
SRC  ?= .
EXCL := .venv,build,dist

ifeq ($(OS),Windows_NT)
	BIN        := $(VENV)/Scripts
	PYTHON     := $(BIN)/python.exe
	SYS_PYTHON := python
else
	BIN        := $(VENV)/bin
	PYTHON     := $(BIN)/python
	SYS_PYTHON := python3
endif

.PHONY: help venv install lint format format-check typecheck complexity test check clean

help:
	@echo "venv          cria o ambiente virtual em $(VENV)"
	@echo "install       instala requirements.txt e requirements-dev.txt"
	@echo "lint          ruff check"
	@echo "format        ruff format (altera os arquivos)"
	@echo "format-check  ruff format --check"
	@echo "typecheck     mypy"
	@echo "complexity    radon: complexidade ciclomática e índice de manutenibilidade"
	@echo "test          pytest"
	@echo "check         lint + format-check + typecheck + complexity + test"
	@echo "clean         remove caches das ferramentas"
	@echo ""
	@echo "SRC=<caminho> restringe a análise (padrão: .)"

$(PYTHON):
	$(SYS_PYTHON) -m venv $(VENV)

venv: $(PYTHON)

install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

lint:
	$(BIN)/ruff check $(SRC)

format:
	$(BIN)/ruff format $(SRC)

format-check:
	$(BIN)/ruff format --check $(SRC)

typecheck:
	$(BIN)/mypy --exclude "^(\.venv|build|dist)/" $(SRC)

complexity:
	$(BIN)/radon cc -s -a -e "$(EXCL)" $(SRC)
	$(BIN)/radon mi -s -e "$(EXCL)" $(SRC)

test:
	$(BIN)/pytest

check: lint format-check typecheck complexity test

clean:
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .ruff_cache, .mypy_cache, .pytest_cache; Get-ChildItem -Recurse -Directory -Filter __pycache__ | Where-Object { $$_.FullName -notmatch '\\$(VENV)\\' } | Remove-Item -Recurse -Force"
else
	rm -rf .ruff_cache .mypy_cache .pytest_cache
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -exec rm -rf {} +
endif