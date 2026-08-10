.PHONY: sync format lint typecheck test test-all compose-config compose-up compose-down

sync:
	uv sync --frozen

format:
	uv run ruff format .

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy src

test:
	uv run pytest -m "not integration and not e2e"

test-all:
	uv run pytest

compose-config:
	docker compose config --quiet

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down
