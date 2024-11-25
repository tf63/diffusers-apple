.PHONY: lint format test
lint:
	uv run ruff check src
	# uv run mypy src

format:
	uv run ruff format src
	uv run ruff check --fix src

test:
	uv run pytest
