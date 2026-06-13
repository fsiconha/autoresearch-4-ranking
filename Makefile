.PHONY: test data run

test:
	uv run python -m pytest tests/ -v

data:
	uv run python src/generate_data.py

run:
	uv run python train.py
