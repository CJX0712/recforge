.PHONY: install test lint demo bench ci clean

install:
	python -m venv envs/recforge
	envs/recforge/bin/pip install -r requirements.txt

lint:
	python -m ruff check .

test:
	python -m pytest -q -W ignore::UserWarning

demo:
	python examples/run_demo.py

bench: demo

ci: lint test demo

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ .ruff_cache
	rm -f benchmark.json
