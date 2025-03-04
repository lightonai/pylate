test:
	pytest pylate --durations=5 -n auto
	pytest tests --durations=5 -n auto

ruff:
	ruff format .

lint:
	ruff check .

livedoc:
	python docs/parse
	mkdocs build --clean
	mkdocs serve --dirtyreload

deploydoc:
	mkdocs gh-deploy --force


install: 
	pip install -e ".[dev]"

install-test:
	python -m pip install --upgrade pip
	pip install ".[dev]"