test:
	pytest pylate --durations=5 -n auto
	pytest tests --durations=5 -n auto

lint:
	pre-commit run --all-files

builddoc:
	python docs/parse
	mkdocs build --clean

livedoc: builddoc
	mkdocs serve --dirtyreload

deploydoc:
	mkdocs gh-deploy --force

install:
	pip install -e ".[dev]"
	pre-commit install

install-test:
	python -m pip install --upgrade pip
	pip install ".[dev]"
