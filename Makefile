test:
	pytest pylate
	pytest tests

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
