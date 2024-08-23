test:
	pytest pylate
	pytest tests

ruff:
	ruff format pylate

lint:
	ruff check pylate

livedoc:
	python docs/parse
	mkdocs build --clean
	mkdocs serve --dirtyreload

deploydoc:
	mkdocs gh-deploy --force