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

release:
	@if [ -z "$(VERSION)" ]; then echo "Usage: make release VERSION=X.Y.Z"; exit 1; fi
	@echo "Bumping version to $(VERSION)..."
	git checkout -b release/v$(VERSION)
	python3 -c "v='$(VERSION)'; t=', '.join(v.split('.')); open('pylate/__version__.py','w').write('from __future__ import annotations\n\nVERSION = ('+t+')\n\n__version__ = \".\".join(map(str, VERSION))\n')"
	git add pylate/__version__.py
	git commit -m "Bump version to $(VERSION)"
	git push -u origin release/v$(VERSION)
	gh pr create --title "Release v$(VERSION)" --body "Bump version to $(VERSION)"
	@echo ""
	@echo "PR created. After merge, run: make tag VERSION=$(VERSION)"

tag:
	@if [ -z "$(VERSION)" ]; then echo "Usage: make tag VERSION=X.Y.Z"; exit 1; fi
	git checkout main
	git pull
	git tag "v$(VERSION)"
	git push --tags
	@echo "Tag v$(VERSION) pushed. PyPI publish workflow triggered."
