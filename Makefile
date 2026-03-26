.PHONY: sync
sync:
	uv sync --group dev

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run pyright

.PHONY: check
check: format type

.PHONY: test
test:
	uv run pytest

.PHONY: build
build:
	uv build
	uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
	uv run --isolated --no-project --with dist/*.tar.gz tests/smoke_test.py
	@echo "Build and smoke test successful"

.PHONY: publish-test
publish-test: build
	uv publish --publish-url https://test.pypi.org/legacy/

.PHONY: publish
publish: build
	uv publish
