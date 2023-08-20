format:
	@poetry run black .

format-check:
	@poetry run black --check .

lint:
	@poetry run mypy --strict swr2_asr
	@poetry run pylint swr2_asr