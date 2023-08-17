format:
	@poetry run black .

lint:
	@poetry run mypy --strict .
	@poetry run pylint swr2_asr