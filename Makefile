format:
	@poetry run black .

lint:
	@poetry run mypy --strict swr2_asr
	@poetry run pylint swr2_asr