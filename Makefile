format:
	@black .

lint:
	@mypy --strict swr2_asr
	@pylint swr2_asr