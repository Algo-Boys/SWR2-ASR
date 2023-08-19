FROM python:3.10

# install python poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY readme.md mypy.ini poetry.lock pyproject.toml ./
COPY swr2_asr ./swr2_asr
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
RUN /root/.local/bin/poetry --no-interaction install --without dev

ENTRYPOINT [ "/root/.local/bin/poetry", "run", "python", "-m", "swr2_asr" ]
