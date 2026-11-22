
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY audit_rl_sim ./audit_rl_sim
COPY tests ./tests

RUN python -m pip install --no-cache-dir -e ".[dev]"

CMD ["pytest", "-q"]
