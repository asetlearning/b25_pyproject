# Created by: Alex Myasnikov
# Date: February 10, 2026

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

RUN apt-get update && apt-get install -y \
    cmake g++ git ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy only code files (not data)
COPY pyproject.toml uv.lock .python-version CMakeLists.txt ./
COPY cpp/ ./cpp/
COPY algorithms/ ./algorithms/
COPY data/relators_io.py data/__init__.py ./data/
COPY experiments/ ./experiments/

RUN git submodule update --init --recursive

# Install runtime + build deps into .venv (no project yet — needs C++ submodule)
RUN uv sync --frozen --no-install-project --group build

# Build & install project (--no-build-isolation: uses build deps from .venv)
RUN uv pip install -e . --no-build-isolation

# Create mount points for volumes
RUN mkdir -p /app/data /app/output

# Default: run rewrite_experiment, but allow override
ENTRYPOINT ["uv", "run", "python", "-u"]
CMD ["experiments/rewrite_experiment.py"]
