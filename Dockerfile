# Created by: Alex Myasnikov
# Date: February 10, 2026

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    cmake g++ git ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy only code files (not data)
COPY pyproject.toml CMakeLists.txt ./
COPY cpp/ ./cpp/
COPY algorithms/ ./algorithms/
COPY data/relators_io.py data/__init__.py ./data/
COPY experiments/ ./experiments/

RUN git submodule update --init --recursive

# Install dependencies
RUN pip install --no-cache-dir -e . --no-build-isolation

# Create mount points for volumes
RUN mkdir -p /app/data /app/output

# Default: run rewrite_experiment, but allow override
ENTRYPOINT ["python3", "-u"]
CMD ["experiments/rewrite_experiment.py"]
