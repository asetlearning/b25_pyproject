# Created by: Alex Myasnikov
# Date: February 10, 2026

FROM python:3.12-slim

WORKDIR /app

# Copy only code files (not data)
COPY pyproject.toml ./
COPY algorithms/ ./algorithms/
COPY data/relators_io.py data/__init__.py ./data/
COPY experiments/ ./experiments/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Create mount points for volumes
RUN mkdir -p /app/data /app/output

# Default: run rewrite_experiment, but allow override
ENTRYPOINT ["python3", "-u"]
CMD ["experiments/rewrite_experiment.py"]
