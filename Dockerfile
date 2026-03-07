FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY src/ ./src/
COPY train.py evaluate.py neat_config.cfg ./

# Make neat_trader importable (mirrors pyproject.toml pythonpath=["src"])
ENV PYTHONPATH=/app/src

# Directories that should be mounted at runtime
# (data/cache -> avoid re-downloading; checkpoints -> resume training)
VOLUME ["/app/data", "/app/checkpoints"]

# Default: run full training
# Override with: docker run ... neat-trading python evaluate.py
CMD ["python", "train.py"]
