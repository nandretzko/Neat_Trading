FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY *.py ./
COPY neat_config.cfg .

# Directories that should be mounted at runtime
# (data/cache → avoid re-downloading; checkpoints → resume training)
VOLUME ["/app/data", "/app/checkpoints"]

# Default: run full training
# Override with: docker run ... neat-trading python evaluate.py
CMD ["python", "train.py"]
