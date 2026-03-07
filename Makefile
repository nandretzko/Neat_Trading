.PHONY: help install install-dev train train-fast evaluate test test-cov clean \
        docker-build docker-run

PYTHON := python
IMAGE  := neat-trading

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  NEAT Stock Trading Agent — available targets"
	@echo ""
	@echo "  Setup"
	@echo "    install        Install runtime dependencies"
	@echo "    install-dev    Install runtime + test dependencies"
	@echo ""
	@echo "  Training"
	@echo "    train          Full 2 000-generation training run"
	@echo "    train-fast     Smoke-test run  (50 / 20 / 10 generations)"
	@echo ""
	@echo "  Evaluation"
	@echo "    evaluate       Backtest best_genome.pkl on 100 windows"
	@echo ""
	@echo "  Tests"
	@echo "    test           Run unit-test suite"
	@echo "    test-cov       Run tests with coverage report"
	@echo ""
	@echo "  Docker"
	@echo "    docker-build   Build Docker image"
	@echo "    docker-run     Train inside Docker (mounts data/ and checkpoints/)"
	@echo ""
	@echo "  Maintenance"
	@echo "    clean          Remove all generated files and caches"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov

# ── Training ──────────────────────────────────────────────────────────────────
train:
	$(PYTHON) train.py

train-fast:
	$(PYTHON) train.py --fast

# ── Evaluation ────────────────────────────────────────────────────────────────
evaluate:
	$(PYTHON) evaluate.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/neat_trader --cov-report=term-missing --cov-report=html

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm \
	  -v "$(PWD)/data:/app/data" \
	  -v "$(PWD)/checkpoints:/app/checkpoints" \
	  -v "$(PWD)/best_genome.pkl:/app/best_genome.pkl" \
	  $(IMAGE)

docker-evaluate:
	docker run --rm \
	  -v "$(PWD)/data:/app/data" \
	  -v "$(PWD)/best_genome.pkl:/app/best_genome.pkl" \
	  -v "$(PWD):/app/output" \
	  $(IMAGE) python evaluate.py

# ── Maintenance ───────────────────────────────────────────────────────────────
clean:
	rm -rf data/cache/ checkpoints/
	rm -rf __pycache__/ .pytest_cache/ htmlcov/
	rm -f .coverage coverage.xml
	rm -f best_genome.pkl neat_config_used.pkl
	rm -f comparison_plot.png equity_sample.png
	find . -name "*.pyc" -delete
