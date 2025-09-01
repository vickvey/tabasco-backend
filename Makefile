VENV := .venv
PYTHON := $(VENV)/bin/python
UV := uv

# Create the virtual environment if it doesn't exist
$(PYTHON):
	@echo "📦 Creating virtual environment..."
	$(UV) venv $(VENV)

# Install dependencies
install: $(PYTHON)
	@echo "📦 Installing dependencies from pyproject.toml..."
	$(UV) sync
# 	@echo "📦 Installing torch (CPU version only)..."
# 	$(UV) pip install -r requirements-torch.txt

# Run dev server with autoreload and docs
dev: install
	@echo "🔧 Running FastAPI in development mode..."
	$(UV) run fastapi dev ./src/server.py

# Run production server with Docker
prod:
	@echo "🚀 Building and running FastAPI in production mode with Docker..."
	docker build -t fastapi-app .
	docker run -p 8000:80 fastapi-app

# Run tests via pytest
test: install
	@echo "🧪 Running tests..."
	$(UV) run pytest

clean:
	rm -rf venv .venv __pycache__ .pytest_cache