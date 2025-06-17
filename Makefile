include .env
export $(shell sed 's/=.*//' .env)

VENV := .venv

# Create the virtual environment if it doesn't exist
$(VENV)/bin/python:
	@echo "📦 Creating virtual environment..."
	uv venv $(VENV)

# Install dependencies
install: $(VENV)/bin/python
	@echo "📦 Installing dependencies from requirements-torch.txt..."
	uv pip install -r requirements-torch.txt

# Run the server using .env config
run: install
	@echo "🚀 Running with .env config (ENVIRONMENT = $(ENVIRONMENT))..."
	uv run uvicorn app.server:app --host $(HOST) --port $(PORT)

# Dev server with reload and docs
dev: install
	@echo "🔧 Running in development mode (ENVIRONMENT=development)..."
	ENVIRONMENT=development uv run uvicorn app.server:app --reload --host $(HOST) --port $(PORT)

# Production server with multiple workers
prod: install
	@echo "🚀 Running in production mode (ENVIRONMENT=production)..."
	ENVIRONMENT=production uv run uvicorn app.server:app --workers 4 --host $(HOST) --port $(PORT)
