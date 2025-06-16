VENV := .venv

run: $(VENV)/bin/activate
	uv pip install -r requirements-torch.txt
	uv run fastapi dev ./main.py

$(VENV)/bin/activate:
	@echo "Creating virtual environment..."
	uv venv $(VENV)
