FROM python:3.12-slim

# Install uv from the multi-stage build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install core dependencies from pyproject.toml using uv
RUN uv sync --frozen --no-cache

# Install additional requirements not tracked by uv (like torch)
RUN uv pip install -r requirements-torch.txt --no-cache

# Expose port
EXPOSE 80

# Run the FastAPI app using uvicorn
CMD ["fastapi", "run", "app/main.py", "--port", "80"]
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
