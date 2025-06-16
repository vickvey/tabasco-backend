FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally via pip
RUN pip install --no-cache-dir uv

# Copy pyproject.toml and install deps via uv
COPY pyproject.toml .
RUN uv pip install --system --no-cache-dir .

# Install torch-specific deps
COPY requirements-torch.txt .
RUN uv pip install --system --no-cache-dir -r requirements-torch.txt

# Copy the rest of the app
COPY . .

EXPOSE 8000

# Run the app via uv CLI like in Makefile
CMD ["uv", "run", "fastapi", "dev", "./main.py"]
