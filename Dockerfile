FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy pyproject and install dependencies
COPY pyproject.toml ./
RUN uv pip install --no-cache-dir --system .

# Install additional torch/ML dependencies
COPY requirements-torch.txt ./
RUN uv pip install --no-cache-dir --system -r requirements-torch.txt

# Copy full project
COPY . .

# Run app using uvicorn
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
