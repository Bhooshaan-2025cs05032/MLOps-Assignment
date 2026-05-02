# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]