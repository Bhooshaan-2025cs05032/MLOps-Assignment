#!/bin/bash

set -e

IMAGE_NAME="heart-disease-api"
CONTAINER_NAME="heart-disease-container"

# -----------------------------
# Load .env automatically
# -----------------------------
if [ -f .env ]; then
  echo "📦 Loading environment variables from .env..."
  
  # Export variables safely (ignore comments and empty lines)
  export $(grep -v '^#' .env | grep -v '^$' | xargs)
else
  echo "⚠️  No .env file found. Using system environment variables..."
fi

# -----------------------------
# Stop & cleanup
# -----------------------------
echo "🔴 Stopping existing container (if running)..."
docker stop $CONTAINER_NAME 2>/dev/null || true

echo "🗑 Removing existing container..."
docker rm $CONTAINER_NAME 2>/dev/null || true

echo "🧹 Removing existing image..."
docker rmi $IMAGE_NAME 2>/dev/null || true

# -----------------------------
# Build image
# -----------------------------
echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME .

# -----------------------------
# Run container with env vars
# -----------------------------
echo "🚀 Running container..."

docker run -d -p 8000:8000 \
  --name $CONTAINER_NAME \
  -e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  -e MLFLOW_TRACKING_USERNAME="$MLFLOW_TRACKING_USERNAME" \
  -e MLFLOW_TRACKING_PASSWORD="$MLFLOW_TRACKING_PASSWORD" \
  $IMAGE_NAME

# -----------------------------
# Done
# -----------------------------
echo "✅ Done!"
echo "🌐 Open: http://localhost:8000/docs"