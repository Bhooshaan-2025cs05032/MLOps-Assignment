#!/bin/bash

set -e

IMAGE_NAME="heart-disease-api"
CONTAINER_NAME="heart-disease-container"

echo "🔴 Stopping existing container (if running)..."
docker stop $CONTAINER_NAME 2>/dev/null || true

echo "🗑 Removing existing container..."
docker rm $CONTAINER_NAME 2>/dev/null || true

echo "🧹 Removing existing image..."
docker rmi $IMAGE_NAME 2>/dev/null || true

echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME .

echo "🚀 Running container..."
docker run -d -p 8000:8000 --name $CONTAINER_NAME $IMAGE_NAME

echo "✅ Done!"
echo "🌐 Open: http://localhost:8000/docs"