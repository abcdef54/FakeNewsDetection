#!/bin/bash

echo "[1/2] Building Docker image 'viet-fake-news-detector'..."
docker build -t viet-fake-news-detector .
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Docker build failed. Please check if Docker daemon is running and configured correctly."
    exit 1
fi
echo "[SUCCESS] Docker image built successfully."
echo ""

echo "[2/2] Launching container in interactive mode..."
echo "Mounting local persistent volumes:"
echo "  - Checkpoints:   $(pwd)/src/checkpoints"
echo "  - KnowledgeBase: $(pwd)/KnowledgeBase"
echo "  - Cache:         $(pwd)/src/CACHES"
echo ""

docker run -it --gpus all \
  -v "$(pwd)/src/checkpoints:/app/src/checkpoints" \
  -v "$(pwd)/KnowledgeBase:/app/KnowledgeBase" \
  -v "$(pwd)/src/CACHES:/app/src/CACHES" \
  viet-fake-news-detector --interactive

DOCKER_EXIT_CODE=$?

if [ $DOCKER_EXIT_CODE -eq 125 ]; then
    echo ""
    echo "[WARNING] GPU acceleration (--gpus all) is not supported or configured in Docker."
    echo "Retrying in CPU mode..."
    echo ""
    docker run -it \
      -v "$(pwd)/src/checkpoints:/app/src/checkpoints" \
      -v "$(pwd)/KnowledgeBase:/app/KnowledgeBase" \
      -v "$(pwd)/src/CACHES:/app/src/CACHES" \
      viet-fake-news-detector --interactive
fi

echo ""
echo "Container session closed."
