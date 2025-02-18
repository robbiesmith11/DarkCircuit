#!/bin/bash

echo "[*] Starting Ollama server..."
ollama serve &

# Wait for Ollama to be ready
echo "[*] Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 2
done
echo "[*] Ollama is ready!"

# Pull all models from models.txt
if [ -f /ollama/models.txt ]; then
    echo "[*] Pulling models..."
    while IFS= read -r model || [[ -n "$model" ]]; do
        if [[ ! -z "$model" ]]; then
            echo "[*] Pulling model: $model"
            ollama pull "$model" || echo "[!] Failed to pull $model"
        fi
    done < /ollama/models.txt
else
    echo "[!] models.txt not found."
fi

# Keep Ollama running
wait
