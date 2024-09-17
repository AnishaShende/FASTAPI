#!/bin/sh
# Retry download up to 5 times with exponential backoff
retry() {
    local n=1
    local max=5
    local delay=5
    while true; do
        "$@" && break || {
            if [ $n -lt $max ]; then
                ((n++))
                echo "Command failed. Attempt $n/$max:"
                sleep $delay
            else
                echo "The command has failed after $n attempts."
                return 1
            fi
        }
    done
}

# Install Ollama using curl and bash
retry curl -fsSL https://ollama.com/install.sh | bash

# Check installation status
if command -v ollama >/dev/null 2>&1; then
    echo "Ollama installed successfully"
else
    echo "Ollama installation failed"
    exit 1
fi

# Pull Llama3.1 model
retry ollama pull llama3.1

# Start Ollama server in the background
ollama serve &

# Wait until the Ollama server is running
until curl -s http://127.0.0.1:11434 >/dev/null; do
    echo "Waiting for Ollama server to be ready..."
    sleep 2
done

echo "Ollama server is up and running!"
