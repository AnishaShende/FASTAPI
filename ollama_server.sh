#!/bin/sh
# curl -fsSL https://ollama.com/install.sh | sh
curl -fsSL https://ollama.com/install.sh | bash
# Check installation status
if command -v ollama >/dev/null 2>&1; then
    echo "Ollama installed successfully"
else
    echo "Ollama installation failed"
    exit 1
fi
ollama pull llama3.1
ollama serve
