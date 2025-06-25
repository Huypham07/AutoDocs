#!/bin/bash
ollama serve &
sleep 10
ollama pull nomic-embed-text
ollama pull qwen2.5:1.7b
wait