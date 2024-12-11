# Ravens Query Pipeline

This project sets up a local pipeline to query documents related to the Baltimore Ravens using [llama-index](https://github.com/jerryjliu/llama_index), [Chroma](https://github.com/chroma-core/chroma), and [Ollama](https://github.com/ollama/ollama).

## Requirements

1. **Python 3.10**  
   Make sure you have Python 3.10 installed.
   
2. **Dependencies**  
   Install the required Python packages:
   ```bash
   pip install llama-index
   pip install chromadb
   pip install llama-index-llms-ollama
   pip install llama-index-vector-stores-postgres
   pip install llama-index-vector-stores-chroma
   pip install llama_index.embeddings.ollama


# Ollama Setup

## Introduction

Ollama is a local large language model inference solution that allows you to easily run, manage, and interact with LLMs on your local machine. This guide will walk you through the installation process and the steps to pull a model into Ollama.

## Installation Steps

1. **Install Ollama**

   To install Ollama, follow the instructions provided in the official Ollama repository:
   - [https://github.com/ollama/ollama](https://github.com/ollama/ollama)

   Ensure you meet any system requirements mentioned in the repository (e.g., macOS version, CPU/GPU specifications, etc.).

2. **Pull a Model With Ollama**

   Once Ollama is successfully installed, you can pull a model. Ollama supports various models, and you can find them listed in the official documentation or community resources. For example, to pull the `llama3.1` model, run:
   ```bash
   ollama pull llama3.1
