# Local RAG (Retrieval Augmented Generation)

### What is RAG?
RAG = Information Retrieval + In-Context Learning

### Basic RAG Workflow

![RAG Workflow Diagram](./diagrams/rag_v2.drawio.png)

### Tools

- Python - Programming Language for this tool
- - Draw.io Integration (extension for flowchart development)
- Ollama - Open-source platform to that runs LLMs (Large Language Models) locally
- Langchain - framework of tools to connect LLMs to other data sources
- Langsmith - Monitoring tool
- Streamlit - Used to build the UI
- ChromaDB - Creates the vector database to store new information

> pip install -r requirements.txt

The primary goal of this demo is to build a local data pipeline to analyze new data using RAG.

The secondary goal is to structure the pipeline to simplify testing different models and prompts.

### Ollama models used here

- llama3.2:3b (primary llm)
- mxbai-embed-large (embeddings model)
- nomic-embed-text:v1.5 (alternate embedder)

- RAG Evaluation Framework / Metrics / Techniques
- - qllama/bge-reranker-v2-m3:latest (prompt reranker)

### References

- https://www.youtube.com/watch?v=c5jHhMXmXyo - local RAG Chatbot
- https://www.youtube.com/watch?v=gcqp3Fbv4_o - RAG Pipeline
- https://www.e2enetworks.com/blog/guide-to-building-a-rag-based-llm-application RAG Guide from 2023
- https://medium.com/@krtarunsingh/an-in-depth-exploration-of-rag-retrieval-augmented-generation-611a0ddaba81 June 2024
- https://www.youtube.com/watch?v=uFhDXUOrgO0
- https://www.youtube.com/watch?v=vf9emNxXWdA & https://www.youtube.com/watch?v=g8OViy0xOoI RAG Walkthrough for 2025
- https://www.youtube.com/watch?v=sVcwVQRHIc8 LangChain walkthrough
- https://www.youtube.com/watch?v=wVdiT78-wS8 Video on chunking data

#### Test data source
https://www.congress.gov/browse

## Future Project

### Context Engineering

- [Context Engineering Tutorial](https://www.youtube.com/watch?v=Egeuql3Lrzg)
