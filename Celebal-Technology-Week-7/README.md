# RAG Document Question Answering System

A simple Retrieval-Augmented Generation (RAG) project that lets you ask questions about custom text documents.

## What it does
- Loads a `.txt` or `.pdf` document
- Splits the document into smaller chunks
- Creates embeddings and stores them in a FAISS vector index
- Retrieves the most relevant chunks for a user question
- Generates an answer using the retrieved context

## Project files
- `app.py` – main entry point for running the app
- `rag_pipeline.py` – core RAG pipeline implementation
- `sample_document.txt` – example document for testing

## Requirements
Install the required packages in your virtual environment:

```bash
pip install numpy faiss-cpu sentence-transformers PyMuPDF transformers torch
```

## Run the project
From the project folder, run:

```bash
python app.py
```

When prompted, enter a document filename such as:

```text
sample_document.txt
```

You can then ask questions interactively.

## Notes
- The app works with plain text and PDF documents.
- If a heavier language-model backend is unavailable, it falls back to a lightweight answer generation approach so the project still runs.
