# DriveWise

DriveWise is a conversational AI assistant for car brochures. It helps users ask natural-language questions about a vehicle and receive brochure-grounded answers with source attribution, making the information more transparent and trustworthy.

## Features

- Natural-language question answering over car brochure content
- Retrieval-Augmented Generation (RAG) pipeline
- Metadata-aware filtering by brand and model
- Structured chunking and re-ranking for better retrieval quality
- Source attribution so users can see which brochure section informed the answer
- Simple Streamlit web interface for interactive use
- Evaluation support for testing answer quality

## Project Structure

- app.py — Streamlit web app entry point
- main.py — CLI/demo entry point
- Core/ — core RAG engine and logging/evaluation modules
- Data/ — brochure data generation and storage
- interface/ — CLI chat interface
- tests/ — evaluation scripts

## Setup

Install the required packages:

```bash
pip install numpy pandas streamlit openai sentence-transformers faiss-cpu langchain
```

## Run

### 1. Setup and build the brochure index

```bash
python main.py --setup
```

### 2. Run the demo queries

```bash
python main.py --demo
```

### 3. Launch the Streamlit web app

```bash
python -m streamlit run app.py --server.headless true --server.port 8501
```

Then open:

```text
http://localhost:8501
```

## Notes

- The current version uses a brochure-grounded rule-based answer generator for the demo flow.
- In a production version, this can be upgraded to call an LLM API such as OpenAI while keeping the same retrieval pipeline.
