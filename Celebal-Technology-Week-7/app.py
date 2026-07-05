# app.py
import math
import os
import re
from collections import Counter


def load_document(file_path):
    print(f"\n📄 Loading document → {file_path}")
    if file_path.endswith('.pdf'):
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required for PDF files.") from exc
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError("Only .txt and .pdf files are supported!")
    print(f"   ✅ Loaded {len(text)} characters")
    return text


def chunk_text(text, chunk_size=100, overlap=20):
    print("\n✂️  Chunking text...")
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap
    print(f"   ✅ Created {len(chunks)} chunks")
    return chunks


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def cosine_similarity(vec_a, vec_b):
    shared = set(vec_a) | set(vec_b)
    dot = sum(vec_a.get(token, 0) * vec_b.get(token, 0) for token in shared)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def create_embeddings(chunks):
    print("\n🔢 Creating lightweight embeddings...")
    return [Counter(tokenize(chunk)) for chunk in chunks]


def retrieve_context(query, embeddings, chunks, top_k=2):
    print(f"\n🔍 Retrieving context for: {query}")
    query_vector = Counter(tokenize(query))
    scored = []
    for idx, chunk_vector in enumerate(embeddings):
        scored.append((cosine_similarity(query_vector, chunk_vector), idx))
    scored.sort(reverse=True)
    retrieved = []
    for rank, (score, idx) in enumerate(scored[:top_k], start=1):
        retrieved.append({
            'rank': rank,
            'chunk': chunks[idx],
            'distance': round(1 - score, 4),
            'index': idx
        })
    return retrieved


def generate_answer(query, retrieved_chunks):
    print("\n💬 Generating answer...")
    context = "\n\n".join([f"[Chunk {r['rank']}]: {r['chunk']}" for r in retrieved_chunks])
    query_words = set(query.lower().split())
    best_sentence = ""
    best_score = 0
    for chunk in [r['chunk'] for r in retrieved_chunks]:
        for sent in re.split(r'(?<=[.!?])\s+', chunk):
            sent_words = set(sent.lower().split())
            overlap = len(query_words & sent_words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sent
    answer = (
        f"Based on the document:\n\n"
        f"{best_sentence or 'No relevant sentence found in the retrieved context.'}\n\n"
        f"--- Retrieved Context ---\n{context}"
    )
    return answer, context


class RAGSystem:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.is_ready = False

    def ingest(self, file_path, chunk_size=100, overlap=20):
        text = load_document(file_path)
        self.chunks = chunk_text(text, chunk_size, overlap)
        self.embeddings = create_embeddings(self.chunks)
        self.is_ready = True
        print("\n✅ RAG Pipeline ready")
        return self

    def ask(self, query, top_k=2):
        if not self.is_ready:
            return "❌ Please ingest a document first!"
        retrieved = retrieve_context(query, self.embeddings, self.chunks, top_k)
        answer, context = generate_answer(query, retrieved)
        return answer, retrieved, context


def main():
    print("Simple RAG Document Q&A")
    print("=======================")
    rag = RAGSystem()

    files = [f for f in os.listdir('.') if f.endswith(('.txt', '.pdf'))]
    print("Available files:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")

    filename = input("\nEnter filename: ").strip() or "sample_document.txt"
    if not os.path.exists(filename):
        filename = "sample_document.txt"

    rag.ingest(filename)

    while True:
        query = input("\nQuestion (or 'exit'): ").strip()
        if query.lower() in {'exit', 'quit', 'q'}:
            break
        answer, _, _ = rag.ask(query)
        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
