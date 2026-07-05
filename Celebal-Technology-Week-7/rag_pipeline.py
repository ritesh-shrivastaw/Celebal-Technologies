# rag_pipeline.py
# Lightweight RAG pipeline for document Q&A

import math
import re
from collections import Counter


def load_document(file_path):
    """Load text from .txt or .pdf file."""
    print(f"\n📄 Stage 1: Loading document → {file_path}")

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

    print(f"   ✅ Loaded {len(text)} characters from document")
    return text


def chunk_text(text, chunk_size=200, overlap=50):
    """Split text into overlapping word chunks."""
    print(f"\n✂️  Stage 2: Chunking text...")
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap

    print(f"   ✅ Created {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})")
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
    """Create lightweight bag-of-words embeddings for each chunk."""
    print("\n🔢 Stage 3: Creating lightweight embeddings...")
    return [Counter(tokenize(chunk)) for chunk in chunks]


def retrieve_context(query, embeddings, chunks, top_k=3):
    """Retrieve the top-k matching chunks for a query."""
    print(f"\n🔍 Stage 5+6: Retrieving context for query...")
    print(f"   Query: '{query}'")

    query_vector = Counter(tokenize(query))
    scored = []
    for idx, chunk_vector in enumerate(embeddings):
        similarity = cosine_similarity(query_vector, chunk_vector)
        scored.append((similarity, idx))

    scored.sort(reverse=True)
    retrieved = []
    for rank, (score, idx) in enumerate(scored[:top_k], start=1):
        chunk = chunks[idx]
        retrieved.append({
            'rank': rank,
            'chunk': chunk,
            'distance': round(1 - score, 4),
            'index': idx
        })
        print(f"   Rank {rank} | Distance: {1 - score:.4f} | Chunk #{idx}: {chunk[:60]}...")

    return retrieved


def generate_answer(query, retrieved_chunks):
    """Generate a grounded answer using retrieved chunks."""
    print(f"\n💬 Stage 7: Generating answer...")
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

    def ingest(self, file_path, chunk_size=200, overlap=50):
        text = load_document(file_path)
        self.chunks = chunk_text(text, chunk_size, overlap)
        self.embeddings = create_embeddings(self.chunks)
        self.is_ready = True
        print("\n✅ RAG Pipeline ready! You can now ask questions.")
        return self

    def ask(self, query, top_k=3):
        if not self.is_ready:
            return "❌ Please ingest a document first!"

        retrieved = retrieve_context(query, self.embeddings, self.chunks, top_k)
        answer, context = generate_answer(query, retrieved)
        return answer, retrieved, context