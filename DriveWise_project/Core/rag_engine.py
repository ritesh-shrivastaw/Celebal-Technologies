"""
core/rag_engine.py
==================
DriveWise RAG Engine — handles:
  1. Structured chunking by brochure section
  2. Metadata tagging (brand, model, section, page, version)
  3. TF-IDF vector store with metadata filtering
  4. Re-ranking of retrieved chunks
  5. Context window control
  6. Source attribution

Production upgrade path:
  - Replace SimpleVectorStore with FAISS + sentence-transformers
  - Replace re-ranker with Cohere Rerank or cross-encoder
  - Replace TF-IDF embeddings with OpenAI text-embedding-3-small
"""

import os, json, pickle, re, math
import numpy as np
from datetime import datetime

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_PATH = os.path.join(ROOT, "vector_store", "drivewise_store.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHUNKER — splits brochure sections into coherent chunks
# ═══════════════════════════════════════════════════════════════════════════════
class BrochureChunker:
    """
    Structured chunking: one chunk per brochure section (semantic boundaries).
    Falls back to word-window chunking for very long sections.

    Production: Use LangChain RecursiveCharacterTextSplitter with
                chunk_size=400, chunk_overlap=80
    """

    SECTION_ORDER = [
        "Overview", "Engine and Performance", "Mileage and Fuel Efficiency",
        "Dimensions", "Safety", "Interior and Comfort",
        "Infotainment and Connectivity", "Pricing",
    ]

    def chunk(self, brand: str, model: str, version: str, sections: dict) -> list[dict]:
        chunks = []
        page_num = 1

        for section_name in self.SECTION_ORDER:
            text = sections.get(section_name, "").strip()
            if not text:
                continue

            words = text.split()

            # If section fits in one chunk (≤ 300 words), keep whole
            if len(words) <= 300:
                chunks.append(self._make_chunk(
                    text, brand, model, version, section_name, page_num, 0
                ))
            else:
                # Split into 200-word sub-chunks with 40-word overlap
                start, sub = 0, 0
                while start < len(words):
                    end   = min(start + 200, len(words))
                    chunk_text = " ".join(words[start:end])
                    chunks.append(self._make_chunk(
                        chunk_text, brand, model, version,
                        section_name, page_num, sub
                    ))
                    sub   += 1
                    start += 160  # 200 - 40 overlap

            page_num += 1

        return chunks

    def _make_chunk(self, text, brand, model, version,
                    section, page, sub_idx) -> dict:
        chunk_id = f"{brand}_{model}_{section}_{sub_idx}".replace(" ", "_")
        return {
            "chunk_id"  : chunk_id,
            "text"      : text.strip(),
            "metadata"  : {
                "brand"          : brand,
                "model"          : model,
                "version"        : version,
                "section"        : section,
                "page_number"    : page,
                "sub_index"      : sub_idx,
                "doc_name"       : f"{brand} {model} Brochure {version}",
                "char_count"     : len(text),
                "word_count"     : len(text.split()),
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VECTOR STORE — TF-IDF embeddings with metadata filtering
# ═══════════════════════════════════════════════════════════════════════════════
class DriveWiseVectorStore:
    """
    TF-IDF based vector store with full metadata support.

    Production replacement:
        from sentence_transformers import SentenceTransformer
        import faiss
        model   = SentenceTransformer('all-MiniLM-L6-v2')
        vectors = model.encode([c['text'] for c in chunks])
        index   = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
    """

    def __init__(self):
        self.chunks  : list[dict]  = []
        self.vocab   : dict        = {}
        self.vectors : np.ndarray  = None
        self.idf     : np.ndarray  = None

    # ── Vocabulary + IDF ──────────────────────────────────────────────────────
    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_vocab(self):
        for chunk in self.chunks:
            for tok in self._tokenize(chunk["text"]):
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)

    def _compute_idf(self):
        N   = len(self.chunks)
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        for chunk in self.chunks:
            seen = set(self._tokenize(chunk["text"]))
            for tok in seen:
                if tok in self.vocab:
                    idf[self.vocab[tok]] += 1
        self.idf = np.log((N + 1) / (idf + 1)) + 1   # smooth IDF

    def _tfidf_vector(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        tf     = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for t, count in tf.items():
            if t in self.vocab:
                idx        = self.vocab[t]
                vec[idx]   = (count / max(len(tokens), 1)) * self.idf[idx]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ── Build index ───────────────────────────────────────────────────────────
    def add_chunks(self, chunks: list[dict]):
        self.chunks.extend(chunks)

    def build_index(self):
        self._build_vocab()
        self._compute_idf()
        self.vectors = np.array(
            [self._tfidf_vector(c["text"]) for c in self.chunks],
            dtype=np.float32
        )
        print(f"  ✓ Index built — {len(self.chunks)} chunks | vocab {len(self.vocab):,} terms")

    # ── Search with metadata filtering ────────────────────────────────────────
    def search(self, query: str, brand: str = None, model: str = None,
               top_k: int = 6) -> list[dict]:
        """
        Metadata-aware search:
          1. Filter by brand + model (if provided)
          2. Run cosine similarity only on filtered subset
          3. Return top_k results with scores
        """
        # Step 1: metadata filter
        if brand and model:
            indices = [
                i for i, c in enumerate(self.chunks)
                if c["metadata"]["brand"] == brand
                and c["metadata"]["model"] == model
            ]
        elif brand:
            indices = [
                i for i, c in enumerate(self.chunks)
                if c["metadata"]["brand"] == brand
            ]
        else:
            indices = list(range(len(self.chunks)))

        if not indices:
            return []

        # Step 2: embed query
        q_vec = self._tfidf_vector(query)
        if len(q_vec) < self.vectors.shape[1]:
            q_vec = np.pad(q_vec, (0, self.vectors.shape[1] - len(q_vec)))

        # Step 3: cosine similarity on filtered subset
        subset  = self.vectors[indices]
        scores  = subset @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            chunk = self.chunks[indices[idx]].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results

    # ── Persist ───────────────────────────────────────────────────────────────
    def save(self, path: str = STORE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  ✓ Vector store saved → {path}")

    @staticmethod
    def load(path: str = STORE_PATH) -> "DriveWiseVectorStore":
        with open(path, "rb") as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RE-RANKER — keyword + position-aware re-scoring
# ═══════════════════════════════════════════════════════════════════════════════
class ReRanker:
    """
    Re-ranks retrieved chunks using:
      - Term overlap with query (precision-focused)
      - Section relevance bonus (e.g. "mileage" query → mileage section scores higher)
      - Length normalisation penalty for very short chunks

    Production: Replace with:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        scores = model.predict([(query, chunk['text']) for chunk in chunks])
    """

    SECTION_QUERY_MAP = {
        "mileage"       : ["Mileage and Fuel Efficiency"],
        "fuel"          : ["Mileage and Fuel Efficiency"],
        "efficiency"    : ["Mileage and Fuel Efficiency"],
        "kmpl"          : ["Mileage and Fuel Efficiency"],
        "engine"        : ["Engine and Performance"],
        "power"         : ["Engine and Performance"],
        "torque"        : ["Engine and Performance"],
        "horsepower"    : ["Engine and Performance"],
        "performance"   : ["Engine and Performance"],
        "turbo"         : ["Engine and Performance"],
        "speed"         : ["Engine and Performance"],
        "safety"        : ["Safety"],
        "airbag"        : ["Safety"],
        "ncap"          : ["Safety"],
        "adas"          : ["Safety"],
        "brake"         : ["Safety"],
        "dimension"     : ["Dimensions"],
        "size"          : ["Dimensions"],
        "length"        : ["Dimensions"],
        "width"         : ["Dimensions"],
        "height"        : ["Dimensions"],
        "boot"          : ["Dimensions"],
        "clearance"     : ["Dimensions"],
        "seat"          : ["Interior and Comfort"],
        "interior"      : ["Interior and Comfort"],
        "comfort"       : ["Interior and Comfort"],
        "sunroof"       : ["Interior and Comfort"],
        "infotainment"  : ["Infotainment and Connectivity"],
        "screen"        : ["Infotainment and Connectivity"],
        "android"       : ["Infotainment and Connectivity"],
        "bluetooth"     : ["Infotainment and Connectivity"],
        "connected"     : ["Infotainment and Connectivity"],
        "price"         : ["Pricing"],
        "cost"          : ["Pricing"],
        "lakh"          : ["Pricing"],
        "variant"       : ["Pricing", "Overview"],
        "colour"        : ["Overview"],
        "color"         : ["Overview"],
        "overview"      : ["Overview"],
        "seating"       : ["Dimensions", "Interior and Comfort"],
        "capacity"      : ["Dimensions", "Interior and Comfort"],
    }

    def rerank(self, query: str, chunks: list[dict], top_k: int = 4) -> list[dict]:
        q_words = set(re.findall(r"[a-z0-9]+", query.lower()))

        # Identify preferred sections from query keywords
        preferred_sections = set()
        for word in q_words:
            for kw, sections in self.SECTION_QUERY_MAP.items():
                if kw in word or word in kw:
                    preferred_sections.update(sections)

        scored = []
        for chunk in chunks:
            base_score    = chunk.get("score", 0.0)
            c_words       = set(re.findall(r"[a-z0-9]+", chunk["text"].lower()))
            overlap       = len(q_words & c_words) / max(len(q_words), 1)
            section_bonus = 0.3 if chunk["metadata"]["section"] in preferred_sections else 0.0
            length_pen    = min(chunk["metadata"]["word_count"] / 50, 1.0)  # reward longer chunks
            final_score   = (base_score * 0.5) + (overlap * 0.3) + section_bonus + (length_pen * 0.05)
            chunk         = chunk.copy()
            chunk["rerank_score"] = round(final_score, 4)
            scored.append(chunk)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CONTEXT BUILDER — assembles final prompt context
# ═══════════════════════════════════════════════════════════════════════════════
class ContextBuilder:
    """
    Controls the context window:
      - Limits total tokens sent to the LLM
      - Deduplicates overlapping chunks
      - Assembles context string with source labels
    """

    MAX_WORDS = 600  # approximate context window limit

    def build(self, chunks: list[dict]) -> tuple[str, list[dict]]:
        """Returns (context_string, source_list)"""
        seen_sections = set()
        used_chunks   = []
        total_words   = 0

        for chunk in chunks:
            section_key = f"{chunk['metadata']['brand']}_{chunk['metadata']['model']}_{chunk['metadata']['section']}"
            if section_key in seen_sections:
                continue  # deduplicate same section
            word_count = chunk["metadata"]["word_count"]
            if total_words + word_count > self.MAX_WORDS:
                break
            used_chunks.append(chunk)
            seen_sections.add(section_key)
            total_words += word_count

        if not used_chunks:
            return "", []

        parts = []
        for i, chunk in enumerate(used_chunks, 1):
            m = chunk["metadata"]
            parts.append(
                f"[Source {i}: {m['doc_name']} | Section: {m['section']} | Page {m['page_number']}]\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(parts)
        return context, used_chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ANSWER GENERATOR — template-based (swap with LLM in production)
# ═══════════════════════════════════════════════════════════════════════════════
class AnswerGenerator:
    """
    Generates grounded answers from retrieved context.

    Rule-based extraction here. In production, replace with:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content
    """

    SYSTEM_PROMPT = """You are DriveWise, an expert car brochure assistant.
Answer the user's question using ONLY the provided brochure context.
Be specific, accurate, and cite exact numbers/specs. If info is not in context, say so.
Format numbers and specs clearly. Keep answers under 200 words."""

    def generate(self, query: str, context: str, brand: str, model: str) -> str:
        if not context:
            return (f"I don't have enough information about the {brand} {model} "
                    f"to answer your question. Please try rephrasing.")

        q_lower = query.lower()
        lines   = context.split("\n")

        # Extract relevant sentences by keyword matching
        relevant = []
        q_words  = set(re.findall(r"[a-z0-9]+", q_lower))

        for line in lines:
            line = line.strip()
            if not line or line.startswith("[Source"):
                continue
            l_words = set(re.findall(r"[a-z0-9]+", line.lower()))
            overlap = len(q_words & l_words)
            if overlap >= 2:
                relevant.append((overlap, line))

        relevant.sort(reverse=True)
        top_lines = [l for _, l in relevant[:8]]

        if not top_lines:
            # Fall back: return first meaningful paragraph
            for line in lines:
                if len(line.strip()) > 40 and not line.startswith("["):
                    top_lines.append(line.strip())
                if len(top_lines) >= 4:
                    break

        intro  = f"Based on the **{brand} {model}** brochure:\n\n"
        body   = "\n".join(f"• {l}" for l in top_lines)
        return intro + body

    def generate_llm(self, query: str, context: str, brand: str, model: str) -> str:
        """
        PRODUCTION: Call OpenAI / Anthropic here.
        Requires: pip install openai && OPENAI_API_KEY in .env

        from openai import OpenAI
        client = OpenAI()
        prompt = (
            f"You are a car brochure expert assistant for {brand} {model}.\n"
            f"Answer based ONLY on this context:\n\n{context}\n\n"
            f"Question: {query}"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=300, temperature=0.2
        )
        return resp.choices[0].message.content.strip()
        """
        raise NotImplementedError("Set OPENAI_API_KEY and uncomment LLM code above.")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
class DriveWiseRAG:
    """
    End-to-end RAG pipeline:
      query + brand/model → metadata filter → retrieve → rerank → context → answer
    """

    def __init__(self, store: DriveWiseVectorStore = None):
        self.store    = store or DriveWiseVectorStore.load()
        self.reranker = ReRanker()
        self.context_builder = ContextBuilder()
        self.generator       = AnswerGenerator()

    def query(self, question: str, brand: str, model: str,
              top_k_retrieve: int = 8, top_k_rerank: int = 4) -> dict:
        """
        Full pipeline execution. Returns dict with answer + sources + metadata.
        """
        import time
        t0 = time.time()

        # Step 1: Retrieve (with metadata filter)
        raw_chunks = self.store.search(question, brand=brand, model=model, top_k=top_k_retrieve)

        # Step 2: Re-rank
        reranked = self.reranker.rerank(question, raw_chunks, top_k=top_k_rerank)

        # Step 3: Build context (window control)
        context, used_chunks = self.context_builder.build(reranked)

        # Step 4: Generate answer
        answer = self.generator.generate(question, context, brand, model)

        # Step 5: Assemble sources
        sources = [
            {
                "doc_name"    : c["metadata"]["doc_name"],
                "section"     : c["metadata"]["section"],
                "page_number" : c["metadata"]["page_number"],
                "chunk_id"    : c["chunk_id"],
                "score"       : round(c.get("rerank_score", c.get("score", 0)), 4),
            }
            for c in used_chunks
        ]

        latency_ms = round((time.time() - t0) * 1000, 1)

        return {
            "question"    : question,
            "brand"       : brand,
            "model"       : model,
            "answer"      : answer,
            "sources"     : sources,
            "context_text": context,
            "raw_chunks"  : used_chunks,
            "chunks_used" : len(used_chunks),
            "latency_ms"  : latency_ms,
            "timestamp"   : datetime.now().isoformat(),
        }

    def get_brands(self) -> list[str]:
        return sorted({c["metadata"]["brand"] for c in self.store.chunks})

    def get_models(self, brand: str) -> list[str]:
        return sorted({
            c["metadata"]["model"] for c in self.store.chunks
            if c["metadata"]["brand"] == brand
        })


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD FUNCTION — called from main.py
# ═══════════════════════════════════════════════════════════════════════════════
def build_vector_store() -> DriveWiseVectorStore:
    """Load all brochures, chunk them, build and save the vector store."""
    import sys
    sys.path.insert(0, ROOT)
    from Data.brochure_generator import BROCHURE_DATA

    chunker = BrochureChunker()
    store   = DriveWiseVectorStore()
    total   = 0

    for brand, models in BROCHURE_DATA.items():
        for model, data in models.items():
            chunks = chunker.chunk(
                brand    = brand,
                model    = model,
                version  = data["version"],
                sections = data["sections"],
            )
            store.add_chunks(chunks)
            total += len(chunks)
            print(f"  ✓ {brand} {model}: {len(chunks)} chunks")

    store.build_index()
    store.save()
    print(f"\n  Total chunks: {total}")
    return store


if __name__ == "__main__":
    print("\n[DriveWise] Building vector store...")
    store = build_vector_store()

    # Quick test
    rag = DriveWiseRAG(store)
    result = rag.query("What is the mileage of the Creta?", "Hyundai", "Creta")
    print(f"\n  Test query: {result['question']}")
    print(f"  Answer    : {result['answer'][:200]}")
    print(f"  Sources   : {[s['section'] for s in result['sources']]}")
    print(f"  Latency   : {result['latency_ms']} ms\n")