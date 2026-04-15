"""
RAG Engine for Chat with Transcript.

Handles transcript ingestion (chunking + embedding + ChromaDB storage)
and querying (embed question → retrieve context → LLM answer).

Uses Ollama for local LLM and embeddings — no API costs.
"""
import re
import sys
import hashlib
from pathlib import Path

import chromadb
import ollama
from config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
)


# ---------------------------------------------------------------------------
# Transcript-aware chunking
# ---------------------------------------------------------------------------

def _parse_transcript_line(line: str) -> dict | None:
    """
    Parse a transcript line like:
      [0.00s -> 3.90s] [Speaker 1]: Some text here
    Returns dict with start, end, speaker, text — or None if unparseable.
    """
    pattern = r"\[(\d+\.\d+)s\s*->\s*(\d+\.\d+)s\]\s*\[([^\]]+)\]:\s*(.*)"
    match = re.match(pattern, line.strip())
    if not match:
        return None
    return {
        "start": float(match.group(1)),
        "end": float(match.group(2)),
        "speaker": match.group(3),
        "text": match.group(4).strip(),
    }


def _chunk_transcript(file_path: str) -> list[dict]:
    """
    Read a transcript file and produce chunks for embedding.
    
    Strategy:
    - Group consecutive lines from the same speaker into blocks
    - Split blocks that exceed CHUNK_SIZE with CHUNK_OVERLAP
    - Each chunk carries metadata: speaker, start_time, end_time, source_file
    """
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()

    # Parse all lines
    parsed = []
    for line in lines:
        entry = _parse_transcript_line(line)
        if entry and entry["text"]:
            parsed.append(entry)

    if not parsed:
        # Fallback: treat as plain text (no timestamps)
        raw_text = path.read_text(encoding="utf-8").strip()
        chunks = []
        for i in range(0, len(raw_text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = raw_text[i : i + CHUNK_SIZE]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source_file": path.name,
                        "speaker": "Unknown",
                        "start_time": 0.0,
                        "end_time": 0.0,
                    },
                })
        return chunks

    # Group consecutive lines by speaker
    groups = []
    current_group = {
        "speaker": parsed[0]["speaker"],
        "start": parsed[0]["start"],
        "end": parsed[0]["end"],
        "texts": [parsed[0]["text"]],
    }

    for entry in parsed[1:]:
        if entry["speaker"] == current_group["speaker"]:
            current_group["end"] = entry["end"]
            current_group["texts"].append(entry["text"])
        else:
            groups.append(current_group)
            current_group = {
                "speaker": entry["speaker"],
                "start": entry["start"],
                "end": entry["end"],
                "texts": [entry["text"]],
            }
    groups.append(current_group)

    # Convert groups to chunks, splitting large ones
    chunks = []
    for group in groups:
        combined = f"[{group['speaker']}]: " + " ".join(group["texts"])
        
        if len(combined) <= CHUNK_SIZE:
            chunks.append({
                "text": combined,
                "metadata": {
                    "source_file": path.name,
                    "speaker": group["speaker"],
                    "start_time": group["start"],
                    "end_time": group["end"],
                },
            })
        else:
            # Split with overlap
            for i in range(0, len(combined), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_text = combined[i : i + CHUNK_SIZE]
                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "source_file": path.name,
                            "speaker": group["speaker"],
                            "start_time": group["start"],
                            "end_time": group["end"],
                        },
                    })

    return chunks


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using Ollama."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return response["embeddings"]


# ---------------------------------------------------------------------------
# ChromaDB management
# ---------------------------------------------------------------------------

def _get_chroma_client() -> chromadb.ClientAPI:
    """Get a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def _collection_name_from_file(file_path: str) -> str:
    """
    Generate a valid ChromaDB collection name from a file path.
    ChromaDB requires: 3-63 chars, starts/ends with alphanumeric, 
    only alphanumeric, underscores, hyphens allowed.
    """
    name = Path(file_path).stem
    # Replace dots and spaces with underscores
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # Ensure it starts and ends with alphanumeric
    name = name.strip("_-")
    if not name:
        name = "transcript"
    # Ensure minimum length
    if len(name) < 3:
        name = name + "_transcript"
    # Truncate to max length
    return name[:63]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_transcript(file_path: str) -> str:
    """
    Ingest a transcript file into ChromaDB.
    
    1. Parse and chunk the transcript
    2. Generate embeddings via Ollama
    3. Store in ChromaDB with metadata
    
    Returns the collection name.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {file_path}")

    print(f"Parsing transcript: {path.name}")
    chunks = _chunk_transcript(file_path)

    if not chunks:
        raise ValueError(f"No content found in {file_path}")

    print(f"Created {len(chunks)} chunks")

    # Generate embeddings
    print(f"Generating embeddings with {EMBEDDING_MODEL}...")
    texts = [c["text"] for c in chunks]
    embeddings = _get_embeddings(texts)

    # Store in ChromaDB
    collection_name = _collection_name_from_file(file_path)
    client = _get_chroma_client()
    
    # Delete existing collection if re-indexing
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"source_file": path.name},
    )

    # Generate unique IDs for each chunk
    ids = []
    for i, chunk in enumerate(chunks):
        content_hash = hashlib.md5(chunk["text"].encode()).hexdigest()[:8]
        ids.append(f"{collection_name}_{i}_{content_hash}")

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"Indexed {len(chunks)} chunks into collection '{collection_name}'")
    return collection_name


def query_transcript(
    question: str,
    collection_name: str,
    top_k: int = TOP_K_RESULTS,
) -> dict:
    """
    Query a transcript collection with a natural language question.
    
    Returns dict with:
      - answer: str (LLM response)
      - sources: list of dicts with speaker, start_time, end_time, text
    """
    # Embed the question
    q_embedding = _get_embeddings([question])[0]

    # Retrieve similar chunks
    client = _get_chroma_client()
    collection = client.get_collection(collection_name)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Build context for the LLM
    context_parts = []
    sources = []
    for doc, meta in zip(documents, metadatas):
        start = meta.get("start_time", 0)
        end = meta.get("end_time", 0)
        speaker = meta.get("speaker", "Unknown")
        context_parts.append(f"[{start:.1f}s - {end:.1f}s] {doc}")
        sources.append({
            "speaker": speaker,
            "start_time": start,
            "end_time": end,
            "text": doc,
        })

    context = "\n\n".join(context_parts)

    # Ask the LLM
    prompt = f"""You are a helpful assistant that answers questions about a conversation transcript.
Use ONLY the provided context to answer. If the answer is not in the context, say "I couldn't find that in the transcript."
Always cite the speaker and timestamp when possible.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )

    answer = response["message"]["content"]

    return {
        "answer": answer,
        "sources": sources,
    }


def list_collections() -> list[dict]:
    """List all indexed transcript collections."""
    client = _get_chroma_client()
    collections = client.list_collections()
    result = []
    for col in collections:
        # In newer ChromaDB, list_collections returns Collection objects
        col_name = col if isinstance(col, str) else col.name
        collection = client.get_collection(col_name)
        count = collection.count()
        meta = collection.metadata or {}
        result.append({
            "name": col_name,
            "source_file": meta.get("source_file", "unknown"),
            "chunk_count": count,
        })
    return result


def delete_collection(collection_name: str) -> None:
    """Delete an indexed transcript collection."""
    client = _get_chroma_client()
    client.delete_collection(collection_name)
    print(f"Deleted collection '{collection_name}'")


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python rag_engine.py ingest <transcript_file>")
        print("  python rag_engine.py query <collection_name> <question>")
        print("  python rag_engine.py list")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python rag_engine.py ingest <transcript_file>")
            sys.exit(1)
        col_name = ingest_transcript(sys.argv[2])
        print(f"\nReady! Use collection name: {col_name}")

    elif command == "query":
        if len(sys.argv) < 4:
            print("Usage: python rag_engine.py query <collection_name> <question>")
            sys.exit(1)
        col_name = sys.argv[2]
        question = " ".join(sys.argv[3:])
        result = query_transcript(question, col_name)
        print(f"\n{'='*60}")
        print(f"Answer: {result['answer']}")
        print(f"\n{'='*60}")
        print("Sources:")
        for src in result["sources"]:
            print(f"  [{src['start_time']:.1f}s - {src['end_time']:.1f}s] {src['speaker']}: {src['text'][:80]}...")

    elif command == "list":
        collections = list_collections()
        if not collections:
            print("No indexed transcripts found.")
        else:
            print("Indexed transcripts:")
            for col in collections:
                print(f"  {col['name']} ({col['source_file']}, {col['chunk_count']} chunks)")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
