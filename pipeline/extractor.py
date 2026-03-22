"""
pipeline/extractor.py
DarkSyntax — Document Extraction Module
Person A owns this file.

Handles:
  - PDF extraction via PyMuPDF
  - Plain text / CSV loading
  - MTSamples corpus builder
  - Chunking for ScaleDown retrieval

Versions: PyMuPDF>=1.23, pandas>=2.0
"""

import os
import re
import fitz          # PyMuPDF
import pandas as pd
from pathlib import Path


# ─── PDF Extractor ────────────────────────────────────────────────────────────

def extract_pdf(path: str) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.
    Fastest PDF reader available — handles 200 pages in ~1 second.
    """
    doc  = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text.strip()


def load_document(path: str) -> str:
    """
    Universal loader — detects file type and returns raw text.
    Supports: .pdf, .txt, .md
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix.lower() == ".pdf":
        return extract_pdf(str(path))

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


# ─── Chunker ──────────────────────────────────────────────────────────────────

def chunk_document(
    text: str,
    chunk_size: int = 400,
    overlap:    int = 50,
) -> list:
    """
    Split text into overlapping word-level chunks.

    chunk_size: words per chunk (400 words ≈ ~500 tokens)
    overlap:    words shared between adjacent chunks (maintains context)

    Returns list of chunk strings.
    """
    words  = text.split()
    chunks = []
    step   = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def chunk_to_file(text: str, output_path: str, **kwargs) -> int:
    """
    Chunk a document and write each chunk as a line to a file.
    Used to prepare corpus for ScaleDown SemanticOptimizer.
    Returns number of chunks written.
    """
    chunks = chunk_document(text, **kwargs)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.replace("\n", " ") + "\n")
    return len(chunks)


# ─── MTSamples corpus builder ─────────────────────────────────────────────────

def build_mtsamples_corpus(
    csv_path:    str,
    output_path: str = "data/mtsamples.txt",
    specialties: list = None,
    max_notes:   int  = 80,
) -> str:
    """
    Load MTSamples CSV and build a focused corpus text file.

    csv_path:    path to downloaded mtsamples.csv
    output_path: where to save the combined corpus
    specialties: list of specialty keywords to filter
                 (default: ER, Cardiology, Neurology, Surgery)
    max_notes:   cap on number of notes to include (~200 pages at 80)

    Returns the corpus text.
    """
    if specialties is None:
        specialties = ["Emergency", "Cardio", "Neurology",
                       "Surgery", "Internal Medicine", "Orthopedic"]

    df = pd.read_csv(csv_path)

    # Filter by specialty
    pattern  = "|".join(specialties)
    filtered = df[
        df["medical_specialty"].str.contains(pattern, case=False, na=False)
    ]

    # Drop rows with empty transcriptions
    filtered = filtered.dropna(subset=["transcription"])
    filtered = filtered[filtered["transcription"].str.len() > 100]

    # Take top N notes
    notes = filtered["transcription"].tolist()[:max_notes]

    corpus = "\n\n--- NOTE ---\n\n".join(notes)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    word_count = len(corpus.split())
    print(f"[Extractor] Corpus built: {len(notes)} notes, "
          f"~{word_count:,} words → {output_path}")

    return corpus


# ─── Corpus stats ─────────────────────────────────────────────────────────────

def corpus_stats(text: str) -> dict:
    """Returns quick stats about a corpus — useful for demo metrics."""
    words  = len(text.split())
    chars  = len(text)
    pages  = round(words / 250)   # ~250 words per page
    tokens = round(words * 1.3)   # rough token estimate

    return {
        "words":           words,
        "characters":      chars,
        "estimated_pages": pages,
        "estimated_tokens": tokens,
    }


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("DarkSyntax — Extractor Self-Test")
    print("=" * 55)

    # Test 1: chunker on synthetic text
    sample = " ".join([f"word{i}" for i in range(1000)])
    chunks = chunk_document(sample, chunk_size=100, overlap=20)
    print(f"\n[Chunker] 1000 words → {len(chunks)} chunks")
    print(f"  First chunk length : {len(chunks[0].split())} words")
    print(f"  Last chunk length  : {len(chunks[-1].split())} words")

    # Test 2: MTSamples if CSV exists
    csv_path = "data/mtsamples.csv"
    if os.path.exists(csv_path):
        corpus = build_mtsamples_corpus(csv_path)
        stats  = corpus_stats(corpus)
        print(f"\n[MTSamples] Corpus stats:")
        for k, v in stats.items():
            print(f"  {k}: {v:,}")
    else:
        print(f"\n[MTSamples] Skipped — {csv_path} not found.")
        print("  Download from mtsamples.com and place in data/")

    # Test 3: PDF if one exists in data/
    pdfs = list(Path("data/").glob("*.pdf")) if Path("data/").exists() else []
    if pdfs:
        text = extract_pdf(str(pdfs[0]))
        stats = corpus_stats(text)
        print(f"\n[PDF] {pdfs[0].name}: {stats['estimated_pages']} pages, "
              f"{stats['estimated_tokens']:,} tokens")
    else:
        print("\n[PDF] Skipped — no PDFs found in data/")

    print("\nExtractor ready.")
