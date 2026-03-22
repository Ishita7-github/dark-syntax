"""
pipeline/compressor.py
DarkSyntax — ScaleDown Compression Module
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

SCALEDOWN_URL = "https://api.scaledown.xyz/compress/raw/"

def compress_for_query(query: str, doc_path: str, top_k: int = 15) -> dict:

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document not found: {doc_path}")

    start        = time.time()
    full_context = open(doc_path, encoding="utf-8").read()
    full_tokens  = len(full_context.split())

    # ── Pre-filter ────────────────────────────────
    query_words   = query.lower().split()
    base_keywords = [
        'cardiac', 'chest', 'heart', 'bp', 'ecg',
        'troponin', 'pain', 'breath', 'pulse', 'blood',
        'pressure', 'rate', 'oxygen', 'medication',
        'allergy', 'history', 'diagnosis', 'emergency'
    ]
    all_keywords = list(set(query_words + base_keywords))
    lines        = full_context.split('\n')
    relevant     = [l for l in lines if any(k in l.lower() for k in all_keywords)]

    if len(relevant) > 50:
        context = '\n'.join(relevant[:300])       # ← context defined here
        print(f'[FILTER] {len(lines)} lines → {len(relevant[:300])} relevant')
    else:
        words   = full_context.split()
        context = ' '.join(words[:5000])          # ← context defined here
        print(f'[FILTER] Fallback: using first 5000 words')

    filtered_tokens = len(context.split())
    print(f'[DEBUG] Original: {full_tokens} words')
    print(f'[DEBUG] After filter: {filtered_tokens} words')
    print(f'[COMPRESS] Sending {filtered_tokens} tokens to ScaleDown')

    t_send = time.time()

    response = requests.post(
        SCALEDOWN_URL,
        headers={
            "x-api-key":    os.getenv("SCALEDOWN_API_KEY"),
            "Content-Type": "application/json"
        },
        json={
            "context":   context,              # ← now defined
            "prompt":    query,
            "scaledown": {"rate": "auto"}
        },
        timeout=10
    )

    t_receive         = time.time()
    result            = response.json()
    compressed        = result.get("compressed_prompt", context[:500])
    compressed_tokens = len(compressed.split())
    api_ms            = round((t_receive - t_send) * 1000)
    tokens_removed    = full_tokens - compressed_tokens
    compress_speed    = round(tokens_removed / max((t_receive - t_send), 0.001))

    print(f'[COMPRESS] Done: {filtered_tokens} → {compressed_tokens} tokens in {api_ms}ms')

    return {
        "compressed_context":       compressed,
        "original_tokens":          full_tokens,
        "filtered_tokens":          filtered_tokens,
        "compressed_tokens":        compressed_tokens,
        "compression_ratio":        round(full_tokens / max(compressed_tokens, 1), 2),
        "latency_ms":               round((time.time() - start) * 1000),
        "scaledown_api_ms":         api_ms,
        "scaledown_tokens_per_sec": compress_speed,
    }
def compress_batch(queries: list, doc_path: str) -> list:
    return [compress_for_query(q, doc_path) for q in queries]