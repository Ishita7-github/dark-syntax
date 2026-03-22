"""
pipeline/triage.py
DarkSyntax — Core Triage Engine (FINAL FIXED)
"""

import os
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv

from pipeline.ner import extract_entities
from pipeline.compressor import compress_for_query

load_dotenv()


def get_llm_client():
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise ValueError("LLM_API_KEY not set in .env")
    return Groq(api_key=key)


COT_PROMPT = """Emergency triage physician. Respond ONLY in JSON.

PATIENT: {patient_input}
SYMPTOMS: {symptoms}
ANATOMY: {anatomy}
MEDS: {chemicals}
VITALS: {vitals}
SEVERITY: {severity_flags}
DENIED: {negations}
DURATION: {duration_hints}
CONTEXT: {compressed_context}

{{
  "diagnosis": "",
  "urgency": "critical/high/moderate/low",
  "confidence": 0.0,
  "reasoning": "",
  "red_flags": [],
  "recommended_action": ""
}}"""

INSTANT_CRITICAL_KEYWORDS = [
    "not breathing", "no pulse", "unconscious",
    "severe bleeding", "stroke", "heart attack"
]

MEDICAL_KEYWORDS = [
    'patient', 'bp', 'heart', 'pain', 'chest',
    'history', 'diagnosis', 'treatment', 'medication',
    'ecg', 'pulse', 'oxygen', 'fever', 'symptom',
    'clinical', 'blood', 'pressure', 'respiratory'
]


def _instant_critical_check(text: str) -> bool:
    return any(k in text.lower() for k in INSTANT_CRITICAL_KEYWORDS)


def _call_llm(prompt: str, client) -> dict:
    t_send = time.time()
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    
    t_receive = time.time()
    
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
    
    # Calculate tokens per second
    output_tokens = len(raw.split())
    total_groq_ms = round((t_receive - t_send) * 1000)
    tokens_per_sec = round(output_tokens / max((t_receive - t_send), 0.001))
    
    print(f"[GROQ] Request sent → response received: {total_groq_ms}ms")
    print(f"[GROQ] Output tokens: {output_tokens} | Speed: {tokens_per_sec} tokens/sec")
    
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "diagnosis": "Unable to parse",
            "urgency": "high",
            "confidence": 0.5,
            "reasoning": raw[:300],
            "red_flags": [],
            "recommended_action": "Escalate to physician immediately"
        }
    
    result["_groq_ms"]          = total_groq_ms
    result["_groq_tokens_out"]  = output_tokens
    result["_groq_tokens_sec"]  = tokens_per_sec
    
    return result


def run_triage(patient_input: str, doc_path="data/mtsamples.txt", pdf_context=""):

    start_total = time.time()
    timings = {}  # ← track every step

    # ── Critical shortcut ─────────────────────────────────────────────────────
    if _instant_critical_check(patient_input):
        return {
        "diagnosis":          "Life-threatening emergency",
        "urgency":            "critical",
        "confidence":         0.99,
        "reasoning":          "Critical keyword detected — auto response",
        "red_flags":          ["auto-detected critical keyword"],
        "recommended_action": "Call emergency services immediately (112/911)",
        "negated_symptoms":   [],
        "medications":        [],
        "vitals":             {},
        "latency_ms":         round((time.time() - start_total) * 1000),
        "original_tokens":    0,
        "filtered_tokens":    0,
        "compressed_tokens":  0,
        "compression_ratio":  0.0,
        "compression_latency_ms": 0,
        "entities":           {},
        "pdf_blocked":        False,
        "timings":            {"total_ms": round((time.time() - start_total) * 1000)},
    }

    # ── Step 1: NER ───────────────────────────────────────────────────────────
    t1 = time.time()
    ner = extract_entities(patient_input)
    timings["ner_ms"] = round((time.time() - t1) * 1000)
    query = ner["query"] or patient_input[:200]
    print(f'[STEP 1] NER: {timings["ner_ms"]}ms')

    # ── Step 2: PDF loading ───────────────────────────────────────────────────
    t2 = time.time()
    if pdf_context.strip():
        temp_path = "data/temp_context.txt"
        os.makedirs("data", exist_ok=True)
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(pdf_context)
        timings["pdf_load_ms"] = round((time.time() - t2) * 1000)
        print(f'[STEP 2] PDF load: {timings["pdf_load_ms"]}ms')

        # ── Step 3: ScaleDown compression ─────────────────────────────────────
        t3 = time.time()
        compressed = compress_for_query(query, temp_path)
        timings["scaledown_ms"] = round((time.time() - t3) * 1000)
        print(f'[STEP 3] ScaleDown: {timings["scaledown_ms"]}ms')
    else:
        timings["pdf_load_ms"] = 0
        t3 = time.time()
        compressed = compress_for_query(query, doc_path)
        timings["scaledown_ms"] = round((time.time() - t3) * 1000)
        print(f'[STEP 3] ScaleDown: {timings["scaledown_ms"]}ms')

    # ── Step 3.5: Relevance Gate ──────────────────────────────────────────────
    t35 = time.time()
    if pdf_context.strip():
        original_text = pdf_context.lower()
    elif os.path.exists(doc_path):
        with open(doc_path, encoding="utf-8") as f:
            original_text = f.read().lower()
    else:
        original_text = ""

    medical_matches = sum(1 for kw in MEDICAL_KEYWORDS if kw in original_text)
    if not original_text:
        medical_matches = 5
    timings["relevance_ms"] = round((time.time() - t35) * 1000)
    print(f'[STEP 3.5] Relevance gate: {timings["relevance_ms"]}ms — {medical_matches} keywords')

    if medical_matches < 2:
        return {
            "diagnosis":          "BLOCKED — Non-medical document",
            "urgency":            "critical",
            "confidence":         0.0,
            "reasoning":          f"Only {medical_matches} keywords found",
            "red_flags":          ["invalid_input"],
            "recommended_action": "Upload correct medical document",
            "negated_symptoms":   [],
            "medications":        [],
            "vitals":             {},
            "timings":            timings,
            "latency_ms":         round((time.time() - start_total) * 1000),
            "original_tokens":    compressed.get("original_tokens", 0),
            "compressed_tokens":  compressed.get("compressed_tokens", 0),
            "compression_ratio":  compressed.get("compression_ratio", 0),
            "compression_latency_ms": compressed.get("latency_ms", 0),
            "entities":           ner,
            "pdf_blocked":        True,
            "block_reason":       f"Only {medical_matches}/18 medical keywords found"
        }

    # ── Step 4: Build prompt ──────────────────────────────────────────────────
    t4 = time.time()
    prompt = COT_PROMPT.format(
        patient_input     = patient_input,
        symptoms          = ner["symptoms"],
        anatomy           = ner["anatomy"],
        chemicals         = ner["chemicals"],
        vitals            = ner["vitals"],
        severity_flags    = ner["severity_flags"],
        negations         = ner["negations"],
        duration_hints    = ner["duration_hints"],
        compressed_context= compressed["compressed_context"]
    )
    timings["prompt_build_ms"] = round((time.time() - t4) * 1000)
    print(f'[STEP 4] Prompt build: {timings["prompt_build_ms"]}ms')

    # ── Step 5: LLM call ──────────────────────────────────────────────────────
    t5 = time.time()
    client = get_llm_client()
    llm_result = _call_llm(prompt, client)
    timings["llm_ms"] = round((time.time() - t5) * 1000)
    print(f'[STEP 5] LLM: {timings["llm_ms"]}ms')

    total_ms = round((time.time() - start_total) * 1000)
    timings["total_ms"] = total_ms

    # Print full summary
    print(f"""
╔══════════════════════════════════════╗
║         LATENCY BREAKDOWN            ║
╠══════════════════════════════════════╣
║ NER extraction    : {timings['ner_ms']:>6}ms          ║
║ PDF loading       : {timings['pdf_load_ms']:>6}ms          ║
║ ScaleDown API     : {timings['scaledown_ms']:>6}ms          ║
║ Relevance check   : {timings['relevance_ms']:>6}ms          ║
║ Prompt build      : {timings['prompt_build_ms']:>6}ms          ║
║ Groq LLM          : {timings['llm_ms']:>6}ms          ║
╠══════════════════════════════════════╣
║ TOTAL             : {total_ms:>6}ms          ║
╚══════════════════════════════════════╝
    """)

    return {
        "diagnosis":              llm_result.get("diagnosis",          "Unknown"),
        "urgency":                llm_result.get("urgency",            "high"),
        "confidence":             llm_result.get("confidence",         0.0),
        "reasoning":              llm_result.get("reasoning",          ""),
        "red_flags":              llm_result.get("red_flags",          []),
        "recommended_action":     llm_result.get("recommended_action", ""),
        "negated_symptoms":       ner["negations"],
        "medications":            ner["chemicals"],
        "vitals":                 ner["vitals"],
        "latency_ms":             total_ms,
        "original_tokens":        compressed.get("original_tokens",    0),
        "filtered_tokens":        compressed.get("filtered_tokens",    0),
        "compressed_tokens":      compressed.get("compressed_tokens",  0),
        "compression_ratio":      compressed.get("compression_ratio",  0),
        "compression_latency_ms": compressed.get("latency_ms",         0),
        "entities":               ner,
        "pdf_blocked":            False,
        "timings":                timings,
    }