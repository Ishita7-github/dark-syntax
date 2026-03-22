"""
pipeline/api.py
DarkSyntax — FastAPI Endpoint
"""

from fastapi import FastAPI
from pydantic import BaseModel
from pipeline.triage import run_triage
from pipeline.ner import get_ner  # ← ADD

app = FastAPI()

# ── Pre-load NER model ONCE at startup ────────────
@app.on_event("startup")
async def startup_event():
    print("[STARTUP] Pre-loading NER model...")
    get_ner()  # loads once, cached in singleton
    print("[STARTUP] NER ready! All requests will be fast.")

class TriageRequest(BaseModel):
    symptoms: str
    doc_path: str = "data/mtsamples.txt"
    context:  str = ""

@app.post("/triage")
def triage(request: TriageRequest):
    result = run_triage(
        patient_input = request.symptoms,
        doc_path      = request.doc_path,
        pdf_context   = request.context
    )
    return result

@app.get("/health")
def health():
    return {"status": "ok"}
