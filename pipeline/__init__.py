"""
pipeline/__init__.py
DarkSyntax — Pipeline package exports

Public API for Person B (FastAPI) and Person C (Streamlit):
    from pipeline.triage import run_triage
"""

from pipeline.triage     import run_triage
from pipeline.ner        import extract_entities
from pipeline.extractor  import load_document, chunk_document
from pipeline.compressor import compress_for_query

__all__ = [
    "run_triage",
    "extract_entities",
    "load_document",
    "chunk_document",
    "compress_for_query",
]
