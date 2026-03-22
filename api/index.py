import sys
import os

# Fix import path for Vercel
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel
from pipeline.triage import run_triage
from pipeline.ner import get_ner

app = FastAPI()