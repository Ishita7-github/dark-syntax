# run.py  ← in ROOT folder, not inside pipeline/
import sys
import os

# Tell Python where to find pipeline/
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.triage import run_triage
print("Import works!")