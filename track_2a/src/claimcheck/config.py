"""Zentrale Einstellungen. Alles, was man zum Experimentieren drehen will, steht hier."""
import os
from pathlib import Path

from .contracts import LABEL_NAMES

# ---- Pfade ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # track_2a/
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
INDEX_DIR = DATA_DIR / "index"
