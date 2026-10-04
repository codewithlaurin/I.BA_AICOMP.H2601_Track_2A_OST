"""Zentrale Einstellungen. Alles, was man zum Experimentieren drehen will, steht hier."""
import os
from pathlib import Path

# ---- Pfade ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # track_2a/
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"            # Original-Broschüren
PARSED_DIR = DATA_DIR / "parsed"      # *.docling.json + *.md
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
INDEX_DIR = DATA_DIR / "index"        # BM25 + Embeddings

# ---- Parsing (Docling) ---------------------------------------------------
DESCRIBE_PICTURES = True              # VLM-Bildbeschreibungen (langsam ohne GPU)
PICTURE_AREA_THRESHOLD = 0.05         # Bilder < 5 % der Seite überspringen (Logos, Icons)
IMAGES_SCALE = 2.0

# ---- Retrieval -----------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-m3"
BM25_TOP_K = 50
DENSE_TOP_K = 5

# ---- LLM (OpenAI-kompatibler Endpoint, vom Template vorgegeben) ----------
LLM_NAME = os.getenv("LLM_NAME", "swiss-ai/Apertus-8B-Instruct-2509")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "none")

LABELS = ("ENTAIL", "CONTRADICT", "NEUTRAL")
