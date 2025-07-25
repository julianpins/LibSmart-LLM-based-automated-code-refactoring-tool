import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"

# ChromaDB
CHROMA_DB_DIR = str(DOCS_DIR / "chroma_db")
COLLECTION_NAME = "numpy_docs"

# RAG
TOP_K_RESULTS = 3
CHUNK_SIZE = 500
SIMILARITY_THRESHOLD = 0.4

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "NumPy Code Modernization API"
API_VERSION = "2.0.0"

# NumPy
NUMPY_ALIASES = ["np", "numpy"]