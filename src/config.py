import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage settings
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", BASE_DIR / "chroma_db"))

# Text chunking settings
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Embedding and Model settings
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
