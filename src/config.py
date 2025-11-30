from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 128))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
DEFAULT_TOP_P = float(os.getenv("TOP_P", 0.95))
DEFAULT_TOP_K = int(os.getenv("TOP_K", 50))
DEFAULT_DEVICE = os.getenv("DEVICE", "auto")
STATIC_DIR = BASE_DIR / "src" / "static"
