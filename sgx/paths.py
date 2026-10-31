"""Folder layout. Everything lives inside the project folder; SGX_RESULTS can redirect results (used by the smoke test)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CACHE = os.path.join(ROOT, "data", "cache")
CONFIGS = os.path.join(ROOT, "configs")
RESULTS = os.path.join(ROOT, os.environ.get("SGX_RESULTS", "results"))
CKPT = os.path.join(RESULTS, "ckpt")
LOGS = os.path.join(RESULTS, "logs")


def ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
