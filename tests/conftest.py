"""Make sure the sample corpus and index exist before tests run."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Tests must be deterministic and fast: never talk to Ollama or Claude.
os.environ["FINLENS_LLM_PROVIDER"] = "none"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def corpus():
    from finlens.config import DB_PATH, RAW_DIR
    if not any(RAW_DIR.glob("*.pdf")):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_data.py")], check=True)
    if not DB_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "ingest.py")], check=True)
