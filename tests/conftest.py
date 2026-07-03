"""
pytest configuration for VantaLLM.

Only test_smoke.py is written as pytest-native cases. The other files in this
directory are standalone, script-style suites (``if __name__ == "__main__"``)
that drive the FastAPI server, SSE streaming, and RAG paths — they are run
directly (and in CI) rather than collected by pytest. Excluding them here keeps
``pytest tests/`` fast and green without duplicating that harness.
"""

import os

# Ensure the tiny CPU config is used if a test imports the model without setting it.
os.environ.setdefault("TINY_MODEL", "1")

collect_ignore = [
    "test_system.py",
    "test_e2e.py",
    "test_stream.py",
    "test_rag_logit_probe.py",
    "test_rag_readiness.py",
    "test_task3.py",
]
