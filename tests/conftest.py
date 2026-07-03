"""
pytest configuration for VantaLLM.

``test_smoke.py`` holds the pytest-native unit cases, and
``test_suites_pytest.py`` wraps the CPU-safe, checkpoint-free script suites
(``test_task3.py`` and ``test_system.py`` phases 1–2) as subprocess-driven
pytest tests so ``pytest tests/`` reports real coverage rather than just the
smoke set. The remaining files here are standalone, script-style suites
(``if __name__ == "__main__"``) that drive the live FastAPI server, SSE
streaming, LFS-checkpoint, and RAG paths — they are run directly (and in CI)
rather than collected by pytest. Excluding them here keeps ``pytest tests/``
fast, hermetic, and green without duplicating that harness.
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
