"""
test_e2e.py — End-to-End System Smoke Test
=================================================
Validates the entire pipeline:
  Tokenizer ↔ FastAPI app ↔ Unified VantaTransformer
"""

import importlib
import json
import os
import sys
import time

from fastapi.testclient import TestClient

TESTS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_client() -> TestClient:
    os.environ["TINY_MODEL"] = "1"
    os.environ["QUANT_MODE"] = "fp16"
    os.environ["MAX_BATCH_SIZE"] = "2"
    os.environ["TOKENIZER_PATH"] = os.path.join(
        PROJECT_ROOT, "vantallm", "tokenizer", "cybersec_tokenizer"
    )

    checkpoint_path = os.environ.get("MODEL_PATH", "").strip()
    if checkpoint_path and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"MODEL_PATH does not exist: {checkpoint_path}")
    if not checkpoint_path:
        os.environ.pop("MODEL_PATH", None)

    server_module = importlib.import_module("vantallm.inference.server")
    server_module = importlib.reload(server_module)
    return TestClient(server_module.app)


def test_generation(client: TestClient) -> bool:
    payload = {
        "prompt": "What is the impact of CVE-2024-1234?",
        "route": "hybrid",
        "context_chunks": [
            "CVE-2024-1234 is a critical heap buffer overflow in the main memory allocator.",
            "It allows remote attackers to execute arbitrary code via malformed packets.",
        ],
        "chunk_ids": ["DOC-001", "DOC-002"],
        "max_new_tokens": 15,
        "temperature": 0.7,
    }

    print("\n[Test] Sending request to /generate")
    print(f"[Test] Route: {payload['route']} | Chunks injected: {len(payload['context_chunks'])}")
    print("-" * 50)

    start_time = time.time()

    response = client.post("/generate", json=payload)
    if response.status_code != 200:
        print(f"❌ Server returned error: {response.status_code}")
        print(response.text)
        return False

    event = response.json()
    print(event.get("answer", ""))

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"✅ Generation successful ({elapsed:.2f} seconds).")
    return True


if __name__ == "__main__":
    print("Starting End-to-End Architecture Test...")
    model_path = os.environ.get("MODEL_PATH", "").strip()
    if model_path:
        print(f"Using checkpoint: {model_path}")
    else:
        print("Running with random tiny weights...")

    with build_client() as client:
        health = client.get("/health")
        if health.status_code == 200:
            print("✅ Server is healthy.")
            success = test_generation(client)
            if success:
                print("\n🎉 ALL SYSTEMS GO: Tokenizer ↔ Server ↔ Unified LLM")
            else:
                print("\n❌ TEST FAILED.")
                sys.exit(1)
        else:
            print(f"❌ /health failed: {health.status_code}")
            sys.exit(1)
