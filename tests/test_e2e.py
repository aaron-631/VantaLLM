"""
test_e2e.py — End-to-End System Smoke Test
=================================================
Validates the entire pipeline:
  Tokenizer ↔ Inference Server ↔ Unified VantaTransformer
"""

import os
import sys
import time
import json
import requests
import multiprocessing
import uvicorn
from contextlib import closing

TESTS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def check_lfs_stub(path):
    """Returns True if the file at path is a Git-LFS stub rather than the real file."""
    if not os.path.exists(path):
        return False
    # LFS stubs are very small (usually < 1KB) and contain specific headers
    if os.path.getsize(path) < 1024:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read(100)
                if "version https://git-lfs.github.com/spec/v1" in content:
                    return True
        except Exception:
            return False
    return False


def run_server():
    """Runs the FastAPI server with a tiny CPU-friendly config."""
    # Force tiny model and fp16 mode (matching showcase_checkpoint.pt)
    os.environ["TINY_MODEL"] = "1"
    os.environ["QUANT_MODE"] = "fp16"
    os.environ["MAX_BATCH_SIZE"] = "2"

    # Point to the real tokenizer we built
    os.environ["TOKENIZER_PATH"] = os.path.join(
        PROJECT_ROOT, "vantallm", "tokenizer", "cybersec_tokenizer"
    )

    # Point to a valid checkpoint for deterministic E2E validation
    os.environ["MODEL_PATH"] = os.path.join(
        PROJECT_ROOT, "checkpoints", "showcase_checkpoint.pt"
    )

    from vantallm.inference.server import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")


def wait_for_server(url="http://127.0.0.1:8000/health", timeout=30):
    """Blocks until the server /health endpoint returns 200."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def test_generation():
    """Sends a streaming request through the hybrid RAG route."""
    url = "http://127.0.0.1:8000/generate/stream"

    payload = {
        "prompt": "What is the impact of CVE-2024-1234?",
        "route": "hybrid",
        "context_chunks": [
            "CVE-2024-1234 is a critical heap buffer overflow in the main memory allocator.",
            "It allows remote attackers to execute arbitrary code via malformed packets."
        ],
        "chunk_ids": ["DOC-001", "DOC-002"],
        "max_new_tokens": 15,
        "temperature": 0.7
    }

    print(f"\n[Test] Sending streaming request to {url}")
    print(f"[Test] Route: {payload['route']} | Chunks injected: {len(payload['context_chunks'])}")
    print("-" * 50)

    start_time = time.time()

    with closing(requests.post(url, json=payload, stream=True)) as r:
        if r.status_code != 200:
            print(f"❌ Server returned error: {r.status_code}")
            print(r.text)
            return False

        for line in r.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    event = json.loads(decoded_line[6:])

                    if event.get("done"):
                        print("\n\n[Test] Stream complete.")
                        break

                    token = event.get("token", "")
                    # The mock model will output random garbage, but it should stream successfully
                    print(token, end="", flush=True)

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"✅ Generation successful ({elapsed:.2f} seconds).")
    return True


if __name__ == "__main__":
    print("Starting End-to-End Architecture Test...")

    # Check for LFS stubs before starting
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "showcase_checkpoint.pt")
    if check_lfs_stub(checkpoint_path):
        print("\n❌ CRITICAL ERROR: Checkpoint is a Git-LFS stub.")
        print("Please run: git lfs pull")
        print("E2E tests require real model weights to initialize the inference server.\n")
        sys.exit(1)

    print("Spinning up inference server (Tiny Model / INT8 Mode)...")

    # Start server in a background process
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()

    try:
        if wait_for_server():
            print("✅ Server is healthy.")

            # Run the actual test
            success = test_generation()

            if success:
                print("\n🎉 ALL SYSTEMS GO: Tokenizer ↔ Server ↔ Unified LLM")
            else:
                print("\n❌ TEST FAILED.")
        else:
            print("❌ Server failed to start within timeout.")

    finally:
        print("Shutting down server...")
        server_process.terminate()
        server_process.join()
