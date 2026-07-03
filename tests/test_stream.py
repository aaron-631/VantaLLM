"""
test_stream.py — VantaLM Streaming Test
==========================================
Run AFTER the server is up:

    set CUDA_VISIBLE_DEVICES=-1
    set TINY_MODEL=1
    set QUANT_MODE=int8
    python -m inference.server

Then in a second terminal:

    python test_stream.py

Expected output:
    --- STREAM START [general] persona=technical_educator ---
    <tokens printed live, one by one>
    --- STREAM DONE | tokens=N ---
"""

import json
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("Install requests first:  pip install requests")

BASE_URL = "http://127.0.0.1:8000"


def stream_generate(prompt: str, route: str = "general", max_new_tokens: int = 30):
    """
    POST to /generate/stream and print tokens as they arrive.
    Returns (token_count, elapsed_seconds).
    """
    payload = {
        "prompt": prompt,
        "route": route,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }

    token_count = 0
    t0 = time.perf_counter()

    with requests.post(
        f"{BASE_URL}/generate/stream",
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue  # blank line between SSE events — skip

            line = raw_line.decode("utf-8")
            if not line.startswith("data: "):
                continue  # ignore SSE comment lines

            payload_str = line[len("data: "):]

            try:
                event = json.loads(payload_str)
            except json.JSONDecodeError:
                # Shouldn't happen with the updated server, but guard anyway
                print(f"\n[WARN] Non-JSON SSE event: {payload_str!r}", file=sys.stderr)
                continue

            if event.get("done"):
                break  # clean sentinel

            if "error" in event:
                print(f"\n[ERROR from server] {event['error']}", file=sys.stderr)
                break

            if "token" in event:
                print(event["token"], end="", flush=True)
                token_count += 1

    elapsed = time.perf_counter() - t0
    return token_count, elapsed


def check_health():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        info = r.json()
        print(f"[health] {info}")
        return True
    except Exception as e:
        print(f"[health] Server not reachable: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TESTS = [
    {
        "label": "general",
        "prompt": "What is SQL injection?",
        "route": "general",
        "max_new_tokens": 25,
    },
    {
        "label": "cyber (hybrid route)",
        "prompt": "Explain how a buffer overflow exploit works.",
        "route": "hybrid",
        "max_new_tokens": 30,
    },
    {
        "label": "OOS refusal",
        "prompt": "What is the weather today?",
        "route": "oos",
        "max_new_tokens": 10,
    },
]


def main():
    print("=" * 60)
    print("VantaLM  —  SSE Streaming Test Suite")
    print("=" * 60)

    if not check_health():
        sys.exit(1)

    all_passed = True

    for test in TESTS:
        label = test["label"]
        print(f"\n--- STREAM START [{label}] ---")

        try:
            n_tokens, elapsed = stream_generate(
                prompt=test["prompt"],
                route=test["route"],
                max_new_tokens=test["max_new_tokens"],
            )
            tps = n_tokens / max(elapsed, 1e-4)
            print(f"\n--- STREAM DONE | tokens={n_tokens} | {elapsed:.2f}s | {tps:.1f} tok/s ---")
        except requests.HTTPError as e:
            print(f"\n[FAIL] HTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
            all_passed = False
        except Exception as e:
            print(f"\n[FAIL] {e}", file=sys.stderr)
            all_passed = False

    print("\n" + "=" * 60)
    print("All tests passed ✓" if all_passed else "Some tests FAILED — see errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
