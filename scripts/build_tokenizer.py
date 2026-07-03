"""
build_tokenizer.py — Custom BPE Tokenizer for Cybersecurity LLM
================================================================
Task 3 — Token & Output Engineer

This script does everything in one run:
  Step 1 — Download cybersecurity dataset from HuggingFace
  Step 2 — Clean and prepare the data
  Step 3 — Train custom BPE tokenizer (vocab size = 151,000)
  Step 4 — Save tokenizer files
  Step 5 — Test the tokenizer with cybersecurity terms

Requirements:
    pip install datasets tokenizers

Usage:
    python build_tokenizer.py

Output:
    cybersec_tokenizer/
    ├── vocab.json     ← all 151,000 tokens
    └── merges.txt     ← BPE merge rules
"""

import os
import re
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
VOCAB_SIZE       = 151_000       # must match LLMConfig in transformer.py
MIN_FREQUENCY    = 2             # ignore tokens appearing less than 2 times
OUTPUT_DIR       = "tokenizer/cybersec_tokenizer"
CLEAN_DATA_FILE  = "./cybersec_data.txt"
# Using multiple datasets to reach 151,000 vocab size.
# Revisions are pinned for tokenizer-build reproducibility.
DATASETS = [
    {
        "name": "vanta-llm/corpus",
        "revision": "fd7967ddda760281a2f01f4367f7b78bd128f3ec",
    },
    {
        "name": "AlicanKiraz0/All-CVE-Records-Training-Dataset",
        "revision": "c2704709be647eabd75305416994278f9a8ec3fc",
    },
]
MAX_RECORDS      = 300_000       # increased to get more unique tokens


# ==============================================================================
# Step 1 — Download Dataset
# ==============================================================================

def download_dataset() -> list:
    """
    Download cybersecurity dataset from HuggingFace.
    Returns a list of text strings.
    """
    log.info("Step 1 — Downloading cybersecurity datasets from HuggingFace...")
    log.info(
        "Datasets: %s",
        [f"{d['name']}@{d['revision'][:12]}" for d in DATASETS],
    )

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not installed. Run: pip install datasets"
        )

    # Extract text from multiple datasets
    texts = []
    for dataset_spec in DATASETS:
        if len(texts) >= MAX_RECORDS:
            break
        dataset_name = dataset_spec["name"]
        dataset_revision = dataset_spec["revision"]
        log.info(f"  Loading: {dataset_name}")
        try:
            dataset = load_dataset(dataset_name, revision=dataset_revision)
            log.info(f"  Loaded successfully!")
        except Exception as e:
            log.warning(f"  Failed to load {dataset_name}: {e} — skipping")
            continue

        for split in dataset.keys():
            log.info(f"  Processing split: {split}")
            sample = dataset[split][0]
            log.info(f"  Dataset fields: {list(sample.keys())}")

            for record in dataset[split]:
                combined = ""
                for field in ["system", "user", "assistant", "System",
                             "User", "Assistant", "text", "content",
                             "description", "input", "output", "body"]:
                    if field in record and isinstance(record[field], str):
                        val = record[field].strip()
                        if val:
                            combined += val + " "

                if combined.strip():
                    texts.append(combined.strip())

                if len(texts) >= MAX_RECORDS:
                    break

            if len(texts) >= MAX_RECORDS:
                break

    log.info(f"Total records collected: {len(texts):,}")

    if not texts:
        raise RuntimeError(
            "No dataset records were collected from HuggingFace. "
            "Aborting tokenizer build instead of producing empty artifacts."
        )

    log.info(f"Step 1 complete — {len(texts):,} text records downloaded ✓")
    return texts


# ==============================================================================
# Step 2 — Clean Data
# ==============================================================================

def clean_text(text: str) -> str:
    """
    Clean a single text record.
    Keeps cybersecurity terms intact while removing noise.
    """
    # Remove URLs (keep domain names as they may be relevant)
    text = re.sub(r'http\S+', '', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove non-ASCII characters except common cybersecurity symbols
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove empty brackets and parentheses
    text = re.sub(r'\[\s*\]|\(\s*\)', '', text)

    return text.strip()


def clean_and_save(texts: list) -> str:
    """
    Clean all texts and save to a plain text file for tokenizer training.
    Returns the path to the saved file.
    """
    log.info("Step 2 — Cleaning and preparing data...")

    cleaned_count = 0
    skipped_count = 0

    with open(CLEAN_DATA_FILE, "w", encoding="utf-8") as f:
        for text in texts:
            cleaned = clean_text(text)

            # Skip very short texts (less than 10 characters)
            if len(cleaned) < 10:
                skipped_count += 1
                continue

            f.write(cleaned + "\n")
            cleaned_count += 1

    log.info(f"Step 2 complete — {cleaned_count:,} records saved, "
             f"{skipped_count:,} skipped ✓")
    log.info(f"Clean data saved to: {CLEAN_DATA_FILE}")

    if cleaned_count == 0:
        raise RuntimeError(
            "All dataset records were filtered out during cleaning. "
            "Aborting tokenizer build."
        )

    return CLEAN_DATA_FILE


# ==============================================================================
# Step 3 — Train BPE Tokenizer
# ==============================================================================

def train_tokenizer(data_file: str):
    """
    Train a Byte-Level BPE tokenizer on the cleaned cybersecurity data.
    Returns the trained tokenizer.
    """
    log.info("Step 3 — Training custom BPE tokenizer...")
    log.info(f"  Vocab size  : {VOCAB_SIZE:,}")
    log.info(f"  Min frequency: {MIN_FREQUENCY}")
    log.info("  This may take 20-40 minutes on a laptop CPU — please wait...")

    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError:
        raise ImportError(
            "tokenizers library not installed. Run: pip install tokenizers"
        )

    # Initialize BPE tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Special tokens important for LLM
    special_tokens = [
        "<pad>",    # padding token
        "<eos>",    # end of sequence
        "<bos>",    # beginning of sequence
        "<unk>",    # unknown token
        "<mask>",   # mask token (for training)
        # Cybersecurity special tokens
        "<CVE>",    # CVE identifier marker
        "<CWE>",    # CWE identifier marker
        "<EXPLOIT>",# exploit marker
        "<VULN>",   # vulnerability marker
    ]

    # Train the tokenizer
    tokenizer.train(
        files=[data_file],
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=special_tokens,
        show_progress=True,
    )

    log.info("Step 3 complete — BPE tokenizer trained successfully ✓")
    return tokenizer


# ==============================================================================
# Step 4 — Save Tokenizer
# ==============================================================================

def save_tokenizer(tokenizer) -> str:
    """
    Save the trained tokenizer to disk.
    Creates vocab.json and merges.txt files.
    """
    log.info("Step 4 — Saving tokenizer files...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer.save_model(OUTPUT_DIR)

    # Save metadata
    metadata = {
        "vocab_size"    : VOCAB_SIZE,
        "min_frequency" : MIN_FREQUENCY,
        "dataset"       : DATASETS,
        "type"          : "ByteLevelBPE",
        "description"   : "Custom cybersecurity BPE tokenizer for the VantaLM architecture",
        "special_tokens": {
            "pad_token"  : "<pad>",
            "eos_token"  : "<eos>",
            "bos_token"  : "<bos>",
            "unk_token"  : "<unk>",
            "mask_token" : "<mask>",
        }
    }

    with open(os.path.join(OUTPUT_DIR, "tokenizer_config.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Step 4 complete — Tokenizer saved to: {OUTPUT_DIR}/ ✓")
    log.info(f"  Files created:")
    log.info(f"    {OUTPUT_DIR}/vocab.json")
    log.info(f"    {OUTPUT_DIR}/merges.txt")
    log.info(f"    {OUTPUT_DIR}/tokenizer_config.json")
    return OUTPUT_DIR


# ==============================================================================
# Step 5 — Test Tokenizer
# ==============================================================================

def test_tokenizer(tokenizer_dir: str):
    """
    Test the trained tokenizer with real cybersecurity terms.
    Shows how the tokenizer splits cybersecurity text vs general text.
    """
    log.info("Step 5 — Testing tokenizer with cybersecurity terms...")

    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.txt"),
    )

    # Test cases — cybersecurity terms
    test_cases = [
        "CVE-2024-1234 SQL injection vulnerability",
        "buffer overflow exploit in memory management",
        "XSS attack via cross-site scripting",
        "ransomware encrypted files with AES-256",
        "zero-day vulnerability in kernel module",
        "penetration testing with Metasploit framework",
        "MITRE ATT&CK technique T1059 command execution",
        "DDoS attack using botnet infrastructure",
    ]

    print("\n" + "=" * 60)
    print("TOKENIZER TEST RESULTS")
    print("=" * 60)

    for text in test_cases:
        encoded = tokenizer.encode(text)
        print(f"\nInput   : {text}")
        print(f"Tokens  : {encoded.tokens}")
        print(f"Token count: {len(encoded.tokens)}")

    print("\n" + "=" * 60)
    print(f"Vocab size: {tokenizer.get_vocab_size():,}")
    print("Step 5 complete — All tests passed ✓")
    print("=" * 60)


# ==============================================================================
# Main — Run all steps
# ==============================================================================

def main():
    print("\n" + "=" * 60)
    print("Custom Cybersecurity BPE Tokenizer Builder")
    print("Task 3 — Token & Output Engineer")
    print("=" * 60 + "\n")

    # Check required libraries
    log.info("Checking required libraries...")
    try:
        import datasets
        import tokenizers
        log.info("All libraries found ✓")
    except ImportError as e:
        log.error(f"Missing library: {e}")
        log.error("Run: pip install datasets tokenizers")
        return

    # Run all steps
    try:
        # Step 1 — Download
        texts = download_dataset()

        # Step 2 — Clean
        data_file = clean_and_save(texts)

        # Step 3 — Train
        tokenizer = train_tokenizer(data_file)

        # Step 4 — Save
        tokenizer_dir = save_tokenizer(tokenizer)

        # Step 5 — Test
        test_tokenizer(tokenizer_dir)

        print("\n✅ Custom BPE tokenizer built successfully!")
        print(f"📁 Tokenizer saved to: {tokenizer_dir}/")
        print("\nNext step: Update tokenizer.py to load from this directory")
        print(f"  set TOKENIZER_PATH={os.path.abspath(tokenizer_dir)}")

    except Exception as e:
        log.error(f"Error during tokenizer build: {e}")
        raise


if __name__ == "__main__":
    main()
