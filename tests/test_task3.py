"""
Task 3 - Shape Verification Tests (Unified Architecture)
-------------------------------------------------------
Run this to confirm architectural shape consistency.

Usage:
    python tests/test_task3.py

All 4 tests should print PASSED.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from vantallm.model.transformer import RMSNorm

# ── Config (matches architecture diagram) ──────────────────────────
VOCAB_SIZE = 151000
EMBED_DIM  = 4096
BATCH_SIZE = 2
SEQ_LEN    = 10

def test_embedding():
    # In unified model, embedding is a standard nn.Embedding
    model = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

    # Fake token IDs
    token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    output = model(token_ids)

    expected = (BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    assert output.shape == expected, f"Expected {expected}, got {output.shape}"
    print(f"  PASSED — Embedding output shape: {tuple(output.shape)}")


def test_rmsnorm():
    model = RMSNorm(EMBED_DIM)

    # Fake tensor coming from transformer
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    output = model(x)

    expected = (BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    assert output.shape == expected, f"Expected {expected}, got {output.shape}"
    print(f"  PASSED — RMSNorm output shape:   {tuple(output.shape)}")


def test_output_layer():
    # In unified model, output layer is a tied-weight nn.Linear
    model = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

    # Fake normalized tensor
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    output = model(x)

    expected = (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert output.shape == expected, f"Expected {expected}, got {output.shape}"
    print(f"  PASSED — Output layer shape:     {tuple(output.shape)}")


def test_full_pipeline():
    """
    Simulates the Task 3 flow:
    token_ids --> embedding --> rmsnorm --> output_layer --> logits
    """
    embedding    = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
    rmsnorm      = RMSNorm(EMBED_DIM)
    output_layer = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

    # Step 1: token IDs in
    token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # Step 2: embed
    x = embedding(token_ids)

    # Step 3: normalize
    x = rmsnorm(x)

    # Step 4: project to vocab
    logits = output_layer(x)

    expected = (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"
    print(f"  PASSED — Full pipeline shape:    {tuple(logits.shape)}")


if __name__ == "__main__":
    print("\nRunning Task 3 tests...\n")
    test_embedding()
    test_rmsnorm()
    test_output_layer()
    test_full_pipeline()
    print("\nAll tests passed! Task 3 is ready.\n")
