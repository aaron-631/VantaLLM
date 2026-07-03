"""
manifest_utils.py — Shared pipeline manifest schema helpers
===========================================================
Keeps pipeline state durable even when intermediate `.pt` files are deleted.
"""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime, UTC
from typing import Any, Dict, Optional


MANIFEST_SCHEMA_VERSION = 2

STATUS_TOKENIZED_PENDING_SHARD = "tokenized_pending_shard"
STATUS_SHARDED = "sharded"
STATUS_MISSING_TOKENIZED_ARTIFACT = "missing_tokenized_artifact"
STATUS_RECOVERED_MISSING_INTERMEDIATE = "recovered_missing_intermediate"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def default_manifest() -> Dict[str, Any]:
    return {
        "pipeline_schema_version": MANIFEST_SCHEMA_VERSION,
        "processed_raw_files": {},
        "processed_normalized_files": {},
        "processed_tokenized_files": {},
        "shard_history": {},
        "last_shard_index": 0,
        "residual_tokens": [],
    }


def infer_normalized_filename(tokenized_filename: str) -> str:
    return tokenized_filename[:-3] if tokenized_filename.endswith(".pt") else tokenized_filename


def _default_tokenized_entry(tokenized_filename: str, token_dir: str) -> Dict[str, Any]:
    normalized_filename = infer_normalized_filename(tokenized_filename)
    return {
        "source_normalized_file": normalized_filename,
        "source_normalized_hash": None,
        "tokenized_path": os.path.join(token_dir, tokenized_filename),
        "tokenized_exists": False,
        "status": STATUS_MISSING_TOKENIZED_ARTIFACT,
        "sequence_count": None,
        "sequence_length": None,
        "dtype": None,
        "shard_ids": [],
        "tokenized_at": None,
        "last_sharded_at": None,
        "notes": None,
    }


def normalize_manifest(
    manifest: Optional[Dict[str, Any]],
    token_dir: str = "data_pipeline/tokenized",
) -> Dict[str, Any]:
    """
    Backward-compatible manifest migration.
    Old boolean `processed_tokenized_files` entries are converted into richer
    ledger records without losing the current pipeline meaning.
    """
    merged = default_manifest()
    if isinstance(manifest, dict):
        merged.update(manifest)

    for key in (
        "processed_raw_files",
        "processed_normalized_files",
        "processed_tokenized_files",
        "shard_history",
    ):
        if not isinstance(merged.get(key), dict):
            merged[key] = {}

    if not isinstance(merged.get("residual_tokens"), list):
        merged["residual_tokens"] = []

    normalized_hashes = merged["processed_normalized_files"]
    migrated: Dict[str, Dict[str, Any]] = {}
    for tokenized_filename, value in merged["processed_tokenized_files"].items():
        tokenized_path = os.path.join(token_dir, tokenized_filename)
        tokenized_exists = os.path.exists(tokenized_path)
        entry = _default_tokenized_entry(tokenized_filename, token_dir)

        if isinstance(value, dict):
            entry.update(deepcopy(value))
            if not entry.get("source_normalized_file"):
                entry["source_normalized_file"] = infer_normalized_filename(tokenized_filename)
        else:
            # Legacy bool schema: False = tokenized but not sharded yet, True = already sharded.
            entry["status"] = STATUS_SHARDED if bool(value) else STATUS_TOKENIZED_PENDING_SHARD

        normalized_filename = entry["source_normalized_file"]
        entry["source_normalized_hash"] = normalized_hashes.get(
            normalized_filename,
            entry.get("source_normalized_hash"),
        )
        entry["tokenized_path"] = tokenized_path
        entry["tokenized_exists"] = bool(tokenized_exists)
        entry["shard_ids"] = sorted(set(entry.get("shard_ids", [])))

        if entry["status"] == STATUS_TOKENIZED_PENDING_SHARD and not tokenized_exists:
            entry["status"] = STATUS_MISSING_TOKENIZED_ARTIFACT
        if entry["status"] == STATUS_SHARDED and not entry["notes"] and not tokenized_exists:
            entry["notes"] = "Intermediate tokenized artifact deleted after sharding."

        migrated[tokenized_filename] = entry

    merged["processed_tokenized_files"] = migrated
    merged["pipeline_schema_version"] = MANIFEST_SCHEMA_VERSION
    return merged


def upsert_tokenized_entry(
    manifest: Dict[str, Any],
    tokenized_filename: str,
    *,
    normalized_filename: Optional[str] = None,
    normalized_hash: Optional[str] = None,
    tokenized_path: Optional[str] = None,
    tokenized_exists: Optional[bool] = None,
    sequence_count: Optional[int] = None,
    sequence_length: Optional[int] = None,
    dtype: Optional[str] = None,
    status: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    token_dir = os.path.dirname(tokenized_path) if tokenized_path else "data_pipeline/tokenized"
    entry = deepcopy(
        manifest["processed_tokenized_files"].get(
            tokenized_filename,
            _default_tokenized_entry(tokenized_filename, token_dir),
        )
    )

    if normalized_filename is not None:
        entry["source_normalized_file"] = normalized_filename
    if normalized_hash is not None:
        entry["source_normalized_hash"] = normalized_hash
    if tokenized_path is not None:
        entry["tokenized_path"] = tokenized_path
    if tokenized_exists is not None:
        entry["tokenized_exists"] = tokenized_exists
    if sequence_count is not None:
        entry["sequence_count"] = int(sequence_count)
    if sequence_length is not None:
        entry["sequence_length"] = int(sequence_length)
    if dtype is not None:
        entry["dtype"] = dtype
    if status is not None:
        entry["status"] = status
    if notes is not None:
        entry["notes"] = notes

    if entry["tokenized_at"] is None and entry["status"] in (
        STATUS_TOKENIZED_PENDING_SHARD,
        STATUS_SHARDED,
    ):
        entry["tokenized_at"] = utc_now_iso()

    manifest["processed_tokenized_files"][tokenized_filename] = entry
    return entry


def mark_entry_sharded(
    manifest: Dict[str, Any],
    tokenized_filename: str,
    shard_ids: list[int],
) -> Dict[str, Any]:
    entry = manifest["processed_tokenized_files"][tokenized_filename]
    entry["status"] = STATUS_SHARDED
    entry["last_sharded_at"] = utc_now_iso()
    entry["shard_ids"] = sorted(set(entry.get("shard_ids", []) + list(shard_ids)))
    if not entry.get("tokenized_exists"):
        entry["notes"] = "Intermediate tokenized artifact deleted after sharding."
    return entry

