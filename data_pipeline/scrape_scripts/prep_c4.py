"""
prep_c4.py — C4 / RedPajama Preprocessing Script for VantaLLM
===============================================================
Converts a raw RedPajama C4 JSONL shard into a clean pipeline-ready file
that can be dropped into data_pipeline/raw/ and consumed by run_pipeline.sh

Input schema (RedPajama C4):
    {"text": "...", "meta": {"url": "...", "timestamp": "...", ...}, "red_pajama_subset": "c4"}

Output schema (VantaLLM raw/ format):
    {"text": "..."}

Preprocessing steps applied:
    1.  Schema extraction      — extract only the "text" field, drop meta/subset
    2.  Length filter          — drop documents shorter than min_chars OR fewer words than min_words
    3.  Language/script filter — heuristic ASCII ratio (proxy for non-Latin-script content)
    4.  Boilerplate filter     — scan last N lines for footer patterns (cookie banners, nav menus)
    5.  Repetition filter      — drop docs where top n-grams dominate the text (SEO spam)
    6.  Punctuation density    — drop structured-data dumps (nav trees, product listings)
    7.  URL density            — drop link farms and reference-list pages
    8.  Text cleaning          — normalize whitespace, strip control chars
    9.  Deduplication          — exact-hash dedup within this run; cross-file dedup is handled
                                 by the pipeline's global_hash_log.json

Memory note:
    seen_hashes holds one MD5 digest (16 raw bytes) per kept+deduped document in RAM.
    For single-shard runs (~365K docs) this is ~20MB — fine.
    For large --input_dir runs (10+ shards) RAM usage can exceed ~200MB.
    If that is a concern, swap seen_hashes for a sqlite3 or shelve-backed set.

Usage:
    python data_pipeline/scrape_scripts/prep_c4.py \\
        --input  path/to/c4-train.00000-of-01024.jsonl \\
        --output data_pipeline/raw/c4_shard_00000.jsonl

    # Or process all shards in a directory:
    python data_pipeline/scrape_scripts/prep_c4.py \\
        --input_dir path/to/c4_shards/ \\
        --output    data_pipeline/raw/c4_combined.jsonl

    # Overwrite output instead of appending:
    python data_pipeline/scrape_scripts/prep_c4.py \\
        --input  path/to/shard.jsonl \\
        --output data_pipeline/raw/c4_shard_00000.jsonl \\
        --overwrite

    # Dry-run: print filter stats without writing any output:
    python data_pipeline/scrape_scripts/prep_c4.py \\
        --input  path/to/shard.jsonl \\
        --output data_pipeline/raw/c4_shard_00000.jsonl \\
        --dry-run

    # Widen the boilerplate tail window:
    python data_pipeline/scrape_scripts/prep_c4.py \\
        --input  path/to/shard.jsonl \\
        --output data_pipeline/raw/c4_shard_00000.jsonl \\
        --boilerplate_tail 30
"""

import io
import os
import re
import sys
import json
import hashlib
import logging
import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("prep_c4")

# How often to emit a progress line during file processing (records)
_PROGRESS_INTERVAL = 50_000

# How many records to write before flushing the output buffer to disk.
# Prevents silent data loss if the process is killed mid-run (OOM, cluster
# preemption) before Python's 8KB default write buffer is flushed.
_FLUSH_INTERVAL = 10_000


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class FilterConfig:
    """All tuneable thresholds in one place — passed explicitly, never mutated globally."""

    # ── Length ───────────────────────────────────────────────────────────────
    min_chars:        int   = 200      # Drop very short snippets (nav bars, cookie notices)
    max_chars:        int   = 50_000   # Drop very long docs — they are usually low-quality web
                                       # dumps (not a tokenizer packing concern; the tokenizer
                                       # splits any doc into 2048-token windows automatically).
    min_words:        int   = 40       # Independent word-count floor (space+newline count proxy)

    # ── Language ─────────────────────────────────────────────────────────────
    min_ascii_ratio:  float = 0.85     # Fraction of printable ASCII — filters non-Latin scripts

    # ── Boilerplate ───────────────────────────────────────────────────────────
    boilerplate_tail: int   = 15       # Number of trailing lines to scan for footer boilerplate

    # ── Repetition (SEO spam / templated content) ─────────────────────────────
    # If the most frequent trigram accounts for more than max_trigram_ratio of
    # all trigrams in the document, the document is likely spam/template.
    # Industry reference: LLaMA-2 uses 0.20; 0.25 is slightly more permissive.
    max_trigram_ratio:  float = 0.25

    # ── Punctuation density (structured-data dumps) ───────────────────────────
    # Nav trees, product listings, form dumps have abnormally high colon/pipe/slash density.
    max_punct_ratio:    float = 0.12   # Fraction of chars that are :|/\

    # ── URL density (link farms / reference lists) ────────────────────────────
    # If more than max_url_ratio of lines start with http/www or contain bare URLs,
    # the document is likely a link aggregator, not natural text.
    max_url_line_ratio: float = 0.20   # Fraction of lines that are URL-dominated

    # Digit ratio filter intentionally omitted — cybersecurity text legitimately
    # contains CVE IDs, IPs, port numbers, hashes. A ratio cap would drop valid content.


# Instantiate once as a sentinel for CLI defaults.
# argparse default= requires a concrete value, not a class field reference.
_DEFAULTS = FilterConfig()


# ─── Boilerplate patterns ─────────────────────────────────────────────────────

# Compiled once at module load — scanning is done on the tail of each document,
# not the full body, to avoid false positives in articles that quote privacy law.
_BOILERPLATE_PATTERNS = [
    r"(?i)click here to (accept|enable|allow) cookies",
    r"(?i)this site uses cookies",
    r"(?i)privacy policy\s*\|\s*terms",
    r"(?i)all rights reserved\.?\s*$",
    r"(?i)^\s*(home|about|contact|login|sign up|menu|navigation)\s*$",
    r"(?i)subscribe to our (newsletter|mailing list)",
    r"(?i)^\W{0,5}(404|403|500)\W",
    r"(?i)javascript\s+is\s+(disabled|required|not enabled)",
]
_BOILERPLATE_RE = [re.compile(p) for p in _BOILERPLATE_PATTERNS]

# Pre-compiled URL pattern used by the URL density filter
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


# ─── Filters ──────────────────────────────────────────────────────────────────

def extract_text(record: dict) -> Optional[str]:
    """Pull the text field from a RedPajama C4 record."""
    val = record.get("text", "")
    return val.strip() if val and isinstance(val, str) else None


def is_too_short(text: str, cfg: FilterConfig) -> bool:
    # Char check and word check are independent thresholds.
    # A document can pass the char check but still fail on word count
    # (e.g. a 200-char string of repeated punctuation with no spaces).
    # Word count is approximated via space+newline count — O(n) in C via str.count,
    # avoids the allocation cost of str.split().
    if len(text) < cfg.min_chars:
        return True
    word_approx = text.count(" ") + text.count("\n") + 1
    if word_approx < cfg.min_words:
        return True
    return False


def is_too_long(text: str, cfg: FilterConfig) -> bool:
    return len(text) > cfg.max_chars


def is_non_latin_script(text: str, cfg: FilterConfig) -> bool:
    """
    Heuristic: if fewer than min_ascii_ratio of chars are printable ASCII,
    the document is likely non-Latin-script (Arabic, Chinese, Cyrillic, etc.).

    Note: this does NOT reliably distinguish English from other Latin-script
    languages like French or Spanish — it's intentionally a coarse filter.

    str.encode("ascii", errors="ignore") delegates to C internals and is
    ~10–20x faster than a Python-level character loop on large documents.
    """
    if not text:
        return True
    printable_ascii = len(text.encode("ascii", errors="ignore"))
    return (printable_ascii / len(text)) < cfg.min_ascii_ratio


def has_boilerplate(text: str, cfg: FilterConfig) -> bool:
    """Scan only the last cfg.boilerplate_tail lines for footer/boilerplate content."""
    tail = "\n".join(text.splitlines()[-cfg.boilerplate_tail:])
    for pattern in _BOILERPLATE_RE:
        if pattern.search(tail):
            return True
    return False


# Pre-compiled punctuation stripper for the repetition filter.
# Strips everything that is not a word character or whitespace so that
# "Buy." and "buy" resolve to the same token — matching the LLaMA-2 pipeline.
_PUNCT_RE = re.compile(r"[^\w\s]")


def has_excessive_repetition(text: str, cfg: FilterConfig) -> bool:
    """
    Repetition filter — catches SEO spam, templated product pages, auto-generated text.

    Method (same as LLaMA-2 data pipeline):
      1. Lowercase and strip punctuation so "Buy." and "buy" are the same token.
      2. Split into word-level trigrams.
      3. If the single most common trigram accounts for more than max_trigram_ratio
         of ALL trigram occurrences, the document is considered repetitive.

    Example of what this catches:
      "Buy now. Buy now. Buy now." → tokens ["buy","now","buy","now",...]
                                    → top trigram ("buy","now","buy") = 100% → dropped.
      Normal prose → top trigram rarely exceeds 5–10% → kept.
    """
    # Strip punctuation before splitting — without this, "Buy." ≠ "buy",
    # making the filter ~10–15% less sensitive on punctuated repetitive text.
    normalized = _PUNCT_RE.sub("", text.lower())
    words = normalized.split()
    if len(words) < 3:
        return False
    # At this point trigrams is guaranteed non-empty (len(words) >= 3 → at least 1 trigram)
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    top_count = Counter(trigrams).most_common(1)[0][1]
    return (top_count / len(trigrams)) > cfg.max_trigram_ratio


def has_high_punct_density(text: str, cfg: FilterConfig) -> bool:
    """
    Punctuation density filter — catches structured-data dumps masquerading as text.

    Targets chars: : | / \\ (common in nav trees, product attrs, form dumps, URL lists).
    These chars are legitimate in normal prose but should never dominate it.

    Example of what this catches:
      "Home > Category > Sub > Product | Color: Red | Size: M | Price: $29"
      "Name: ___ | Email: ___ | Phone: ___ | Address: ___"
    """
    if not text:
        return False
    punct_count = sum(1 for c in text if c in ":|/\\")
    return (punct_count / len(text)) > cfg.max_punct_ratio


def has_high_url_density(text: str, cfg: FilterConfig) -> bool:
    """
    URL density filter — catches link farms, reference-aggregation pages, sitemaps.

    A line is considered "URL-dominated" if it contains a URL pattern and the URL
    makes up the majority of the line content (url_len > 60% of line length).
    If more than max_url_line_ratio of all lines are URL-dominated, drop the doc.

    This is stricter than just counting URL occurrences — it won't penalise a
    normal article that cites one or two sources per paragraph.
    """
    lines = text.splitlines()
    if not lines:
        return False
    url_dominated = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        match = _URL_RE.search(stripped)
        if match and len(match.group()) / max(len(stripped), 1) > 0.60:
            url_dominated += 1
    non_empty_lines = sum(1 for l in lines if l.strip())
    if non_empty_lines == 0:
        return False
    return (url_dominated / non_empty_lines) > cfg.max_url_line_ratio


def clean_text(text: str) -> str:
    """
    Normalize whitespace and strip control characters.

    Pass 1 (list comprehension): for each line — strip non-printable chars
           (keeping tabs), then rstrip trailing whitespace.
    Pass 2 (regex): collapse 3+ consecutive blank lines into 2.
    Pass 3 (replace): normalize surviving tabs to 4 spaces.
    """
    lines = [
        "".join(c for c in line if c.isprintable() or c == "\t").rstrip()
        for line in text.splitlines()
    ]
    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = result.replace("\t", "    ")
    return result.strip()


def doc_hash(text: str) -> bytes:
    """
    Return a 16-byte MD5 digest of the text (raw bytes, not hex string).
    Using .digest() instead of .hexdigest() halves per-entry memory in seen_hashes:
    16 bytes vs 32-char string (~85 bytes in CPython due to object overhead).
    """
    return hashlib.md5(text.encode("utf-8")).digest()


# ─── Stats ────────────────────────────────────────────────────────────────────

def _empty_stats() -> dict:
    return {
        "total":             0,
        "kept":              0,
        "dropped_malformed": 0,  # JSON parse failures — distinct from empty-field drops
        "dropped_empty":     0,
        "dropped_short":     0,
        "dropped_long":      0,
        "dropped_non_latin": 0,
        "dropped_boiler":    0,
        "dropped_repetition":0,  # SEO spam / templated content
        "dropped_punct":     0,  # Structured-data dumps
        "dropped_url":       0,  # Link farms / reference lists
        "dropped_dedup":     0,
    }


def print_stats(stats: dict, label: str = "") -> None:
    """
    Stats go to stdout, not the log stream.
    This ensures they are always visible even when stderr/log is redirected to a file.
    """
    total = max(stats["total"], 1)
    sep = "─" * 52
    print(sep)
    if label:
        print(f"  File : {label}")
    print(f"  Total records       : {stats['total']:>8,}")
    print(f"  ✅ Kept             : {stats['kept']:>8,}  ({stats['kept']/total*100:.1f}%)")
    print(f"  ❌ Malformed JSON   : {stats['dropped_malformed']:>8,}")
    print(f"  ❌ Empty field      : {stats['dropped_empty']:>8,}")
    print(f"  ❌ Too short        : {stats['dropped_short']:>8,}")
    print(f"  ❌ Too long         : {stats['dropped_long']:>8,}")
    print(f"  ❌ Non-Latin script : {stats['dropped_non_latin']:>8,}")
    print(f"  ❌ Boilerplate      : {stats['dropped_boiler']:>8,}")
    print(f"  ❌ Repetition spam  : {stats['dropped_repetition']:>8,}")
    print(f"  ❌ Punct density    : {stats['dropped_punct']:>8,}")
    print(f"  ❌ URL density      : {stats['dropped_url']:>8,}")
    print(f"  ❌ Duplicate        : {stats['dropped_dedup']:>8,}")
    print(sep)


# ─── Core processing ──────────────────────────────────────────────────────────

def _open_output(path: str, mode: str, dry_run: bool):
    """
    Return a writable file-like object for the output.

    In dry-run mode, returns an in-memory buffer (io.StringIO) instead of a
    real file. This avoids the os.devnull + encoding=utf-8 incompatibility on
    Windows, where 'nul' is a binary device and raises ValueError when opened
    with a text encoding.
    """
    if dry_run:
        return io.StringIO()
    return open(path, mode, encoding="utf-8")


def process_file(
    input_path: str,
    output_path: str,
    seen_hashes: Set[bytes],
    cfg: FilterConfig,
    overwrite: bool,
    dry_run: bool = False,
) -> dict:
    stats = _empty_stats()
    write_mode = "w" if overwrite else "a"

    with open(input_path, "r", encoding="utf-8", errors="replace") as f_in, \
         _open_output(output_path, write_mode, dry_run) as f_out:

        for line in f_in:
            stats["total"] += 1

            # Progress heartbeat — confirms liveness during long shard processing
            if stats["total"] % _PROGRESS_INTERVAL == 0:
                log.info(
                    f"  ... {stats['total']:,} records processed "
                    f"({stats['kept']:,} kept so far)"
                )

            # ── Parse ──────────────────────────────────────────────────────────
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["dropped_malformed"] += 1
                continue

            # 1. Extract text
            text = extract_text(record)
            if not text:
                stats["dropped_empty"] += 1
                continue

            # 2. Length filters — cheap char+word checks before any regex/ngram work
            if is_too_short(text, cfg):
                stats["dropped_short"] += 1
                continue
            if is_too_long(text, cfg):
                stats["dropped_long"] += 1
                continue

            # 3. Language/script filter — fast C-level encode path
            if is_non_latin_script(text, cfg):
                stats["dropped_non_latin"] += 1
                continue

            # 4. Boilerplate filter — tail-only regex scan
            if has_boilerplate(text, cfg):
                stats["dropped_boiler"] += 1
                continue

            # 5. Repetition filter — catches SEO spam and templated pages
            if has_excessive_repetition(text, cfg):
                stats["dropped_repetition"] += 1
                continue

            # 6. Punctuation density — catches nav trees and structured-data dumps
            if has_high_punct_density(text, cfg):
                stats["dropped_punct"] += 1
                continue

            # 7. URL density — catches link farms and reference aggregators
            if has_high_url_density(text, cfg):
                stats["dropped_url"] += 1
                continue

            # 8. Clean text
            text = clean_text(text)
            if not text:
                stats["dropped_empty"] += 1
                continue

            # 9. Dedup within this run (pipeline handles cross-file dedup via global_hash_log)
            h = doc_hash(text)
            if h in seen_hashes:
                stats["dropped_dedup"] += 1
                continue
            seen_hashes.add(h)

            # ── Write pipeline-ready record ────────────────────────────────────
            f_out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            stats["kept"] += 1

            # Periodic flush: prevents silent data loss if the process is killed
            # before Python's default 8KB write buffer is flushed to disk.
            # No-op in dry-run mode (StringIO has no flush side-effects).
            if not dry_run and stats["kept"] % _FLUSH_INTERVAL == 0:
                f_out.flush()

    return stats


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess RedPajama C4 JSONL shards for VantaLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",     type=str, help="Single input JSONL file")
    group.add_argument("--input_dir", type=str, help="Directory of JSONL files to process")

    parser.add_argument("--output",   type=str, required=True,
                        help="Output JSONL path (drop into data_pipeline/raw/ after)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output file if it exists (default: append)")
    parser.add_argument("--dry-run",  action="store_true", dest="dry_run",
                        help="Print filter stats without writing any output file")

    parser.add_argument("--min_chars",          type=int,   default=_DEFAULTS.min_chars)
    parser.add_argument("--max_chars",          type=int,   default=_DEFAULTS.max_chars)
    parser.add_argument("--min_words",          type=int,   default=_DEFAULTS.min_words)
    parser.add_argument("--min_ascii",          type=float, default=_DEFAULTS.min_ascii_ratio)
    parser.add_argument("--boilerplate_tail",   type=int,   default=_DEFAULTS.boilerplate_tail,
                        help="Number of trailing lines to scan for boilerplate footers")
    parser.add_argument("--max_trigram_ratio",  type=float, default=_DEFAULTS.max_trigram_ratio,
                        help="Max fraction of trigrams that can be the same (repetition filter)")
    parser.add_argument("--max_punct_ratio",    type=float, default=_DEFAULTS.max_punct_ratio,
                        help="Max fraction of chars that are :|/\\ (punctuation density filter)")
    parser.add_argument("--max_url_line_ratio", type=float, default=_DEFAULTS.max_url_line_ratio,
                        help="Max fraction of lines that are URL-dominated (link farm filter)")

    args = parser.parse_args()

    cfg = FilterConfig(
        min_chars          = args.min_chars,
        max_chars          = args.max_chars,
        min_words          = args.min_words,
        min_ascii_ratio    = args.min_ascii,
        boilerplate_tail   = args.boilerplate_tail,
        max_trigram_ratio  = args.max_trigram_ratio,
        max_punct_ratio    = args.max_punct_ratio,
        max_url_line_ratio = args.max_url_line_ratio,
    )

    # Validate input path(s) upfront — before creating/touching the output file
    if args.input:
        if not os.path.isfile(args.input):
            log.error(f"Input file not found: {args.input}")
            sys.exit(1)
        input_files = [args.input]
    else:
        if not os.path.isdir(args.input_dir):
            log.error(f"Input directory not found: {args.input_dir}")
            sys.exit(1)
        input_files = sorted(str(p) for p in Path(args.input_dir).glob("*.jsonl"))
        if not input_files:
            log.error(f"No .jsonl files found in: {args.input_dir}")
            sys.exit(1)

    output_path = args.output

    if args.dry_run:
        log.info("Dry-run mode — no output will be written.")
    elif os.path.exists(output_path) and not args.overwrite:
        log.warning(
            f"Output file '{output_path}' already exists — appending. "
            f"Pass --overwrite to replace it."
        )
    elif os.path.exists(output_path) and args.overwrite:
        log.info(f"--overwrite set: replacing '{output_path}'")

    if not args.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    seen_hashes: Set[bytes] = set()
    totals = _empty_stats()

    for i, fpath in enumerate(input_files):
        log.info(f"[{i+1}/{len(input_files)}] Processing: {fpath}")
        # Only overwrite on the very first file; all subsequent files append
        should_overwrite = args.overwrite and i == 0
        stats = process_file(
            fpath, output_path, seen_hashes, cfg,
            overwrite=should_overwrite,
            dry_run=args.dry_run,
        )
        print_stats(stats, label=os.path.basename(fpath))
        for k in totals:
            totals[k] += stats[k]

    if len(input_files) > 1:
        print("=== TOTALS ACROSS ALL FILES ===")
        print_stats(totals)

    if args.dry_run:
        print("\n⚡ Dry-run complete — no files written.")
    else:
        print(f"\n✅ Output written to: {output_path}")
        print(f"   Drop into data_pipeline/raw/ then run: ./run_pipeline.sh")


if __name__ == "__main__":
    main()
