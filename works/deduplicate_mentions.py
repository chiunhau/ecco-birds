#!/usr/bin/env python3
"""
Remove near-duplicate mentions caused by multiple OCR scans or revised editions
of the same underlying book.

Strategy:
  1. Group rows by matched_term (don't cross-compare different animals).
  2. Within each group, represent each row as a set of character 3-grams of its
     normalised context text.  Character n-grams tolerate OCR substitutions
     better than word-level comparison.
  3. Compute pairwise Jaccard similarity.  Pairs above --threshold are
     considered duplicates.
  4. Use Union-Find to cluster all duplicate rows together.
  5. Keep one representative per cluster (the row with the most metadata —
     title, author, year — present; ties broken by first occurrence).
  6. Write the deduplicated CSV and print a short report.

Usage:
  python3 deduplicate_mentions.py
  python3 deduplicate_mentions.py --threshold 0.75 --dry-run
  python3 deduplicate_mentions.py \\
      --input  keywords_mentions.csv \\
      --output keywords_mentions_deduped.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

DEFAULT_INPUT     = "keywords_mentions.csv"
DEFAULT_OUTPUT    = "keywords_mentions_deduped.csv"
DEFAULT_THRESHOLD = 0.80
DEFAULT_NGRAM     = 3


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ngrams(text: str, n: int) -> set[str]:
    """Character n-gram shingles — robust to single-character OCR errors."""
    text = normalize(text)
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        self.parent[self.find(x)] = self.find(y)

    def clusters(self, indices: list[int]) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for i in indices:
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return groups


# ---------------------------------------------------------------------------
# Representative selection
# ---------------------------------------------------------------------------

def metadata_score(row: dict) -> int:
    """Prefer rows that have more metadata filled in."""
    return sum(1 for k in ("title", "author", "year") if row.get(k, "").strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove near-duplicate mentions from classified CSV."
    )
    parser.add_argument("--input",     default=DEFAULT_INPUT)
    parser.add_argument("--output",    default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Jaccard similarity threshold for duplicate detection (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--ngram", type=int, default=DEFAULT_NGRAM,
        help=f"Character n-gram size (default: {DEFAULT_NGRAM})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print report without writing output file",
    )
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"File not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    with open(in_path, newline="", encoding="utf-8") as f:
        reader    = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows      = list(reader)

    total = len(rows)
    print(f"Loaded {total:,} rows from {in_path}", file=sys.stderr)

    # Pre-compute fingerprints
    fingerprints = [ngrams(row.get("context", ""), args.ngram) for row in rows]

    # Group row indices by matched_term so we only compare within same animal
    by_term: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        term = row.get("matched_term", "")
        by_term.setdefault(term, []).append(i)

    uf = UnionFind(total)
    n_pairs_checked = 0
    n_duplicates    = 0

    for term, indices in by_term.items():
        n = len(indices)
        for a in range(n):
            for b in range(a + 1, n):
                ia, ib = indices[a], indices[b]
                sim = jaccard(fingerprints[ia], fingerprints[ib])
                n_pairs_checked += 1
                if sim >= args.threshold:
                    uf.union(ia, ib)
                    n_duplicates += 1

    # Build clusters and pick one representative per cluster
    all_indices = list(range(total))
    clusters    = uf.clusters(all_indices)
    kept_indices: list[int] = []
    duplicate_groups: list[list[int]] = []

    for root, members in clusters.items():
        if len(members) == 1:
            kept_indices.append(members[0])
        else:
            # Pick the row with most metadata; tie-break by earliest position
            best = max(members, key=lambda i: (metadata_score(rows[i]), -i))
            kept_indices.append(best)
            duplicate_groups.append(members)

    kept_indices.sort()
    n_kept    = len(kept_indices)
    n_removed = total - n_kept

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print(f"\nPairs compared : {n_pairs_checked:,}")
    print(f"Duplicate pairs: {n_duplicates:,}")
    print(f"Clusters merged: {len(duplicate_groups):,}")
    print(f"Rows kept      : {n_kept:,} / {total:,}")
    print(f"Rows removed   : {n_removed:,}  ({n_removed / total * 100:.1f}%)")

    if duplicate_groups:
        print("\nDuplicate clusters (showing up to 10):")
        for group in duplicate_groups[:10]:
            print(f"  cluster of {len(group)}:")
            for i in group:
                row   = rows[i]
                title = (row.get("title") or "")[:60] or "(no title)"
                ctx   = (row.get("context") or "")[:60]
                kept  = "✓ kept" if i in set(kept_indices) else "  removed"
                print(f"    [{i:>4}] {kept}  {title}")
                print(f"           context: {ctx} …")

    if args.dry_run:
        print("\n(dry run — no file written)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in kept_indices:
            writer.writerow(rows[i])

    print(f"\nWritten → {out_path}")


if __name__ == "__main__":
    main()
