#!/usr/bin/env python3
"""
Filter all ECCO books to those containing at least one keyword from a keywords file.

Streams the full JSONL data store in a single pass, applies the same text
normalisation used by extract_mentions.py (long-s, ligatures, NFKD), then
checks for whole-word matches.  Matching documents are written to a single
output JSONL file in the same format produced by prepare_book_samples.py.

By default, work-ID deduplication is applied after the keyword scan: for each
ESTC work that appears multiple times (different editions / revisions), only
the document with the earliest publication year is kept.  Pass --no-dedup to
disable this step and keep all matched documents.

Usage examples:
  # Default paths – keywords.txt, writes books/keywords_match.jsonl
  python3 filter_books_by_keywords.py

  # Custom keywords file and output
  python3 filter_books_by_keywords.py --keywords my_birds.txt --output books/birds.jsonl

  # Restrict to a publication-year range (optional)
  python3 filter_books_by_keywords.py --year-start 1740 --year-end 1800

  # Use 8 parallel workers
  python3 filter_books_by_keywords.py --workers 8

  # Skip work-ID deduplication
  python3 filter_books_by_keywords.py --no-dedup
"""

import argparse
import json
import logging
import multiprocessing as mp
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults  (mirror prepare_book_samples.py)
# ---------------------------------------------------------------------------
DEFAULT_DATA_FILE   = "../../../laczakol/shared/ecco_downloaded/ecco_downloaded.jsonl"
DEFAULT_KEYWORDS    = "keywords.txt"
DEFAULT_OUTPUT      = "books/keywords_match.jsonl"
DEFAULT_DB_FILE     = "/scratch/project_2017429/laczakol/shared/experimental/estc_metadata_db/estc.db"

# ---------------------------------------------------------------------------
# Text normalisation  (mirrors extract_mentions.py)
# ---------------------------------------------------------------------------
LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
    "\u00c6": "AE", "\u00e6": "ae",
    "\u0152": "OE", "\u0153": "oe",
}

_DEHYPHEN  = re.compile(r"(\w)-\n(\w)")
_COMBINING = re.compile(r"[\u0300-\u036f\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]")


def normalize(text: str) -> str:
    text = text.replace("\u017f", "s")   # long-s → s
    for lig, repl in LIGATURES.items():
        text = text.replace(lig, repl)
    text = unicodedata.normalize("NFKD", text)
    text = _COMBINING.sub("", text)      # strip combining diacritics via regex (fast)
    return text


def load_keywords(keywords_file: str) -> set[str]:
    """Load normalised lowercase keywords from a plain-text file.

    Supports two formats:
      - One keyword per line:          canary
      - Group with aliases (new):      canary: canary,canaries
    """
    path = Path(keywords_file)
    keywords: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Handle "label: alias1,alias2,..." format
        if ": " in line:
            _, _, aliases_part = line.partition(": ")
            parts = [a.strip() for a in aliases_part.split(",")]
        else:
            parts = [line]
        for part in parts:
            word = normalize(part).lower()
            if word:
                keywords.add(word)
    return keywords


def build_pattern(keywords: set[str]) -> re.Pattern:
    """Build a compiled word-boundary regex that matches any keyword."""
    alt = "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))
    return re.compile(rf"\b(?:{alt})\b")


# ---------------------------------------------------------------------------
# Work-ID deduplication  (keep earliest edition per work)
# ---------------------------------------------------------------------------

def _metadata_for_ids(document_ids: list[str], db_file: str) -> dict:
    """Return a {ecco_id: {work_id, publication_year, finalWorkField}} dict."""
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        placeholders = ",".join("?" for _ in document_ids)
        cur.execute(
            f"""
            SELECT
                ip.document_id,
                m.publication_year,
                w.work_id,
                m.finalWorkField
            FROM idpairs ip
            LEFT JOIN metadata m
                ON ip.estc_id_student_edition = m.id
            LEFT JOIN works w
                ON w.estc_id = ip.estc_id_student_edition
            WHERE ip.document_id IN ({placeholders})
            """,
            document_ids,
        )
        return {row["document_id"]: dict(row) for row in cur.fetchall()}


def _work_key(meta: dict, doc_id: str) -> str:
    """Stable grouping key: work_id if available, else finalWorkField, else doc_id."""
    wid = meta.get("work_id")
    return wid if wid is not None else (meta.get("finalWorkField") or doc_id)


def _pub_year(meta: dict) -> float:
    year = meta.get("publication_year")
    try:
        return int(year) if year is not None else float("inf")
    except (ValueError, TypeError):
        return float("inf")


def deduplicate_to_earliest(metadata_map: dict) -> set[str]:
    """
    Given {ecco_id: metadata}, return the set of ecco_ids that represent
    the earliest known edition for each unique work.
    """
    best: dict = {}  # work_key -> (doc_id, year)
    for doc_id, meta in metadata_map.items():
        key = _work_key(meta, doc_id)
        year = _pub_year(meta)
        if key not in best or year < best[key][1]:
            best[key] = (doc_id, year)
    return {doc_id for doc_id, _ in best.values()}


# ---------------------------------------------------------------------------
# Worker (runs in each subprocess)
# ---------------------------------------------------------------------------
_KEYWORDS:   set[str]          = set()
_PATTERN:    re.Pattern | None = None
_YEAR_START: int | None        = None
_YEAR_END:   int | None        = None


def _worker_init(
    keywords:   set[str],
    pattern_src: str,
    year_start: int | None,
    year_end:   int | None,
) -> None:
    global _KEYWORDS, _PATTERN, _YEAR_START, _YEAR_END
    _KEYWORDS   = keywords
    _PATTERN    = re.compile(pattern_src)
    _YEAR_START = year_start
    _YEAR_END   = year_end


def _process_line(line: str) -> str | None:
    """Return the JSON line if the doc matches, else None."""
    line = line.strip()
    if not line:
        return None
    try:
        doc = json.loads(line)
    except json.JSONDecodeError:
        return None

    # Optional year filter
    if _YEAR_START or _YEAR_END:
        year = doc.get("year") or doc.get("publication_year")
        try:
            y = int(year)
            if _YEAR_START and y < _YEAR_START:
                return None
            if _YEAR_END and y > _YEAR_END:
                return None
        except (TypeError, ValueError):
            pass  # no parseable year → pass through

    raw_text = doc.get("text", "")
    if not raw_text:
        return None

    # Fast pre-filter: plain substring check before expensive normalization
    raw_lower = raw_text.lower()
    if not any(kw in raw_lower for kw in _KEYWORDS):
        return None

    # Full normalisation + word-boundary match
    norm_text = normalize(raw_text).lower()
    norm_text = _DEHYPHEN.sub(r"\1\2", norm_text)
    if not _PATTERN.search(norm_text):
        return None

    return json.dumps(doc, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter ECCO JSONL to books containing keywords (no time split)."
    )
    parser.add_argument(
        "--data-file", default=DEFAULT_DATA_FILE,
        help=f"Path to the ECCO JSONL data file (default: {DEFAULT_DATA_FILE})",
    )
    parser.add_argument(
        "--keywords", default=DEFAULT_KEYWORDS,
        help=f"Plain-text keywords file, one word per line (default: {DEFAULT_KEYWORDS})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--year-start", type=int, default=None, metavar="YEAR",
        help="Only include documents published from this year onwards (optional)",
    )
    parser.add_argument(
        "--year-end", type=int, default=None, metavar="YEAR",
        help="Only include documents published up to and including this year (optional)",
    )
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count(),
        help=f"Parallel worker processes (default: all CPUs = {mp.cpu_count()})",
    )
    parser.add_argument(
        "--db-file", default=DEFAULT_DB_FILE,
        help=f"ESTC SQLite database for work-ID deduplication (default: {DEFAULT_DB_FILE})",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Skip work-ID deduplication and keep all matched documents",
    )
    args = parser.parse_args()

    # Validate input paths
    for path, name in [
        (args.data_file, "--data-file"),
        (args.keywords,  "--keywords"),
    ]:
        if not Path(path).exists():
            logger.error(f"{name} not found: {path}")
            sys.exit(1)

    keywords = load_keywords(args.keywords)
    if not keywords:
        logger.error(f"No keywords found in {args.keywords}")
        sys.exit(1)
    logger.info(f"Keywords ({len(keywords)}): {sorted(keywords)}")

    pattern_src = build_pattern(keywords).pattern

    output_path = Path(args.output)
    if output_path.exists():
        logger.info(f"Output already exists — skipping. ({output_path})")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    year_start = args.year_start
    year_end   = args.year_end
    if year_start or year_end:
        logger.info(
            f"Year filter: "
            f"{year_start if year_start else '–'} – "
            f"{year_end   if year_end   else '–'}"
        )

    do_dedup = not args.no_dedup
    if do_dedup and not Path(args.db_file).exists():
        logger.warning(
            f"ESTC DB not found ({args.db_file}); skipping deduplication. "
            "Pass --no-dedup to silence this warning."
        )
        do_dedup = False

    logger.info(f"Workers: {args.workers}")

    # ------------------------------------------------------------------
    # Phase 1 — parallel keyword scan; buffer all matched docs in memory.
    # The matched subset is a tiny fraction of the full corpus so this is
    # fine.  We need the full JSON later anyway (to write to output).
    # ------------------------------------------------------------------
    matched_docs: dict[str, str] = {}   # ecco_id -> raw JSON line
    scanned = 0

    logger.info(f"Phase 1/2: streaming {args.data_file} …")
    with (
        open(args.data_file, encoding="utf-8") as data_f,
        mp.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(keywords, pattern_src, year_start, year_end),
        ) as pool,
    ):
        for result in pool.imap(_process_line, data_f, chunksize=50):
            scanned += 1
            if scanned % 10_000 == 0:
                logger.info(
                    f"  {scanned:,} docs scanned, {len(matched_docs):,} matched …"
                )
            if result is not None:
                doc_id = json.loads(result).get("id")
                if doc_id:
                    matched_docs[doc_id] = result

    logger.info(
        f"Scan complete. Scanned {scanned:,} docs — {len(matched_docs):,} matched."
    )

    # ------------------------------------------------------------------
    # Phase 2 — work-ID deduplication via ESTC DB.
    # ------------------------------------------------------------------
    if do_dedup:
        logger.info(
            f"Phase 2/2: querying ESTC metadata for {len(matched_docs):,} docs …"
        )
        metadata_map = _metadata_for_ids(list(matched_docs.keys()), args.db_file)
        keep_ids = deduplicate_to_earliest(metadata_map)

        # Docs absent from the DB are kept (no work_id → treated as unique works).
        absent = set(matched_docs.keys()) - set(metadata_map.keys())
        if absent:
            logger.info(
                f"  {len(absent):,} docs not found in ESTC DB — kept as-is."
            )
        keep_ids |= absent

        logger.info(
            f"  {len(matched_docs):,} matched docs → "
            f"{len(keep_ids):,} after deduplication "
            f"({len(matched_docs) - len(keep_ids):,} duplicate editions removed)."
        )
    else:
        keep_ids = set(matched_docs.keys())
        logger.info("Deduplication skipped (--no-dedup).")

    # ------------------------------------------------------------------
    # Phase 3 — write kept docs to output JSONL.
    # ------------------------------------------------------------------
    written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for doc_id, line in matched_docs.items():
            if doc_id in keep_ids:
                out_f.write(line + "\n")
                written += 1

    logger.info(f"Done. {written:,} docs written → {output_path}")


if __name__ == "__main__":
    main()
