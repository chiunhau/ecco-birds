#!/usr/bin/env python3
"""
Filter all ECCO books to those containing at least one keyword from a keywords file.

Streams the full JSONL data store in a single pass, applies the same text
normalisation used by extract_mentions.py (long-s, ligatures, NFKD), then
checks for whole-word matches.  Matching documents are written to a single
output JSONL file in the same format produced by prepare_book_samples.py.

Usage examples:
  # Default paths – keywords.txt, writes books/keywords_match.jsonl
  python3 filter_books_by_keywords.py

  # Custom keywords file and output
  python3 filter_books_by_keywords.py --keywords my_birds.txt --output books/birds.jsonl

  # Restrict to a publication-year range (optional)
  python3 filter_books_by_keywords.py --year-start 1740 --year-end 1800

  # Use 8 parallel workers
  python3 filter_books_by_keywords.py --workers 8
"""

import argparse
import json
import logging
import multiprocessing as mp
import re
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
    """Load normalised lowercase keywords from a plain-text file (one per line)."""
    path = Path(keywords_file)
    keywords: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        word = normalize(line.strip()).lower()
        if word:
            keywords.add(word)
    return keywords


def build_pattern(keywords: set[str]) -> re.Pattern:
    """Build a compiled word-boundary regex that matches any keyword."""
    alt = "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))
    return re.compile(rf"\b(?:{alt})\b")


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

    logger.info(f"Workers: {args.workers}")

    matched = 0
    scanned = 0

    logger.info(f"Streaming {args.data_file} …")
    with (
        open(args.data_file, encoding="utf-8") as data_f,
        open(output_path, "w", encoding="utf-8") as out_f,
        mp.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(keywords, pattern_src, year_start, year_end),
        ) as pool,
    ):
        for result in pool.imap(_process_line, data_f, chunksize=50):
            scanned += 1
            if scanned % 1_000 == 0:
                logger.info(f"  {scanned:,} docs scanned, {matched:,} matched …")
            if result is not None:
                out_f.write(result + "\n")
                matched += 1

    logger.info(f"Done. Scanned {scanned:,} docs — {matched:,} matched → {output_path}")


if __name__ == "__main__":
    main()
