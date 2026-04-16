#!/usr/bin/env python3
"""
Extract in-context mentions of target animal terms from ECCO JSONL book files.

For each occurrence of a target word, emits one CSV row with:
  - doc_id, title, year, author (if present in the JSON)
  - the matched term, the raw token (original capitalisation)
  - token index within the document's token stream
  - ±window context (whitespace-joined tokens)

Usage examples:
  # Single animal, default book file
  python3 extract_mentions.py --animals linnet

  # Multiple animals, custom window and output
  python3 extract_mentions.py --animals linnet sparrow finch \\
      --input books/1780_1799.jsonl --window 30 --output linnet_contexts.csv

  # Multiple book files, use all CPU cores
  python3 extract_mentions.py --animals horse ox \\
      --input books/1700_1719.jsonl books/1720_1739.jsonl --workers 8

  # Canary the bird, not Canary Islands
  python3 extract_mentions.py --animals canary \\
      --exclude-if island islands isle isles

  # Only keep mentions that look bird-related
  python3 extract_mentions.py --animals canary \\
      --require-any cage sing song feather nest bird wings

  # Per-keyword phrase exclusion via a text file (e.g. exclude_phrases.txt):
  #   # lines starting with # are comments
  #   canary: Canary Islands
  #   canary: Canary Island
  #   canaries: the Canaries
  #   canaries: Canary Islands
  python3 extract_mentions.py --exclude-phrases exclude_phrases.txt
"""

import argparse
import csv
import json
import multiprocessing as mp
import re
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT   = "books/keywords_match.jsonl"
DEFAULT_WINDOW  = 20
DEFAULT_OUTPUT  = "keywords_mentions.csv"

# ---------------------------------------------------------------------------
# Text normalisation (mirrors train_diachronic_embeddings.py)
# ---------------------------------------------------------------------------
LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
    "\u00c6": "AE", "\u00e6": "ae",
    "\u0152": "OE", "\u0153": "oe",
}

# Compiled once at module level — shared across workers via fork
_DEHYPHEN    = re.compile(r"(\w)-\n(\w)")
_STRIP_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")


def normalize(text: str) -> str:
    text = text.replace("\u017f", "s")   # long-s → s
    for lig, repl in LIGATURES.items():
        text = text.replace(lig, repl)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


# ---------------------------------------------------------------------------
# Per-document worker
# ---------------------------------------------------------------------------
# These module-level globals are set in each worker process via initializer,
# avoiding repeated pickling across the process pool.
_TARGETS:         set[str]              = set()
_WINDOW:          int                   = 20
_PATTERN:         re.Pattern | None     = None
_EXCLUDE_IF:      set[str]              = set()
_REQUIRE_ANY:     set[str]              = set()
_LOWERCASE_ONLY:  bool                  = False
_EXCLUDE_PHRASES: dict[str, list[tuple[str, bool]]] = {}  # (phrase, case_sensitive)
_TERM_MAP:        dict[str, str]        = {}  # normalised variant → label

# Number of tokens on each side used for local phrase matching (phrases are short)
_PHRASE_WINDOW = 6


def _worker_init(
    targets:         set[str],
    window:          int,
    pattern_src:     str,
    exclude_if:      set[str],
    require_any:     set[str],
    lowercase_only:  bool,
    exclude_phrases: dict[str, list[tuple[str, bool]]],
    term_map:        dict[str, str],
) -> None:
    global _TARGETS, _WINDOW, _PATTERN, _EXCLUDE_IF, _REQUIRE_ANY, _LOWERCASE_ONLY, _EXCLUDE_PHRASES, _TERM_MAP
    _TARGETS         = targets
    _WINDOW          = window
    _PATTERN         = re.compile(pattern_src)
    _EXCLUDE_IF      = exclude_if
    _REQUIRE_ANY     = require_any
    _LOWERCASE_ONLY  = lowercase_only
    _EXCLUDE_PHRASES = exclude_phrases
    _TERM_MAP        = term_map


def _process_line(args: tuple[int, str, str]) -> list[dict]:
    """Process one JSONL line.  Called in a worker process."""
    lineno, line, source_file = args
    line = line.strip()
    if not line:
        return []
    try:
        doc = json.loads(line)
    except json.JSONDecodeError:
        return []

    raw_text = doc.get("text", "")
    if not raw_text:
        return []

    # --- Normalize entire text once (not per-token) ---
    norm_text = normalize(raw_text).lower()

    # --- Pre-filter: skip documents with no target word at all ---
    if not _PATTERN.search(norm_text):
        return []

    # --- Tokenize (both raw for display, norm for matching) ---
    norm_text = _DEHYPHEN.sub(r"\1\2", norm_text)
    raw_text  = _DEHYPHEN.sub(r"\1\2", raw_text)
    norm_tokens = norm_text.split()
    raw_tokens  = raw_text.split()
    # Guard against length mismatch caused by edge-case whitespace differences
    n = min(len(norm_tokens), len(raw_tokens))
    norm_tokens = norm_tokens[:n]
    raw_tokens  = raw_tokens[:n]

    # Strip leading/trailing punctuation from normalised tokens for matching
    clean_tokens = [_STRIP_PUNCT.sub("", t) for t in norm_tokens]

    doc_id = doc.get("document_id") or doc.get("id") or doc.get("ecco_id") or ""
    title  = doc.get("title", "")
    year   = doc.get("year") or doc.get("publication_year") or ""
    author = doc.get("author", "")

    rows: list[dict] = []
    for idx, clean in enumerate(clean_tokens):
        if clean not in _TARGETS:
            continue
        if _LOWERCASE_ONLY and not raw_tokens[idx][0].islower():
            continue
        label = _TERM_MAP.get(clean, clean)
        if _EXCLUDE_PHRASES.get(label):
            local_start  = max(0, idx - _PHRASE_WINDOW)
            local_end    = min(n, idx + _PHRASE_WINDOW + 1)
            local_joined = normalize(" ".join(raw_tokens[local_start:local_end]))
            local_lower  = local_joined.lower()
            def _phrase_hit(ph: str, case_sensitive: bool) -> bool:
                norm_ph = normalize(ph)
                return norm_ph in local_joined if case_sensitive else norm_ph.lower() in local_lower
            if any(_phrase_hit(ph, cs) for ph, cs in _EXCLUDE_PHRASES[label]):
                continue
        start   = max(0, idx - _WINDOW)
        end     = min(n, idx + _WINDOW + 1)
        context_tokens = raw_tokens[start:end]
        context        = " ".join(context_tokens)

        # Context filter: normalise the window tokens for word matching
        context_words = {_STRIP_PUNCT.sub("", t).lower() for t in context_tokens}
        if _EXCLUDE_IF and context_words & _EXCLUDE_IF:
            continue
        if _REQUIRE_ANY and not (context_words & _REQUIRE_ANY):
            continue

        rows.append({
            "doc_id":       doc_id,
            "title":        title,
            "year":         year,
            "author":       author,
            "matched_term": label,
            "raw_token":    raw_tokens[idx],
            "token_index":  idx,
            "context":      context,
        })
    return rows


# ---------------------------------------------------------------------------
# File-level scanner
# ---------------------------------------------------------------------------

def extract_from_file(
    jsonl_path:      Path,
    targets:         set[str],
    window:          int,
    workers:         int,
    exclude_if:      set[str],
    require_any:     set[str],
    lowercase_only:  bool,
    exclude_phrases: dict[str, list[tuple[str, bool]]],
    term_map:        dict[str, str],
) -> list[dict]:
    # Build a word-boundary regex that pre-filters whole documents cheaply
    alt         = "|".join(re.escape(t) for t in sorted(targets, key=len, reverse=True))
    pattern_src = rf"\b(?:{alt})\b"

    source_file = jsonl_path.name
    rows: list[dict] = []
    processed = 0

    def _progress(done: int) -> None:
        print(f"\r    {source_file}: {done:,} docs processed …",
              end="", flush=True, file=sys.stderr)

    def _line_iter():
        with open(jsonl_path, encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                yield (i, ln, source_file)

    if workers == 1:
        _worker_init(targets, window, pattern_src, exclude_if, require_any, lowercase_only, exclude_phrases, term_map)
        for args in _line_iter():
            rows.extend(_process_line(args))
            processed += 1
            if processed % 500 == 0:
                _progress(processed)
    else:
        with mp.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(targets, window, pattern_src, exclude_if, require_any, lowercase_only, exclude_phrases, term_map),
        ) as pool:
            for batch in pool.imap(_process_line, _line_iter(), chunksize=200):
                rows.extend(batch)
                processed += 1
                if processed % 500 == 0:
                    _progress(processed)

    _progress(processed)
    print(file=sys.stderr)  # newline after progress line
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract in-context mentions of animal terms from ECCO JSONL files."
    )
    parser.add_argument(
        "--keywords", default="keywords.txt", metavar="FILE",
        help="Plain-text file of target words, one per line (default: keywords.txt)",
    )
    parser.add_argument(
        "--animals", nargs="+", default=[], metavar="WORD",
        help="Additional target word(s) on the command line (merged with --keywords)",
    )
    parser.add_argument(
        "--input", nargs="+", default=[DEFAULT_INPUT], metavar="JSONL",
        help=f"JSONL book file(s) to search (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help=f"Tokens on each side of the match (default: {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, metavar="CSV",
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--workers", type=int, default=min(mp.cpu_count(), 8),
        help=f"Parallel worker processes (default: min(CPUs, 8) = {min(mp.cpu_count(), 8)})",
    )
    parser.add_argument(
        "--exclude-if", nargs="+", default=[], metavar="WORD",
        help="Skip a mention if any of these words appear in the context window "
             "(e.g. --exclude-if island islands isle isles)",
    )
    parser.add_argument(
        "--require-any", nargs="+", default=[], metavar="WORD",
        help="Skip a mention unless at least one of these words appears in the "
             "context window (e.g. --require-any cage sing song feather nest)",
    )
    parser.add_argument(
        "--lowercase-only", action="store_true",
        help="Only match tokens that appear lowercase in the original text "
             "(e.g. 'canary' matches but 'Canary' does not)",
    )
    parser.add_argument(
        "--exclude-phrases", default="exclude_phrases.txt", metavar="FILE",
        help="Text file of per-keyword phrase exclusions (default: exclude_phrases.txt). "
             "Each non-blank, non-comment line has the form:  label: phrase one, phrase two  "
             "(e.g. 'canary: Canary Islands, the Canaries'). Lines starting with # are ignored.",
    )
    args = parser.parse_args()

    targets:  set[str]       = set()
    term_map: dict[str, str] = {}   # normalised variant → label

    if Path(args.keywords).exists():
        for lineno, line in enumerate(Path(args.keywords).read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                # label: variant1, variant2, ...
                label, _, rest = line.partition(":")
                label = normalize(label.strip()).lower()
                for variant in rest.split(","):
                    v = normalize(variant.strip()).lower()
                    if v:
                        targets.add(v)
                        term_map[v] = label
            else:
                # bare word — label equals the word itself
                w = normalize(line).lower()
                if w:
                    targets.add(w)
                    term_map[w] = w
    elif args.keywords != "keywords.txt":
        print(f"[error] Keywords file not found: {args.keywords}", file=sys.stderr)
        sys.exit(1)

    for w in args.animals:
        v = normalize(w).lower()
        targets.add(v)
        term_map.setdefault(v, v)
    if not targets:
        print("[error] No targets — provide a keywords.txt or use --animals", file=sys.stderr)
        sys.exit(1)
    exclude_if  = {normalize(w).lower() for w in args.exclude_if}
    require_any = {normalize(w).lower() for w in args.require_any}

    # Parse per-keyword phrase exclusions
    exclude_phrases: dict[str, list[tuple[str, bool]]] = {}
    ep_path = Path(args.exclude_phrases)
    if not ep_path.exists() and args.exclude_phrases != "exclude_phrases.txt":
        print(f"[error] Exclude-phrases file not found: {ep_path}", file=sys.stderr)
        sys.exit(1)
    if ep_path.exists():
        for lineno, raw in enumerate(ep_path.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            if ":" not in raw:
                print(f"[warning] {ep_path}:{lineno}: skipping malformed line (no colon): {raw!r}",
                      file=sys.stderr)
                continue
            label, _, rest = raw.partition(":")
            label = normalize(label.strip()).lower()
            for token in rest.split(","):
                token = token.strip()
                if not token:
                    continue
                if token.startswith('"') and token.endswith('"') and len(token) > 1:
                    exclude_phrases.setdefault(label, []).append((token[1:-1], True))
                else:
                    exclude_phrases.setdefault(label, []).append((token, False))

    print(f"Searching for:  {sorted(targets)}", file=sys.stderr)
    if args.lowercase_only:
        print(f"Lowercase only: on (skipping capitalised tokens)", file=sys.stderr)
    if exclude_if:
        print(f"Excluding if:   {sorted(exclude_if)}", file=sys.stderr)
    if require_any:
        print(f"Requiring any:  {sorted(require_any)}", file=sys.stderr)
    if exclude_phrases:
        for kw, phrases in sorted(exclude_phrases.items()):
            fmt = [f'"{ph}"' if cs else ph for ph, cs in phrases]
            print(f"Exclude phrases ({kw}): {fmt}", file=sys.stderr)
    print(f"Workers: {args.workers}", file=sys.stderr)

    all_rows: list[dict] = []
    for path_str in args.input:
        p = Path(path_str)
        if not p.exists():
            print(f"  [error] File not found: {p}", file=sys.stderr)
            sys.exit(1)
        print(f"  Scanning {p} …", file=sys.stderr)
        rows = extract_from_file(
            p, targets, args.window, args.workers,
            exclude_if, require_any, args.lowercase_only, exclude_phrases, term_map,
        )
        print(f"    → {len(rows):,} mentions found in {p.name}", file=sys.stderr)
        all_rows.extend(rows)

    print(f"Total mentions: {len(all_rows):,}", file=sys.stderr)

    if not all_rows:
        print("No matches found — CSV not written.", file=sys.stderr)
        sys.exit(0)

    fieldnames = [
        "doc_id", "title", "year", "author",
        "matched_term", "raw_token", "token_index",
        "context",
    ]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Written → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
