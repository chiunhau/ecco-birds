#!/usr/bin/env python3
"""
Classify bird-mention contexts using a vLLM server (Tague framing categories).

Reads the CSV produced by extract_mentions.py, runs each context through the
model, parses Classification + Key Evidence from the output, and writes a new
CSV with those columns appended.

Skips already-classified rows so the script is safe to resume after interruption.

Usage:
  python3 classify_mentions_vllm.py
  python3 classify_mentions_vllm.py --model Qwen/Qwen2.5-32B-Instruct --input keywords_mentions.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT  = "keywords_mentions.csv"
DEFAULT_OUTPUT = "keywords_mentions_classified_vllm.csv"
DEFAULT_MODEL  = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_URL    = "http://localhost:8000/v1/chat/completions"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Classify the 18th-century text snippet into ONE of the three categories below based on how the bird's situation is described.

### Labels:
- Label 1: SLAVERY/DEPRIVATION/FREEDOM. The text focuses on the bird's natural right to be free or the cruelty of its confinement. Key themes include "stolen" liberty, "unfortunate" imprisonment, or the "distress" of being caged.
- Label 2: HOSPITALITY/CARING/REFUGE. The text justifies the cage as a positive place of safety. Key themes include providing "food," "drink," or "care" to the bird, or describing the owner's "kindness" in protecting a "foreign" or "helpless" creature.
- Label 3: OTHER. The text is not related to any of the above categories, or is a neutral scientific description, or not related to birds.

### Context:
"{CONTEXT_WINDOW}"

### Response Format:
Label: [Number only]
Evidence: [Short quote from the text]
"""

# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "1": "SLAVERY/DEPRIVATION/FREEDOM",
    "2": "HOSPITALITY/CARING/REFUGE",
    "3": "OTHER",
}
VALID_CATEGORIES = set(LABEL_MAP.values())

_CLS_RE = re.compile(r"Label\s*:\s*([1-3])", re.IGNORECASE)
_EVI_RE = re.compile(r"Evidence\s*:\s*(.+)",  re.IGNORECASE)


def parse_output(text: str) -> tuple[str, str]:
    cls_m = _CLS_RE.search(text)
    evi_m = _EVI_RE.search(text)
    label          = cls_m.group(1).strip() if cls_m else None
    classification = LABEL_MAP.get(label, "PARSE_ERROR") if label else "PARSE_ERROR"
    evidence       = evi_m.group(1).strip() if evi_m else ""
    return classification, evidence


# ---------------------------------------------------------------------------
# vLLM query
# ---------------------------------------------------------------------------

def query_vllm(url: str, model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify bird mentions with a vLLM server."
    )
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"Model name served by vLLM (default: {DEFAULT_MODEL})")
    parser.add_argument("--url",    default=DEFAULT_URL,
                        help=f"vLLM chat completions endpoint (default: {DEFAULT_URL})")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    # Load input
    with open(in_path, newline="", encoding="utf-8") as f:
        reader    = csv.DictReader(f)
        in_fields = list(reader.fieldnames or [])
        rows      = list(reader)
    print(f"Loaded {len(rows):,} rows from {in_path}", file=sys.stderr)

    # Resume: find already-classified row indices
    done_indices: set[int] = set()
    if out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done_indices.add(int(r["row_index"]))
        print(f"Resuming: {len(done_indices):,} rows already classified", file=sys.stderr)

    todo = [(i, r) for i, r in enumerate(rows) if i not in done_indices]
    print(f"Rows to classify: {len(todo):,}", file=sys.stderr)
    if not todo:
        print("Nothing to do.", file=sys.stderr)
        return

    # Output CSV
    out_fields   = ["row_index"] + in_fields + ["classification", "key_evidence", "raw_output"]
    write_header = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=out_fields)
        if write_header:
            writer.writeheader()

        for i, (row_idx, row) in enumerate(todo, 1):
            print(f"  [{i}/{len(todo)}] {row.get('title', '')[:60] or row.get('doc_id', '')} …",
                  file=sys.stderr)

            prompt = SYSTEM_PROMPT.format(CONTEXT_WINDOW=row["context"])

            raw_out = query_vllm(args.url, args.model, prompt)
            classification, evidence = parse_output(raw_out)

            print(f"    → {classification}", file=sys.stderr)

            writer.writerow({
                "row_index":      row_idx,
                **row,
                "classification": classification,
                "key_evidence":   evidence,
                "raw_output":     raw_out.replace("\n", " "),
            })
            out_f.flush()

    print(f"\nDone. Written → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
