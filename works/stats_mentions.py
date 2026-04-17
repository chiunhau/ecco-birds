#!/usr/bin/env python3
"""Statistics for keywords_mentions.csv"""

import csv
import sys
from collections import Counter
from pathlib import Path

INPUT = Path(__file__).parent / "keywords_mentions.csv"


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def top_n(counter, n=15):
    for item, count in counter.most_common(n):
        print(f"  {count:>6}  {item}")


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT
    rows = load(path)

    total = len(rows)
    unique_docs = len({r["doc_id"] for r in rows})
    unique_terms = len({r["matched_term"].lower() for r in rows})

    section("OVERVIEW")
    print(f"  Total mentions   : {total:,}")
    print(f"  Unique documents : {unique_docs:,}")
    print(f"  Unique terms     : {unique_terms}")

    # --- Mentions per term ---
    section("MENTIONS PER TERM (top 30)")
    term_counts = Counter(r["matched_term"].lower() for r in rows)
    top_n(term_counts, 30)

    # --- Documents per term ---
    section("UNIQUE DOCUMENTS PER TERM (top 30)")
    term_docs = {}
    for r in rows:
        term_docs.setdefault(r["matched_term"].lower(), set()).add(r["doc_id"])
    doc_counts = Counter({t: len(d) for t, d in term_docs.items()})
    top_n(doc_counts, 30)

    # --- Year distribution ---
    section("MENTIONS BY DECADE")
    decades: Counter = Counter()
    bad_years = 0
    for r in rows:
        try:
            y = int(r["year"])
            decades[f"{(y // 10) * 10}s"] += 1
        except (ValueError, TypeError):
            bad_years += 1
    for decade in sorted(decades):
        bar = "#" * (decades[decade] // 20)
        print(f"  {decade}  {decades[decade]:>5}  {bar}")
    if bad_years:
        print(f"  (unparseable years: {bad_years})")

    # --- Most prolific documents per term ---
    all_terms = [t for t, _ in term_counts.most_common()]
    for term in all_terms:
        section(f"TOP DOCUMENTS BY MENTION COUNT — {term.upper()} (top 100)")
        doc_counter: Counter = Counter()
        doc_meta: dict = {}
        for r in rows:
            if r["matched_term"].lower() != term:
                continue
            did = r["doc_id"]
            doc_counter[did] += 1
            doc_meta[did] = (r["year"], r["title"][:70])
        print(f"  {'COUNT':>5}  {'YEAR':<6}  {'DOC_ID':<12}  TITLE")
        print(f"  {'-'*5}  {'-'*6}  {'-'*12}  {'-'*50}")
        for did, count in doc_counter.most_common(100):
            year, title = doc_meta[did]
            print(f"  {count:>5}  {year:<6}  [{did}]  {title}")

    # --- Documents mentioning both birds ---
    section("ALL DOCUMENTS MENTIONING BOTH BIRDS")
    docs_by_term: dict = {}
    doc_meta_all: dict = {}
    for r in rows:
        t = r["matched_term"].lower()
        did = r["doc_id"]
        docs_by_term.setdefault(t, Counter())[did] += 1
        doc_meta_all[did] = (r["year"], r["title"][:70])
    if len(all_terms) >= 2:
        t1, t2 = all_terms[0], all_terms[1]
        shared = set(docs_by_term.get(t1, {})) & set(docs_by_term.get(t2, {}))
        combined = Counter({did: docs_by_term[t1][did] + docs_by_term[t2][did] for did in shared})
        print(f"  {'TOTAL':>5}  {t1.upper():>8}  {t2.upper():>8}  {'YEAR':<6}  {'DOC_ID':<12}  TITLE")
        print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*12}  {'-'*50}")
        for did, total in combined.most_common():
            year, title = doc_meta_all[did]
            c1 = docs_by_term[t1][did]
            c2 = docs_by_term[t2][did]
            print(f"  {total:>5}  {c1:>8}  {c2:>8}  {year:<6}  [{did}]  {title}")

        # Export mentions from both-bird documents to CSV
        out_path = path.parent / "keywords_mentions_both_birds.csv"
        both_rows = [r for r in rows if r["doc_id"] in shared]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=both_rows[0].keys())
            writer.writeheader()
            writer.writerows(both_rows)
        print(f"\n  -> Exported {len(both_rows):,} mentions to {out_path.name}")

    # --- raw_token case variants ---
    section("RAW TOKEN VARIANTS PER TERM (sample)")
    for term, _ in term_counts.most_common(10):
        variants = Counter(
            r["raw_token"] for r in rows if r["matched_term"].lower() == term
        )
        top_v = ", ".join(f"{v}({c})" for v, c in variants.most_common(5))
        print(f"  {term:<20} {top_v}")


if __name__ == "__main__":
    main()
