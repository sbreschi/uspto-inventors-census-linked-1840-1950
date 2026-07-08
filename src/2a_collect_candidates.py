#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2a: collect candidate pairs from the blocking outputs.

For one census year, read the blocking candidate sets across the requested
genders, keep the best score per (histid, PatInvID, pubnumber, inventor_name,
census_name), and outer-merge them into a single deduplicated candidate table.

Four blocking sources are selectable via --sources (default: all four):
  * cosine_county   FAISS cosine, county-level         (1a)
  * leven_county    RapidFuzz Levenshtein, county      (1b)
  * leven_state     RapidFuzz Levenshtein, state       (1b)
  * cosine_country  FAISS cosine, nationwide IVF-PQ    (1c) -- folded in BY DEFAULT
Drop any source by omitting it from --sources.

Output feeds 2b_refine_filter.py.

How "best score" works (cosine and Levenshtein are NOT comparable)
------------------------------------------------------------------
The best score is taken *within each metric independently* -- the two are never
compared or combined onto a common scale:
  * cosine      -> max  (similarity; higher = more similar, 1 = identical).
                  TF-IDF cosine is computed per blocking batch, so the same name
                  pair can score differently across batches; max keeps the
                  strongest signal. Note the nationwide 1c source recomputes an
                  *exact* cosine in [-1, 1], so cosine_score may be negative.
  * Levenshtein -> min  (normalized distance; lower = more similar, 0 = identical).
                  It is a pure function of the two strings (batch-independent),
                  so min is effectively just deduplication.
The reduced sources are then outer-merged into *separate* columns
(`cosine_score`, `levenshtein_score`); a pair found by only one method has the
other score missing. These scores only assemble the union of candidate pairs --
2b recomputes its own similarity features and drops both columns before the
matched_candidates feature table, so they never reach the classifier as such.

Input file-name convention (as written by the blocking scripts), all inside
`--blocking-dir`:
    cosine_match_county_{year}_{gender}.parquet         (cosine_county, 1a)
    levenshtein_match_county_{year}_{gender}.parquet    (leven_county, 1b)
    levenshtein_match_state_{year}_{gender}.parquet     (leven_state, 1b)
    cosine_match_country_{year}_{gender}.parquet        (cosine_country, 1c)
Missing files are skipped, so a requested source whose files are absent (e.g. 1c
never ran for a wave) simply contributes nothing.
"""

import argparse
import os

from collect_common import collect_and_write

# Selectable blocking sources: name -> (filename template, raw score column, metric).
# `cosine_country` (1c) is included in the default selection like the others.
SOURCE_SPECS = {
    "cosine_county":  ("cosine_match_county_{year}_{g}.parquet",      "Cosine_Similarity", "cosine"),
    "cosine_country": ("cosine_match_country_{year}_{g}.parquet",     "Cosine_Similarity", "cosine"),
    "leven_county":   ("levenshtein_match_county_{year}_{g}.parquet", "score",             "leven"),
    "leven_state":    ("levenshtein_match_state_{year}_{g}.parquet",  "score",             "leven"),
}
ALL_SOURCES = ["cosine_county", "leven_county", "leven_state", "cosine_country"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Collect + deduplicate blocking candidates for one census year (polars).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--blocking-dir", required=True,
                   help="Directory holding the blocking output parquet files.")
    p.add_argument("--census-year", type=int, required=True, help="Census year.")
    p.add_argument("--genders", nargs="+", default=["M", "F", "U"],
                   help="Inventor genders to collect.")
    p.add_argument("--sources", nargs="+", choices=ALL_SOURCES, default=ALL_SOURCES,
                   help="Which blocking outputs to collect from --blocking-dir "
                        "(default: all four, incl. the nationwide 1c cosine). Drop a "
                        "source by omitting it, e.g. --sources cosine_county "
                        "leven_county leven_state.")
    p.add_argument("--output", required=True, help="Output candidates parquet path.")
    p.add_argument("--false-coinvs", default=None,
                   help="Optional CSV with a PatInvID column listing false co-inventors to "
                        "drop from the candidates (excluded from the published deposit).")
    return p.parse_args(argv)


def build_sources(blocking_dir, year, genders, sources):
    """(cosine, leven) lists of (path, raw_score_column) for the selected sources.

    Each selected source contributes one file per gender; cosine and Levenshtein
    sources are split into the two lists collect_and_write expects. Missing files
    are skipped downstream, so naming a source whose files are absent is harmless."""
    cosine, leven = [], []
    for name in sources:
        tmpl, raw, metric = SOURCE_SPECS[name]
        for g in genders:
            path = os.path.join(blocking_dir, tmpl.format(year=year, g=g))
            (cosine if metric == "cosine" else leven).append((path, raw))
    return cosine, leven


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cosine, leven = build_sources(args.blocking_dir, args.census_year, args.genders,
                                  args.sources)
    collect_and_write(
        cosine_sources=cosine,
        leven_sources=leven,
        year=args.census_year,
        output=args.output,
        false_coinvs_path=args.false_coinvs,
    )


if __name__ == "__main__":
    main()
