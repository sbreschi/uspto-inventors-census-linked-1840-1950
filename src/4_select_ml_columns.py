#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4: project the stage-3 feature table to the ML/publication column subset.

Reads the stage-3 output (`3_merge_features.py` -> `matched_candidates_features.parquet`,
a single parquet or a per-year directory) and writes a slimmed copy keeping ONLY the
columns the downstream ML model (`ml_model/`) actually consumes: the identifiers/keys,
the feature-source columns, the golden labels, and the two optional/label-source
columns. Everything else (any geo/gender passthroughs or bulky `occ_desc_*`/`pat_desc_*`
strings retained when stage 3 ran with `--keep-all-columns`) is dropped.

This is the table intended for the Zenodo deposit accompanying the Scientific Data
submission: it is a deliberate, documented "publication" projection rather than the
upstream default, small enough to share yet complete enough that a third party can
(a) rebuild the ML training table (`ml_model/1_build_training.py` reads exactly this
column set) and (b) score/run the classifier over the full candidate universe.

ALL rows are retained -- the projection is columns-only, byte-stable (no reorder, no
dedup), so the deposit reproduces both training and full prediction.

The raw inventor/census NAME strings are NOT published. `diff_length` (the absolute
name-length gap) is the only model feature derived from them, so it is precomputed
here -- exactly as stage 2's `build_features` would (`abs(len(inventor_name) -
len(census_name))`) -- and the two name columns are dropped. The downstream ML stages
consume this precomputed `diff_length` directly (`1_build_training.py` /
`2_train_classifier.py` were made to accept it), so nothing personal-name-bearing
leaves the pipeline yet every training feature is present.

The kept set (up to 23 columns): 4 keys + `diff_length` + 13 feature sources + 2 optional
+ 2 labels + `source`. `occ` (1850-1880 & 1940 only) and `is_hard_negative` are optional so older
feature tables without them still project. The required-column guard mirrors stage 1's
(the two name columns ARE required in the INPUT so diff_length can be derived), so a
table missing a needed feature source fails loudly here rather than downstream.

A per-row provenance column `source` is appended last:
"https://patents.google.com/patent/US<pubnumber> | https://doi.org/10.18128/D014.V4.0R"
(the Google Patents page of the patent | the IPUMS full-count DOI). It is a pure function
of `pubnumber`, is never read by the ML stages (they select their own column lists), and
leaves every other column unchanged.

NOTE (IPUMS redistribution): the kept set still includes census-derived fields
(`occstr`, `histid`) that originate in restricted IPUMS USA extracts. Confirm your
IPUMS redistribution terms before depositing this table publicly.

All inputs/outputs are CLI arguments; nothing is hardcoded.
"""

import argparse
import os
import time

import polars as pl

# KEEP IN SYNC with src/ml_model/1_build_training.py -- the digit-prefixed stage-1 module
# is not cleanly importable, so the canonical column set is mirrored here. stage 1 reads
# these from the features table; stage 2's build_features() derives every model feature
# from them. If you edit the ML column set, edit it in both places.
KEY_COLUMNS = ["PatInvID", "pubnumber", "histid", "census_year"]

# Raw inventor/census name strings: REQUIRED in the INPUT (to derive diff_length) but
# deliberately EXCLUDED from the published output -- diff_length is the only model feature
# derived from them and is precomputed below, so no personal name strings are published.
NAME_COLUMNS = ["inventor_name", "census_name"]
DERIVED_COLUMNS = ["diff_length"]  # precomputed here from NAME_COLUMNS; ships in their place

FEATURE_SOURCE_COLUMNS = [
    "leven_norm", "tkratio", "tksortratio",   # name-similarity metrics (already numeric)
    "surname_logfreq",                         # census-population surname rarity
    "n_candidates",                            # group-size feature (stage-3 merge)
    "county_distance",                         # geography
    "pubyear", "birthyr",                      # -> age_at_pat
    "ind1950", "occ1950", "occstr",           # -> 14 occupation dummies (stage 2)
    "relatedness_short", "relatedness_long",   # patent<->occupation cosine
]
LABEL_COLUMNS = ["true_histid", "true_match"]
# `occ` present only for 1850-1880 & 1940; `is_hard_negative` is a label-SOURCE column
# (deceased-inventor flag), carried for the ML training builder's group logic, never a
# model feature. Both optional so tables lacking them still project.
OPTIONAL_COLUMNS = ["occ", "is_hard_negative"]
# Per-row provenance, derived from `pubnumber`: Google Patents URL | IPUMS full-count DOI.
PATENT_URL_PREFIX = "https://patents.google.com/patent/US"
IPUMS_DOI = "https://doi.org/10.18128/D014.V4.0R"
SOURCE_COLUMNS = ["source"]

# Required in the INPUT table (names included -- needed to derive diff_length).
REQUIRED_COLUMNS = KEY_COLUMNS + NAME_COLUMNS + FEATURE_SOURCE_COLUMNS + LABEL_COLUMNS
# Present in the OUTPUT: names dropped, diff_length added in their place, source last.
OUTPUT_COLUMNS = (KEY_COLUMNS + DERIVED_COLUMNS + FEATURE_SOURCE_COLUMNS
                  + OPTIONAL_COLUMNS + LABEL_COLUMNS + SOURCE_COLUMNS)
# Built here (not read from the input), so always kept.
GENERATED_COLUMNS = DERIVED_COLUMNS + SOURCE_COLUMNS


def _lap(label, t0):
    """Print elapsed seconds since t0 for a pipeline stage; return a fresh timestamp."""
    now = time.perf_counter()
    print(f"[timing] {label}: {now - t0:.1f}s", flush=True)
    return now


def _scan(path):
    """Lazily scan the stage-3 features output, whether a single parquet file or a
    directory of per-year parquets."""
    if os.path.isdir(path):
        return pl.scan_parquet(os.path.join(path, "**/*.parquet"))
    return pl.scan_parquet(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Project the stage-3 feature table to the ML/publication column subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="Stage-3 merged-features output (3_merge_features.py): "
                        "matched_candidates_features.parquet or a per-year directory.")
    p.add_argument("--output", required=True,
                   help="Output parquet path (suggested deposit name: "
                        "matched_candidates_ml.parquet).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()

    lf = _scan(args.input)

    # Required-column guard (mirrors 1_build_training.py): every feature source, key,
    # label, and the two name columns (needed for diff_length) must be present.
    # Optional columns may be absent.
    present = lf.collect_schema().names()
    present_set = set(present)
    missing = [c for c in REQUIRED_COLUMNS if c not in present_set]
    if missing:
        raise SystemExit(
            f"[4] input is missing required column(s): {missing}\n"
            f"    present columns: {sorted(present_set)}")

    # Precompute diff_length (abs name-length gap) exactly as stage 2's build_features
    # would (pandas .str.len() == char count -> polars str.len_chars(); cast to Int64
    # before subtracting so the UInt32 char counts don't underflow), then drop the names.
    # `source` = Google Patents URL of the patent | IPUMS DOI (pubnumber is Int64, non-null).
    lf = lf.with_columns(
        (pl.col("inventor_name").str.len_chars().cast(pl.Int64)
         - pl.col("census_name").str.len_chars().cast(pl.Int64)).abs().alias("diff_length"),
        (pl.lit(PATENT_URL_PREFIX) + pl.col("pubnumber").cast(pl.String)
         + pl.lit(" | " + IPUMS_DOI)).alias("source"))

    keep = [c for c in OUTPUT_COLUMNS if c in GENERATED_COLUMNS or c in present_set]
    dropped = [c for c in present if c not in set(keep)]
    print(f"[4] keeping {len(keep)}/{len(present) + len(GENERATED_COLUMNS)} columns "
          f"(+diff_length +source, -{len(NAME_COLUMNS)} names); dropping {len(dropped)}: "
          f"{dropped}", flush=True)

    lf.select(keep).sink_parquet(args.output, compression="zstd")
    t0 = _lap(f"projected + wrote -> {args.output}", t0)

    # Cheap row-count read-back for the log (lazy count, not a full materialise).
    n = pl.scan_parquet(args.output).select(pl.len()).collect().item()
    print(f"[4] wrote {n:,} rows x {len(keep)} columns -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
