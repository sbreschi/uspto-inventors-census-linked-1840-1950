#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML model stage 1: build the labeled training table from the relatedness output.

Almost all of the work is already done upstream: the stage-2b `matched_candidates`
table (run with `--golden`) carries the `true_histid` / `true_match` labels and
all name/geo/age/occupation feature inputs, and the relatedness add-on appends
`relatedness_short` / `relatedness_long` (and the occupation/patent strings).
So this stage only has to (a) keep the golden-labeled candidate rows and (b)
select the groups.

Input is the stage-3 merged-features output (`3_merge_features.py` ->
`matched_candidates_features.parquet`, or a per-year directory) -- the 2b
matched_candidates table enriched with the relatedness scores and the name-rarity
feature `surname_logfreq`. It is a 5xx-million-row table whose labeled subset is
tiny (`true_match` is non-null only for the ~hundreds-of-thousands of golden
PatInvID candidates); it is scanned lazily with polars and filtered to the labeled
rows before collecting, so memory stays bounded.

This stage is a TAGGER, not a filter: after dropping histid-null rows it keeps EVERY
labeled `(PatInvID, pubnumber, census_year)` group and adds a `patinv_type` column so the
three kinds are distinguishable downstream. The decision of which kinds the classifier
actually fits on is made at stage 2 (`--exclude-orphans` / `--exclude-hard-negatives`),
NOT here. The three types (over the grain below):
  * `matched`       -- the true match survived blocking+filtering (`sum(true_match) >= 1`);
  * `orphan`        -- golden (`true_histid` known) but the true match was lost downstream
                       (`sum(true_match) == 0`, not a hard negative) -- competing
                       candidates only;
  * `hard_negative` -- deceased-inventor group (stage-3 `is_hard_negative == 1`, provably
                       absent from this wave, `true_histid` null).
The partition is exhaustive: after the `true_match`-not-null filter every non-hard-negative
row is golden (`true_match` is non-null iff `true_histid` is non-null, per refine_common).

The grain includes `census_year`: on this all-years merged table a bare `(PatInvID,
pubnumber)` grain sums a multi-wave inventor's per-wave true matches to >=2, so
`census_year` keeps the intended one-true-per-wave unit, matching the stage-2
`(PatInvID, census_year)` ranking group.

All inputs/outputs are CLI arguments; nothing is hardcoded.
"""

import argparse
import os
import time

import polars as pl

# Source columns the stage-2 classifier (and its feature build) consumes, plus the
# pair keys and golden labels. `occ` is optional (present only for 1850-1880 & 1940);
# everything else must be present in the relatedness output.
KEY_COLUMNS = ["PatInvID", "pubnumber", "histid", "census_year"]
FEATURE_SOURCE_COLUMNS = [
    "leven_norm", "tkratio", "tksortratio",  # name metrics
    "surname_logfreq",                        # census-population surname rarity
    "n_candidates",                           # group-size feature (full set, stage-3 merge)
    "county_distance",                        # geography
    "pubyear", "birthyr",                     # -> age_at_pat
    "ind1950", "occ1950", "occstr",          # -> 14 occupation dummies (stage 2)
    "relatedness_short", "relatedness_long",  # patent<->occupation cosine
]
# `diff_length` (abs name-length gap) reaches stage 2 one of two ways: the raw name
# strings (`inventor_name`, `census_name`) are present and stage 2 computes it, OR it is
# precomputed and the names are absent -- the stage-4 publication table
# (4_select_ml_columns.py) drops the personal name strings and ships `diff_length` in
# their place. Exactly one form must be present (checked in main); whichever exists is
# carried through. Both groups are OPTIONAL at the schema level for that reason.
NAME_COLUMNS = ["inventor_name", "census_name"]
DERIVED_COLUMNS = ["diff_length"]
LABEL_COLUMNS = ["true_histid", "true_match"]
# `is_hard_negative` (stage-3 deceased-inventor flag) is a label-SOURCE column, not a
# model feature -- it is carried through so the group-selection step can exempt the
# all-negative deceased groups from the orphan drop, but it must NOT enter
# FEATURE_SOURCE_COLUMNS (the classifier would leak the label). Optional so a
# feature table without it still builds.
OPTIONAL_COLUMNS = ["occ", "is_hard_negative"]

REQUIRED_COLUMNS = KEY_COLUMNS + FEATURE_SOURCE_COLUMNS + LABEL_COLUMNS
TRAIN_COLUMNS = (KEY_COLUMNS + FEATURE_SOURCE_COLUMNS + NAME_COLUMNS + DERIVED_COLUMNS
                 + OPTIONAL_COLUMNS + LABEL_COLUMNS)


def _lap(label, t0):
    """Print elapsed seconds since t0 for a pipeline stage; return a fresh timestamp."""
    now = time.perf_counter()
    print(f"[timing] {label}: {now - t0:.1f}s", flush=True)
    return now


def _scan(path):
    """Lazily scan the relatedness output, whether a single parquet file or a
    directory of per-year parquets."""
    if os.path.isdir(path):
        return pl.scan_parquet(os.path.join(path, "**/*.parquet"))
    return pl.scan_parquet(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build the labeled ML training table from the relatedness output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--relatedness", required=True,
                   help="Stage-3 merged-features output (3_merge_features.py): "
                        "matched_candidates_features.parquet or a per-year directory "
                        "(carries relatedness scores + surname_logfreq).")
    p.add_argument("--output", required=True, help="Output training parquet path.")
    p.add_argument("--years", type=int, nargs="*", default=None,
                   help="Optional subset of census years (default: all present).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()

    lf = _scan(args.relatedness)

    # Feature-presence check: every required source column must exist upstream.
    present = set(lf.collect_schema().names())
    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    if missing:
        raise SystemExit(
            f"[1] relatedness output is missing required column(s): {missing}\n"
            f"    present columns: {sorted(present)}")
    # diff_length must be derivable: either precomputed (stage-4 publication table) or
    # both name strings present (full feature table) so stage 2 can build it.
    if "diff_length" not in present and not all(c in present for c in NAME_COLUMNS):
        raise SystemExit(
            "[1] cannot build diff_length: need a precomputed 'diff_length' column "
            f"or both name columns {NAME_COLUMNS}; present: {sorted(present)}")
    keep_cols = [c for c in TRAIN_COLUMNS if c in present]

    # Keep only the golden-labeled candidate rows (true_match in {0, 1}); the
    # unlabeled bulk (true_match null) is dropped before collecting.
    lf = lf.filter(pl.col("true_match").is_not_null())
    if args.years:
        lf = lf.filter(pl.col("census_year").is_in(args.years))

    labeled = lf.select(keep_cols).collect(engine="streaming")
    t0 = _lap(f"scan + filter labeled rows ({labeled.height:,} rows)", t0)

    # Drop histid-null rows.
    labeled = labeled.filter(pl.col("histid").is_not_null())

    # Tag each (PatInvID, pubnumber, census_year) group with its type; keep ALL of them.
    # Grain matches the stage-2 (PatInvID, census_year) ranking unit (census_year matters:
    # a multi-wave inventor sums per-wave true matches to >=2 over the bare (PatInvID,
    # pubnumber)). The classifier (stage 2) decides which types it fits on.
    grain = ["PatInvID", "pubnumber", "census_year"]
    hn_group = ((pl.col("is_hard_negative").max().over(grain) == 1)
                if "is_hard_negative" in labeled.columns else pl.lit(False))
    labeled = labeled.with_columns(
        pl.when(hn_group).then(pl.lit("hard_negative"))
          .when(pl.col("true_match").sum().over(grain) >= 1).then(pl.lit("matched"))
          .otherwise(pl.lit("orphan")).alias("patinv_type"))

    # Deterministic, byte-stable output: dedup then sort on the unique candidate key
    # (PatInvID, pubnumber, histid) so the row order does not vary run-to-run (the
    # downstream XGBoost `hist` builder is row-order sensitive).
    labeled = labeled.unique().sort(["PatInvID", "pubnumber", "histid"])

    # Group-size feature `n_candidates` (distinct candidate census persons per PatInvID
    # per wave, a name-ambiguity proxy) is computed upstream by the stage-3 merge
    # (3_merge_features.py) over the FULL candidate set and carried through here -- NOT
    # recomputed on this labeled subset (which would undercount). Kept via FEATURE_SOURCE_COLUMNS.

    pos = int(labeled.select(pl.col("true_match").eq(1).sum()).item())
    print(f"[1] training rows: {labeled.height:,} "
          f"({pos:,} true matches / {labeled.height - pos:,} negatives)", flush=True)
    # Per-type row + group breakdown (matched / orphan / hard_negative).
    by_type = (labeled.group_by("patinv_type")
               .agg(pl.len().alias("rows"),
                    pl.struct(grain).n_unique().alias("groups"))
               .sort("patinv_type"))
    for r in by_type.iter_rows(named=True):
        print(f"[1]   {r['patinv_type']:14s} rows={r['rows']:>9,}  groups={r['groups']:>6,}",
              flush=True)
    print(f"[1] groups (PatInvID, census_year): "
          f"{labeled.select(['PatInvID', 'census_year']).n_unique():,}", flush=True)

    labeled.write_parquet(args.output, compression="zstd")
    _lap(f"wrote training table -> {args.output}", t0)


if __name__ == "__main__":
    main()
