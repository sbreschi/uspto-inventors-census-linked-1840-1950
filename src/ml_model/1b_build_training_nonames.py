#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML model stage 1b: build the labeled training table WITHOUT the raw name strings.

Identical to `1_build_training.py` in every respect except that the raw inventor/census
name strings (`inventor_name`, `census_name`) are NOT carried into the training table:
their sole model contribution -- `diff_length`, the absolute name-length gap -- is
precomputed here (exactly as stage 2's `build_features` would:
`abs(len(inventor_name) - len(census_name))`) and shipped in their place, then the two
name columns are dropped. Everything else (row selection, `patinv_type` tagging, the
dedup + byte-stable sort, all other feature-source/label columns) is unchanged, so the
output is the SAME training data the plain stage 1 produces -- just name-free.

Use this to build a training table directly from the full stage-3
`matched_candidates_features.parquet` (which still carries names) without publishing or
retaining any personal name strings. It is equivalent to running plain stage 1 on the
stage-4 publication table (`4_select_ml_columns.py` output): both yield an identical,
name-free, `diff_length`-bearing training table. If the input already has a precomputed
`diff_length` and no names (i.e. the stage-4 table), this stage is a no-op on that front
and simply passes it through.

Input is the stage-3 merged-features output (`3_merge_features.py` ->
`matched_candidates_features.parquet`, or a per-year directory). It is a 5xx-million-row
table whose labeled subset is tiny (`true_match` is non-null only for the
~hundreds-of-thousands of golden PatInvID candidates); it is scanned lazily with polars
and filtered to the labeled rows before collecting, so memory stays bounded.

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
# The raw name strings are read (to derive diff_length) but DROPPED from the output --
# that is the sole difference from `1_build_training.py`. `diff_length` is precomputed in
# their place (or passed through if the input already carries it and no names).
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
# Read the names too (needed to derive diff_length); they are dropped after the compute.
READ_COLUMNS = (KEY_COLUMNS + FEATURE_SOURCE_COLUMNS + NAME_COLUMNS + DERIVED_COLUMNS
                + OPTIONAL_COLUMNS + LABEL_COLUMNS)

# Default output lands in this ml_model/ folder (relative to the script, like
# 2_train_classifier.py's DEF_OCC_DICT -- overridable via --output, not a hardcoded path).
DEF_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "training_nonames.parquet")


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
        description="Build the labeled ML training table (name strings dropped, "
                    "diff_length precomputed) from the relatedness output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--relatedness", required=True,
                   help="Stage-3 merged-features output (3_merge_features.py): "
                        "matched_candidates_features.parquet or a per-year directory "
                        "(carries relatedness scores + surname_logfreq).")
    p.add_argument("--output", default=DEF_OUTPUT,
                   help="Output training parquet path (default: alongside this script "
                        "in the ml_model/ folder).")
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
            f"[1b] relatedness output is missing required column(s): {missing}\n"
            f"    present columns: {sorted(present)}")
    # diff_length must be derivable: either precomputed (stage-4 publication table) or
    # both name strings present so we can build it here before dropping the names.
    if "diff_length" not in present and not all(c in present for c in NAME_COLUMNS):
        raise SystemExit(
            "[1b] cannot build diff_length: need a precomputed 'diff_length' column "
            f"or both name columns {NAME_COLUMNS}; present: {sorted(present)}")
    keep_cols = [c for c in READ_COLUMNS if c in present]

    # Keep only the golden-labeled candidate rows (true_match in {0, 1}); the
    # unlabeled bulk (true_match null) is dropped before collecting.
    lf = lf.filter(pl.col("true_match").is_not_null())
    if args.years:
        lf = lf.filter(pl.col("census_year").is_in(args.years))

    labeled = lf.select(keep_cols).collect(engine="streaming")
    t0 = _lap(f"scan + filter labeled rows ({labeled.height:,} rows)", t0)

    # --- The one difference from stage 1 -----------------------------------------------
    # Precompute diff_length from the names (exactly as stage 2's build_features would),
    # then drop the raw name strings so no personal names enter the training table. Done
    # BEFORE the tag/dedup/sort below so the result is byte-identical to running plain
    # stage 1 on the stage-4 (already name-free) publication table.
    if "diff_length" not in labeled.columns:
        labeled = labeled.with_columns(
            (pl.col("inventor_name").str.len_chars().cast(pl.Int64)
             - pl.col("census_name").str.len_chars().cast(pl.Int64)).abs().alias("diff_length"))
    labeled = labeled.drop([c for c in NAME_COLUMNS if c in labeled.columns])
    # -----------------------------------------------------------------------------------

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
    print(f"[1b] training rows: {labeled.height:,} "
          f"({pos:,} true matches / {labeled.height - pos:,} negatives)", flush=True)
    # Per-type row + group breakdown (matched / orphan / hard_negative).
    by_type = (labeled.group_by("patinv_type")
               .agg(pl.len().alias("rows"),
                    pl.struct(grain).n_unique().alias("groups"))
               .sort("patinv_type"))
    for r in by_type.iter_rows(named=True):
        print(f"[1b]   {r['patinv_type']:14s} rows={r['rows']:>9,}  groups={r['groups']:>6,}",
              flush=True)
    print(f"[1b] groups (PatInvID, census_year): "
          f"{labeled.select(['PatInvID', 'census_year']).n_unique():,}", flush=True)

    labeled.write_parquet(args.output, compression="zstd")
    _lap(f"wrote training table -> {args.output}", t0)


if __name__ == "__main__":
    main()
