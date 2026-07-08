#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3: merge per-year feature sources into the final matched_candidates table.

Concatenates the per-year relatedness outputs (relatedness add-on stage 4,
`by_year/matched_candidates_rel_{year}.parquet` -- the 2b matched_candidates rows
carrying `relatedness_short` / `relatedness_long`), attaches the census-population
surname-rarity feature `surname_logfreq`, and adds the group-size feature
`n_candidates`, producing the single table the downstream ML model (`ml_model/`)
consumes.

`n_candidates` is the number of distinct candidate census persons (`histid`) competing
for each `PatInvID` in its wave -- a cheap name-ambiguity proxy (common names -> larger
groups). It is counted on distinct histid (not rows) so multiple patents of the same
inventor don't inflate it, over the FULL candidate set in each per-year file (each file
is one wave, so grouping by PatInvID here is the (PatInvID, census_year) disambiguation
unit used downstream).

`surname_logfreq` (rarer surname -> higher value) is precomputed once per wave by
`name_rarity/1_surname_rarity.py` as a small reference keyed by surname `clean_ln`.
The relatedness rows carry `histid` but not the bare surname, so per year we bridge
`histid -> clean_ln` through the cleaned census (`US{year}P.parquet`, which is keyed
by histid and unique per person) and join the reference on `clean_ln`, yielding a
`histid -> surname_logfreq` map left-joined onto that year's rows. Both histid sides
are census-derived, so they share case and the join is exact.

Attaching rarity is optional: pass BOTH `--cleaned-census-dir` and
`--surname-rarity-dir` to add `surname_logfreq`; pass neither to fall back to a plain
relatedness concat (a warning is printed -- the ML model expects the feature).

`--deceased-negatives` (optional) wires in the golden_negatives add-on's deceased
hard negatives. For each confirmed-deceased inventor in the reviewed CSV
(`golden_negatives/deceased_inventors_reviewed.csv`, `confirmed == YES`), all of its
candidate rows in its `eligible_census_year` -- the census taken after the patent, in
which a dead inventor is provably absent -- are flipped to `true_match = 0` and flagged
`is_hard_negative = 1`. Marking is collision-safe: only currently-unlabeled rows
(`true_match` null) are flipped, so the golden sample's own labels are never
overwritten (collisions are counted and reported). `is_hard_negative` is always emitted
(0 when `--deceased-negatives` is omitted or a row is not a deceased candidate); it is a
label-SOURCE column carried for the ML training builder's group-retention logic, NOT a
model feature. The golden sample CSV is not modified, so the stage-2c golden diagnostics
are untouched.

By default the merge projects each relatedness file to only the columns the ML model
consumes (`_ML_INPUT_COLUMNS`), dropping the geo/gender passthroughs and the bulky
`occ_desc_*`/`pat_desc_*` description strings (used only to compute relatedness). Each
wave is processed independently to a part file and the parts are concatenated, so peak
memory is bounded to one wave -- the full-data single-shot concat OOMs. `--keep-all-columns`
retains the full schema.

All inputs/outputs are CLI arguments; nothing is hardcoded.
"""

import argparse
import glob
import os
import re
import shutil

import polars as pl

_REL_GLOB = "matched_candidates_rel_*.parquet"
_YEAR_RE = re.compile(r"matched_candidates_rel_(\d{4})\.parquet$")

# Column names read from the deceased-inventors review file
# (golden_negatives/deceased_inventors_reviewed.csv). These are that file's SCHEMA,
# not paths -- the path itself is the `--deceased-negatives` CLI argument.
_DEC_ID_COL = "PatInvID"            # inventor id (matches the merged table key)
_DEC_YEAR_COL = "eligible_census_year"  # census wave the inventor is provably absent in
_DEC_CONFIRM_COL = "confirmed"      # manual review verdict ("YES" admits the row)

# Columns the downstream ML model (`ml_model/1_build_training.py` TRAIN_COLUMNS) actually
# consumes -- MINUS the two this stage adds (`surname_logfreq`, `n_candidates`). By default
# the merge projects each relatedness file to these before joining, dropping the columns the
# model never reads: the geo/gender passthroughs (county_*/state_*/gender_*) and -- the big
# win -- the bulky `occ_desc_*`/`pat_desc_*` description strings that existed only to COMPUTE
# the relatedness cosine. Carrying those through the per-wave joins is what blows memory.
# `--keep-all-columns` retains the full schema. Year-varying columns (occ/occstr) are kept
# only where present (intersection), and the final concat is `diagonal_relaxed`.
_ML_INPUT_COLUMNS = [
    "PatInvID", "pubnumber", "histid", "census_year",
    "inventor_name", "census_name",
    "leven_norm", "tkratio", "tksortratio",
    "county_distance", "pubyear", "birthyr",
    "ind1950", "occ1950", "occstr", "occ",
    "relatedness_short", "relatedness_long",
    "true_histid", "true_match",
]


def _scan_census(path):
    """Lazily scan a census parquet -- single `.parquet` file or partitioned
    *directory* (`part.*.parquet`). Mirrors `refine_common._scan_census`."""
    p = str(path)
    if os.path.isdir(p):
        return pl.scan_parquet(p.rstrip("/\\") + "/**/*.parquet")
    return pl.scan_parquet(p)


def _histid_logfreq(cleaned_census_dir, surname_rarity_dir, year):
    """Lazy (histid, surname_logfreq) map for one wave: census histid->clean_ln joined
    to the precomputed surname reference (clean_ln->surname_logfreq)."""
    census = _scan_census(os.path.join(cleaned_census_dir, f"US{year}P.parquet")) \
        .select(["histid", "clean_ln"])
    ref = pl.scan_parquet(os.path.join(surname_rarity_dir, f"surname_rarity_{year}.parquet")) \
        .select(["clean_ln", "surname_logfreq"])
    return (census.join(ref, on="clean_ln", how="left")
                  .select(["histid", "surname_logfreq"])
                  .unique(subset=["histid"]))


def _load_deceased_map(path):
    """Build {eligible_census_year: [PatInvID, ...]} from the reviewed deceased-inventors
    CSV, keeping only `confirmed == YES` rows. A self-deceased inventor cannot appear in
    the census wave after the patent, so every candidate matched to that PatInvID in its
    `eligible_census_year` is a guaranteed non-match (a hard negative). Returns {} if the
    file has no admitted rows. PatInvID is cast to Int64 to match the merged-table key."""
    df = pl.read_csv(path, infer_schema_length=10000)
    missing = [c for c in (_DEC_ID_COL, _DEC_YEAR_COL, _DEC_CONFIRM_COL)
               if c not in df.columns]
    if missing:
        raise SystemExit(f"[3] --deceased-negatives {path} is missing column(s): {missing}")
    df = (df.filter(pl.col(_DEC_CONFIRM_COL).cast(pl.Utf8).str.strip_chars()
                    .str.to_uppercase() == "YES")
            .select([pl.col(_DEC_ID_COL).cast(pl.Int64),
                     pl.col(_DEC_YEAR_COL).cast(pl.Int64)])
            .drop_nulls()
            .unique())
    out = {}
    for year, sub in df.group_by(_DEC_YEAR_COL):
        out[int(year[0])] = sub[_DEC_ID_COL].to_list()
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Merge per-year relatedness + name-rarity features into matched_candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input-dir", required=True,
                   help="Directory with per-year relatedness parquets "
                        "(matched_candidates_rel_{year}.parquet).")
    p.add_argument("--cleaned-census-dir", default=None,
                   help="Directory with cleaned US{year}P.parquet (stage-0 output); needed "
                        "with --surname-rarity-dir to attach surname_logfreq.")
    p.add_argument("--surname-rarity-dir", default=None,
                   help="Directory with per-wave surname_rarity_{year}.parquet "
                        "(name_rarity/1_surname_rarity.py); needed with --cleaned-census-dir.")
    p.add_argument("--keep-all-columns", action="store_true",
                   help="Keep every column from the relatedness files. Default: project to "
                        "the columns the ML model consumes (drops geo/gender passthroughs and "
                        "the bulky occ/patent description strings) so the merge stays "
                        "memory-bounded.")
    p.add_argument("--deceased-negatives", default=None,
                   help="Reviewed deceased-inventors CSV "
                        "(golden_negatives/deceased_inventors_reviewed.csv). For every "
                        "confirmed=YES inventor, all candidates in its eligible_census_year "
                        "are marked true_match=0 (hard negatives) and flagged "
                        "is_hard_negative=1. Only currently-unlabeled rows are flipped "
                        "(golden labels are preserved); collisions are reported. Omit to "
                        "skip (is_hard_negative is then all 0).")
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Rarity attachment is all-or-nothing: both dirs, or neither.
    attach_rarity = bool(args.cleaned_census_dir) and bool(args.surname_rarity_dir)
    if bool(args.cleaned_census_dir) != bool(args.surname_rarity_dir):
        raise SystemExit("[3] --cleaned-census-dir and --surname-rarity-dir must be "
                         "given together (or both omitted to skip surname_logfreq).")

    files = sorted(glob.glob(os.path.join(args.input_dir, _REL_GLOB)))
    if not files:
        raise SystemExit(f"[3] no {_REL_GLOB} under {args.input_dir}")
    print(f"[3] found {len(files)} per-year relatedness files; "
          f"surname_logfreq={'on' if attach_rarity else 'OFF (relatedness only)'}; "
          f"columns={'all' if args.keep_all_columns else 'ML subset'}", flush=True)
    if not attach_rarity:
        print("[3] WARNING: surname_logfreq not attached; the ML model expects this feature.",
              flush=True)

    # Deceased-inventor hard negatives: {eligible_census_year: [PatInvID, ...]}. Empty
    # ({}) when --deceased-negatives is omitted -- is_hard_negative is then all 0.
    deceased_map = _load_deceased_map(args.deceased_negatives) if args.deceased_negatives else {}
    if args.deceased_negatives:
        n_inv = sum(len(v) for v in deceased_map.values())
        print(f"[3] deceased hard negatives: {n_inv:,} confirmed inventors across "
              f"{len(deceased_map)} waves from {args.deceased_negatives}", flush=True)
    dec_marked_total = dec_collision_total = 0

    # Process each wave INDEPENDENTLY to its own part file, then concatenate the parts.
    # Doing all waves in one `pl.concat(...).sink_parquet()` builds every wave's group-by
    # and census-join hash tables simultaneously and OOMs on the full data (~250 GB); a
    # per-wave sink bounds peak memory to a single wave's hash tables. The final concat is a
    # light column-union with no joins.
    parts_dir = os.path.join(os.path.dirname(args.output) or ".", "_merge_parts")
    os.makedirs(parts_dir, exist_ok=True)
    part_files = []
    for i, f in enumerate(files):
        # Wave year is needed for both rarity attach and deceased-negative marking.
        m = _YEAR_RE.search(os.path.basename(f))
        if not m:
            raise SystemExit(f"[3] cannot parse census year from filename: {f}")
        year = int(m.group(1))
        lf = pl.scan_parquet(f)
        present = lf.collect_schema().names()
        if not args.keep_all_columns:
            lf = lf.select([c for c in _ML_INPUT_COLUMNS if c in present])
        # n_candidates: distinct candidate census persons (histid) per PatInvID in this
        # wave. Computed via group-by + left-join (streaming-safe) rather than a window
        # so it composes with sink_parquet's streaming path.
        ncand = lf.group_by("PatInvID").agg(pl.col("histid").n_unique().alias("n_candidates"))
        lf = lf.join(ncand, on="PatInvID", how="left")
        if attach_rarity:
            rarity = _histid_logfreq(args.cleaned_census_dir, args.surname_rarity_dir, year)
            lf = lf.join(rarity, on="histid", how="left")
        # Deceased hard negatives: flag candidates of confirmed-deceased inventors whose
        # eligible_census_year is THIS wave and flip their label to 0. Collision-safe --
        # only currently-unlabeled rows (true_match null) are flipped, so golden labels
        # survive untouched; is_hard_negative is always emitted (0 when no marking).
        # Marking requires the true_match label column (present when 2b ran with --golden).
        can_mark = "true_match" in present
        dec_ids = deceased_map.get(year, []) if can_mark else []
        if deceased_map and not can_mark:
            print(f"[3] {year}: WARNING no true_match column; skipping deceased marking "
                  f"(was 2b run without --golden?)", flush=True)
        if can_mark:
            is_dec = pl.col("PatInvID").is_in(dec_ids) if dec_ids else pl.lit(False)
            lf = lf.with_columns(
                (is_dec & pl.col("true_match").is_null()).cast(pl.Int8).alias("is_hard_negative"))
            lf = lf.with_columns(
                pl.when(pl.col("is_hard_negative") == 1).then(pl.lit(0, dtype=pl.Int64))
                  .otherwise(pl.col("true_match")).alias("true_match"))
        else:
            lf = lf.with_columns(pl.lit(0, dtype=pl.Int8).alias("is_hard_negative"))
        if dec_ids:
            # One cheap streaming pass for stats: rows marked vs. golden collisions skipped.
            # A collision is a deceased-PatInvID row that already carried a golden label
            # (is_hard_negative stayed 0 because true_match was non-null); those are left
            # as-is. Test on is_hard_negative -- NOT on `true_match is not null`, which now
            # also holds for the rows we just flipped to 0.
            stats = lf.select([
                pl.col("is_hard_negative").sum().alias("marked"),
                (is_dec & (pl.col("is_hard_negative") == 0)).sum().alias("collisions"),
            ]).collect(engine="streaming")
            marked, collisions = int(stats["marked"][0]), int(stats["collisions"][0])
            dec_marked_total += marked
            dec_collision_total += collisions
            print(f"[3] {year}: deceased hard negatives marked={marked:,} "
                  f"(golden-label collisions left untouched={collisions:,})", flush=True)
        part = os.path.join(parts_dir, f"part_{i:02d}.parquet")
        lf.sink_parquet(part, compression="zstd")
        part_files.append(part)
        print(f"[3] merged {os.path.basename(f)} -> {os.path.basename(part)}", flush=True)

    # Light final concat (column-union, no joins). diagonal_relaxed tolerates per-year
    # schema drift (e.g. occ vs occstr years).
    pl.concat([pl.scan_parquet(p) for p in part_files], how="diagonal_relaxed") \
      .sink_parquet(args.output, compression="zstd")
    shutil.rmtree(parts_dir, ignore_errors=True)

    n = pl.scan_parquet(args.output).select(pl.len()).collect().item()
    print(f"[3] wrote {n:,} rows -> {args.output}", flush=True)
    if args.deceased_negatives:
        print(f"[3] deceased hard negatives total: marked={dec_marked_total:,} rows "
              f"(true_match set to 0); golden-label collisions skipped="
              f"{dec_collision_total:,}", flush=True)
    if attach_rarity:
        miss = (pl.scan_parquet(args.output)
                .select(pl.col("surname_logfreq").is_null().sum())
                .collect(engine="streaming").item())
        print(f"[3] surname_logfreq null for {miss:,} rows ({miss / n:.2%})", flush=True)


if __name__ == "__main__":
    main()
