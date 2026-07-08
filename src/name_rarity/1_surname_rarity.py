#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Name-rarity add-on, stage 1: per-wave census surname-rarity reference table.

A candidate whose census surname is rare in the population (e.g. *Cattaneo*) is a
far more trustworthy match than one with a common surname (e.g. *Smith*), regardless
of edit distance. This stage measures that, ONCE, from the full cleaned census so the
heavy count is never recomputed on a normal pipeline run.

For one census year it scans the entire cleaned census (`US{year}P.parquet`, the
stage-0 output -- a partitioned parquet *directory*), counts how many individuals
bear each surname `clean_ln`, and emits a small reference table

    clean_ln, n, surname_logfreq

where, with N = number of records carrying a (non-empty) surname in the wave,

    surname_logfreq = -log10(n / N)   ==   log10(N) - log10(n)

i.e. the **negated log relative frequency**: rarer surnames score HIGHER (a unique
surname n=1 -> log10(N) ~ 7.4 for N~25M; the commonest surnames -> ~2), always >= 0.

Stage 2b (`refine_common.refine_and_write`) joins this reference onto each candidate
by `clean_ln` (the authoritative census surname it already loads per histid), adding
`surname_logfreq` to the matched_candidates table. The relatedness add-on carries it
through, and the ML model (`ml_model/2_train_classifier.py`) consumes it as a feature.

All inputs/outputs are CLI arguments; nothing is hardcoded.

"""

import argparse
import os
import time

import polars as pl


def _lap(label, t0):
    """Print elapsed seconds since t0 for a pipeline stage; return a fresh timestamp."""
    now = time.perf_counter()
    print(f"[timing] {label}: {now - t0:.1f}s", flush=True)
    return now


def _scan_census(path):
    """Lazily scan a census parquet -- single `.parquet` file or partitioned
    *directory* (`part.*.parquet`). Mirrors `refine_common._scan_census`."""
    p = str(path)
    if os.path.isdir(p):
        return pl.scan_parquet(p.rstrip("/\\") + "/**/*.parquet")
    return pl.scan_parquet(p)


def surname_rarity(cleaned_census_path):
    """Per-wave surname reference (clean_ln, n, surname_logfreq) from the full
    cleaned census. Keeps records with a non-null, non-empty `clean_ln` (matching the
    surname filter in `blocking_common.census_full_names`)."""
    lf = (_scan_census(cleaned_census_path)
          .select("clean_ln")
          .filter(pl.col("clean_ln").is_not_null() & (pl.col("clean_ln").str.len_chars() > 0)))
    # Counts per surname. Collected via the streaming engine: the cleaned census is a
    # partitioned-directory scan (thousands of part-file chunks), and polars' in-memory
    # group-by/join path can panic on that layout ("index out of bounds" in
    # single_keys.rs); the streaming path is unaffected (house style for heavy collects).
    counts = (lf.group_by("clean_ln").agg(pl.len().alias("n"))
                .collect(engine="streaming"))
    n_total = int(counts.select(pl.col("n").sum()).item())
    if n_total == 0:
        raise SystemExit(f"[1] no surnames found in {cleaned_census_path}")
    # surname_logfreq = -log10(n / N) = log10(N) - log10(n); rarer -> larger, always >= 0.
    import math
    log10_n_total = math.log10(n_total)
    out = (counts
           .with_columns((pl.lit(log10_n_total) - pl.col("n").cast(pl.Float64).log10())
                         .alias("surname_logfreq"))
           .sort(["surname_logfreq", "clean_ln"], descending=[True, False]))
    return out, n_total


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build a per-wave census surname-rarity reference table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cleaned-census", required=True,
                   help="Cleaned census parquet (stage-0 output US{year}P.parquet).")
    p.add_argument("--census-year", type=int, required=True, help="Census year (for logging).")
    p.add_argument("--output", required=True,
                   help="Output reference parquet (surname_rarity_{year}.parquet).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()

    out, n_total = surname_rarity(args.cleaned_census)
    t0 = _lap(f"surname counts for {args.census_year} "
              f"({out.height:,} distinct surnames over {n_total:,} records)", t0)

    out.write_parquet(args.output, compression="zstd")
    rng = out.select([pl.col("surname_logfreq").min().alias("lo"),
                      pl.col("surname_logfreq").max().alias("hi")]).row(0)
    print(f"[1] surname_logfreq range [{rng[0]:.3f}, {rng[1]:.3f}] "
          f"-> {args.output}", flush=True)
    _lap("wrote reference table", t0)


if __name__ == "__main__":
    main()
