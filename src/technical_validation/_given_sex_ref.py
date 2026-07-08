#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Precompute the given-name sex reference for the household-relative validation's
name-corrected sex gate (`4_relative_attach.py --given-sex-ref`).

Paper-analysis helper (Scientific Data descriptor) -- NOT a pipeline stage. Scans one
or more RAW IPUMS person waves (streaming) and writes a small parquet
(`given, n, n_f, female_share`) over the first alpha token of `namefrst`, keeping
names with at least --min-count observations. Stage 4 uses it to correct a
representative's role-word gender when the given name is decisively of the opposite
sex (patent role words are noisy: OCR'd plurals like "ADMINISTRATORS" broadcast over
mixed-sex representative pairs, and the masculine form is sometimes used for women --
the Eugenia-the-"executor" pattern). The reference is population-level census
information, independent of any specific link.

The output derives from restricted IPUMS extracts -- keep it on scratch, never in
the repo. All paths are CLI arguments; nothing is hardcoded.

Usage:
    python src/technical_validation/_given_sex_ref.py \
        --raw-census-dir /scratch/breschi/census --years 1900 \
        --output /scratch/breschi/ipums/prediction/technical_validation/given_name_sex.parquet
"""

import argparse
import os
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build the given-name sex reference (given, n, n_f, "
                    "female_share) from raw IPUMS person waves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-census-dir", required=True,
                   help="Dir of raw IPUMS person files (US{year}P.parquet).")
    p.add_argument("--person-template", default="US{year}P.parquet")
    p.add_argument("--years", type=int, nargs="+", default=[1900],
                   help="Waves to pool (1900 alone covers ~25k names at n>=50).")
    p.add_argument("--min-count", type=int, default=50,
                   help="Minimum observations for a name to enter the reference.")
    p.add_argument("--output", required=True, help="Output parquet.")
    p.add_argument("--spill-dir", default=None,
                   help="polars spill dir (set before import; default: output dir).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    spill = args.spill_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.output)), ".polars_tmp")
    os.makedirs(spill, exist_ok=True)
    os.environ["POLARS_TEMP_DIR"] = spill
    os.environ["TMPDIR"] = spill
    import polars as pl

    t0 = time.perf_counter()
    parts = []
    for year in sorted(args.years):
        path = os.path.join(args.raw_census_dir,
                            args.person_template.format(year=year))
        scan = pl.scan_parquet(path.rstrip("/\\") + "/**/*.parquet") \
            if os.path.isdir(path) else pl.scan_parquet(path)
        parts.append(
            scan.select([pl.col("namefrst").cast(pl.Utf8),
                         pl.col("sex").cast(pl.Int64, strict=False)])
                .with_columns(pl.col("namefrst").str.extract(r"([A-Za-z]+)", 1)
                              .str.to_uppercase().alias("given"))
                .drop_nulls()
                .group_by("given")
                .agg([(pl.col("sex") == 2).sum().alias("n_f"),
                      pl.len().alias("n")]))
    ref = (pl.concat(parts)
           .group_by("given")
           .agg([pl.col("n_f").sum(), pl.col("n").sum()])
           .filter(pl.col("n") >= args.min_count)
           .with_columns((pl.col("n_f") / pl.col("n")).alias("female_share"))
           .sort("given")
           .collect(engine="streaming"))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    ref.write_parquet(args.output)
    print(f"[sexref] wrote {ref.height:,} given names (waves {sorted(args.years)}, "
          f"n>={args.min_count}) -> {args.output}", flush=True)
    print(f"[timing] done: {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
