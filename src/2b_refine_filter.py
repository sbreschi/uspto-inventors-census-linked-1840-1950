#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2b: refine and filter the collected candidates (polars).

Loads the candidate table from 2a, normalises names, computes detailed
name-similarity metrics, applies the 13-filter cascade (see refine_common.py),
enriches the survivors with patent geography, census geography/demographics and
the inter-county distance, and writes the `matched_candidates` feature table
used by the downstream classifier.

The output is the trimmed model-feature table: only the columns the downstream ML
model consumes (keys, names, the three name metrics, geography + county distance, the
age/occupation/gender inputs) plus the golden labels. The inter-county distance
is recovered two ways: a priority cascade over every county-distance crosswalk in
`--county-distances-dir`, and a household-file fallback (`--household-census`,
default the raw-census path with `US{year}P`->`US{year}H`) that reconstructs the
census county from `statefip`+`countyicp` where stage-0's place-geomatch left it
null -- the sole cause of a missing distance (see refine_common.py).

Passing `--golden` adds two label columns: `true_histid` (the correct histid for
golden patent-inventors, else null) and `true_match` (1 true match / 0 competing
candidate of a golden PatInvID / null unlabeled). The golden-sample diagnostics
live in the separate stage-2c script.

All inputs are CLI arguments; nothing is hardcoded. Census inputs are restricted
IPUMS extracts (see README).

"""

import argparse
import os
from datetime import datetime

import polars as pl

from refine_common import load_patents, refine_and_write

# Trimmed to exactly what the downstream ML model consumes, plus the
# explicit keeps (inventor/census state & county, the golden true_histid and the
# true_match label). `occstr` exists for 1900-1930 and `occ` for 1850-1880 & 1940;
# both are listed and the missing one is silently dropped by refine_and_write's
# schema filter.
OUTPUT_COLUMNS = [
    # keys
    "PatInvID", "pubnumber", "histid", "census_year",
    # names (model: diff_length; also downstream name checks)
    "inventor_name", "census_name",
    # name metrics (model features)
    "leven_norm", "tkratio", "tksortratio",
    # geography (user keep + distance feature)
    "county_inv", "state_inv", "county_census", "state_census", "county_distance",
    # household-file county (countyicp->FIPS) for all rows; also fills county_census nulls
    "county_census_hh", "state_census_hh",
    # age inputs -> age_at_pat = pubyear - birthyr
    "pubyear", "birthyr",
    # occupation -> occXtech binaries
    "ind1950", "occ1950", "occstr", "occ",
    # gender (safety-margin keep)
    "gender_inv", "gender_census",
    # labels
    "true_histid", "true_match",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compute metrics, filter, and enrich candidate matches (polars).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--candidates", required=True, help="Candidates parquet from stage 2a.")
    p.add_argument("--inventors", required=True, help="Published patent-inventor TSV.")
    p.add_argument("--cleaned-census", required=True, help="Cleaned census parquet (US{year}P.parquet).")
    p.add_argument("--raw-census", required=True, help="Raw IPUMS census parquet (occupation vars).")
    p.add_argument("--household-census", default=None,
                   help="IPUMS household parquet (US{year}H) for county_distance recovery via "
                        "countyicp. Default: derive from --raw-census by swapping P->H.")
    p.add_argument("--county-distances-dir", required=True,
                   help="Directory of county-distance crosswalk .csv.zip files (cascade recovery).")
    p.add_argument("--census-year", type=int, required=True, help="Census year.")
    p.add_argument("--output", required=True, help="Output matched_candidates parquet path.")
    p.add_argument("--golden", default=None,
                   help="Optional golden CSV; adds true_histid value + true_match label.")
    p.add_argument("--missing-distance-out", default=None,
                   help="Optional path to write rows with no county distance (diagnostic).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    year = args.census_year
    start = datetime.now()

    patents = load_patents(args.inventors)

    # Household file for county_distance recovery: default to the raw-census path with the
    # person-file 'P' swapped for the household-file 'H' (US{year}P -> US{year}H).
    household_census = args.household_census or args.raw_census.replace("P.parquet", "H.parquet")

    # Streaming hot path + streaming enrichment: normalise names, compute metrics, drop
    # pairs failing the pair-level cascade (0-8, 10-12) + the record-level age filter (9),
    # then enrich (patent/census geography, inter-county distance with household-file
    # countyicp fallback, occupation, golden) and sink straight to parquet. Kept streaming
    # end-to-end so peak RAM stays bounded on the ~238M-row waves.
    refine_and_write(
        candidates_path=args.candidates, patents=patents,
        cleaned_census_path=args.cleaned_census, raw_census_path=args.raw_census,
        county_dist_dir=args.county_distances_dir, year=year, output=args.output,
        output_columns=OUTPUT_COLUMNS, golden_path=args.golden,
        missing_distance_out=args.missing_distance_out, household_path=household_census)
    print(f"Runtime: {datetime.now() - start}")


if __name__ == "__main__":
    main()
