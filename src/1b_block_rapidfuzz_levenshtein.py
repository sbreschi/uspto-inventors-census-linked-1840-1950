#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Blocking (candidate generation) by fuzzy string matching using RapidFuzz.

For one census year and one inventor gender, this script compares every
patent-inventor name against census names within a geographic block
(county / state) using the normalised Levenshtein distance, and -- at the
coarser state level -- additionally with the token-sort ratio.

This is stage 1 ("blocking") of the linkage pipeline. Its parquet output is one
of several candidate sets that the downstream refinement step merges before a
classifier scores each pair.

Geographic level governs the candidate budget:
    county  -> Levenshtein only,        limit 500
    state   -> Levenshtein + token_sort, limit 250
These defaults can be overridden with --limit / --token-sort / --no-token-sort.

Requirements: rapidfuzz, pyarrow, pandas (see environment.yml).

"""

import argparse
import gc
import os
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein

from blocking_common import (
    GEO_LEVELS,
    cleanother,
    parallelize_dataframe,
    load_inventors,
    load_census,
    census_full_names,
    iter_geo_values,
    format_elapsed,
)

# Production defaults per geographic level: (Levenshtein limit, token_sort on?)
GEO_DEFAULTS = {
    "county": (500, False),
    "state": (250, True),
}


# ---------------------------------------------------------------------------#
# Worker functions (module-level so they pickle for ProcessPoolExecutor)
# ---------------------------------------------------------------------------#

def _levenshtein_batch(names_batch, names_list, cenid_map, score_cutoff, limit):
    out = []
    for name in names_batch:
        for matched_name, score, _ in process.extract(
                name, names_list, scorer=Levenshtein.normalized_distance,
                score_cutoff=score_cutoff, limit=limit):
            out.append((name, cenid_map[matched_name], score))
    return out


def _tokensort_batch(names_batch, names_list, cenid_map, score_cutoff, limit):
    out = []
    for name in names_batch:
        for matched_name, score, _ in process.extract(
                name, names_list, scorer=fuzz.token_sort_ratio,
                score_cutoff=score_cutoff, limit=limit):
            out.append((name, cenid_map[matched_name], score))
    return out


def _run_parallel(worker, unique_names, names_tofind, score_cutoff, limit,
                  value_col, max_workers, batch_size=20):
    """Score all inventor names against census names in parallel batches.

    Returns a DataFrame with columns [invidx, cenidx, <value_col>].
    """
    names_list = unique_names["clean_name"].tolist()
    cenid_map = unique_names.set_index("clean_name")["cenidx"].to_dict()
    invidx_map = names_tofind.set_index("clean_name")["invidx"].to_dict()

    batches = [names_tofind["clean_name"][i:i + batch_size].tolist()
               for i in range(0, len(names_tofind), batch_size)]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, b, names_list, cenid_map, score_cutoff, limit)
                   for b in batches]
        for future in as_completed(futures):
            for name, cenidx, score in future.result():
                results.append({"invidx": invidx_map[name], "cenidx": cenidx, value_col: score})

    cols = ["invidx", "cenidx", value_col]
    df = pd.DataFrame(results, columns=cols)
    if df.empty:
        # Force numeric dtypes on the empty frame; otherwise the columns are
        # object dtype and an all-NA object column would later trip a pandas
        # FutureWarning at the outer merge / final concat.
        df = df.astype({"invidx": "int64", "cenidx": "int64", value_col: "float64"})
    return df


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="RapidFuzz Levenshtein blocking of patent inventors against census names.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--inventors", required=True,
                   help="Path to the published patent-inventor TSV "
                        "(USPTO_patent_inventors_1840_1950.tsv).")
    p.add_argument("--census", required=True,
                   help="Path to one cleaned census parquet file (single year).")
    p.add_argument("--census-year", type=int, required=True,
                   help="Census year; selects patents in a +/-10 year window and names the output.")
    p.add_argument("--gender", choices=["M", "F", "U"], required=True,
                   help="Inventor gender to match (M, F, or U=unknown).")
    p.add_argument("--geo-level", choices=list(GEO_LEVELS), default="county",
                   help="Geographic blocking key.")
    p.add_argument("--output-dir", required=True,
                   help="Directory for the output parquet (created if absent).")
    p.add_argument("--score-cutoff", type=float, default=0.2,
                   help="Max normalised Levenshtein distance to keep a candidate (0=identical).")
    p.add_argument("--limit", type=int, default=None,
                   help="Max census candidates per inventor name (default: per geo-level).")
    p.add_argument("--token-sort", dest="token_sort", action="store_true", default=None,
                   help="Also compute the token-sort ratio (default: on for state).")
    p.add_argument("--no-token-sort", dest="token_sort", action="store_false",
                   help="Disable the token-sort ratio pass.")
    p.add_argument("--token-sort-cutoff", type=float, default=90,
                   help="Min token-sort ratio (0-100) to keep a candidate.")
    p.add_argument("--max-workers", type=int, default=4,
                   help="Worker processes for parallel matching.")
    p.add_argument("--n-cores", type=int, default=8,
                   help="Process count for parallel name cleaning.")
    p.add_argument("--max-counties", type=int, default=None,
                   help="Process at most this many geographic buckets (for quick smoke tests).")
    return p.parse_args(argv)


def match_one_year(inventors_year, census, args, limit, use_token_sort):
    """Run RapidFuzz blocking for all geographic buckets of one census year."""
    matched = []
    geo_values = iter_geo_values(census, max_buckets=args.max_counties)

    for geo_value in geo_values:
        census_names, unique_names = census_full_names(census, geo_value)
        if len(unique_names) == 0:
            continue

        geo_inv = inventors_year[inventors_year["geo_key"] == geo_value]
        if geo_inv.shape[0] == 0:
            continue

        year_range = range(int(geo_inv["year"].min()), int(geo_inv["year"].max()) + 1)
        for year in year_range:
            clean_names = geo_inv[geo_inv["year"] == year].copy()
            clean_names = clean_names[clean_names["clean_name"].notnull()]
            clean_names = clean_names[["clean_name"]].drop_duplicates()
            if len(clean_names) == 0:
                continue
            clean_names["invidx"] = range(1, len(clean_names) + 1)
            clean_names = pd.merge(clean_names, geo_inv, how="left", on="clean_name")

            names_tofind = clean_names[clean_names["clean_name"].notnull()]
            names_tofind = names_tofind[["invidx", "clean_name"]].drop_duplicates()
            names_tofind = names_tofind.reset_index(drop=True)
            if len(names_tofind) == 0:
                continue

            lev = _run_parallel(_levenshtein_batch, unique_names, names_tofind,
                                args.score_cutoff, limit, "score", args.max_workers)
            lev = lev.drop_duplicates(["invidx", "cenidx"])

            if use_token_sort:
                tks = _run_parallel(_tokensort_batch, unique_names, names_tofind,
                                    args.token_sort_cutoff, limit, "token_sort", args.max_workers)
                tks = tks.drop_duplicates(["invidx", "cenidx"])
                pairs = pd.merge(lev, tks, how="outer", on=["invidx", "cenidx"])
            else:
                pairs = lev

            if pairs.shape[0] == 0:
                del clean_names, names_tofind
                gc.collect()
                continue

            # Attach census and inventor metadata to each candidate pair.
            pairs = pd.merge(pairs, census_names[["cenidx", "histid", "clean_name"]],
                             how="left", on="cenidx")
            pairs = pd.merge(pairs, clean_names[["invidx", "PatInvID", "pubnumber", "clean_name"]],
                             how="left", on="invidx")
            pairs = pairs.drop_duplicates()
            pairs.rename(columns={"clean_name_x": "census_name",
                                  "clean_name_y": "inventor_name"}, inplace=True)
            pairs.drop(["cenidx", "invidx"], axis=1, inplace=True)
            pairs["census_year"] = args.census_year
            pairs["geo_key"] = geo_value
            matched.append(pairs)

            del clean_names, names_tofind, lev, pairs
            gc.collect()

    if not matched:
        cols = ["census_name", "inventor_name", "score", "histid",
                "PatInvID", "pubnumber", "census_year", "geo_key"]
        if use_token_sort:
            cols.insert(3, "token_sort")
        return pd.DataFrame(columns=cols)
    return pd.concat(matched, axis=0).drop_duplicates()


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    default_limit, default_token_sort = GEO_DEFAULTS[args.geo_level]
    limit = args.limit if args.limit is not None else default_limit
    use_token_sort = args.token_sort if args.token_sort is not None else default_token_sort

    start_overall = datetime.now()
    print(f"RapidFuzz blocking | year={args.census_year} gender={args.gender} "
          f"geo={args.geo_level} limit={limit} token_sort={use_token_sort}")
    sys.stdout.flush()

    inventors = load_inventors(args.inventors, args.gender, args.geo_level)
    inventors_year = inventors.loc[
        (inventors["year"] >= args.census_year - 10) &
        (inventors["year"] <= args.census_year + 10)
    ]
    inventors_year = parallelize_dataframe(inventors_year, cleanother, n_cores=args.n_cores)
    print(f"Inventor records (gender {args.gender}): {len(inventors_year)}")

    census = load_census(args.census, args.gender, args.geo_level)
    print(f"Census records (gender {args.gender}): {len(census)}")
    sys.stdout.flush()

    matched = match_one_year(inventors_year, census, args, limit, use_token_sort)

    out_path = os.path.join(
        args.output_dir,
        f"levenshtein_match_{args.geo_level}_{args.census_year}_{args.gender}.parquet")
    matched.to_parquet(out_path, engine="pyarrow", index=False, compression="gzip")
    print(f"Wrote {len(matched)} candidate pairs -> {out_path}")
    print(f"Runtime: {format_elapsed(datetime.now() - start_overall)}")


if __name__ == "__main__":
    main()
