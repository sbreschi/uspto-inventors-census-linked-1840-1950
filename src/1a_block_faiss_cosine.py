#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Blocking (candidate generation) by approximate name similarity using FAISS.

For one census year and one inventor gender, this script finds, for every
patent-inventor name, its `k` nearest census names within a geographic block
(county / state). Names are vectorised with character 3-gram TF-IDF,
L2-normalised, indexed with a FAISS flat L2 index, and the L2 distances are
converted to cosine similarities.

This is stage 1 ("blocking") of the linkage pipeline. Its parquet output is one
of several candidate sets that the downstream refinement step merges before a
classifier scores each pair.

Requirements: faiss (faiss-gpu for --device gpu), scikit-learn, pyarrow, pandas
(see environment.yml).

"""

import argparse
import gc
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

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


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="FAISS cosine-similarity blocking of patent inventors against census names.",
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
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                   help="Run the FAISS index on GPU (production) or CPU (portable; identical results).")
    p.add_argument("--k", type=int, default=30,
                   help="Number of nearest census neighbours to retrieve per inventor name.")
    p.add_argument("--batch-size", type=int, default=250,
                   help="Number of census vectors added to the index per batch.")
    p.add_argument("--n-cores", type=int, default=8,
                   help="Process count for parallel name cleaning.")
    p.add_argument("--max-counties", type=int, default=None,
                   help="Process at most this many geographic buckets (for quick smoke tests).")
    return p.parse_args(argv)


def build_index(dimension, device, n_gpu=4):
    """Create a flat L2 index on CPU or sharded across GPUs.

    Returns (index, gpu_resources). `gpu_resources` is a list to keep alive on
    GPU (empty on CPU).
    """
    cpu_index = faiss.IndexFlatL2(dimension)
    if device == "cpu":
        return cpu_index, []

    if not hasattr(faiss, "StandardGpuResources"):
        raise ImportError(
            "FAISS GPU support is not available (faiss.StandardGpuResources missing). "
            "Install faiss-gpu, or re-run with --device cpu."
        )
    resources = [faiss.StandardGpuResources() for _ in range(n_gpu)]
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co)
    return gpu_index, resources


def match_one_year(inventors_year, census, args):
    """Run FAISS blocking for all geographic buckets of one census year."""
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
            clean_names["invidx"] = np.arange(1, len(clean_names) + 1)
            clean_names = pd.merge(clean_names, geo_inv, how="left", on="clean_name")

            names_tofind = clean_names[clean_names["clean_name"].notnull()]
            names_tofind = names_tofind[["invidx", "clean_name"]].drop_duplicates()
            names_tofind = names_tofind.reset_index(drop=True)
            if len(names_tofind) == 0:
                continue

            # Vectorise names with character 3-gram TF-IDF.
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
            vectors_names_to_find = vectorizer.fit_transform(names_tofind["clean_name"])
            vectors_unique_names = vectorizer.transform(unique_names["clean_name"])

            dimension = vectors_names_to_find.shape[1]
            index, resources = build_index(dimension, args.device)

            # Add the census (database) vectors in batches.
            n_db = vectors_unique_names.shape[0]
            n_batches = n_db // args.batch_size + (n_db % args.batch_size > 0)
            for i in range(n_batches):
                start = i * args.batch_size
                end = min((i + 1) * args.batch_size, n_db)
                batch = vectors_unique_names[start:end].toarray().astype(np.float32)
                faiss.normalize_L2(batch)
                index.add(batch)

            # Search for the k nearest census names per inventor name.
            query = vectors_names_to_find.toarray().astype(np.float32)
            faiss.normalize_L2(query)
            D, I = index.search(query, args.k)

            # Convert (squared) L2 distances to cosine similarities. See module docstring.
            cosine_similarities = 1 - 0.5 * D

            rows = []
            for i in range(query.shape[0]):
                for j in range(args.k):
                    similar_idx = I[i][j]
                    if similar_idx < 0:
                        continue
                    rows.append({
                        "Original_Name": names_tofind["clean_name"][i],
                        "Matched_Name": unique_names["clean_name"].iloc[similar_idx],
                        "Cosine_Similarity": cosine_similarities[i][j],
                    })

            matches = pd.DataFrame(rows)
            if len(matches) > 0:
                matches = pd.merge(
                    matches, clean_names[["PatInvID", "pubnumber", "clean_name"]],
                    how="left", left_on="Original_Name", right_on="clean_name")
                matches = pd.merge(
                    matches, census_names[["histid", "clean_name"]],
                    how="left", left_on="Matched_Name", right_on="clean_name")
                matches = matches.drop(["clean_name_x", "clean_name_y"], axis=1)
                matches.rename(columns={"Matched_Name": "census_name",
                                        "Original_Name": "inventor_name"}, inplace=True)
                matches["census_year"] = args.census_year
                matches["geo_key"] = geo_value
                matched.append(matches)

            # Release GPU/CPU index resources for this (geo, year) block.
            for resource in resources:
                resource.noTempMemory()
            del index, resources
            del vectorizer, vectors_names_to_find, vectors_unique_names, query, D, I
            del matches, cosine_similarities, clean_names, names_tofind
            gc.collect()

    if not matched:
        return pd.DataFrame(columns=["inventor_name", "census_name", "Cosine_Similarity",
                                     "PatInvID", "pubnumber", "histid", "census_year", "geo_key"])
    return pd.concat(matched, axis=0)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    start_overall = datetime.now()
    print(f"FAISS blocking | year={args.census_year} gender={args.gender} "
          f"geo={args.geo_level} device={args.device}")
    sys.stdout.flush()

    # Patent inventors, filtered to the +/-10 year window and cleaned.
    inventors = load_inventors(args.inventors, args.gender, args.geo_level)
    inventors_year = inventors.loc[
        (inventors["year"] >= args.census_year - 10) &
        (inventors["year"] <= args.census_year + 10)
    ]
    inventors_year = parallelize_dataframe(inventors_year, cleanother, n_cores=args.n_cores)
    print(f"Inventor records (gender {args.gender}): {len(inventors_year)}")

    # Census individuals for this year/gender.
    census = load_census(args.census, args.gender, args.geo_level)
    print(f"Census records (gender {args.gender}): {len(census)}")
    sys.stdout.flush()

    matched = match_one_year(inventors_year, census, args)

    out_path = os.path.join(
        args.output_dir,
        f"cosine_match_{args.geo_level}_{args.census_year}_{args.gender}.parquet")
    matched.to_parquet(out_path, engine="pyarrow", index=False, compression="gzip")
    print(f"Wrote {len(matched)} candidate pairs -> {out_path}")
    print(f"Runtime: {format_elapsed(datetime.now() - start_overall)}")


if __name__ == "__main__":
    main()
