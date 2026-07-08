#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Relatedness stage 4: cosine similarity between patent and occupation embeddings.

For each requested census year it streams the stage-2 with-strings candidate file in
record batches, looking up the (already L2-normalised) patent and occupation
embeddings and writing, per candidate:
  - relatedness_short = cos(emb patent short,  emb occupation short)
  - relatedness_long  = cos(emb patent long,   emb occupation long)

Embeddings are normalised at encode time, so cosine = dot product. A missing
patent OR occupation embedding yields a NaN score (serialised as null).

Streaming keeps memory bounded to one batch plus the per-year embedding lookups,
instead of materialising a 1024-dim vector on every one of (e.g.) ~238M rows.

Writes one parquet per year: <out-dir>/matched_candidates_rel_{year}.parquet
"""

import argparse
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ALL_YEARS = [1850, 1860, 1870, 1880, 1900, 1910, 1920, 1930, 1940]


def stack_embeddings(values):
    """List/array of per-row float arrays (or None) -> (matrix (N+1, dim) float32,
    key_to_row resolver). Row N is an all-NaN sentinel for missing/unknown keys."""
    dim = None
    for v in values:
        if v is not None and isinstance(v, np.ndarray) and v.size:
            dim = v.shape[0]
            break
    if dim is None:
        return None, 0
    mat = np.full((len(values) + 1, dim), np.nan, dtype=np.float32)
    for i, v in enumerate(values):
        if v is not None and isinstance(v, np.ndarray) and v.size:
            mat[i] = v.astype(np.float32)
    return mat, dim


def build_lookup(keys, *emb_cols):
    """Map each key to its row index; build a NaN-padded matrix per embedding col.
    Returns (key_to_idx dict, [matrices], sentinel_idx)."""
    key_to_idx = {k: i for i, k in enumerate(keys)}
    sentinel = len(keys)
    mats = []
    dim = 0
    for col in emb_cols:
        mat, d = stack_embeddings(list(col))
        mats.append(mat)
        dim = dim or d
    return key_to_idx, mats, sentinel, dim


def cosine_rows(a, b):
    """Row-wise dot of two (B, dim) float32 matrices; NaN where either is NaN."""
    return np.sum(a * b, axis=1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Score patent<->occupation relatedness per year.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Stage-2 directory of matched_candidates_with_strings_{year}.parquet.")
    p.add_argument("--embeddings-dir", required=True, help="Stage-3 embeddings directory.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--years", type=int, nargs="*", default=ALL_YEARS)
    p.add_argument("--batch-size", type=int, default=500_000)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    patents = pq.read_table(os.path.join(args.embeddings_dir, "patent_embeddings.parquet"),
                            columns=["pubnumber", "emb_short", "emb_long"]).to_pandas()
    print(f"Patents loaded: {len(patents):,}")
    pat_keys, (pat_short, pat_long), pat_sent, _ = build_lookup(
        patents["pubnumber"].tolist(), patents["emb_short"], patents["emb_long"])
    del patents

    emb_short_dir = os.path.join(args.embeddings_dir, "occupation_embeddings_short")
    emb_long_dir = os.path.join(args.embeddings_dir, "occupation_embeddings_long")

    start = time.time()
    for year in args.years:
        in_file = os.path.join(args.input, f"matched_candidates_with_strings_{year}.parquet")
        if not os.path.exists(in_file):
            print(f"[{year}] no with-strings input; skipping")
            continue
        print(f"\n=== census_year {year} ===")

        occ_short = pq.read_table(emb_short_dir, filters=[("census_year", "=", year)],
                                  columns=["histid", "emb_occ_desc_short"]).to_pandas()
        occ_long = pq.read_table(emb_long_dir, filters=[("census_year", "=", year)],
                                 columns=["histid", "emb_occ_desc_long"]).to_pandas()
        occ = occ_short.merge(occ_long, on="histid", how="outer")
        occ_keys, (occ_short_m, occ_long_m), occ_sent, _ = build_lookup(
            occ["histid"].tolist(), occ["emb_occ_desc_short"], occ["emb_occ_desc_long"])
        del occ_short, occ_long, occ

        pf = pq.ParquetFile(in_file)
        out_file = os.path.join(args.out_dir, f"matched_candidates_rel_{year}.parquet")
        writer = None
        n = 0
        for batch in pf.iter_batches(batch_size=args.batch_size):
            tbl = pa.Table.from_batches([batch])
            pub = tbl.column("pubnumber").to_pylist()
            hid = tbl.column("histid").to_pylist()
            p_idx = np.fromiter((pat_keys.get(x, pat_sent) for x in pub), dtype=np.int64, count=len(pub))
            o_idx = np.fromiter((occ_keys.get(x, occ_sent) for x in hid), dtype=np.int64, count=len(hid))

            rel_short = cosine_rows(pat_short[p_idx], occ_short_m[o_idx]) if pat_short is not None and occ_short_m is not None else np.full(len(pub), np.nan, np.float32)
            rel_long = cosine_rows(pat_long[p_idx], occ_long_m[o_idx]) if pat_long is not None and occ_long_m is not None else np.full(len(pub), np.nan, np.float32)

            # Some None values were serialised as the literal 'NONE'; null those + their score.
            for col, score in [("occ_desc_short", rel_short), ("occ_desc_long", rel_long)]:
                if col in tbl.column_names:
                    is_none = np.array([v == "NONE" for v in tbl.column(col).to_pylist()])
                    score[is_none] = np.nan

            tbl = tbl.append_column("relatedness_short",
                                    pa.array(rel_short, type=pa.float32(), mask=np.isnan(rel_short)))
            tbl = tbl.append_column("relatedness_long",
                                    pa.array(rel_long, type=pa.float32(), mask=np.isnan(rel_long)))
            if writer is None:
                writer = pq.ParquetWriter(out_file, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            n += tbl.num_rows
        if writer is not None:
            writer.close()
        print(f"[{year}] scored {n:,} rows -> {out_file}")

    print(f"\nDone in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
