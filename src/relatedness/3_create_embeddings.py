#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Relatedness stage 3: embed patent and occupation strings with JobBERT-v2.

Patents are embedded ONCE from the global stage-1 patent_strings
file (already unique per pubnumber); occupations are embedded one census year at
a time, reading the per-year stage-2 with-strings files via polars so only the
deduped (histid, census_year) string set for a single year is held in memory.

Writes:
  <out-dir>/patent_embeddings.parquet                 (pubnumber, emb_short, emb_long)
  <out-dir>/occupation_embeddings_short/  (partitioned by census_year, emb_occ_desc_short)
  <out-dir>/occupation_embeddings_long/   (partitioned by census_year, emb_occ_desc_long)

--device {cpu,auto,cuda[:N]}: defaults to 'cpu'. The GPU is opt-in. 
CPU runs the embedding workload fine (it is tiny next to the blocking stages). 
Pass --device cuda to opt in. 'auto' uses the GPU only.

(needs torch + sentence-transformers).
"""

import argparse
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

ALL_YEARS = [1850, 1860, 1870, 1880, 1900, 1910, 1920, 1930, 1940]


def resolve_device(requested):
    """Pick a usable device. 'auto' uses CUDA only if torch has kernels for the
    GPU's compute capability; otherwise CPU. Never returns a CUDA device the
    installed torch cannot actually run on."""
    if requested == "cpu":
        return "cpu"
    if requested in ("auto", "cuda") or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            if requested != "auto":
                print(f"  requested {requested} but CUDA is unavailable; using CPU")
            return "cpu"
        idx = int(requested.split(":", 1)[1]) if requested.startswith("cuda:") else 0
        cap = torch.cuda.get_device_capability(idx)
        dev_cc = cap[0] * 10 + cap[1]
        sm = f"sm_{dev_cc}"
        arch_list = torch.cuda.get_arch_list()

        def _cc(arch):  # "sm_90a" -> 90, "compute_80" -> 80, else None
            digits = "".join(c for c in arch if c.isdigit())
            return int(digits) if digits else None

        # A cubin built for sm_AB runs on a device of the same major with >= minor.
        supported = any((cc := _cc(a)) is not None and cc // 10 == cap[0] and cc <= dev_cc
                        for a in arch_list)
        if not supported:
            print(f"  GPU {torch.cuda.get_device_name(idx)} ({sm}) not supported by this "
                  f"torch (archs: {arch_list}); using CPU")
            return "cpu"
        return f"cuda:{idx}"
    return requested


def encode_series(series, model, batch_size=256):
    """Encode a Series of texts, leaving NaN as pd.NA. Returns list of float16 arrays / pd.NA."""
    mask = series.notna()
    out = [pd.NA] * len(series)
    if mask.any():
        embs = model.encode(series[mask].tolist(), convert_to_tensor=True,
                            batch_size=batch_size, show_progress_bar=True,
                            normalize_embeddings=True)
        embs = [e.cpu().numpy().astype(np.float16) for e in embs]
        j = 0
        for i, keep in enumerate(mask):
            if keep:
                out[i] = embs[j]
                j += 1
    return out


def encode_unique(values, model, batch_size=256):
    """Embed each DISTINCT non-null string ONCE (occupation/title text repeats
    heavily across people and patents). Returns {string: float16 array}; the caller
    broadcasts it back to every row. Empty/all-null input -> {}."""
    distinct = pd.unique(pd.Series(values).dropna())
    if len(distinct) == 0:
        return {}
    embs = encode_series(pd.Series(distinct), model=model, batch_size=batch_size)
    return dict(zip(distinct, embs))


def _broadcast(values, emb_map):
    """Map each value to its embedding via emb_map; pd.NA for null / unknown."""
    return [emb_map.get(v, pd.NA) if pd.notna(v) else pd.NA for v in values]


def encode_and_save(df, col, out_path, model, batch_size=256, chunk_size=500_000):
    """Embed the DISTINCT strings in `col` once, then broadcast back per row and
    append to a parquet dataset partitioned by census_year (chunked to bound
    memory). One row per input row is preserved, so the stage-4 histid lookup is
    unchanged."""
    os.makedirs(out_path, exist_ok=True)
    emb_map = encode_unique(df[col], model=model, batch_size=batch_size)
    print(f"Encoding {col}: rows {len(df):,} from {len(emb_map):,} distinct strings")
    for i in range(0, len(df), chunk_size):
        sub = df.iloc[i:i + chunk_size].copy()
        sub[f"emb_{col}"] = _broadcast(sub[col], emb_map)
        table = pa.Table.from_pandas(sub, preserve_index=False)
        pq.write_to_dataset(table, root_path=out_path, partition_cols=["census_year"])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Embed patent + occupation strings (single GPU, per year).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--patent-strings", required=True, help="Stage-1 patent_strings parquet.")
    p.add_argument("--with-strings-dir", required=True,
                   help="Stage-2 directory of matched_candidates_with_strings_{year}.parquet.")
    p.add_argument("--out-dir", required=True, help="Output embeddings directory.")
    p.add_argument("--years", type=int, nargs="*", default=ALL_YEARS)
    p.add_argument("--model", default="TechWolf/JobBERT-v2")
    p.add_argument("--device", default="cpu",
                   help="cpu | auto | cuda | cuda:N (default cpu; GPU opt-in, "
                        "reboots the P40 host — see gpu_crash_analysis.txt)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=500_000)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    model = SentenceTransformer(args.model, device=device)

    start = time.time()

    # --- patents: embed once from the global stage-1 file ---------------------
    patents = pl.read_parquet(args.patent_strings).select(
        ["pubnumber", "pat_desc_short", "pat_desc_long"]).unique(subset=["pubnumber"]).to_pandas()
    print(f"Unique patents: {len(patents):,}")
    short_map = encode_unique(patents["pat_desc_short"], model=model, batch_size=args.batch_size)
    long_map = encode_unique(patents["pat_desc_long"], model=model, batch_size=args.batch_size)
    print(f"Patent strings: {len(short_map):,} distinct short / {len(long_map):,} distinct long")
    patents["emb_short"] = _broadcast(patents["pat_desc_short"], short_map)
    patents["emb_long"] = _broadcast(patents["pat_desc_long"], long_map)
    patents.to_parquet(os.path.join(args.out_dir, "patent_embeddings.parquet"),
                       index=False, compression="zstd")

    # --- occupations: per year, deduped (histid, census_year) -----------------
    short_dir = os.path.join(args.out_dir, "occupation_embeddings_short")
    long_dir = os.path.join(args.out_dir, "occupation_embeddings_long")
    for year in args.years:
        in_file = os.path.join(args.with_strings_dir,
                               f"matched_candidates_with_strings_{year}.parquet")
        if not os.path.exists(in_file):
            print(f"[{year}] no with-strings input; skipping")
            continue
        occ = (pl.scan_parquet(in_file)
                 .select(["histid", "census_year", "occ_desc_short", "occ_desc_long"])
                 .unique(subset=["histid", "census_year"])
                 .collect(engine="streaming").to_pandas())
        print(f"[{year}] unique occupations: {len(occ):,}")
        encode_and_save(occ, "occ_desc_short", short_dir,
                        model=model, batch_size=args.batch_size, chunk_size=args.chunk_size)
        encode_and_save(occ, "occ_desc_long", long_dir,
                        model=model, batch_size=args.batch_size, chunk_size=args.chunk_size)

    print(f"All embeddings written in {time.time() - start:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
