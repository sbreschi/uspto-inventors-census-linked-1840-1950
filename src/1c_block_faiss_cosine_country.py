#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Country-wide (US-wide) blocking by approximate cosine name similarity.

This is the no-geography counterpart to 1a_block_faiss_cosine.py: instead of one
tiny FAISS index per county, it builds a SINGLE approximate index over every
(age 8-80) census name of a given sex nationwide for one wave, and retrieves the
k nearest census names for every patent-inventor regardless of state. It exists
to recover the cross-state movers that geographic blocking cannot.

A flat index is infeasible at national scale (tens of millions of names; a dense
d=256 matrix for 35M names is ~36 GB), so we:
  1. embed names as char-trigram TF-IDF reduced to dense d via TruncatedSVD (one
     shared embedding, fit once on a census sample);
  2. index the census embeddings with a FAISS IVF-PQ index (compressed, ~1 GB for
     35M), streaming the census in batches so the dense matrix is never held whole;
  3. search the inventor embeddings (top-k, nprobe tuned). Cosine == inner product
     of L2-normalised vectors. The IVF-PQ score is approximate (it can overshoot
     1.0), so the retrieved cells get their EXACT cosine recomputed (`exact_cosine`)
     before output -- the index stays a fast recall device; the stored score is the
     true cosine in [-1, 1].

Sex-shared design (one index per sex, census read once per wave):
  For sex in (1=male, 2=female): read that sex's census once, build its index once,
  and query the inventors that target it -- M inventors -> male index, F -> female,
  U (unknown) -> BOTH (its results are accumulated across the two sex passes). This
  reads each wave's census once in total, holds one sex's data + one index at a time
  (bounded RAM), and avoids a separate U-over-both-sexes index.

Per-gender outputs cosine_match_country_{year}_{gender}.parquet use the same schema
as cosine_match/ so 2a_collect_candidates.py can fold them in (geo_key='US').

NOTE on candidate volume: a matched UNIQUE name expands to every histid carrying it.
Nationwide a common name maps to thousands of histids -- both an output explosion and
exactly the precision cost geographic blocking avoided. --max-histids-per-name caps
the expansion (0 = unlimited).

"""

import argparse
import gc
import os
import sys
from datetime import datetime

# Determinism: pin numeric-library thread counts to a fixed value *before* numpy/BLAS
# import so multithreaded reductions (BLAS behind TF-IDF/SVD, FAISS k-means) use a
# deterministic, static work partition -> reproducible run-to-run. A fixed count
# (not necessarily 1) preserves speed; setdefault lets an explicit env win.
DET_THREADS = "8"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, DET_THREADS)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss

from blocking_common import cleanother, parallelize_dataframe, load_inventors, format_elapsed

# Inventor gender -> census sex(es) it should be compared against (mirrors
# blocking_common.GENDER_TO_CENSUS_SEX): M->male index, F->female index, U->both.
GENDER_TO_SEX = {"M": [1], "F": [2], "U": [1, 2]}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Country-wide FAISS IVF-PQ cosine blocking (sex-shared index, GPU-capable).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--inventors", required=True, help="Published patent-inventor TSV.")
    p.add_argument("--census", required=True, help="Cleaned census parquet for one wave.")
    p.add_argument("--census-year", type=int, required=True, help="Census year (+/-10y window).")
    p.add_argument("--genders", nargs="+", default=["M", "F", "U"], choices=["M", "F", "U"],
                   help="Inventor genders to produce candidates for.")
    p.add_argument("--output-dir", required=True, help="Directory for the output parquets.")
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                   help="FAISS device (falls back to cpu if GPU FAISS is unavailable).")
    p.add_argument("--k", type=int, default=50,
                   help="Nearest census names per inventor (>= the per-county 30; the true "
                        "match competes nationwide).")
    p.add_argument("--svd-dim", type=int, default=256, help="TruncatedSVD output dimension.")
    p.add_argument("--pq-m", type=int, default=32, help="IVF-PQ sub-quantizers (bytes/vector).")
    p.add_argument("--nlist", type=int, default=0, help="IVF lists (0 => ~sqrt(N), capped 8192).")
    p.add_argument("--nprobe", type=int, default=32, help="IVF lists probed at search time.")
    p.add_argument("--batch-size", type=int, default=1_000_000, help="Streaming add batch size.")
    p.add_argument("--fit-sample", type=int, default=1_000_000,
                   help="Census names sampled to fit the shared TF-IDF vocabulary.")
    p.add_argument("--svd-sample", type=int, default=300_000,
                   help="Census vectors sampled to fit the SVD and train each index.")
    p.add_argument("--age-min", type=int, default=8)
    p.add_argument("--age-max", type=int, default=80)
    p.add_argument("--max-histids-per-name", type=int, default=0,
                   help="Cap histids expanded per matched census name (0 = unlimited).")
    p.add_argument("--expand-batch-rows", type=int, default=15_000_000,
                   help="Target exploded rows per streamed batch / parquet row group "
                        "(bounds peak RAM of the histid expansion; does not affect output).")
    p.add_argument("--n-cores", type=int, default=8, help="Processes for inventor name cleaning.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------#
# Embedding (shared TF-IDF + SVD) and FAISS helpers
# ---------------------------------------------------------------------------#

def fit_embedder(names, args):
    """Fit the shared char-trigram TF-IDF + TruncatedSVD embedding on a name sample."""
    fit_n = min(args.fit_sample, len(names))
    sample = names.sample(fit_n, random_state=0) if len(names) > fit_n else names
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
    Xs = vec.fit_transform(sample)
    svd = TruncatedSVD(n_components=min(args.svd_dim, Xs.shape[1] - 1), random_state=0)
    svd.fit(Xs[:min(args.svd_sample, Xs.shape[0])])
    return vec, svd


def embed(names_iter, vec, svd):
    """TF-IDF -> SVD -> L2-normalised float32 dense vectors."""
    Y = svd.transform(vec.transform(names_iter)).astype(np.float32)
    faiss.normalize_L2(Y)
    return Y


def make_index(d, nlist, pq_m, device):
    """Build an IVF-PQ index (inner product); move to GPU if requested/available."""
    quant = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quant, d, nlist, pq_m, 8, faiss.METRIC_INNER_PRODUCT)
    gpu = device == "gpu" and hasattr(faiss, "StandardGpuResources")
    if gpu:
        try:
            index = faiss.index_cpu_to_all_gpus(index)
        except Exception as e:  # noqa: BLE001 -- portable fallback
            print(f"  (GPU FAISS unavailable: {e}; using CPU)")
            gpu = False
    return index, gpu


# ---------------------------------------------------------------------------#
# Per-sex census load and candidate assembly
# ---------------------------------------------------------------------------#

def load_census_sex(path, sex, age_min, age_max):
    """Read one sex of the cleaned census (age-filtered), build full clean names,
    and apply the census_full_names gate (non-empty fn/ln, full name length > 3).
    Returns (census_names[histid, clean_name], unique_names[clean_name] in order)."""
    tbl = pq.read_table(
        path, columns=["histid", "clean_fn", "clean_ln", "sex", "age"],
        filters=[("age", ">=", age_min), ("age", "<=", age_max), ("sex", "==", sex)],
    ).to_pandas()
    tbl = tbl[tbl["clean_fn"].notnull() & tbl["clean_ln"].notnull()]
    tbl = tbl[(tbl["clean_fn"].str.len() > 0) & (tbl["clean_ln"].str.len() > 0)]
    tbl["clean_name"] = tbl["clean_fn"] + " " + tbl["clean_ln"]
    tbl = tbl[tbl["clean_name"].str.len() > 3]
    census_names = tbl[["histid", "clean_name"]].reset_index(drop=True)
    unique_names = census_names[["clean_name"]].drop_duplicates().reset_index(drop=True)
    return census_names, unique_names


SEARCH_BATCH = 50_000  # queries per index.search call (bounds GPU temp memory)


def exact_cosine(matches, vec, svd):
    """Overwrite `matches['Cosine_Similarity']` with the *exact* cosine.

    The IVF-PQ search returns an **approximate** inner product (product-quantisation
    reconstruction error), which routinely overshoots 1.0 and gives only an approximate
    within-candidate ranking. Here we re-embed the unique names that were actually
    retrieved (TF-IDF -> SVD -> L2-normalise, identical to the index space) and take the
    true dot product per retrieved cell, so the stored score is the exact cosine in
    [-1, 1] and the candidate ranking is exact. The work is bounded by the number of
    unique retrieved names (far below the full-census embedding done to build the index),
    so it adds only a small fraction to the per-sex runtime."""
    if len(matches) == 0:
        return matches
    inv_names, inv_pos = np.unique(matches["inventor_name"].to_numpy(), return_inverse=True)
    cen_names, cen_pos = np.unique(matches["census_name"].to_numpy(), return_inverse=True)
    inv_emb = embed(pd.Series(inv_names), vec, svd)   # (Ui, d), L2-normalised
    cen_emb = embed(pd.Series(cen_names), vec, svd)   # (Uc, d), L2-normalised
    cos = np.empty(len(matches), dtype=np.float32)
    chunk = 1_000_000  # bound the (chunk x d) gather temporaries
    for s in range(0, len(matches), chunk):
        sl = slice(s, s + chunk)
        cos[sl] = np.einsum("ij,ij->i", inv_emb[inv_pos[sl]], cen_emb[cen_pos[sl]])
    matches = matches.copy()
    matches["Cosine_Similarity"] = cos
    return matches


def search_assemble(index, unique_names, query_df, vec, svd, k, nprobe):
    """Search query names against the index; return (inventor_name, census_name,
    Cosine_Similarity) rows. query_df has a unique 'clean_name' column.

    The search is **batched**: a single index.search() over hundreds of thousands of
    queries overflows FAISS GPU temp memory and segfaults, so queries are searched in
    SEARCH_BATCH chunks. Results are assembled with vectorised numpy indexing (not a
    per-(query,k) Python loop), which is far faster and lower-memory at this scale.

    The IVF-PQ distances are the *approximate* inner product; `exact_cosine` then replaces
    them with the true cosine for the retrieved cells before the frame is returned."""
    index.nprobe = nprobe
    Q = embed(query_df["clean_name"], vec, svd)
    qn = query_df["clean_name"].to_numpy()
    un = unique_names["clean_name"].to_numpy()
    parts = []
    for s in range(0, len(Q), SEARCH_BATCH):
        D, I = index.search(Q[s:s + SEARCH_BATCH], k)
        qidx = np.repeat(np.arange(s, s + I.shape[0]), k)
        flat_i = I.reshape(-1)
        flat_d = D.reshape(-1)
        keep = flat_i >= 0
        parts.append(pd.DataFrame({
            "inventor_name": qn[qidx[keep]],
            "census_name": un[flat_i[keep]],
            "Cosine_Similarity": flat_d[keep],  # approximate; replaced by exact_cosine below
        }))
    if not parts:
        return pd.DataFrame(columns=["inventor_name", "census_name", "Cosine_Similarity"])
    return exact_cosine(pd.concat(parts, ignore_index=True), vec, svd)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    faiss.omp_set_num_threads(int(DET_THREADS))  # deterministic FAISS reductions (see top of file)
    t0 = datetime.now()
    print(f"Country-wide cosine (sex-shared) | year={args.census_year} genders={args.genders} "
          f"device={args.device} k={args.k} svd={args.svd_dim} pq_m={args.pq_m}")
    sys.stdout.flush()

    # Inventors of all requested genders, in the +/-10y window, name-cleaned.
    inv_parts = []
    for g in args.genders:
        gi = load_inventors(args.inventors, g, "county")  # geo_level unused here
        gi = gi.loc[(gi["year"] >= args.census_year - 10) & (gi["year"] <= args.census_year + 10)]
        inv_parts.append(gi)
    inv = pd.concat(inv_parts, axis=0, ignore_index=True)
    inv = parallelize_dataframe(inv, cleanother, n_cores=args.n_cores)
    inv = inv[inv["clean_name"].notnull() & (inv["clean_name"].str.len() > 0)]
    inv_q = inv[["clean_name", "PatInvID", "pubnumber", "gender"]].drop_duplicates()
    print("Inventor (PatInvID,pubnumber) pairs by gender: "
          + ", ".join(f"{g}={inv_q[inv_q.gender==g][['PatInvID','pubnumber']].drop_duplicates().shape[0]:,}"
                      for g in args.genders))
    sys.stdout.flush()

    # Shared embedder, fit once on the first available sex's names.
    embedder = None

    # One streaming parquet writer per requested gender, kept open across both sex
    # passes. U batches from sex 1 and sex 2 are appended straight to the U writer
    # (their histids are disjoint across passes, so per-batch dedup is sufficient).
    schema = build_output_schema(inv_q, args.census)
    writers = {g: StreamWriter(os.path.join(
        args.output_dir, f"cosine_match_country_{args.census_year}_{g}.parquet"), schema)
        for g in args.genders}

    for sex in (1, 2):
        genders_here = [g for g in args.genders if sex in GENDER_TO_SEX[g]]
        if not genders_here:
            continue
        census_names, unique_names = load_census_sex(args.census, sex, args.age_min, args.age_max)
        N = len(unique_names)
        print(f"[sex {sex}] census names: {len(census_names):,} | unique (N): {N:,} "
              f"({format_elapsed(datetime.now() - t0)})")
        sys.stdout.flush()
        if N == 0:
            continue

        if embedder is None:
            embedder = fit_embedder(unique_names["clean_name"], args)
            print(f"  embedder fitted: TF-IDF dim={len(embedder[0].vocabulary_)} "
                  f"-> SVD d={embedder[1].n_components}")
        vec, svd = embedder
        d = svd.n_components

        nlist = args.nlist or min(8192, max(1024, int(np.sqrt(N))))
        index, on_gpu = make_index(d, nlist, args.pq_m, args.device)
        train_n = min(N, max(args.svd_sample, 40 * nlist))
        train_names = (unique_names["clean_name"].sample(train_n, random_state=1)
                       if N > train_n else unique_names["clean_name"])
        index.train(embed(train_names, vec, svd))
        for s in range(0, N, args.batch_size):
            index.add(embed(unique_names["clean_name"].iloc[s:s + args.batch_size], vec, svd))
        print(f"  index built (nlist={nlist}, gpu={on_gpu}) "
              f"({format_elapsed(datetime.now() - t0)})")
        sys.stdout.flush()

        # GPU build is fast, but GPU IVF-PQ *search* is unstable on large indices
        # (segfaults at ~20M+ vectors). Move the populated index back to CPU for the
        # search: CPU search is numerically equivalent and cheap (~57k queries/s), a
        # few seconds even for the largest query sets.
        if on_gpu:
            index = faiss.index_gpu_to_cpu(index)
            print(f"  moved index to CPU for search ({format_elapsed(datetime.now() - t0)})")
            sys.stdout.flush()

        # Cap histid expansion if requested.
        cn = census_names
        if args.max_histids_per_name > 0:
            cn = cn.groupby("clean_name", group_keys=False).head(args.max_histids_per_name)

        for g in genders_here:
            gq = inv_q[inv_q["gender"] == g][["clean_name"]].drop_duplicates().reset_index(drop=True)
            if len(gq) == 0:
                continue
            matches = search_assemble(index, unique_names, gq, vec, svd, args.k, args.nprobe)
            gpairs = inv_q[inv_q["gender"] == g][["clean_name", "PatInvID", "pubnumber"]]
            n = stream_expand_write(matches, cn, gpairs, writers[g], args)
            print(f"  [{g}] {n:,} rows from sex {sex} "
                  f"({format_elapsed(datetime.now() - t0)})")
            sys.stdout.flush()

        del index, census_names, unique_names, cn
        gc.collect()

    # Close every writer (creates an empty, schema-correct file for genders that
    # produced no rows, so the stage-2a direct-path read never sees a missing file).
    for g in args.genders:
        out = writers[g].close()
        print(f"  wrote {writers[g].rows:,} rows ({writers[g].triples:,} triples) -> {out}")

    print(f"Total runtime: {format_elapsed(datetime.now() - t0)}")


# Fixed 8-column output schema (order matters: stage 2a / collect_common select by name).
OUT_COLS = ["inventor_name", "census_name", "Cosine_Similarity",
            "PatInvID", "pubnumber", "histid", "census_year", "geo_key"]


def build_output_schema(inv_q, census_path):
    """Canonical pyarrow schema for the 8 output columns, used for every row group and
    for empty-gender files. Data-dependent field types are pulled from the actual inputs
    (PatInvID/pubnumber from the inventor frame, histid from the census parquet) so the
    per-batch from_pandas casts are value-preserving identities."""
    inv_types = pa.Table.from_pandas(
        inv_q[["PatInvID", "pubnumber"]].iloc[:0], preserve_index=False).schema
    histid_type = pq.read_schema(census_path).field("histid").type
    return pa.schema([
        ("inventor_name", pa.string()),
        ("census_name", pa.string()),
        ("Cosine_Similarity", pa.float32()),
        ("PatInvID", inv_types.field("PatInvID").type),
        ("pubnumber", inv_types.field("pubnumber").type),
        ("histid", histid_type),
        ("census_year", pa.int64()),
        ("geo_key", pa.string()),
    ])


class StreamWriter:
    """Append exploded match batches to one gzip parquet as row groups, never holding the
    whole expansion in RAM. The writer opens lazily on the first non-empty batch; if it is
    never opened, close() still writes a single empty file with the canonical schema so the
    output path always exists. rows/triples are running totals for the console log (exact:
    batches are triple-disjoint, so per-batch counts sum without double-counting)."""

    def __init__(self, path, schema):
        self.path = path
        self.schema = schema
        self._w = None
        self.rows = 0
        self.triples = 0

    def append(self, df):
        if df.empty:
            return
        tbl = pa.Table.from_pandas(df[OUT_COLS], schema=self.schema, preserve_index=False)
        if self._w is None:
            self._w = pq.ParquetWriter(self.path, self.schema, compression="gzip")
        self._w.write_table(tbl)
        self.rows += len(df)
        self.triples += df[["PatInvID", "pubnumber", "histid"]].drop_duplicates().shape[0]

    def close(self):
        if self._w is None:
            # No data for this gender: emit an empty, schema-correct file.
            pq.write_table(self.schema.empty_table(), self.path, compression="gzip")
        else:
            self._w.close()
        return self.path


def stream_expand_write(matches, cn, gpairs, writer, args):
    """Explode `matches` (unique inventor_name/census_name cells) into output rows in
    memory-bounded batches and append each finished batch to `writer`.

    Correctness: each output triple (PatInvID, pubnumber, histid) maps back to exactly one
    (inventor_name, census_name) cell, so distinct cells produce disjoint output rows. We
    therefore partition the cells into contiguous batches sized by *projected expanded rows*
    (histids-per-census_name x pairs-per-inventor_name), expand/filter/dedup each batch on
    its own, and append it. The union over batches equals the single-frame merge exactly.
    Returns the number of output rows written for this call (for the console log)."""
    if len(matches) == 0:
        return 0
    histids_per_name = cn.groupby("clean_name").size()
    pairs_per_invname = gpairs.groupby("clean_name").size()
    ecount = (matches["census_name"].map(histids_per_name).fillna(0).astype("int64")
              * matches["inventor_name"].map(pairs_per_invname).fillna(0).astype("int64"))
    target = max(1, args.expand_batch_rows)
    # Contiguous, non-decreasing batch ids each accumulating ~target expanded rows. A single
    # cell larger than target forms its own batch (inherent minimum granularity).
    batch_id = (ecount.cumsum() // target).to_numpy()

    cn_r = cn.rename(columns={"clean_name": "census_name"})
    gp_r = gpairs.rename(columns={"clean_name": "inventor_name"})
    written = 0
    for bid in pd.unique(batch_id):
        sub = matches.iloc[batch_id == bid]
        sub = sub.merge(cn_r, on="census_name", how="left")
        sub = sub.merge(gp_r, on="inventor_name", how="left")
        sub = sub[sub["histid"].notnull()]
        if sub.empty:
            continue
        sub = sub.copy()
        sub["census_year"] = args.census_year
        sub["geo_key"] = "US"
        sub = sub[OUT_COLS].drop_duplicates()
        writer.append(sub)
        written += len(sub)
    return written


if __name__ == "__main__":
    main()
