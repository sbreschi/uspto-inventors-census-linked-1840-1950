#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML model stage 3: production inference — score every candidate and emit the
accepted links the published dataset ships.

The training stage (2_train_classifier.py) only fits/evaluates on the golden
labeled split. This stage applies the trained classifier to the FULL stage-3
features table (matched_candidates_features.parquet, all inventors; non-golden
rows simply have true_match = NULL) and selects, per (PatInvID, census_year)
group, the single highest-probability candidate — the same group-ranking rule the
training stage evaluates. Accepted links are those whose probability clears
--accept-threshold (the production release passes 0: every group's top-1 is
published with its probability, no operating point is imposed).

Output schema (the manuscript's published-links variable table, in order):

    histid, PatInvID, pubnumber, pubyear, census_year, inventor_name,
    leven_simil, county_inv, state_inv, county_distance, age_at_pat,
    relatedness_short, relatedness_long, probability

`leven_simil` (1 - leven_norm) and `age_at_pat` (pubyear - birthyr) come from the
same build_features() the training stage uses. `county_inv`/`state_inv` are not in
the features table (the stage-3 merge drops them), so they are joined back from
the stage-2b per-year matched_candidates parquets (--matched-candidates) onto the
accepted links. The raw census person name is never emitted.

Memory-bounded: the features table is tens of GB, so scoring streams through it in
pyarrow record batches (the pandas build_features() from stage 2 is reused per
batch, so feature semantics/order are identical), writing per-row scores to a temp
parquet; the per-group top-1 selection is then done out-of-core with polars
streaming, as is the geo join.

All inputs/outputs are CLI arguments; nothing is hardcoded.
"""

import argparse
import importlib.util
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_OCC_DICT = os.path.join(HERE, "..", "..", "data", "misc", "occupation_dict.json")

# Identifier + display columns carried straight through (census_name is read only
# because build_features derives diff_length from the two names; it is not emitted).
ID_COLS = ["PatInvID", "pubnumber", "histid", "census_year",
           "inventor_name", "census_name"]
# Raw columns build_features() consumes (occupation dummies + derived features).
FEAT_SRC = ["leven_norm", "tkratio", "tksortratio", "county_distance",
            "pubyear", "birthyr", "ind1950", "occ1950", "occstr",
            "relatedness_short", "relatedness_long", "surname_logfreq", "n_candidates"]

# Published output columns, in manuscript-table order.
OUT_COLS = ["histid", "PatInvID", "pubnumber", "pubyear", "census_year",
            "inventor_name", "leven_simil", "county_inv", "state_inv",
            "county_distance", "age_at_pat", "relatedness_short",
            "relatedness_long", "probability"]

# Join keys for the county_inv/state_inv attach from the 2b matched_candidates.
GEO_KEYS = ["PatInvID", "pubnumber", "histid", "census_year"]


def _lap(label, t0):
    now = time.perf_counter()
    print(f"[timing] {label}: {now - t0:.1f}s", flush=True)
    return now


def _load_train_module():
    """Import the digit-prefixed 2_train_classifier.py so build_features /
    BASE_FEATURES are the single source of truth for feature construction."""
    path = os.path.join(HERE, "2_train_classifier.py")
    spec = importlib.util.spec_from_file_location("train_classifier", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Score all candidates and emit accepted links (publication schema).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--features", required=True,
                   help="Stage-3 matched_candidates_features.parquet (full table).")
    p.add_argument("--model", required=True, help="Trained classifier pickle (stage 2).")
    p.add_argument("--matched-candidates", required=True,
                   help="Directory of stage-2b matched_candidates_{year}.parquet files "
                        "(source of county_inv/state_inv, which the stage-3 merge drops).")
    p.add_argument("--output", required=True, help="Accepted-links parquet to write.")
    p.add_argument("--occ-dict", default=DEF_OCC_DICT, help="Occupation dictionary JSON.")
    p.add_argument("--occdict-name", default="broad_new_v2",
                   choices=["broad_new_v2", "strict_new_v2"],
                   help="Dictionary variant (must match training).")
    p.add_argument("--accept-threshold", type=float, default=0.5,
                   help="Minimum group-best probability for an accepted link "
                        "(the no-threshold production release passes 0).")
    p.add_argument("--batch-size", type=int, default=2_000_000,
                   help="Rows per scoring batch (memory knob).")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallelism for predict_proba (forest); -1 = all cores. "
                        "Only affects wall-clock, not probabilities (to --round-decimals).")
    p.add_argument("--round-decimals", type=int, default=10,
                   help="Round probability to this many decimals so parallel "
                        "(n_jobs>1) scoring is byte-stable: threaded forest reduction "
                        "wobbles at ~1e-16, far below this. Set <0 to disable rounding.")
    p.add_argument("--scored-tmp", default=None,
                   help="Temp per-row scores parquet (default: <output>.scored.tmp.parquet).")
    p.add_argument("--keep-scored-tmp", action="store_true",
                   help="Keep the per-row scores temp file (default: delete it).")
    p.add_argument("--spill-dir", default=None,
                   help="Directory for polars streaming spill files in Pass 2 "
                        "(sets POLARS_TEMP_DIR/TMPDIR). Default: <output dir>/.polars_tmp. "
                        "MUST be real disk with room for the scored temp: the OS default "
                        "/tmp is often tmpfs (RAM-backed), which OOM-kills the large group-by.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.perf_counter()
    mod = _load_train_module()

    with open(args.occ_dict) as f:
        occ_dict = json.load(f)
    if args.occdict_name not in occ_dict:
        raise SystemExit(f"[3] '{args.occdict_name}' not in {args.occ_dict}")
    dictionary = occ_dict[args.occdict_name]

    with open(args.model, "rb") as f:
        model = pickle.load(f)
    # Parallelize scoring across cores. RandomForest.predict_proba splits over trees;
    # this changes only wall-clock, not the predicted probabilities. Harmless for xgb.
    try:
        model.n_jobs = args.n_jobs
    except AttributeError:
        pass

    # Active feature set/order: identical to training (base 24 + n_candidates).
    ml_features = list(mod.BASE_FEATURES) + ["n_candidates"]

    scored_tmp = args.scored_tmp or (args.output + ".scored.tmp.parquet")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ---- Pass 1: stream-score every candidate row -> per-row scores parquet ----
    pf = pq.ParquetFile(args.features)
    read_cols = list(dict.fromkeys(ID_COLS + FEAT_SRC))
    out_schema = pa.schema([
        ("histid", pa.string()), ("PatInvID", pa.int64()),
        ("pubnumber", pa.int64()), ("pubyear", pa.int64()),
        ("census_year", pa.int64()), ("inventor_name", pa.string()),
        ("leven_simil", pa.float64()), ("county_distance", pa.float64()),
        ("age_at_pat", pa.int64()), ("relatedness_short", pa.float64()),
        ("relatedness_long", pa.float64()), ("probability", pa.float64()),
    ])
    writer = pq.ParquetWriter(scored_tmp, out_schema)
    n_rows = 0
    try:
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=read_cols):
            df = batch.to_pandas()
            df = mod.build_features(df, dictionary)
            proba = model.predict_proba(df[ml_features])[:, 1]
            if args.round_decimals >= 0:
                # Absorb the ~1e-16 thread-order jitter of the parallel forest so the
                # published scores are byte-stable regardless of n_jobs / run.
                proba = np.round(proba, args.round_decimals)
            out = pd.DataFrame({
                "histid": df["histid"].astype("string"),
                "PatInvID": pd.to_numeric(df["PatInvID"]).astype("Int64"),
                "pubnumber": pd.to_numeric(df["pubnumber"]).astype("Int64"),
                "pubyear": pd.to_numeric(df["pubyear"], errors="coerce").astype("Int64"),
                "census_year": pd.to_numeric(df["census_year"]).astype("Int64"),
                "inventor_name": df["inventor_name"].astype("string"),
                "leven_simil": pd.to_numeric(df["leven_simil"],
                                             errors="coerce").astype("float64"),
                "county_distance": pd.to_numeric(df["county_distance"],
                                                 errors="coerce").astype("float64"),
                "age_at_pat": pd.to_numeric(df["age_at_pat"],
                                            errors="coerce").astype("Int64"),
                "relatedness_short": pd.to_numeric(df["relatedness_short"],
                                                   errors="coerce").astype("float64"),
                "relatedness_long": pd.to_numeric(df["relatedness_long"],
                                                  errors="coerce").astype("float64"),
                "probability": proba.astype("float64"),
            })
            writer.write_table(pa.Table.from_pandas(out, schema=out_schema,
                                                    preserve_index=False))
            n_rows += len(out)
            if n_rows % (args.batch_size * 5) < args.batch_size:
                print(f"[3] scored {n_rows:,} rows", flush=True)
    finally:
        writer.close()
    print(f"[3] scored {n_rows:,} candidate rows total", flush=True)
    t0 = _lap("score all candidates", t0)

    # ---- Pass 2: per-(PatInvID, census_year) top-1 + threshold (polars) ----
    # Spill to real disk: the default /tmp is often tmpfs (RAM), so the group-by over the
    # tens-of-GB scored temp OOM-kills there. Set before importing polars so it takes effect.
    spill_dir = args.spill_dir or os.path.join(os.path.dirname(args.output) or ".",
                                               ".polars_tmp")
    os.makedirs(spill_dir, exist_ok=True)
    os.environ["POLARS_TEMP_DIR"] = spill_dir
    os.environ["TMPDIR"] = spill_dir
    import polars as pl
    # Per-group reduction (NOT a global sort): keep, per group, the highest-probability row
    # (tie-break: lowest histid). Equivalent to sort-then-first but bounds memory to one
    # accumulator per group, so it streams without materializing all rows for a global sort.
    carried = ["histid", "pubnumber", "pubyear", "inventor_name", "leven_simil",
               "county_distance", "age_at_pat", "relatedness_short",
               "relatedness_long", "probability"]
    accepted = (
        pl.scan_parquet(scored_tmp)
        .group_by(["PatInvID", "census_year"], maintain_order=False)
        .agg(pl.col(carried)
             .sort_by(["probability", "histid"], descending=[True, False]).first())
        .filter(pl.col("probability") >= args.accept_threshold)
    ).collect(engine="streaming")
    n_acc = accepted.height
    print(f"[3] accepted links: {n_acc:,} (threshold >= {args.accept_threshold})",
          flush=True)
    t0 = _lap("select accepted", t0)

    # ---- Pass 3: attach county_inv/state_inv from the 2b matched_candidates ----
    # The stage-3 merge drops the inventor-side geography, so recover it from the
    # per-year 2b outputs, keyed on the full candidate identity. unique() on the keys
    # guards the few known duplicate triples (same key, different raw name spelling) —
    # the geography is inventor-side, so duplicates agree on county_inv/state_inv.
    geo_glob = os.path.join(args.matched_candidates, "matched_candidates_*.parquet")
    acc_keys = accepted.lazy().select(GEO_KEYS)
    geo = (
        pl.scan_parquet(geo_glob)
        .select([
            pl.col("PatInvID").cast(pl.Int64),
            pl.col("pubnumber").cast(pl.Int64),
            pl.col("histid").cast(pl.Utf8),
            pl.col("census_year").cast(pl.Int64),
            pl.col("county_inv").cast(pl.Utf8),
            pl.col("state_inv").cast(pl.Utf8),
        ])
        # Cut the full candidate universe down to the accepted keys BEFORE the
        # unique(): the semi-join streams the big side against the small key set.
        .join(acc_keys, on=GEO_KEYS, how="semi")
        .unique(subset=GEO_KEYS)
    )
    final = (
        accepted.lazy()
        .join(geo, on=GEO_KEYS, how="left")
        .select(OUT_COLS)
    ).collect(engine="streaming")

    n_nogeo = int(final["county_inv"].is_null().sum())
    if n_nogeo:
        print(f"[3] WARNING: {n_nogeo:,} accepted links found no county_inv/state_inv "
              f"in {geo_glob} (expected 0 — check --matched-candidates)", flush=True)

    final.write_parquet(args.output)
    print(f"[3] wrote {final.height:,} accepted links "
          f"({len(final.columns)} publication columns) -> {args.output}", flush=True)

    if not args.keep_scored_tmp:
        try:
            os.remove(scored_tmp)
        except OSError:
            pass
    _lap("geo join + write accepted", t0)


if __name__ == "__main__":
    main()
