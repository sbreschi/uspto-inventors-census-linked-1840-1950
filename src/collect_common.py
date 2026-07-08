#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared polars core for stage 2a (collect blocking candidates).

The 2a collect entry point (`2a_collect_candidates.py`) reduces a set of blocking
outputs to one deduplicated candidate table. The logic is factored out here so the
source lists can vary while the reduction stays identical:

  * cosine sources  -> best (max) `cosine_score` per (histid, PatInvID, pubnumber)
  * Levenshtein srcs -> best (min) `levenshtein_score` per the same key
  * the two metrics are outer-joined into separate columns (never compared), with
    the inventor/census names coalesced across sources.

cosine and Levenshtein are kept on independent scales (never compared); 2b
recomputes its own features and drops both columns.

"""

import os

import polars as pl

KEYS = ["histid", "PatInvID", "pubnumber", "inventor_name", "census_name"]
MERGE_KEYS = ["histid", "PatInvID", "pubnumber"]
OUT_COLS = ["histid", "PatInvID", "pubnumber", "inventor_name", "census_name",
            "cosine_score", "levenshtein_score"]


def _part_filter(part):
    """Predicate selecting one hash bucket of the merge key, or None for no split.

    `part` is (n_parts, idx). The hash is taken over MERGE_KEYS with a fixed seed
    so the cosine and Levenshtein unions bucket matching keys identically -- every
    row for a given triple lands in the same partition, which is what makes the
    per-partition group-by / join / unique produce output identical to the
    single-pass run."""
    if part is None:
        return None
    n_parts, idx = part
    return (pl.struct(MERGE_KEYS).hash(seed=0) % n_parts) == idx


def _union_metric(sources, out_col, agg, part=None):
    """Reduce a list of (path, raw_score_column) to one lazy frame keyed by
    MERGE_KEYS with the best score (max for cosine, min for Levenshtein) and the
    coalesced names. Missing files are skipped. Returns None if none exist.

    When `part` is given, each source scan is filtered to one hash bucket of the
    merge key *before* the group-by, so the streaming engine only ever holds a
    1/n_parts slice of the keys -- this is what bounds peak memory on the large
    census waves (see collect_and_write)."""
    pred = _part_filter(part)
    frames = []
    for path, raw in sources:
        if not os.path.exists(path):
            print(f"  (skip, not found) {os.path.basename(path)}")
            continue
        f = pl.scan_parquet(path).select(KEYS + [pl.col(raw).alias(out_col)])
        if pred is not None:
            f = f.filter(pred)
        frames.append(f)
    if not frames:
        return None
    u = pl.concat(frames, how="vertical_relaxed")
    best = pl.max(out_col) if agg == "max" else pl.min(out_col)
    # Names are constant within a triple (inventor_name from the patent, census_name
    # from the histid), so first() over MERGE_KEYS coalesces the names across sources.
    return u.group_by(MERGE_KEYS).agg(best, pl.first("inventor_name"), pl.first("census_name"))


def collect_polars(cosine_sources, leven_sources, year,
                   extra_paths=None, false_coinvs_path=None, part=None):
    """Return a polars LazyFrame of collected candidates (OUT_COLS + census_year).

    cosine_sources / leven_sources: lists of (parquet_path, raw_score_column).
    extra_paths: optional pre-built candidate parquets (matched_candidates layout,
                 KEYS columns) folded in with null scores and deduped on KEYS.
    false_coinvs_path: optional CSV with a PatInvID column to anti-join out.
    part: optional (n_parts, idx) hash bucket of the merge key to restrict to; used
          by collect_and_write to process one memory-bounded slice at a time. The
          union of all idx in range(n_parts) reproduces the full (part=None) result.
    """
    pred = _part_filter(part)
    cos = _union_metric(cosine_sources, "cosine_score", "max", part=part)
    lev = _union_metric(leven_sources, "levenshtein_score", "min", part=part)

    if cos is not None and lev is not None:
        overall = cos.join(lev, on=MERGE_KEYS, how="full", coalesce=True, suffix="_lev")
        overall = overall.with_columns(
            pl.coalesce("inventor_name", "inventor_name_lev").alias("inventor_name"),
            pl.coalesce("census_name", "census_name_lev").alias("census_name"),
        )
    elif cos is not None:
        overall = cos.with_columns(pl.lit(None, dtype=pl.Float64).alias("levenshtein_score"))
    elif lev is not None:
        overall = lev.with_columns(pl.lit(None, dtype=pl.Float32).alias("cosine_score"))
    else:
        raise SystemExit(f"No blocking outputs found for year {year}")
    overall = overall.select(OUT_COLS)

    # Fold in any pre-built candidate sets (e.g. the 1950 patents), null scores,
    # deduped on the full key keeping the scored (overall) version first.
    for path in (extra_paths or []):
        if not os.path.exists(path):
            print(f"  (extra, not found) {os.path.basename(path)}")
            continue
        ex = (pl.scan_parquet(path).select(KEYS)
              .filter(pl.col("histid").is_not_null() & pl.col("census_name").is_not_null()))
        if pred is not None:
            ex = ex.filter(pred)
        ex = (ex.with_columns(pl.lit(None, dtype=pl.Float32).alias("cosine_score"),
                              pl.lit(None, dtype=pl.Float64).alias("levenshtein_score"))
              .select(OUT_COLS))
        print(f"  (extra) folding {os.path.basename(path)}")
        overall = pl.concat([overall, ex], how="vertical_relaxed").unique(subset=KEYS, keep="first")

    if false_coinvs_path and os.path.exists(false_coinvs_path):
        bad = (pl.scan_csv(false_coinvs_path).select(pl.col("PatInvID").cast(pl.Int64))
               .drop_nulls().unique())
        overall = overall.join(bad, on="PatInvID", how="anti")

    return overall.with_columns(pl.lit(year).alias("census_year")).unique()


def _count_file(path):
    """(rows, unique triples) for a candidate parquet, via streaming scans (the
    frame is never materialised in full just to be measured)."""
    rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    trip = (pl.scan_parquet(path).select(MERGE_KEYS).unique()
            .select(pl.len()).collect(engine="streaming").item())
    return rows, trip


def write_candidates(lazyframe, output, engine="streaming"):
    """Stream a single candidate LazyFrame to disk and print row/triple counts.

    `sink_parquet` writes in chunks rather than returning a DataFrame, but note it
    does NOT by itself bound peak memory: the streaming engine still builds the full
    group-by / outer-join / unique state in committed memory for this query. On the
    large waves (1900/1910 with the country-wide 2g source) that peaks well past
    64 GB, so prefer collect_and_write, which splits the work into memory-bounded
    hash partitions. This single-pass path is fine for the small waves.
    `engine` is accepted for backward compatibility and ignored.
    """
    lazyframe.sink_parquet(output, compression="zstd")
    rows, trip = _count_file(output)
    print(f"Collected {rows} candidate pairs ({trip} unique triples) -> {output}")
    return rows, trip


def collect_and_write(cosine_sources, leven_sources, year, output,
                      extra_paths=None, false_coinvs_path=None, n_parts=None):
    """Collect candidates and stream them to `output`, bounding peak memory.

    The collect query (concat -> group-by -> full outer join -> unique) holds its
    whole hash state in RAM, which overflows 64 GB on the large waves once the 2g
    country source is folded in. To bound it, the work is split into `n_parts` hash
    buckets of the merge key (pushed down to the source scans), each processed and
    sunk independently so the engine only ever holds ~1/n_parts of the keys. Buckets
    are disjoint by construction, so concatenating them reproduces the exact same
    output as a single pass -- and the per-bucket counts simply sum.

    n_parts defaults to the CAND_PARTITIONS env var (else 8). n_parts<=1 falls back
    to the single-pass write_candidates.
    """
    if n_parts is None:
        n_parts = int(os.environ.get("CAND_PARTITIONS", "8"))
    if n_parts <= 1:
        lf = collect_polars(cosine_sources, leven_sources, year,
                            extra_paths=extra_paths, false_coinvs_path=false_coinvs_path)
        return write_candidates(lf, output)

    base, ext = os.path.splitext(output)
    parts, rows, trip = [], 0, 0
    for i in range(n_parts):
        part_path = f"{base}.part{i:02d}{ext}"
        lf = collect_polars(cosine_sources, leven_sources, year,
                            extra_paths=extra_paths, false_coinvs_path=false_coinvs_path,
                            part=(n_parts, i))
        lf.sink_parquet(part_path, compression="zstd")
        # Per-bucket counts are bounded (~1/n_parts) and, because buckets are disjoint
        # on the merge key, sum to the exact global rows / unique-triple totals.
        prows, ptrip = _count_file(part_path)
        rows += prows
        trip += ptrip
        parts.append(part_path)
        print(f"  partition {i + 1}/{n_parts}: {prows} pairs ({ptrip} triples)")

    # Concatenate the buckets into the single contracted output (pure streaming
    # passthrough -- no join/group, so memory stays flat), then drop the parts.
    pl.scan_parquet(parts).sink_parquet(output, compression="zstd")
    for p in parts:
        os.remove(p)
    print(f"Collected {rows} candidate pairs ({trip} unique triples) -> {output}")
    return rows, trip