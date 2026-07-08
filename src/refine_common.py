#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared helpers for stage 2b (refine / filter) of the linkage pipeline -- polars.

Stage 2b takes the candidate pairs from blocking, computes detailed name-similarity
metrics, drops obvious non-matches with a cascade of 13 structural filters, and
enriches survivors with patent/census geography, demographics and the inter-county
distance. Shared by 2a (collect), 2b (refine/filter) and the 2c golden diagnostics.

"""

import os
import re
import time

import pandas as pd
import polars as pl


def _lap(label, t0):
    """Print elapsed seconds since t0 for a pipeline stage; return a fresh timestamp."""
    now = time.perf_counter()
    print(f"[timing] {label}: {now - t0:.1f}s", flush=True)
    return now
from rapidfuzz import fuzz
from rapidfuzz import distance
from metaphone import doublemetaphone

GENDERS = ("M", "F", "U")

METRIC_COLUMNS = [
    "leven_norm", "tkratio", "tksortratio",
    "similar_double_metaphone", "token_inventor", "token_census",
]
# Metric columns the structural (pair-level) filters read off the unique-pairs table.
_PAIR_METRIC_COLS = ["similar_double_metaphone", "tkratio", "tksortratio"]


# ---------------------------------------------------------------------------#
# Name normalisation (computed on unique names in pandas, joined back in polars)
# ---------------------------------------------------------------------------#

NAME_ABBREVIATIONS = [
    (re.compile(r"\baug\b"), "augustus"), (re.compile(r"\balexn\b"), "alexander"),
    (re.compile(r"\bbenj\b"), "benjamin"), (re.compile(r"\bbenjn\b"), "benjamin"),
    (re.compile(r"\bchas\b"), "charles"), (re.compile(r"\bdanl\b"), "daniel"),
    (re.compile(r"\bedwd\b"), "edward"), (re.compile(r"\bfredk\b"), "frederick"),
    (re.compile(r"\bgeo\b"), "george"), (re.compile(r"\bjas\b"), "james"),
    (re.compile(r"\bjno\b"), "john"), (re.compile(r"\bjos\b"), "joseph"),
    (re.compile(r"\bnathl\b"), "nathaniel"), (re.compile(r"\brichd\b"), "richard"),
    (re.compile(r"\brobt\b"), "robert"), (re.compile(r"\bsaml\b"), "samuel"),
    (re.compile(r"\bthos\b"), "thomas"), (re.compile(r"\bwm\b"), "william"),
    (re.compile(r"\bedw\b"), "edward"),
]

_SUFFIX_PATTERNS = [re.compile(s, re.IGNORECASE) for s in [
    r",? jr\.?$", r",? sr\.?$", r",? senior$", r",? junior$",
    r",? II\.?$", r",? III\.?$", r",? IV\.?$", r",? V\.?$",
    r"^prof\.? dr\.? ", r"^professor dr\.? ", r"^prof ", r"^professor ",
    r"^dr\.? ", r"^doctor ",
    r" chief engineer$", r" civil engineer$", r" mechanical engineer$", r"engineer$",
]]


def strip_suffixes(series):
    for pattern in _SUFFIX_PATTERNS:
        series = series.str.replace(pattern, "", regex=True)
    return series.str.strip()


def shorten_middle_names(name):
    parts = name.split()
    if len(parts) > 2:
        for i in range(1, len(parts) - 1):
            parts[i] = parts[i][0].lower()
    return " ".join(parts)


def _normalize_inventor(names_pd):
    s = strip_suffixes(names_pd)
    s = s.str.replace(r"^miss ", "", regex=True)
    return s.map(shorten_middle_names)


def _normalize_census(names_pd):
    s = names_pd.copy()
    for pattern, repl in NAME_ABBREVIATIONS:
        s = s.str.replace(pattern, repl, regex=True)
    return s


def normalize_candidate_names(df):
    """Normalise inventor / census names. Computed on the unique names (pandas)
    and joined back, so it is independent of row count rather than a full-column
    normalisation."""
    inv = df.select("inventor_name").unique().to_pandas()
    inv["__inv"] = _normalize_inventor(inv["inventor_name"])
    cen = df.select("census_name").unique().to_pandas()
    cen["__cen"] = _normalize_census(cen["census_name"])
    df = (df.join(pl.from_pandas(inv), on="inventor_name", how="left")
            .join(pl.from_pandas(cen), on="census_name", how="left")
            .drop(["inventor_name", "census_name"])
            .rename({"__inv": "inventor_name", "__cen": "census_name"}))
    return df


# ---------------------------------------------------------------------------#
# Similarity metrics (rapidfuzz on the unique pairs; merged back in polars)
# ---------------------------------------------------------------------------#

def _metaphone_match(dm_inv, dm_cen):
    a1, a2 = dm_inv
    b1, b2 = dm_cen
    return ((a1 == b1 and a1 != "") or (a1 == b2 and a1 != "" and b2 != "") or
            (a2 == b1 and a2 != "" and b1 != "") or (a2 == b2 and a2 != "" and b2 != ""))


def compute_name_metrics(pairs):
    """Add metric columns to a unique (inventor_name, census_name) pandas table."""
    pairs = pairs.copy()
    inv = pairs["inventor_name"].tolist()
    cen = pairs["census_name"].tolist()
    pairs["leven_norm"] = [distance.Levenshtein.normalized_distance(a, b) for a, b in zip(inv, cen)]
    pairs["tkratio"] = [fuzz.ratio(a, b) for a, b in zip(inv, cen)]
    pairs["tksortratio"] = [fuzz.token_sort_ratio(a, b) for a, b in zip(inv, cen)]
    pairs["similar_double_metaphone"] = [
        _metaphone_match(doublemetaphone(a), doublemetaphone(b)) for a, b in zip(inv, cen)]
    pairs["token_inventor"] = [len(a.split()) for a in inv]
    pairs["token_census"] = [len(b.split()) for b in cen]
    return pairs


def add_metrics(candidates):
    """Attach metric columns to the polars candidate table (computed once per pair)."""
    pairs = candidates.select(["inventor_name", "census_name"]).unique().to_pandas()
    pairs = compute_name_metrics(pairs)
    return candidates.join(pl.from_pandas(pairs), on=["inventor_name", "census_name"], how="left")


# ---------------------------------------------------------------------------#
# Pair-level filter helpers (pandas bad-pair decision on the small unique table)
# ---------------------------------------------------------------------------#

def _ratios(a_list, b_list):
    return [fuzz.ratio(a, b) for a, b in zip(a_list, b_list)]


def _str_len(series):
    return series.map(lambda x: len(x) if isinstance(x, str) else 0)


def _token_count(series):
    return series.str.split().str.len()


def _surname_keep(tmp, sim_col):
    return (tmp[sim_col] > 80) | \
           ((tmp[sim_col] < 70) & (tmp["similar_double_metaphone"] == 1)) | \
           ((tmp[sim_col] < 70) & (tmp["tkratio"] >= 90)) | \
           ((tmp[sim_col] < 70) & (tmp["tksortratio"] >= 90))


def _apply(df, badfn):
    """Run a pandas bad-pair decision on the unique pairs, remove via polars anti-join."""
    cols = ["inventor_name", "census_name"] + [c for c in _PAIR_METRIC_COLS if c in df.columns]
    uniq = df.select(cols).unique(subset=["inventor_name", "census_name"]).to_pandas()
    bad = badfn(uniq)
    if bad is None or len(bad) == 0:
        return df
    bad_pl = pl.from_pandas(bad[["inventor_name", "census_name"]].drop_duplicates())
    return df.join(bad_pl, on=["inventor_name", "census_name"], how="anti")


# --- bad-pair decision functions ---

def _bad1(pairs):
    inv = pairs["inventor_name"].str.split()
    cen = pairs["census_name"].str.split()
    mask = []
    for p1, p2 in zip(inv, cen):
        mask.append(len(p1) == 3 and len(p2) == 3 and p1[0] == p2[0] and p1[-1] == p2[-1]
                    and len(p1[0]) > 1 and len(p1[-1]) > 1
                    and len(p1[1]) == 1 and len(p2[1]) == 1 and p1[1] != p2[1])
    return pairs[pd.Series(mask, index=pairs.index)]


def _two_word_bad(pairs, same_token, diff_token, sim_col):
    tmp = pairs[(_token_count(pairs["inventor_name"]) == 2) & (_token_count(pairs["census_name"]) == 2)].copy()
    a_inv = tmp["inventor_name"].str.split()
    a_cen = tmp["census_name"].str.split()
    tmp["same_inv"] = [p[same_token] for p in a_inv]
    tmp["same_cen"] = [p[same_token] for p in a_cen]
    tmp = tmp[tmp["same_inv"] == tmp["same_cen"]]
    tmp["cmp_inv"] = [p[diff_token] for p in tmp["inventor_name"].str.split()]
    tmp["cmp_cen"] = [p[diff_token] for p in tmp["census_name"].str.split()]
    tmp = tmp[(_str_len(tmp["cmp_inv"]) > 1) & (_str_len(tmp["cmp_cen"]) > 1)]
    if same_token == 0:
        tmp = tmp[(_str_len(tmp["same_inv"]) > 1) & (_str_len(tmp["same_cen"]) > 1)]
    tmp[sim_col] = _ratios(tmp["cmp_inv"], tmp["cmp_cen"])
    return tmp[~_surname_keep(tmp, sim_col)]


def _bad2(pairs):
    return _two_word_bad(pairs, 0, -1, "similarity_last")


def _bad3(pairs):
    return _two_word_bad(pairs, -1, 0, "similarity_first")


def _three_two_bad(pairs, inv_tokens, cen_tokens, same_token, diff_token, sim_col, middle_side):
    tmp = pairs[(_token_count(pairs["inventor_name"]) == inv_tokens) &
                (_token_count(pairs["census_name"]) == cen_tokens)].copy()
    if tmp.empty:
        return tmp
    inv = tmp["inventor_name"].str.split()
    cen = tmp["census_name"].str.split()
    if middle_side == "inventor":
        tmp = tmp[[len(p[1]) == 1 for p in inv]]
    else:
        tmp = tmp[[len(p[1]) == 1 for p in cen]]
    inv = tmp["inventor_name"].str.split()
    cen = tmp["census_name"].str.split()
    tmp["same_inv"] = [p[same_token] for p in inv]
    tmp["same_cen"] = [p[same_token] for p in cen]
    tmp = tmp[tmp["same_inv"] == tmp["same_cen"]]
    tmp["cmp_inv"] = [p[diff_token] for p in tmp["inventor_name"].str.split()]
    tmp["cmp_cen"] = [p[diff_token] for p in tmp["census_name"].str.split()]
    tmp = tmp[(_str_len(tmp["cmp_inv"]) > 1) & (_str_len(tmp["cmp_cen"]) > 1)]
    tmp = tmp[(_str_len(tmp["same_inv"]) > 1) & (_str_len(tmp["same_cen"]) > 1)]
    tmp[sim_col] = _ratios(tmp["cmp_inv"], tmp["cmp_cen"])
    return tmp[~_surname_keep(tmp, sim_col)]


def _bad4(pairs):
    return _three_two_bad(pairs, 3, 2, 0, -1, "similarity_last", "inventor")


def _bad5(pairs):
    return _three_two_bad(pairs, 2, 3, 0, -1, "similarity_last", "census")


def _bad6(pairs):
    return _three_two_bad(pairs, 3, 2, -1, 0, "similarity_first", "inventor")


def _bad7(pairs):
    tmp = pairs[(_token_count(pairs["inventor_name"]) == 3) & (_token_count(pairs["census_name"]) == 3)].copy()
    if tmp.empty:
        return tmp
    inv = tmp["inventor_name"].str.split()
    cen = tmp["census_name"].str.split()
    tmp = tmp[[len(pi[1]) == 1 and len(pc[1]) == 1 for pi, pc in zip(inv, cen)]]
    inv = tmp["inventor_name"].str.split()
    cen = tmp["census_name"].str.split()
    tmp["last_inv"] = [p[-1] for p in inv]
    tmp["last_cen"] = [p[-1] for p in cen]
    tmp["mid_inv"] = [p[1] for p in inv]
    tmp["mid_cen"] = [p[1] for p in cen]
    tmp = tmp[(tmp["last_inv"] == tmp["last_cen"]) & (tmp["mid_inv"] == tmp["mid_cen"])]
    tmp["first_inv"] = [p[0] for p in tmp["inventor_name"].str.split()]
    tmp["first_cen"] = [p[0] for p in tmp["census_name"].str.split()]
    tmp["similarity_first"] = _ratios(tmp["first_inv"], tmp["first_cen"])
    keep = (tmp["similarity_first"] > 80) | \
           ((tmp["similarity_first"] < 50) & (tmp["similar_double_metaphone"] == 1)) | \
           ((tmp["similarity_first"] < 50) & (tmp["tkratio"] >= 90)) | \
           ((tmp["similarity_first"] < 50) & (tmp["tksortratio"] >= 90))
    return tmp[~keep]


def _bad8(pairs):
    tmp = pairs[(_token_count(pairs["inventor_name"]) == 3) & (_token_count(pairs["census_name"]) == 3)].copy()
    if tmp.empty:
        return tmp
    inv = tmp["inventor_name"].str.split()
    cen = tmp["census_name"].str.split()
    tmp = tmp[[len(pi[1]) == 1 and len(pc[1]) == 1 for pi, pc in zip(inv, cen)]]
    inv = tmp["inventor_name"].str.split()
    cen = tmp["census_name"].str.split()
    tmp["first_inv"] = [p[0] for p in inv]
    tmp["first_cen"] = [p[0] for p in cen]
    tmp["mid_inv"] = [p[1] for p in inv]
    tmp["mid_cen"] = [p[1] for p in cen]
    tmp["last_inv"] = [p[-1] for p in inv]
    tmp["last_cen"] = [p[-1] for p in cen]
    tmp = tmp[(tmp["first_inv"] == tmp["first_cen"]) & (tmp["mid_inv"] != tmp["mid_cen"])]
    tmp["leven_last"] = [distance.Levenshtein.distance(a, b)
                         for a, b in zip(tmp["last_inv"], tmp["last_cen"])]
    return tmp[tmp["leven_last"] > 1]


def _bad10(pairs):
    mask = []
    for n1, n2 in zip(pairs["inventor_name"], pairs["census_name"]):
        a, b = n1.split(), n2.split()
        fa, ma = (a[0], a[1] if len(a) > 1 else "")
        fb, mb = (b[0], b[1] if len(b) > 1 else "")
        mask.append(fa == fb and ma != mb and len(fa) > 1 and len(fb) > 1
                    and len(ma) == 1 and len(mb) == 1)
    return pairs[pd.Series(mask, index=pairs.index)]


def _bad11(pairs):
    mask = []
    for n1, n2 in zip(pairs["inventor_name"], pairs["census_name"]):
        a, b = n1.split(), n2.split()
        mask.append(len(a) >= 2 and len(b) >= 2 and len(a[0]) == 1 and len(b[0]) == 1
                    and len(a[1]) == 1 and len(b[1]) == 1 and a[0] != b[0] and a[1] != b[1])
    return pairs[pd.Series(mask, index=pairs.index)]


def _bad12(pairs):
    fn_inv = pairs["inventor_name"].str.split().str[0]
    fn_cen = pairs["census_name"].str.split().str[0]
    p = pairs.copy()
    p["fn_ratio"] = _ratios(fn_inv, fn_cen)
    return p[(p["fn_ratio"] < 40) & (p["tksortratio"] < 90)]


# ---------------------------------------------------------------------------#
# The 13 filters (polars DataFrame -> polars DataFrame)
# ---------------------------------------------------------------------------#

def filter0(df):
    """Drop pairs with weak overall similarity (pure polars, vectorised)."""
    return df.filter(~((pl.col("leven_norm") > 0.2) & (pl.col("tkratio") < 90) &
                       (pl.col("tksortratio") < 90)))


def filter1(df):
    return _apply(df, _bad1)


def filter2(df):
    return _apply(df, _bad2)


def filter3(df):
    return _apply(df, _bad3)


def filter4(df):
    return _apply(df, _bad4)


def filter5(df):
    return _apply(df, _bad5)


def filter6(df):
    return _apply(df, _bad6)


def filter7(df):
    return _apply(df, _bad7)


def filter8(df):
    return _apply(df, _bad8)


def filter9(df):
    """Drop if age at patent outside [18, 80] (record-level; pure polars).
    Null age_at_pat is dropped."""
    return df.filter((pl.col("age_at_pat") >= 18) & (pl.col("age_at_pat") <= 80))


def filter10(df):
    return _apply(df, _bad10)


def filter11(df):
    return _apply(df, _bad11)


def filter12(df):
    return _apply(df, _bad12)


FILTERS = [
    (0, "different full name (filter 0)", filter0),
    (1, "different middle name", filter1),
    (2, "different last name", filter2),
    (3, "different first name", filter3),
    (4, "different last name", filter4),
    (5, "different last name", filter5),
    (6, "different first name", filter6),
    (7, "different first name", filter7),
    (8, "different middle and last name", filter8),
    (9, "age out of band", filter9),
    (10, "same first, different middle", filter10),
    (11, "different single letters", filter11),
    (12, "different first name", filter12),
]


# ---------------------------------------------------------------------------#
# Streaming fast path for 2b (refine/filter)
# ---------------------------------------------------------------------------#
#
# The candidate table is large (up to ~238M rows), so this path avoids a full-frame
# pass per filter (name-normalise join-backs, a metric join-back, the age joins, and
# one polars anti-join per pair-level filter).
#
# Every pair-level filter (0-8, 10-12) is a pure function of the (normalised
# inventor_name, census_name) pair and its metrics; removal is monotone and
# pair-local, so order is irrelevant and the final survivor set equals the UNION
# of every filter's bad-pair set, computed once on the small unique-pairs table.
# Filter 9 (age) is the only record-level filter. `build_surviving_pairs` does all
# the pair-level work on that small table (it calls the same `_bad*` and
# `compute_name_metrics`), and `refine_streaming` applies it to the big frame with a
# single streaming inner join + record-level age filter. This is row-for-row
# identical to the sequential cascade but runs in essentially one streaming pass,
# bounding peak RAM. The per-filter FILTERS list is retained for the 2c diagnostics,
# which replay it one filter at a time.

# Pair-level bad-pair decision functions, in filter order (filter 9 is record-level
# and excluded; filter 0 is the vectorised condition handled inline below).
_PAIR_BAD_FUNCS = [
    (1, _bad1), (2, _bad2), (3, _bad3), (4, _bad4), (5, _bad5), (6, _bad6),
    (7, _bad7), (8, _bad8), (10, _bad10), (11, _bad11), (12, _bad12),
]


def build_surviving_pairs(unique_pairs, verbose=True):
    """Pair-level normalise + metrics + filter cascade on the unique pairs.

    `unique_pairs` is a pandas frame of the unique *original* (inventor_name,
    census_name) pairs. Returns a polars frame keyed by those original names, with
    the normalised names (`ninv`, `ncen`) and the six metric columns attached, and
    only the pairs that survive every pair-level filter (0-8, 10-12) retained.
    """
    pairs = unique_pairs[["inventor_name", "census_name"]].copy()

    # Normalise inventor / census names independently (== normalize_candidate_names).
    inv_u = pd.DataFrame({"inventor_name": pairs["inventor_name"].drop_duplicates()})
    inv_u["ninv"] = _normalize_inventor(inv_u["inventor_name"])
    cen_u = pd.DataFrame({"census_name": pairs["census_name"].drop_duplicates()})
    cen_u["ncen"] = _normalize_census(cen_u["census_name"])
    pairs = pairs.merge(inv_u, on="inventor_name", how="left") \
                 .merge(cen_u, on="census_name", how="left")

    # Metrics on the unique *normalised* pairs (== add_metrics' unique-pairs table,
    # which is what every _apply call deduplicates to). reset_index(drop=True) is
    # REQUIRED: drop_duplicates() leaves a non-contiguous index, and the structural
    # filters (`_three_two_bad` / `_two_word_bad`) mix positional list-assignment with
    # index-aligned boolean masking -- on a gappy/non-monotonic index that silently
    # misaligns and can zero out a filter. polars `.to_pandas()` always hands over a
    # clean RangeIndex, so the reset here reproduces that guarantee.
    norm = (pairs[["ninv", "ncen"]].drop_duplicates()
            .rename(columns={"ninv": "inventor_name", "ncen": "census_name"})
            .reset_index(drop=True))
    norm = compute_name_metrics(norm)

    # Union of every pair-level filter's bad pairs (normalised-name keyed).
    bad = set()
    f0 = (norm["leven_norm"] > 0.2) & (norm["tkratio"] < 90) & (norm["tksortratio"] < 90)
    bad.update(zip(norm.loc[f0, "inventor_name"], norm.loc[f0, "census_name"]))
    if verbose:
        print(f"  filter  0 bad pairs: {int(f0.sum())}")
    for fid, fn in _PAIR_BAD_FUNCS:
        b = fn(norm)
        n = 0 if b is None else len(b)
        if n:
            bad.update(zip(b["inventor_name"], b["census_name"]))
        if verbose:
            print(f"  filter {fid:>2} bad pairs: {n}")

    # Attach metrics back to every original pair via its normalised pair, then drop
    # the pairs whose normalised pair was flagged bad by any filter.
    metrics = norm[["inventor_name", "census_name", *METRIC_COLUMNS]] \
        .rename(columns={"inventor_name": "ninv", "census_name": "ncen"})
    pairs = pairs.merge(metrics, on=["ninv", "ncen"], how="left")
    keep = [(a, b) not in bad for a, b in zip(pairs["ninv"], pairs["ncen"])]
    surv = pairs[pd.Series(keep, index=pairs.index)]
    if verbose:
        print(f"  surviving unique pairs: {len(surv)} / {len(unique_pairs)}")
    return pl.from_pandas(surv)


def refine_streaming(candidates_path, patents, cleaned_census_path, census_year):
    """Stream stage 2b's hot path: drop dead pairs + attach metrics + apply the
    age filter in essentially one pass over the (huge) candidate parquet.

    Returns the small survivor frame with normalised inventor_name/census_name and
    the metric columns, ready for the enrichment joins. Row-for-row identical to the
    sequential `prepare_for_filtering` + FILTERS[0..12] cascade.
    """
    t0 = time.perf_counter()
    cand = pl.scan_parquet(candidates_path).filter(pl.col("histid").is_not_null())

    # Two cheap narrow streaming passes for the join inputs (Part A: polars, parallel).
    hist = cand.select("histid").unique().collect(engine="streaming")
    pairs0 = cand.select(["inventor_name", "census_name"]).unique().collect(engine="streaming")
    t0 = _lap("scan + unique (histid, name-pairs) [polars]", t0)

    # Part B: single-threaded pandas/rapidfuzz on the unique pairs (metrics + 12 filters).
    surv = build_surviving_pairs(pairs0.to_pandas())
    t0 = _lap("build_surviving_pairs (metrics + filter cascade) [single-threaded]", t0)

    # Age inputs (birthyr only for the candidate histids; patent year per pair).
    birth = (_scan_census(cleaned_census_path).select(["histid", "birthyr"])
             .join(hist.lazy(), on="histid", how="semi").unique().collect())
    pat_year = patents.select(["PatInvID", "pubnumber", "year"]).unique(
        subset=["PatInvID", "pubnumber"])

    out = (cand
           .join(surv.lazy(), on=["inventor_name", "census_name"], how="inner")
           .drop(["inventor_name", "census_name"])
           .rename({"ninv": "inventor_name", "ncen": "census_name"})
           .join(pat_year.lazy(), on=["PatInvID", "pubnumber"], how="left")
           .join(birth.lazy(), on="histid", how="left")
           .with_columns((pl.col("year") - pl.col("birthyr")).alias("age_at_pat"))
           .filter((pl.col("age_at_pat") >= 18) & (pl.col("age_at_pat") <= 80))
           .drop(["year", "birthyr"])
           .collect(engine="streaming"))
    _lap("big inner-join + age filter (238M-row streaming collect) [polars]", t0)
    return out


# ---------------------------------------------------------------------------#
# Patent loader + census / distance enrichment (polars)
# ---------------------------------------------------------------------------#

def _scan_census(path):
    """Lazily scan a census parquet. Handles both a single `.parquet` file and a
    partitioned parquet *directory* (`part.*.parquet` + `_metadata` /
    `_common_metadata`) -- a bare `scan_parquet(dir)` rejects the latter on mixed
    file extensions, so glob the parquet parts explicitly when given a directory."""
    p = str(path)
    if os.path.isdir(p):
        return pl.scan_parquet(p.rstrip("/\\") + "/**/*.parquet")
    return pl.scan_parquet(p)


def _fips5(col):
    """5-digit zero-padded county FIPS from a string/numeric column (None-safe)."""
    return (pl.col(col).cast(pl.Int64, strict=False).cast(pl.Utf8).str.zfill(5))


def load_patents(path):
    """Load the published patent-inventor TSV (polars). Adds year/pubyear,
    gender_inv, county_inv (5-digit FIPS) and state_inv; keeps rows with a FIPS."""
    p = pl.read_csv(path, separator="\t", infer_schema_length=10000)
    if "pubyear" in p.columns:
        p = p.with_columns(pl.col("pubyear").cast(pl.Int64, strict=False).alias("year"))
    else:
        p = p.with_columns(pl.col("pubdate").cast(pl.Utf8).str.slice(0, 4)
                           .cast(pl.Int64, strict=False).alias("year"))
    p = p.filter(pl.col("FIPS").is_not_null())
    p = p.with_columns(_fips5("FIPS").alias("county_inv"))
    p = p.with_columns(pl.col("county_inv").str.slice(0, 2).alias("state_inv"),
                       pl.col("gender").alias("gender_inv"),
                       pl.col("year").alias("pubyear"))
    return p


def attach_age_at_pat(candidates, patents, cleaned_census_path):
    """Add age_at_pat (patent year - census birthyr). Reads birthyr only for the
    candidates' histids (semi-join) rather than the whole census file."""
    hist = candidates.select("histid").unique()
    birth = (_scan_census(cleaned_census_path).select(["histid", "birthyr"])
             .join(hist.lazy(), on="histid", how="semi").unique().collect())
    pat_year = patents.select(["PatInvID", "pubnumber", "year"]).unique(
        subset=["PatInvID", "pubnumber"])
    out = (candidates
           .join(pat_year, on=["PatInvID", "pubnumber"], how="left")
           .join(birth, on="histid", how="left")
           .with_columns((pl.col("year") - pl.col("birthyr")).alias("age_at_pat"))
           .drop(["year", "birthyr"]))
    return out


def prepare_for_filtering(candidates, patents, cleaned_census_path):
    """Name-prep -> metrics -> age_at_pat (everything the filter cascade needs)."""
    candidates = normalize_candidate_names(candidates)
    candidates = add_metrics(candidates)
    candidates = attach_age_at_pat(candidates, patents, cleaned_census_path)
    return candidates


def load_cleaned_census_geo(path, histids_df):
    """Census geo/demographic columns for the candidate histids (semi-join)."""
    cc = (_scan_census(path)
          .join(histids_df.lazy().unique(), on="histid", how="semi")
          .collect())
    cc = cc.rename({"county_fips_code": "county_census"})
    cc = cc.with_columns(_fips5_expr("county_census").alias("county_census"))
    cc = cc.with_columns(pl.col("county_census").str.slice(0, 2).alias("state_census"),
                         pl.when(pl.col("sex") == 1).then(pl.lit("M"))
                           .when(pl.col("sex") == 2).then(pl.lit("F"))
                           .otherwise(None).alias("gender_census"))
    return cc


def _fips5_expr(col):
    return pl.col(col).cast(pl.Int64, strict=False).cast(pl.Utf8).str.zfill(5)


def raw_census_columns(year):
    base = ["histid", "labforce", "ind1950", "occ1950", "occscore", "occ", "occstr", "sei",
            "edscor50", "erscor50", "school", "sex", "race", "relate", "marst", "bpl"]
    if 1850 <= year <= 1880:
        return ["histid", "lit", "labforce", "ind1950", "occ1950", "occscore", "occ", "occstr",
                "sei", "edscor50", "erscor50", "school", "sex", "race", "relate", "marst", "bpl"]
    if 1900 <= year <= 1930:
        return ["histid", "lit", "labforce", "ind1950", "occ1950", "occscore", "occstr",
                "sei", "edscor50", "erscor50", "school", "sex", "race", "relate", "marst", "bpl"]
    return base


def load_raw_census_occupation(path, year, histids_df):
    return (_scan_census(path).select(raw_census_columns(year))
            .join(histids_df.lazy().unique(), on="histid", how="semi").collect())


# ---------------------------------------------------------------------------#
# Household-file county fallback (recover census county where stage-0 geomatch is null)
# ---------------------------------------------------------------------------#
#
# Stage 0 fills `county_fips_code` (-> county_census) from a probabilistic *place* geomatch,
# which is null for people not in an identifiable city (~16% in 1850, ~5% in 1900) -- and a
# null county is the sole cause of a missing county_distance. The IPUMS *household* file
# (US{year}H.parquet) carries the true enumeration county (statefip + countyicp) for ~100% of
# these (full-count data). We reconstruct a 5-digit FIPS from it, comparable to the distance
# crosswalks, and use it only to fill the gaps (the stage-0 geomatch FIPS always wins).
#
# countyicp is the ICPSR county code, NOT FIPS: ICPSR = FIPS3 x 10 ("added zero digit"), so
# FIPS3 = countyicp // 10, with two documented systematic exceptions:
#   * Maryland (statefip 24): FIPS >= 009 are shifted DOWN by 2 in ICPSR (Baltimore City
#     excepted) -- invert with +2. MANDATORY: without it every MD county >=009 maps to a
#     valid-but-WRONG county (Frederick 24021 -> 24019, ~60mi error).
#   * Nevada (statefip 32): Pershing ICPSR 025 -> FIPS 027; Ormsby ICPSR 051 -> FIPS 025.
#   * D.C. (statefip 11): a single county-equivalent, FIPS 11001. ICPSR usually codes it >0
#     (countyicp //10 -> 1), but the 1880 household file leaves countyicp = 0 ('not identified')
#     for every D.C. record, which would otherwise drop all of D.C. to a null county (the sole
#     driver of 1880's missing county_distance). Force statefip 11 -> FIPS 001 regardless of
#     countyicp (no-op for 1870/1900, where it already resolves to 11001).
# Independent cities (Baltimore/St-Louis/Richmond) are intentionally not special-cased: they
# map to their adjacent county (<=~15mi), which is acceptable for a county-distance feature.


def _icpsr_county_fips_expr():
    """polars expr: ICPSR countyicp -> 3-digit county FIPS (Int), with MD/NV/DC fixes.
    Null when countyicp is null/0 ('not identified') or statefip is null -- except D.C.
    (statefip 11), which is always FIPS 001 even when countyicp is 0 (1880)."""
    base3 = pl.col("countyicp") // 10
    conv = (pl.when((pl.col("statefip") == 24) & (base3 >= 7) & (base3 != 510)).then(base3 + 2)
              .when((pl.col("statefip") == 32) & (base3 == 25)).then(pl.lit(27))
              .when((pl.col("statefip") == 32) & (base3 == 51)).then(pl.lit(25))
              .when(pl.col("statefip") == 11).then(pl.lit(1))
              .otherwise(base3))
    valid = (pl.col("countyicp").is_not_null() & (pl.col("countyicp") > 0)
             & pl.col("statefip").is_not_null())
    # D.C. (single county-equivalent 11001) is valid even with countyicp 0 ('not identified',
    # as the 1880 household file codes it); requires only a non-null statefip.
    valid = valid | (pl.col("statefip") == 11)
    return pl.when(valid).then(conv).otherwise(None)


def load_household_county(raw_p_path, household_path, histids_df):
    """Reconstruct census county FIPS from the household file for the candidate histids.

    histid -> serialp (person file `raw_p_path`) -> serial -> statefip/countyicp (household
    file `household_path`) -> 5-digit FIPS (`_icpsr_county_fips_expr`). Returns a polars frame
    keyed by histid with `county_census_hh` (5-digit FIPS, crosswalk-comparable) and
    `state_census_hh` (2-digit). Person file is semi-joined to the survivor histids first so
    only the relevant rows are read; the household `serial` is unique (1 row per household)."""
    # serialp/serial dtypes differ by year (f64/f64 in 1850-60, i32/f64 in 1870-80,
    # i32/i32 in 1900-40); cast both to Int64 (household serials are whole numbers) so the
    # join keys always match -- otherwise polars raises a SchemaError on the mismatched years.
    pser = (_scan_census(raw_p_path)
            .select(["histid", pl.col("serialp").cast(pl.Int64, strict=False)])
            .join(histids_df.lazy().unique(), on="histid", how="semi"))
    hh = _scan_census(household_path).select(
        [pl.col("serial").cast(pl.Int64, strict=False), "statefip", "countyicp"])
    out = (pser.join(hh, left_on="serialp", right_on="serial", how="left")
              .with_columns(_icpsr_county_fips_expr().alias("_cfips"))
              .with_columns(
                  pl.when(pl.col("_cfips").is_not_null())
                    .then(pl.col("statefip").cast(pl.Int64, strict=False).cast(pl.Utf8).str.zfill(2)
                          + pl.col("_cfips").cast(pl.Int64, strict=False).cast(pl.Utf8).str.zfill(3))
                    .otherwise(None).alias("county_census_hh"))
              .with_columns(pl.col("county_census_hh").str.slice(0, 2).alias("state_census_hh"))
              .select(["histid", "county_census_hh", "state_census_hh"])
              .unique(subset=["histid"])
              # Collect via the streaming engine: the person/household files are
              # partitioned-directory scans (thousands of part-file chunks), and the
              # polars 1.34 in-memory hash join panics on that chunk layout
              # ("index out of bounds: the len is 0" in single_keys.rs). The
              # streaming join path is unaffected. The cast dtype is irrelevant --
              # Int64 and Float64 both trigger the in-memory bug.
              .collect(engine="streaming"))
    return out


# County-distance crosswalks, in recovery priority order (high -> low): the first source
# that has a given (county1, county2) pair wins, later sources only back-fill what earlier
# ones missed. The 500-mile gaz crosswalk is the PRIMARY source; the back-fill sources
# follow. All are read from their `.csv.zip` form (never the uncompressed CSV) for
# reproducibility.
_COUNTY_DISTANCE_FILES = [
    "gaz1990countydistance500miles.csv.zip",   # PRIMARY
    "sf12000countydistance500miles.csv.zip",
    "sf12010countydistance500miles.csv.zip",
    "gaz1990countydistancemiles.csv.zip",
    "sf12000countydistancemiles.csv.zip",
    "sf12010countydistancemiles.csv.zip",
]


def _read_one_crosswalk(path, rank):
    """Read a single county-distance crosswalk (CSV, or .csv.zip) into polars with
    columns (county1, county2, mi_to_county, _rank). FIPS are normalised to the
    5-digit zero-padded string form used for `county_inv`/`county_census`.

    Polars can't read a zipped CSV directly, so zipped sources are read with pandas
    (`compression='zip'`) and handed to polars.
    """
    p = str(path)
    if p.endswith(".zip"):
        pdf = pd.read_csv(p, compression="zip", usecols=["county1", "county2", "mi_to_county"])
        cd = pl.from_pandas(pdf).lazy()
    elif p.endswith(".parquet"):
        cd = pl.scan_parquet(p)
    else:
        cd = pl.scan_csv(p, infer_schema_length=10000)
    return (cd.select([_fips5("county1").alias("county1"),
                       _fips5("county2").alias("county2"),
                       pl.col("mi_to_county").cast(pl.Float64, strict=False)])
              .filter(pl.col("mi_to_county").is_not_null())
              .with_columns(pl.lit(rank).alias("_rank")))


def _load_county_distance_table(county_dist_dir):
    """Coalesced county-distance reference table (county1, county2, mi_to_county).

    Reads every crosswalk in `_COUNTY_DISTANCE_FILES` from `county_dist_dir`, stacks
    them with a priority rank, and keeps the first (highest-priority) distance for each
    (county1, county2) pair -- equivalent to a sequential `fillna` cascade over the
    sources. Sources missing from the directory are skipped.

    The gaz crosswalk stores county FIPS as integers (e.g. 1001 for Autauga Co., AL),
    so they MUST be zero-padded to the 5-digit string form used for `county_inv`
    (patent) and `county_census` (census) -- otherwise every leading-zero state
    (FIPS < 10000) fails the join and gets a null distance. `_fips5` handles that.

    Shannon County, SD was renamed to Oglala Lakota County and recoded 46113 -> 46102;
    the crosswalks predate the change and only carry 46113, so every 46113 row is
    duplicated under 46102 (both the county1 and county2 sides) so candidate counties
    coded 46102 still join.
    """
    parts = []
    for rank, fname in enumerate(_COUNTY_DISTANCE_FILES):
        path = os.path.join(str(county_dist_dir), fname)
        if os.path.exists(path):
            parts.append(_read_one_crosswalk(path, rank))
    if not parts:
        raise FileNotFoundError(
            f"No county-distance crosswalks found in {county_dist_dir} "
            f"(expected one of {_COUNTY_DISTANCE_FILES})")

    cd = (pl.concat(parts, how="vertical")
            .sort(["county1", "county2", "_rank"])
            .unique(subset=["county1", "county2"], keep="first")
            .drop("_rank")
            .collect())

    # Shannon County SD remap: alias 46113 rows onto 46102 on both join sides.
    alias1 = cd.filter(pl.col("county1") == "46113").with_columns(pl.lit("46102").alias("county1"))
    alias2 = cd.filter(pl.col("county2") == "46113").with_columns(pl.lit("46102").alias("county2"))
    cd = (pl.concat([cd, alias1, alias2], how="vertical")
            .unique(subset=["county1", "county2"], keep="first"))
    return cd


def merge_county_distance(candidates, county_dist_path):
    """Attach county_distance (miles). Returns (candidates, missing)."""
    cd = _load_county_distance_table(county_dist_path)
    out = candidates.join(cd, left_on=["county_inv", "county_census"],
                          right_on=["county1", "county2"], how="left")
    out = out.with_columns(
        pl.when(pl.col("county_inv") == pl.col("county_census")).then(0.0)
          .otherwise(pl.col("mi_to_county")).alias("county_distance")).drop("mi_to_county")
    missing = out.filter(pl.col("county_distance").is_null())
    return out, missing


def _county_distance_expr():
    """county_distance: 0 when same county, else the crosswalk miles (null if absent)."""
    return (pl.when(pl.col("county_inv") == pl.col("county_census")).then(0.0)
              .otherwise(pl.col("mi_to_county")).alias("county_distance"))


def refine_and_write(candidates_path, patents, cleaned_census_path, raw_census_path,
                     county_dist_dir, year, output, output_columns,
                     golden_path=None, missing_distance_out=None, household_path=None):
    """Full stage-2b pipeline: streaming filter cascade -> enrichment -> parquet.

    The filter survivors can still be large (e.g. ~82M rows for 1900 with the country
    source), so the enrichment is built as a single lazy graph that **streams** to disk
    via `sink_parquet` rather than cloning the survivor frame across eager joins. The
    right-hand enrichment tables are small (semi-joined on the survivor histids, or
    small reference tables), so they broadcast cheaply. Peak RAM stays well under the
    eager path's. Output is the same matched_candidates feature table as before.
    """
    candidates = refine_streaming(candidates_path, patents, cleaned_census_path, year)
    candidates = candidates.filter(pl.col("leven_norm").is_not_null())
    drop = [c for c in ["cosine_score", "levenshtein_score", "age_at_pat"]
            if c in candidates.columns]
    candidates = candidates.drop(drop).with_columns(pl.lit(year).alias("census_year"))
    print(f"Survivors after filters: {candidates.height} rows for {year}")

    # Small right-side enrichment tables (built from the survivor histids).
    t0 = time.perf_counter()
    hist_df = candidates.select("histid").unique()
    geo = patents.select(["PatInvID", "pubnumber", "pubyear", "gender_inv",
                          "county_inv", "state_inv"]).unique(subset=["PatInvID", "pubnumber"])
    cc = load_cleaned_census_geo(cleaned_census_path, hist_df).select(
        ["histid", "county_census", "state_census", "gender_census", "birthyr"])
    occ = load_raw_census_occupation(raw_census_path, year, hist_df)
    cd = _load_county_distance_table(county_dist_dir)
    # Household-file county (countyicp->FIPS) for the survivor histids; used to fill the
    # ~16%/~5% of census records the stage-0 place-geomatch left without a county.
    hh = load_household_county(raw_census_path, household_path, hist_df) if household_path else None
    t0 = _lap("enrichment right-tables (census geo + raw occupation semi-joins)", t0)

    # Lazy enrichment graph (geo -> census geo -> [household county fill] -> county distance
    # -> occupation -> golden), streamed straight to disk.
    lf = (candidates.lazy()
          .join(geo.lazy(), on=["PatInvID", "pubnumber"], how="left")
          .with_columns(pl.col("pubyear").cast(pl.Int32))
          .join(cc.lazy(), on="histid", how="left"))
    if hh is not None:
        # Expose the household FIPS for ALL rows, then fill county_census/state_census where
        # the stage-0 geomatch is null -- the stage-0 value always wins (listed first).
        lf = (lf.join(hh.lazy(), on="histid", how="left")
                .with_columns(
                    pl.coalesce(["county_census", "county_census_hh"]).alias("county_census"),
                    pl.coalesce(["state_census", "state_census_hh"]).alias("state_census")))
    lf = (lf.join(cd.lazy(), left_on=["county_inv", "county_census"],
                  right_on=["county1", "county2"], how="left")
            .with_columns(_county_distance_expr()).drop("mi_to_county")
            .join(occ.lazy(), on="histid", how="left"))

    if golden_path:
        # Attach the golden sample's *true* histid per patent-inventor, then label each
        # candidate row. `true_histid` is the actual correct histid (a UUID string) for
        # golden PatInvIDs and null otherwise; `true_match` is the training label:
        #   1  -> this candidate is the true match (histid == true_histid),
        #   0  -> a competing candidate of a golden PatInvID (a false positive),
        #   null -> PatInvID not in the golden sample (unlabeled; excluded from training).
        # true_match's fillna(0) only spans golden PatInvIDs.
        golden = pl.read_csv(golden_path)
        # Cast the join keys to Int64 to match the candidate side: some golden-sample
        # CSVs carry PatInvID/pubnumber as ints, others as floats ("12345.0"), which
        # polars reads as Float64 and then refuses to join against the Int64 candidate
        # keys (SchemaError -- no implicit i64/f64 join cast). The cast keeps both
        # layouts safe; strict=False is null-safe.
        gy = (golden.filter(pl.col("census_year") == year)
              .select([pl.col("PatInvID").cast(pl.Int64, strict=False),
                       pl.col("pubnumber").cast(pl.Int64, strict=False),
                       pl.col("histid").alias("true_histid")])
              .unique(subset=["PatInvID", "pubnumber"]))
        # Case-insensitive histid match. The census `histid` case varies by census year
        # (lowercase for 1850 & 1880, UPPERCASE for the other waves); the golden file's
        # case may or may not match it. Comparing lowercased on both sides labels correctly
        # regardless of either side's convention -- a no-op when they already agree.
        lf = (lf.join(gy.lazy(), on=["PatInvID", "pubnumber"], how="left")
                .with_columns(
                    pl.when(pl.col("true_histid").is_null()).then(None)
                      .when(pl.col("histid").str.to_lowercase()
                            == pl.col("true_histid").str.to_lowercase()).then(1)
                      .otherwise(0).cast(pl.Int64).alias("true_match")))

    cols = [c for c in output_columns if c in lf.collect_schema().names()]
    lf.select(cols).unique().sink_parquet(output, compression="zstd")
    t0 = _lap("enrichment joins + unique + sink_parquet [polars streaming]", t0)

    # Cheap streaming summary + optional missing-distance diagnostic.
    n = pl.scan_parquet(output).select(pl.len()).collect().item()
    pairs_n = (pl.scan_parquet(output).select(["PatInvID", "pubnumber"]).unique()
               .select(pl.len()).collect(engine="streaming").item())
    missing_n = (pl.scan_parquet(output).filter(pl.col("county_distance").is_null())
                 .select(pl.len()).collect(engine="streaming").item())
    if missing_n:
        print(f"WARNING: {missing_n} rows without county distance")
        if missing_distance_out:
            (pl.scan_parquet(output).filter(pl.col("county_distance").is_null())
             .with_columns(pl.lit(year).alias("census_year")).collect()
             .write_csv(missing_distance_out))
    print(f"Wrote {n} matched candidates ({pairs_n} inventor-patent pairs) -> {output}")
