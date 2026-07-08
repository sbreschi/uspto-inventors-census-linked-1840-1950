#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke test for stage 2b (`2b_refine_filter.py`) across all census waves.

Stage 2b reads the collected candidate table from 2a, normalises names, applies
the 13-filter cascade (refine_common.py), enriches survivors with patent/census
geography, demographics and the inter-county distance, and writes the
`matched_candidates` feature table. 2b only consumes a 2a candidate parquet (plus
the inventor TSV, cleaned + raw census, and the county-distance crosswalk), so
this test is CHAINED end to end: per wave it builds one tiny shared fixture, runs
all four blocking producers into a single dir, runs 2a over them, slices a raw
census fixture, then runs 2b -- exercising the real stage-1 -> 2a -> 2b seam.

The fixture-building + blocking + collect machinery is reused verbatim from
`smoke_test_stage2a` (imported, not duplicated); 2b adds three things on top:
  * raw-census slice -> the candidate histids' rows from the partitioned raw
    IPUMS person parquet (`raw_census_columns(year)` + `serialp`), so the
    occupation left-join populates and the household-county fallback can join.
  * household-census slice -> the matching `US{year}H` rows (serial, statefip,
    countyicp) so 2b's county_distance recovery via countyicp is exercised.
  * county-distance crosswalk dir + golden sample -> passed straight to 2b.

Per wave:
  * fixture   -> top-K female-inventor counties (2a strategy): inventor TSV +
                 cleaned-census slice. `--census-cap` defaults to 10000 so 1c's
                 IVF-PQ can train.
  * blocking  -> 1a (county cosine) + 1b (county + state Levenshtein) + 1c
                 (nationwide cosine) into one `{year}_blocking` dir. 1a/1b failure
                 is a wave FAIL; 1c failure is NON-FATAL (sparse waves cannot train
                 IVF-PQ) -- 2a then collects from 1a/1b only.
  * collect   -> 2a over that dir (CAND_PARTITIONS=1) -> the candidate parquet.
  * refine    -> 2b is run TWICE over the same inputs; the two outputs must be
                 row-identical after sorting (2b has no partition knob, so this is
                 the byte-stable-output / reproducibility check). `--golden` is
                 passed by default so the two label columns and the golden
                 enrichment join are exercised.

For each wave it validates the 2b output parquet -- the trimmed 26-col
OUTPUT_COLUMNS schema and order (24 cols without golden; the year-dependent
occ/occstr/lit and the 2 golden label columns adjusted), that the score/age helper
columns were dropped, the metric ranges, the record-level age band (filter 9:
pubyear - birthyr in [18,80]), correct census_year, gender_census in {M,F,null},
true_histid Utf8 and true_match in {0,1,null}, and full-row uniqueness. A
schema-valid empty output is a per-wave PASS (tiny fixtures can be filtered to
nothing), but the suite passes only if nothing failed AND at least one wave
produced rows. Exits non-zero if any run fails.

Usage:
    conda activate ipums_match
    python smoke_test_stage2b.py [--years 1900 1910] [--census-cap 10000] ...

NOTE: tiny fixtures do not exercise the streaming out-of-core path. Verify the
memory behaviour separately by running 2b on a real dense wave (e.g. 1900,
~238M candidate rows) with POLARS_TEMP_DIR set (see run_smoke_test_stage2b.sh).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SRC = os.path.join(REPO_ROOT, "src")
SCRIPT_2B = os.path.join(SRC, "2b_refine_filter.py")
LOG_DIR_DEFAULT = os.path.join(HERE, "logs")

# Reuse the production refine helper for the raw-census column set (single source
# of truth -- exactly the columns 2b will request), and the stage-2a smoke-test
# machinery (fixtures + chained 1a/1b/1c -> 2a) so the upstream seam is identical.
sys.path.insert(0, SRC)
from refine_common import raw_census_columns  # noqa: E402

from smoke_test_stage2a import (  # noqa: E402
    GENDER, Tee, load_inventors, pick_counties, run_2a, run_blocking,
    slice_census, write_inventor_fixture,
)

DEFAULT_YEARS = [1850, 1860, 1870, 1880, 1900, 1910, 1920, 1930, 1940]

# Mirrors 2b_refine_filter.OUTPUT_COLUMNS (the trimmed 26-col model table; with
# --golden the two label columns true_histid/true_match are present, without it
# they are absent). 2b's module name ("2b_...") is not importable, so the list is
# duplicated here; keep in sync with src/2b_refine_filter.py:OUTPUT_COLUMNS.
OUTPUT_COLUMNS = [
    "PatInvID", "pubnumber", "histid", "census_year",
    "inventor_name", "census_name",
    "leven_norm", "tkratio", "tksortratio",
    "county_inv", "state_inv", "county_census", "state_census", "county_distance",
    "county_census_hh", "state_census_hh",
    "pubyear", "birthyr",
    "ind1950", "occ1950", "occstr", "occ",
    "gender_inv", "gender_census",
    "true_histid", "true_match",
]
# The two golden-label columns absent when --golden is not passed.
GOLDEN_COLUMNS = ["true_histid", "true_match"]
# Helper columns 2b must drop before writing.
DROPPED_COLUMNS = ["cosine_score", "levenshtein_score", "age_at_pat"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Smoke test 2b_refine_filter.py by chaining 1a+1b+1c -> 2a -> 2b per wave.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--inventors",
                   default=os.path.join(REPO_ROOT, "data",
                                        "USPTO_patent_inventors_1840_1950.tsv"),
                   help="Path to the published patent-inventor TSV.")
    p.add_argument("--census-dir", default="/scratch/breschi/cleaned_census",
                   help="Directory with cleaned US{year}P.parquet files (stage-0 output).")
    p.add_argument("--raw-census-dir", default="/scratch/breschi/census",
                   help="Directory with raw IPUMS US{year}P.parquet (occupation vars).")
    p.add_argument("--county-distances-dir",
                   default=os.path.join(REPO_ROOT, "data", "county_distance_crosswalks"),
                   help="Directory of county-distance crosswalk .csv.zip files (cascade recovery).")
    p.add_argument("--golden",
                   default=os.path.join(REPO_ROOT, "data", "golden_sample_V6.csv"),
                   help="Golden-sample CSV (adds true_histid value + true_match label).")
    p.add_argument("--no-golden", action="store_true",
                   help="Run 2b without --golden (output drops the 2 label columns).")
    p.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS,
                   help="Census waves to test.")
    p.add_argument("--top-counties", type=int, default=3,
                   help="Number of highest-inventor counties to slice into each fixture.")
    p.add_argument("--census-cap", type=int, default=10000,
                   help="Max cleaned-census rows per wave fixture (large enough for 1c IVF-PQ training).")
    p.add_argument("--device", choices=["gpu", "cpu"], default="cpu",
                   help="FAISS device for 1a/1c (cpu is portable; identical results).")
    # 1a / 1c knobs (passed through run_blocking/run_2a; names match stage-2a args).
    p.add_argument("--k", type=int, default=5,
                   help="Nearest census names per inventor (1a/1c --k).")
    p.add_argument("--n-cores", type=int, default=2,
                   help="Process count passed to the blockers for name cleaning.")
    p.add_argument("--max-counties", type=int, default=None,
                   help="Optional cap on geographic buckets 1a/1b process.")
    # 1b knobs (loosened so pairs survive the tiny fixture).
    p.add_argument("--limit", type=int, default=10,
                   help="Max census candidates per inventor name (1b --limit).")
    p.add_argument("--score-cutoff", type=float, default=0.5,
                   help="Max normalised Levenshtein distance to keep a candidate (1b).")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Worker processes passed to 1b for parallel matching.")
    # 1c IVF-PQ knobs (small, self-consistent so training works on the tiny fixture).
    p.add_argument("--nlist", type=int, default=32,
                   help="IVF lists for 1c (set explicitly; 0 would floor at 1024).")
    p.add_argument("--pq-m", type=int, default=8,
                   help="IVF-PQ sub-quantizers for 1c; must divide --svd-dim.")
    p.add_argument("--svd-dim", type=int, default=64,
                   help="TruncatedSVD output dimension for 1c (divisible by --pq-m).")
    p.add_argument("--svd-sample", type=int, default=2000,
                   help="Census vectors sampled to fit the 1c SVD / train the index.")
    p.add_argument("--fit-sample", type=int, default=4000,
                   help="Census names sampled to fit the 1c shared TF-IDF vocabulary.")
    p.add_argument("--nprobe", type=int, default=8,
                   help="IVF lists probed at search time for 1c.")
    p.add_argument("--log-dir", default=LOG_DIR_DEFAULT,
                   help="Directory for the timestamped run log.")
    p.add_argument("--workdir", default=None,
                   help="Where to build fixtures (default: a fresh temp dir).")
    p.add_argument("--keep", action="store_true",
                   help="Keep the fixtures/outputs instead of deleting them.")
    return p.parse_args(argv)


def _scan_raw_glob(raw_dir, year):
    """Lazy scan of the raw census for `year`, handling the partitioned-directory
    layout the same way refine_common._scan_census does."""
    path = os.path.join(raw_dir, f"US{year}P.parquet")
    if os.path.isdir(path):
        return pl.scan_parquet(path.rstrip("/\\") + "/**/*.parquet")
    return pl.scan_parquet(path)


def slice_raw_census(raw_dir, year, cleaned_fixture, out_parquet):
    """Raw-census (person-file) rows for the cleaned-fixture histids only, with the
    columns 2b requests PLUS `serialp` (the household-file join key the county
    fallback needs -- the occupation join still selects only raw_census_columns).
    Bounds the scan to the survivors (semi-join, the same pattern as
    refine_common.load_raw_census_occupation) so the occupation join populates.
    Returns the row count written."""
    hids = pl.read_parquet(cleaned_fixture, columns=["histid"]).unique()
    cols = raw_census_columns(year)
    if "serialp" not in cols:
        cols = cols + ["serialp"]
    out = (_scan_raw_glob(raw_dir, year).select(cols)
           .join(hids.lazy(), on="histid", how="semi")
           .collect(engine="streaming"))
    out.write_parquet(out_parquet)
    return out.height


def slice_household_census(raw_dir, year, raw_person_fixture, out_parquet):
    """IPUMS household-file (US{year}H) rows whose `serial` matches the person
    fixture's `serialp` set, with just (serial, statefip, countyicp) -- the inputs
    refine_common.load_household_county needs to reconstruct census county FIPS for
    records the stage-0 place-geomatch left without a county. Bounding to the
    fixture's serials keeps the scan tiny. Returns the row count written."""
    serials = (pl.read_parquet(raw_person_fixture, columns=["serialp"])
               .select(pl.col("serialp").cast(pl.Int64, strict=False))
               .drop_nulls().unique())
    path = os.path.join(raw_dir, f"US{year}H.parquet")
    scan = (pl.scan_parquet(path.rstrip("/\\") + "/**/*.parquet")
            if os.path.isdir(path) else pl.scan_parquet(path))
    out = (scan.select([pl.col("serial").cast(pl.Int64, strict=False),
                        "statefip", "countyicp"])
           .join(serials.lazy(), left_on="serial", right_on="serialp", how="semi")
           .collect(engine="streaming"))
    out.write_parquet(out_parquet)
    return out.height


def run_2b(candidates, inv_tsv, cleaned_census, raw_census, household_census,
           county_dist_dir, golden, year, out_path):
    """Run 2b once. Returns (rc, last_error_line)."""
    cmd = [sys.executable, SCRIPT_2B,
           "--candidates", candidates,
           "--inventors", inv_tsv,
           "--cleaned-census", cleaned_census,
           "--raw-census", raw_census,
           "--household-census", household_census,
           "--county-distances-dir", county_dist_dir,
           "--census-year", str(year),
           "--output", out_path]
    if golden:
        cmd += ["--golden", golden]
    proc = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return proc.returncode, (tail[-1] if tail else f"exit {proc.returncode}")
    return 0, ""


def validate_output(out_path, year, with_golden):
    """Return a list of failure messages (empty == PASS) plus a stats dict."""
    failures = []
    df = pl.read_parquet(out_path)
    stats = {"out_rows": df.height}

    # The expected schema is year-dependent: refine_and_write keeps only the
    # OUTPUT_COLUMNS that exist in the enrichment schema, and the raw-occupation
    # columns vary by year (occ present 1850-1880 & 1940, occstr present 1900-1930,
    # lit absent in 1940). Drop the raw-optional columns the wave doesn't have, and
    # the golden label columns when --golden was not passed.
    raw_set = set(raw_census_columns(year))
    raw_optional = {"lit", "occ", "occstr"}
    expected = [c for c in OUTPUT_COLUMNS
                if (c not in raw_optional or c in raw_set)
                and (with_golden or c not in GOLDEN_COLUMNS)]
    if df.columns != expected:
        failures.append(f"columns {df.columns} != expected {expected}")
        return failures, stats  # schema wrong: skip value checks

    leaked = [c for c in DROPPED_COLUMNS if c in df.columns]
    if leaked:
        failures.append(f"helper columns not dropped: {leaked}")

    if df.height == 0:
        # Schema-valid empty output is a per-wave PASS (filters can wipe a tiny
        # fixture). The suite-level "no wave produced rows" guard catches a broken
        # chain. No value checks possible.
        return failures, stats

    leven = df["leven_norm"].drop_nulls()
    if leven.len():
        lmin, lmax = float(leven.min()), float(leven.max())
        stats["leven"] = (round(lmin, 3), round(lmax, 3))
        if lmin < -1e-6 or lmax > 1 + 1e-6:
            failures.append(f"leven_norm out of [0,1]: [{lmin}, {lmax}]")

    for col in ("tkratio", "tksortratio"):
        s = df[col].drop_nulls()
        if s.len() and (float(s.min()) < -1e-6 or float(s.max()) > 100 + 1e-6):
            failures.append(f"{col} out of [0,100]: [{float(s.min())}, {float(s.max())}]")

    # filter 9 (record-level age band). age_at_pat itself is dropped, so recompute
    # it from the surviving columns and assert every row is inside [18, 80].
    null_age = df.filter(pl.col("pubyear").is_null() | pl.col("birthyr").is_null())
    if null_age.height:
        failures.append(f"{null_age.height} rows have null pubyear/birthyr (filter 9 needs both)")
    age = df.with_columns((pl.col("pubyear") - pl.col("birthyr")).alias("_age"))
    bad_age = age.filter((pl.col("_age") < 18) | (pl.col("_age") > 80))
    if bad_age.height:
        amin = int(age["_age"].min()) if age.height else None
        amax = int(age["_age"].max()) if age.height else None
        failures.append(f"{bad_age.height} rows with age_at_pat outside [18,80] "
                        f"(range [{amin},{amax}])")

    bad_year = df.filter(pl.col("census_year") != year)
    if bad_year.height:
        failures.append(f"census_year != {year} for {bad_year.height} rows")

    bad_gender = df.filter(
        pl.col("gender_census").is_not_null()
        & (pl.col("gender_census") != "M") & (pl.col("gender_census") != "F"))
    if bad_gender.height:
        failures.append(f"{bad_gender.height} rows with gender_census not in {{M,F,null}}")

    if with_golden:
        # true_histid is the correct histid (a UUID string) for golden patent-inventors,
        # else null -> dtype must be Utf8.
        if df["true_histid"].dtype != pl.Utf8:
            failures.append(f"true_histid dtype {df['true_histid'].dtype} != Utf8")
        # true_match is the training label: 1 (true) / 0 (competing golden candidate) /
        # null (unlabeled) -> non-null values must be in {0,1}.
        tm = df["true_match"].drop_nulls()
        if tm.len() and not set(tm.unique().to_list()).issubset({0, 1}):
            failures.append(f"true_match not in {{0,1,null}}: {sorted(set(tm.unique().to_list()))}")

    # 2b writes .unique() output -> no fully-duplicate rows.
    if df.n_unique() != df.height:
        failures.append(f"non-unique rows: {df.height} rows, {df.n_unique()} distinct")

    # Missing county distance is allowed (2b only warns); report as a stat.
    stats["missing_dist"] = df.filter(pl.col("county_distance").is_null()).height

    return failures, stats


def assert_deterministic(p1, p2):
    """The two 2b runs must be row-identical after sorting (byte-stable output).
    Returns a failure string or None."""
    a = pl.read_parquet(p1)
    b = pl.read_parquet(p2)
    if a.shape != b.shape:
        return f"reruns differ in shape: {a.shape} vs {b.shape}"
    a = a.sort(by=a.columns)
    b = b.sort(by=b.columns)
    if not a.equals(b):
        return "two 2b runs differ after sort (non-deterministic output)"
    return None


def run_wave(year, inv_tsv, census_parquet, args, workdir, prep):
    """Chain 1a+1b+1c -> 2a -> 2b for one wave and validate. Returns a result dict."""
    blocking_dir = os.path.join(workdir, f"{year}_blocking")
    out_dir = os.path.join(workdir, f"{year}_out")
    os.makedirs(blocking_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    stats = dict(prep)
    with_golden = (not args.no_golden) and bool(args.golden)

    t0 = time.time()
    fatal, country_ok = run_blocking(year, inv_tsv, census_parquet, blocking_dir, args)
    stats["country_ok"] = country_ok
    if fatal:
        return {"year": year, "status": "FAIL", "secs": round(time.time() - t0, 1),
                "error": "; ".join(fatal), "stats": stats}

    cand = os.path.join(out_dir, f"candidates_{year}.parquet")
    rc, err = run_2a(blocking_dir, year, cand, 1, args)
    if rc != 0:
        return {"year": year, "status": "FAIL", "secs": round(time.time() - t0, 1),
                "error": f"2a: {err}", "stats": stats}

    raw_fixture = os.path.join(workdir, f"{year}_raw_census.parquet")
    hh_fixture = os.path.join(workdir, f"{year}_household_census.parquet")
    try:
        stats["raw_rows"] = slice_raw_census(args.raw_census_dir, year,
                                             census_parquet, raw_fixture)
        # Household-file slice so 2b's county_distance recovery (countyicp->FIPS) runs.
        stats["hh_rows"] = slice_household_census(args.raw_census_dir, year,
                                                  raw_fixture, hh_fixture)
    except Exception as exc:  # noqa: BLE001 - any fixture failure is a FAIL
        return {"year": year, "status": "FAIL", "secs": round(time.time() - t0, 1),
                "error": f"raw/household-census slice failed: {exc}", "stats": stats}

    golden = args.golden if with_golden else None
    out_1 = os.path.join(out_dir, f"matched_{year}_a.parquet")
    out_2 = os.path.join(out_dir, f"matched_{year}_b.parquet")
    failures = []
    for out_path in (out_1, out_2):
        rc, err = run_2b(cand, inv_tsv, census_parquet, raw_fixture, hh_fixture,
                         args.county_distances_dir, golden, year, out_path)
        if rc != 0:
            failures.append(f"2b: {err}")
            break
    secs = round(time.time() - t0, 1)
    if failures:
        return {"year": year, "status": "FAIL", "secs": secs,
                "error": "; ".join(failures), "stats": stats}

    out_fails, out_stats = validate_output(out_1, year, with_golden)
    stats.update(out_stats)
    if not out_fails:
        det = assert_deterministic(out_1, out_2)
        if det:
            out_fails.append(det)

    return {"year": year, "status": "PASS" if not out_fails else "FAIL",
            "secs": secs, "error": "; ".join(out_fails), "stats": stats}


def run_all(args):
    """Build fixtures and run every wave; return (results, ok)."""
    workdir = args.workdir or tempfile.mkdtemp(prefix="stage2b_smoke_")
    os.makedirs(workdir, exist_ok=True)
    with_golden = (not args.no_golden) and bool(args.golden)
    print(f"Stage-2b smoke test | {len(args.years)} waves | gender {GENDER} "
          f"| top-{args.top_counties} counties | census-cap {args.census_cap} "
          f"| device {args.device} | golden={'on' if with_golden else 'off'} "
          f"| chained 1a+1b(county,state)+1c -> 2a -> 2b | workdir={workdir}")

    if not os.path.exists(args.inventors):
        return [{"year": "-", "status": "FAIL", "secs": 0.0,
                 "error": f"missing inventor TSV: {args.inventors}", "stats": {}}], False
    if not os.path.isdir(args.county_distances_dir):
        return [{"year": "-", "status": "FAIL", "secs": 0.0,
                 "error": f"missing county-distances dir: {args.county_distances_dir}", "stats": {}}], False
    if with_golden and not os.path.exists(args.golden):
        return [{"year": "-", "status": "FAIL", "secs": 0.0,
                 "error": f"missing golden CSV: {args.golden}", "stats": {}}], False

    print("loading inventor table once (gender F) ...", flush=True)
    inv = load_inventors(args.inventors, GENDER, "county")
    print(f"  inventor records (gender {GENDER}, US, known FIPS): {len(inv)}\n", flush=True)

    results = []
    for year in args.years:
        census_src = os.path.join(args.census_dir, f"US{year}P.parquet")
        raw_src = os.path.join(args.raw_census_dir, f"US{year}P.parquet")
        print(f"[{year}] building fixtures ...", flush=True)

        if not os.path.exists(census_src):
            results.append({"year": year, "status": "FAIL", "secs": 0.0,
                            "error": f"missing census: {census_src}", "stats": {}})
            print(f"    -> FAIL  missing census: {census_src}", flush=True)
            continue
        if not os.path.exists(raw_src):
            results.append({"year": year, "status": "FAIL", "secs": 0.0,
                            "error": f"missing raw census: {raw_src}", "stats": {}})
            print(f"    -> FAIL  missing raw census: {raw_src}", flush=True)
            continue

        counties = pick_counties(inv, year, args.top_counties)
        if not counties:
            results.append({"year": year, "status": "SKIP", "secs": 0.0,
                            "error": "no female inventors in window", "stats": {}})
            print("    -> SKIP  no female inventors in +/-10yr window", flush=True)
            continue

        inv_tsv = os.path.join(workdir, f"{year}_inventors.tsv")
        census_parquet = os.path.join(workdir, f"{year}_census.parquet")
        try:
            inv_rows = write_inventor_fixture(inv, counties, inv_tsv)
            cen_rows = slice_census(census_src, counties, args.census_cap, census_parquet)
        except Exception as exc:  # noqa: BLE001 - any fixture failure is a FAIL
            results.append({"year": year, "status": "FAIL", "secs": 0.0,
                            "error": f"fixture build failed: {exc}", "stats": {}})
            print(f"    -> FAIL  fixture build failed: {exc}", flush=True)
            continue

        prep = {"counties": len(counties), "inv_rows": inv_rows, "cen_rows": cen_rows}
        print(f"    fixtures: {len(counties)} counties, {inv_rows} inventors, "
              f"{cen_rows} census rows", flush=True)

        res = run_wave(year, inv_tsv, census_parquet, args, workdir, prep)
        results.append(res)
        s = res["stats"]
        line = (f"    -> {res['status']}  ({res['secs']}s)  "
                f"out_rows={s.get('out_rows', '-')} raw={s.get('raw_rows', '-')} "
                f"hh={s.get('hh_rows', '-')} "
                f"1c={'ok' if s.get('country_ok') else 'skip'}")
        if "leven" in s:
            line += f" leven={list(s['leven'])}"
        if s.get("missing_dist"):
            line += f" missing_dist={s['missing_dist']}"
        print(line, flush=True)
        if res["error"]:
            print(f"       {res['error']}", flush=True)

    _print_summary(results)
    n_pass = sum(r["status"] == "PASS" for r in results)
    n_fail = sum(r["status"] == "FAIL" for r in results)
    n_skip = sum(r["status"] == "SKIP" for r in results)
    n_matched = sum(r["stats"].get("out_rows", 0) > 0 for r in results)

    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped "
          f"| {n_matched} wave(s) produced matched candidates")
    if n_fail == 0 and n_matched == 0:
        print("WARNING: no wave produced any matched candidates; the chain may be broken")

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"fixtures kept in {workdir}")

    ok = (n_fail == 0) and (n_matched > 0)
    return results, ok


def _print_summary(results):
    print("\n" + "=" * 80)
    print(f"{'YEAR':<6}{'STATUS':<8}{'SECS':>7}  "
          f"{'INV':>7} {'CEN':>7} {'RAW':>7} {'OUT_ROWS':>9} {'1C':>4}")
    print("-" * 80)
    for r in results:
        s = r["stats"]
        country = "ok" if s.get("country_ok") else ("-" if "country_ok" not in s else "skip")
        print(f"{str(r['year']):<6}{r['status']:<8}{r['secs']:>7}  "
              f"{s.get('inv_rows', '-'):>7} {s.get('cen_rows', '-'):>7} "
              f"{s.get('raw_rows', '-'):>7} {s.get('out_rows', '-'):>9} {country:>4}")
    print("=" * 80)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.exists(SCRIPT_2B):
        sys.exit(f"cannot find {SCRIPT_2B}")

    # Always persist the run to a timestamped log (tee'd from the console).
    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"stage2b_smoke_{stamp}.log")

    logfile = open(log_path, "w")
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(real_stdout, logfile)
    sys.stderr = Tee(real_stderr, logfile)
    try:
        print(f"log: {log_path}\n")
        results, ok = run_all(args)
        print(f"\nlog written to {log_path}")
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr
        logfile.close()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
