#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke test for stage 2a (`2a_collect_candidates.py`) across all census waves.

Stage 2a collects the blocking outputs into one deduplicated candidate table. It
reads nothing but blocking parquets, so this test is CHAINED: per wave it builds
one tiny shared fixture, runs all four blocking producers into a single dir, then
runs 2a over them -- exercising the real stage-1 -> 2a seam end to end.

Per wave:
  * fixture   -> reuse the 1a/1b/1c strategy: pick the top-K counties by female
                 inventor count in the +/-10yr window (via the production
                 `blocking_common.load_inventors`), write the original-schema
                 inventor TSV for those counties, and slice the same counties'
                 cleaned census. One shared fixture feeds all four blockers.
                 `--census-cap` defaults to 10000 so the female slice yields the
                 >= 256 unique names 1c's IVF-PQ needs to train.
  * blocking  -> 1a (county cosine), 1b (county + state Levenshtein) and 1c
                 (nationwide cosine) all write into one `{year}_blocking` dir.
                 1a/1b failure is a wave FAIL; 1c failure is a NON-FATAL warning
                 (sparse waves cannot train IVF-PQ), and 2a still collects from
                 the 1a/1b sources -- the country source is skipped when absent.
  * collect   -> 2a is run twice over the same dir: once with CAND_PARTITIONS=1
                 (single pass) and once with CAND_PARTITIONS=4 (hash-bucket
                 concat). Both must produce row-identical output (validates the
                 memory-bounding equivalence claim in collect_common). 2a's
                 --sources is left at its default (all four), so the
                 country-by-default fold-in is exercised.

For each wave it validates the 2a output parquet -- the 8-column schema/order,
`cosine_score` in [-1, 1] (1c recomputes an exact cosine, so it may be negative),
`levenshtein_score` in [0, 1], no row missing BOTH scores, correct census_year,
and exactly one row per (histid, PatInvID, pubnumber) triple -- and prints a
summary. The suite passes only if nothing failed AND at least one wave produced
candidates. Exits non-zero if any run fails.

Usage:
    conda activate ipums_match
    python smoke_test_stage2a.py [--years 1900 1910] [--census-cap 10000] ...

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
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SRC = os.path.join(REPO_ROOT, "src")
SCRIPT_1A = os.path.join(SRC, "1a_block_faiss_cosine.py")
SCRIPT_1B = os.path.join(SRC, "1b_block_rapidfuzz_levenshtein.py")
SCRIPT_1C = os.path.join(SRC, "1c_block_faiss_cosine_country.py")
SCRIPT_2A = os.path.join(SRC, "2a_collect_candidates.py")
LOG_DIR_DEFAULT = os.path.join(HERE, "logs")

# Import the production inventor loader so county selection and the fixture rows
# use exactly the same filtering as the blocking scripts under test.
sys.path.insert(0, SRC)
from blocking_common import load_inventors  # noqa: E402


class Tee:
    """Write everything to the real stdout *and* a log file, flushing both."""

    def __init__(self, stream, logfile):
        self._stream = stream
        self._logfile = logfile

    def write(self, data):
        self._stream.write(data)
        self._logfile.write(data)
        self.flush()

    def flush(self):
        self._stream.flush()
        self._logfile.flush()


DEFAULT_YEARS = [1850, 1860, 1870, 1880, 1900, 1910, 1920, 1930, 1940]
GENDER = "F"

# Original published inventor-table columns (written verbatim to the fixture TSV
# so the scripts' own load_inventors reprocesses it identically).
INVENTOR_COLS = ["PatInvID", "pubnumber", "progr", "pubdate", "pubyear", "name",
                 "citizen", "country", "state", "county", "city", "FIPS", "gender"]

# Cleaned-census columns the blocking scripts read (drops `potential_match`).
CENSUS_COLS = ["histid", "clean_fn", "clean_ln", "county_fips_code",
               "age", "birthyr", "sex", "city"]

# 2a's fixed output schema = collect_common.OUT_COLS + census_year (order matters).
EXPECTED_OUT_COLS = ["histid", "PatInvID", "pubnumber", "inventor_name",
                     "census_name", "cosine_score", "levenshtein_score", "census_year"]
MERGE_KEYS = ["histid", "PatInvID", "pubnumber"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Smoke test 2a_collect_candidates.py by chaining 1a+1b+1c -> 2a per wave.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--inventors",
                   default=os.path.join(REPO_ROOT, "data",
                                        "USPTO_patent_inventors_1840_1950.tsv"),
                   help="Path to the published patent-inventor TSV.")
    p.add_argument("--census-dir", default="/scratch/breschi/cleaned_census",
                   help="Directory with cleaned US{year}P.parquet files (stage-0 output).")
    p.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS,
                   help="Census waves to test.")
    p.add_argument("--top-counties", type=int, default=3,
                   help="Number of highest-inventor counties to slice into each fixture.")
    p.add_argument("--census-cap", type=int, default=10000,
                   help="Max cleaned-census rows per wave fixture (large enough for 1c IVF-PQ training).")
    p.add_argument("--device", choices=["gpu", "cpu"], default="cpu",
                   help="FAISS device for 1a/1c (cpu is portable; identical results).")
    # 1a / 1c knobs.
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


def pick_counties(inv, year, top_k):
    """Top-K county FIPS by inventor count inside the wave's +/-10yr window."""
    window = inv[(inv["year"] >= year - 10) & (inv["year"] <= year + 10)]
    counts = window["geo_key"].dropna().value_counts()
    return counts.head(top_k).index.tolist()


def write_inventor_fixture(inv, counties, out_tsv):
    """Write the original-schema inventor rows for `counties` to a small TSV."""
    sub = inv[inv["geo_key"].isin(counties)]
    cols = [c for c in INVENTOR_COLS if c in sub.columns]
    sub[cols].to_csv(out_tsv, sep="\t", index=False)
    return len(sub)


def slice_census(census_path, counties, cap, out_parquet):
    """First `cap` cleaned-census rows restricted to `counties` (overlap-guaranteed)."""
    dataset = ds.dataset(census_path)
    flt = pc.field("county_fips_code").isin(counties)
    batch_size = min(max(cap, 1024), 65536)
    table = dataset.scanner(columns=CENSUS_COLS, filter=flt, batch_size=batch_size).head(cap)
    pq.write_table(table, out_parquet)
    return table.num_rows


def _run(cmd):
    """Run a blocking/collect subprocess; return (returncode, last_error_line)."""
    proc = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return proc.returncode, (tail[-1] if tail else f"exit {proc.returncode}")
    return 0, ""


def run_blocking(year, inv_tsv, census_parquet, blocking_dir, args):
    """Run 1a + 1b(county) + 1b(state) + 1c into `blocking_dir`.

    Returns (failures, country_ok): `failures` lists fatal (1a/1b) errors;
    `country_ok` is False if 1c failed (non-fatal -- 2a collects 1a/1b only)."""
    failures = []

    # 1a -- county cosine.
    cmd_1a = [sys.executable, SCRIPT_1A,
              "--inventors", inv_tsv, "--census", census_parquet,
              "--census-year", str(year), "--gender", GENDER,
              "--geo-level", "county",
              "--device", args.device, "--k", str(args.k),
              "--n-cores", str(args.n_cores),
              "--output-dir", blocking_dir]
    if args.max_counties is not None:
        cmd_1a += ["--max-counties", str(args.max_counties)]
    rc, err = _run(cmd_1a)
    if rc != 0:
        failures.append(f"1a: {err}")

    # 1b -- county and state Levenshtein (same fixture, two geo-levels).
    for geo in ("county", "state"):
        cmd_1b = [sys.executable, SCRIPT_1B,
                  "--inventors", inv_tsv, "--census", census_parquet,
                  "--census-year", str(year), "--gender", GENDER,
                  "--geo-level", geo,
                  "--limit", str(args.limit), "--score-cutoff", str(args.score_cutoff),
                  "--max-workers", str(args.max_workers), "--n-cores", str(args.n_cores),
                  "--output-dir", blocking_dir]
        if args.max_counties is not None:
            cmd_1b += ["--max-counties", str(args.max_counties)]
        rc, err = _run(cmd_1b)
        if rc != 0:
            failures.append(f"1b/{geo}: {err}")

    # 1c -- nationwide cosine (non-fatal: sparse waves cannot train IVF-PQ).
    cmd_1c = [sys.executable, SCRIPT_1C,
              "--inventors", inv_tsv, "--census", census_parquet,
              "--census-year", str(year), "--genders", GENDER,
              "--device", args.device, "--k", str(args.k),
              "--nlist", str(args.nlist), "--pq-m", str(args.pq_m),
              "--svd-dim", str(args.svd_dim), "--svd-sample", str(args.svd_sample),
              "--fit-sample", str(args.fit_sample), "--nprobe", str(args.nprobe),
              "--n-cores", str(args.n_cores),
              "--output-dir", blocking_dir]
    rc, err = _run(cmd_1c)
    country_ok = rc == 0
    if rc != 0:
        print(f"       (1c non-fatal) {err}", flush=True)

    return failures, country_ok


def run_2a(blocking_dir, year, out_path, n_parts, args):
    """Run 2a over `blocking_dir` with CAND_PARTITIONS=n_parts. Returns (rc, err)."""
    cmd = [sys.executable, SCRIPT_2A,
           "--blocking-dir", blocking_dir,
           "--census-year", str(year),
           "--genders", GENDER,
           "--output", out_path]  # --sources defaults to all four (country folded in)
    env = dict(os.environ, CAND_PARTITIONS=str(n_parts))
    proc = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return proc.returncode, (tail[-1] if tail else f"exit {proc.returncode}")
    return 0, ""


def validate_output(out_path, year, expect_country):
    """Return a list of failure messages (empty == PASS) plus a stats dict."""
    failures = []
    df = pl.read_parquet(out_path)
    stats = {"out_rows": df.height}

    if df.columns != EXPECTED_OUT_COLS:
        failures.append(f"columns {df.columns} != expected {EXPECTED_OUT_COLS}")
        return failures, stats  # schema wrong: skip value checks

    if df.height == 0:
        # 1a/1b succeeded over overlapping fixtures, so an empty collect is a bug.
        failures.append("collected 0 candidates (expected >=1 after 1a/1b)")
        return failures, stats

    stats["triples"] = df.select(MERGE_KEYS).n_unique()

    cos = df["cosine_score"].drop_nulls()
    if cos.len():
        cmin, cmax = float(cos.min()), float(cos.max())
        stats["cos_min"], stats["cos_max"] = round(cmin, 3), round(cmax, 3)
        # 1c recomputes the exact cosine (L2-normalised SVD embeddings) so values
        # can be negative; the fold-in means cosine_score ranges over [-1, 1].
        if cmin < -1.0001 or cmax > 1.0001:
            failures.append(f"cosine_score outside [-1,1]: [{cmin}, {cmax}]")

    lev = df["levenshtein_score"].drop_nulls()
    if lev.len():
        lmin, lmax = float(lev.min()), float(lev.max())
        stats["lev_min"], stats["lev_max"] = round(lmin, 3), round(lmax, 3)
        if lmin < -1e-6 or lmax > 1 + 1e-6:
            failures.append(f"levenshtein_score out of [0,1]: [{lmin}, {lmax}]")

    # Every retained pair must come from at least one metric.
    both_null = df.filter(
        pl.col("cosine_score").is_null() & pl.col("levenshtein_score").is_null())
    if both_null.height:
        failures.append(f"{both_null.height} rows have BOTH scores null")

    bad_year = df.filter(pl.col("census_year") != year)
    if bad_year.height:
        failures.append(f"census_year != {year} for {bad_year.height} rows")

    # 2a contract: exactly one row per (histid, PatInvID, pubnumber) triple.
    if stats["triples"] != df.height:
        failures.append(f"non-unique triples: {df.height} rows, {stats['triples']} triples")

    if expect_country and cos.len() == 0:
        failures.append("country source expected but no non-null cosine_score present")

    return failures, stats


def assert_partition_equiv(p1, p4):
    """Single-pass (p1) and 4-bucket (p4) outputs must be row-identical. Returns
    a failure string or None."""
    a = pl.read_parquet(p1).sort(by=EXPECTED_OUT_COLS)
    b = pl.read_parquet(p4).sort(by=EXPECTED_OUT_COLS)
    if a.shape != b.shape:
        return f"partition shapes differ: {a.shape} (parts=1) vs {b.shape} (parts=4)"
    if not a.equals(b):
        return "single-pass vs 4-bucket outputs differ after sort"
    return None


def run_wave(year, inv_tsv, census_parquet, args, workdir, prep):
    """Chain 1a+1b+1c -> 2a for one wave and validate. Returns a result dict."""
    blocking_dir = os.path.join(workdir, f"{year}_blocking")
    out_dir = os.path.join(workdir, f"{year}_out")
    os.makedirs(blocking_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    stats = dict(prep)  # counties / inv_rows / cen_rows carried from fixture build

    t0 = time.time()
    fatal, country_ok = run_blocking(year, inv_tsv, census_parquet, blocking_dir, args)
    stats["country_ok"] = country_ok
    if fatal:
        secs = round(time.time() - t0, 1)
        return {"year": year, "status": "FAIL", "secs": secs,
                "error": "; ".join(fatal), "stats": stats}

    out_p1 = os.path.join(out_dir, f"candidates_{year}_p1.parquet")
    out_p4 = os.path.join(out_dir, f"candidates_{year}_p4.parquet")
    failures = []
    for out_path, n_parts in ((out_p1, 1), (out_p4, 4)):
        rc, err = run_2a(blocking_dir, year, out_path, n_parts, args)
        if rc != 0:
            failures.append(f"2a(parts={n_parts}): {err}")
    secs = round(time.time() - t0, 1)

    if failures:
        return {"year": year, "status": "FAIL", "secs": secs,
                "error": "; ".join(failures), "stats": stats}

    expect_country = country_ok  # --sources default includes cosine_country
    out_fails, out_stats = validate_output(out_p1, year, expect_country)
    stats.update(out_stats)
    if not out_fails:
        equiv = assert_partition_equiv(out_p1, out_p4)
        if equiv:
            out_fails.append(equiv)

    return {"year": year,
            "status": "PASS" if not out_fails else "FAIL",
            "secs": secs, "error": "; ".join(out_fails), "stats": stats}


def run_all(args):
    """Build fixtures and run every wave; return (results, ok)."""
    workdir = args.workdir or tempfile.mkdtemp(prefix="stage2a_smoke_")
    os.makedirs(workdir, exist_ok=True)
    print(f"Stage-2a smoke test | {len(args.years)} waves | gender {GENDER} "
          f"| top-{args.top_counties} counties | census-cap {args.census_cap} "
          f"| device {args.device} | chained 1a+1b(county,state)+1c -> 2a "
          f"| workdir={workdir}")

    if not os.path.exists(args.inventors):
        print(f"missing inventor TSV: {args.inventors}")
        return [{"year": "-", "status": "FAIL", "secs": 0.0,
                 "error": f"missing inventor TSV: {args.inventors}", "stats": {}}], False

    print("loading inventor table once (gender F) ...", flush=True)
    inv = load_inventors(args.inventors, GENDER, "county")
    print(f"  inventor records (gender {GENDER}, US, known FIPS): {len(inv)}\n", flush=True)

    results = []
    for year in args.years:
        census_src = os.path.join(args.census_dir, f"US{year}P.parquet")
        print(f"[{year}] building fixtures ...", flush=True)

        if not os.path.exists(census_src):
            results.append({"year": year, "status": "FAIL", "secs": 0.0,
                            "error": f"missing census: {census_src}", "stats": {}})
            print(f"    -> FAIL  missing census: {census_src}", flush=True)
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
                f"out_rows={s.get('out_rows', '-')} triples={s.get('triples', '-')} "
                f"1c={'ok' if s.get('country_ok') else 'skip'}")
        if "cos_min" in s:
            line += f" cos=[{s['cos_min']},{s['cos_max']}]"
        if "lev_min" in s:
            line += f" lev=[{s['lev_min']},{s['lev_max']}]"
        print(line, flush=True)
        if res["error"]:
            print(f"       {res['error']}", flush=True)

    _print_summary(results)
    n_pass = sum(r["status"] == "PASS" for r in results)
    n_fail = sum(r["status"] == "FAIL" for r in results)
    n_skip = sum(r["status"] == "SKIP" for r in results)
    n_matched = sum(r["stats"].get("out_rows", 0) > 0 for r in results)

    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped "
          f"| {n_matched} wave(s) produced candidates")
    if n_fail == 0 and n_matched == 0:
        print("WARNING: no wave produced any candidates; the chain may be broken")

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"fixtures kept in {workdir}")

    # Suite passes only if nothing failed AND at least one wave produced candidates.
    ok = (n_fail == 0) and (n_matched > 0)
    return results, ok


def _print_summary(results):
    print("\n" + "=" * 78)
    print(f"{'YEAR':<6}{'STATUS':<8}{'SECS':>7}  "
          f"{'INV':>7} {'CEN':>7} {'OUT_ROWS':>9} {'TRIPLES':>9} {'1C':>4}")
    print("-" * 78)
    for r in results:
        s = r["stats"]
        country = "ok" if s.get("country_ok") else ("-" if "country_ok" not in s else "skip")
        print(f"{str(r['year']):<6}{r['status']:<8}{r['secs']:>7}  "
              f"{s.get('inv_rows', '-'):>7} {s.get('cen_rows', '-'):>7} "
              f"{s.get('out_rows', '-'):>9} {s.get('triples', '-'):>9} {country:>4}")
    print("=" * 78)


def main(argv=None):
    args = parse_args(argv)

    for script in (SCRIPT_1A, SCRIPT_1B, SCRIPT_1C, SCRIPT_2A):
        if not os.path.exists(script):
            sys.exit(f"cannot find {script}")

    # Always persist the run to a timestamped log (tee'd from the console).
    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"stage2a_smoke_{stamp}.log")

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
