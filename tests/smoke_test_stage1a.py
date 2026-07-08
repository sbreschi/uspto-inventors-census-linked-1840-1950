#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke test for stage 1a (`1a_block_faiss_cosine.py`) across all census waves.

A full 1a run can take hours: it builds a *flat* (brute-force) FAISS index whose
cost is O(n_queries x n_database), and cleans every inventor in a +/-10 year
window. To check "does 1a run end-to-end and write a schema-correct parquet" on
every census wave in a few minutes, this harness builds a tiny coherent fixture
per wave and runs the unmodified script via subprocess.

The stage-0 "first N rows" slice trick does NOT work here: the cleaned census is
serial-ordered by FIPS, so the first N rows land on one rural county with ~zero
inventor overlap (empty output). Instead, for each wave the fixture is built
data-driven so census and inventors share real geography:

  * pick counties -> the top-K county FIPS by *female* inventor count inside the
                     wave's +/-10yr window (so the bucket is guaranteed to have
                     inventors). Selection reuses the production
                     `blocking_common.load_inventors` loader (no logic drift).
  * inventors     -> the original-schema TSV rows for those counties (tiny; the
                     script re-reads and reprocesses it identically).
  * census        -> first `--census-cap` cleaned rows of those same counties
                     (`scanner(filter=...).head(N)`), so both sides overlap.

Coverage: gender F (far fewer inventors than M -> tiny, fast) at both
`--geo-level county` and `--geo-level state`. The same per-wave fixtures serve
both geo-levels (the script rolls county FIPS up to state itself).

For each (wave, geo-level) it validates the output parquet (8-column schema,
non-empty cosine values in [0, 1], correct census_year) and prints a summary.
A schema-valid but empty output is a PASS (early waves can have ~no female
inventors); the suite still asserts that at least one run produced matches.
Exits non-zero if any run fails.

Usage:
    conda activate ipums_match
    python smoke_test_stage1a.py [--years 1900 1910] [--top-counties 3] ...

"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SRC = os.path.join(REPO_ROOT, "src")
SCRIPT = os.path.join(SRC, "1a_block_faiss_cosine.py")
LOG_DIR_DEFAULT = os.path.join(HERE, "logs")

# Import the production inventor loader so county selection and the fixture rows
# use exactly the same filtering as the script under test.
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
# so the script's own load_inventors reprocesses it identically).
INVENTOR_COLS = ["PatInvID", "pubnumber", "progr", "pubdate", "pubyear", "name",
                 "citizen", "country", "state", "county", "city", "FIPS", "gender"]

# Cleaned-census columns the script actually reads (drops `potential_match`).
CENSUS_COLS = ["histid", "clean_fn", "clean_ln", "county_fips_code",
               "age", "birthyr", "sex", "city"]

EXPECTED_OUT_COLS = ["inventor_name", "census_name", "Cosine_Similarity",
                     "PatInvID", "pubnumber", "histid", "census_year", "geo_key"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Smoke test 1a_block_faiss_cosine.py on a tiny fixture of every census wave.",
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
    p.add_argument("--geo-levels", nargs="+", default=["county", "state"],
                   choices=["county", "state"],
                   help="Geographic blocking keys to exercise per wave.")
    p.add_argument("--top-counties", type=int, default=3,
                   help="Number of highest-inventor counties to slice into each fixture.")
    p.add_argument("--census-cap", type=int, default=3000,
                   help="Max cleaned-census rows per wave fixture (across the picked counties).")
    p.add_argument("--k", type=int, default=5,
                   help="Nearest census neighbours per inventor name (smaller = faster).")
    p.add_argument("--n-cores", type=int, default=2,
                   help="Process count passed to the script for name cleaning.")
    p.add_argument("--device", choices=["gpu", "cpu"], default="cpu",
                   help="FAISS device for the script (cpu is portable; identical results).")
    p.add_argument("--max-counties", type=int, default=None,
                   help="Optional cap on geographic buckets the script processes.")
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


def validate_output(out_path, year):
    """Return a list of failure messages (empty == PASS) plus a stats dict."""
    failures = []
    df = pq.read_table(out_path).to_pandas()
    stats = {"out_rows": len(df)}

    if list(df.columns) != EXPECTED_OUT_COLS:
        failures.append(f"columns {list(df.columns)} != expected {EXPECTED_OUT_COLS}")
        return failures, stats  # schema wrong: skip value checks

    if len(df) == 0:
        # Schema-valid but empty is acceptable (sparse wave); record and move on.
        return failures, stats

    cos = df["Cosine_Similarity"].astype(float)
    stats["cos_min"] = round(float(cos.min()), 3)
    stats["cos_max"] = round(float(cos.max()), 3)
    if cos.min() < -1e-6 or cos.max() > 1 + 1e-6:
        failures.append(f"Cosine_Similarity out of [0,1]: [{cos.min()}, {cos.max()}]")

    bad_year = df[df["census_year"] != year]
    if len(bad_year) > 0:
        failures.append(f"census_year != {year} for {len(bad_year)} rows")

    return failures, stats


def run_combo(year, geo, inv_tsv, census_parquet, args, workdir, prep):
    """Run the script for one (wave, geo-level) and validate. Returns a result dict."""
    out_dir = os.path.join(workdir, f"{year}_{geo}_out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cosine_match_{geo}_{year}_{GENDER}.parquet")

    cmd = [sys.executable, SCRIPT,
           "--inventors", inv_tsv,
           "--census", census_parquet,
           "--census-year", str(year),
           "--gender", GENDER,
           "--geo-level", geo,
           "--device", args.device,
           "--k", str(args.k),
           "--n-cores", str(args.n_cores),
           "--output-dir", out_dir]
    if args.max_counties is not None:
        cmd += ["--max-counties", str(args.max_counties)]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
    secs = round(time.time() - t0, 1)
    stats = dict(prep)  # inv_rows / cen_rows / counties carried from fixture build

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"year": year, "geo": geo, "status": "FAIL", "secs": secs,
                "error": tail[-1] if tail else f"exit {proc.returncode}", "stats": stats}

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return {"year": year, "geo": geo, "status": "FAIL", "secs": secs,
                "error": f"no output written: {out_path}", "stats": stats}

    failures, out_stats = validate_output(out_path, year)
    stats.update(out_stats)
    return {"year": year, "geo": geo,
            "status": "PASS" if not failures else "FAIL",
            "secs": secs, "error": "; ".join(failures), "stats": stats}


def run_all(args):
    """Build fixtures and run every (wave, geo-level); return (results, n_pass)."""
    workdir = args.workdir or tempfile.mkdtemp(prefix="stage1a_smoke_")
    os.makedirs(workdir, exist_ok=True)
    print(f"Stage-1a smoke test | {len(args.years)} waves | gender {GENDER} "
          f"| geo {args.geo_levels} | top-{args.top_counties} counties "
          f"| census-cap {args.census_cap} | device {args.device} | workdir={workdir}")

    if not os.path.exists(args.inventors):
        print(f"missing inventor TSV: {args.inventors}")
        return [{"year": "-", "geo": "-", "status": "FAIL", "secs": 0.0,
                 "error": f"missing inventor TSV: {args.inventors}", "stats": {}}], 0

    print("loading inventor table once (gender F) ...", flush=True)
    inv = load_inventors(args.inventors, GENDER, "county")
    print(f"  inventor records (gender {GENDER}, US, known FIPS): {len(inv)}\n", flush=True)

    results = []
    for year in args.years:
        census_src = os.path.join(args.census_dir, f"US{year}P.parquet")
        print(f"[{year}] building fixtures ...", flush=True)

        if not os.path.exists(census_src):
            for geo in args.geo_levels:
                results.append({"year": year, "geo": geo, "status": "FAIL", "secs": 0.0,
                                "error": f"missing census: {census_src}", "stats": {}})
            print(f"    -> FAIL  missing census: {census_src}", flush=True)
            continue

        counties = pick_counties(inv, year, args.top_counties)
        if not counties:
            for geo in args.geo_levels:
                results.append({"year": year, "geo": geo, "status": "SKIP", "secs": 0.0,
                                "error": "no female inventors in window", "stats": {}})
            print("    -> SKIP  no female inventors in +/-10yr window", flush=True)
            continue

        inv_tsv = os.path.join(workdir, f"{year}_inventors.tsv")
        census_parquet = os.path.join(workdir, f"{year}_census.parquet")
        try:
            inv_rows = write_inventor_fixture(inv, counties, inv_tsv)
            cen_rows = slice_census(census_src, counties, args.census_cap, census_parquet)
        except Exception as exc:  # noqa: BLE001 - any fixture failure is a FAIL
            for geo in args.geo_levels:
                results.append({"year": year, "geo": geo, "status": "FAIL", "secs": 0.0,
                                "error": f"fixture build failed: {exc}", "stats": {}})
            print(f"    -> FAIL  fixture build failed: {exc}", flush=True)
            continue

        prep = {"counties": len(counties), "inv_rows": inv_rows, "cen_rows": cen_rows}
        print(f"    fixtures: {len(counties)} counties, {inv_rows} inventors, "
              f"{cen_rows} census rows", flush=True)

        for geo in args.geo_levels:
            res = run_combo(year, geo, inv_tsv, census_parquet, args, workdir, prep)
            results.append(res)
            s = res["stats"]
            line = (f"    -> [{geo}] {res['status']}  ({res['secs']}s)  "
                    f"out_rows={s.get('out_rows', '-')}")
            if "cos_min" in s:
                line += f" cos=[{s['cos_min']},{s['cos_max']}]"
            print(line, flush=True)
            if res["error"]:
                print(f"       {res['error']}", flush=True)

    _print_summary(results)
    n_pass = sum(r["status"] == "PASS" for r in results)
    n_fail = sum(r["status"] == "FAIL" for r in results)
    n_skip = sum(r["status"] == "SKIP" for r in results)
    n_matched = sum(r["stats"].get("out_rows", 0) > 0 for r in results)

    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped "
          f"| {n_matched} run(s) produced matches")
    if n_fail == 0 and n_matched == 0:
        print("WARNING: no run produced any matches; the blocking path may be broken")

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"fixtures kept in {workdir}")

    # Suite passes only if nothing failed AND at least one run produced matches.
    ok = (n_fail == 0) and (n_matched > 0)
    return results, ok


def _print_summary(results):
    print("\n" + "=" * 78)
    print(f"{'YEAR':<6}{'GEO':<8}{'STATUS':<8}{'SECS':>7}  "
          f"{'INV':>7} {'CEN':>7} {'OUT_ROWS':>9}")
    print("-" * 78)
    for r in results:
        s = r["stats"]
        print(f"{str(r['year']):<6}{r['geo']:<8}{r['status']:<8}{r['secs']:>7}  "
              f"{s.get('inv_rows', '-'):>7} {s.get('cen_rows', '-'):>7} "
              f"{s.get('out_rows', '-'):>9}")
    print("=" * 78)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.exists(SCRIPT):
        sys.exit(f"cannot find {SCRIPT}")

    # Always persist the run to a timestamped log (tee'd from the console).
    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"stage1a_smoke_{stamp}.log")

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
