#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke test for stage 0 (`0_preclean_census.py`) across all census years.

The raw IPUMS person/household files are huge directory-format parquet datasets
(tens of millions of rows). To check "does stage 0 run and produce the right
output" on every year *without* loading those files, this harness builds a tiny
coherent fixture per year and runs the unmodified `0_preclean_census.py` on it:

  * person   -> first N rows of US{year}P.parquet (cheap `scanner().head(N)`);
  * household-> first 2N rows of US{year}H.parquet. Both datasets are
                serial-ordered, so these slices share serials and the city join
                populates;
  * crosswalk-> the real data/geo_census_crosswalk/{year}_csv.zip (read in full
                by the script, as in production), or --synthetic-crosswalk to
                fabricate a tiny one from the slice histids for a seconds-long run.

For each year it then validates the output parquet (schema, non-empty, that name
cleaning ran and the city / county-FIPS joins populated) and prints a summary.
Exits non-zero if any year fails.

Usage:
    conda activate faiss_env
    python smoke_test_stage0.py [--synthetic-crosswalk] [--years 1900 1910] ...

"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import pyarrow.dataset as ds
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SRC = os.path.join(REPO_ROOT, "src")
SCRIPT = os.path.join(SRC, "0_preclean_census.py")
LOG_DIR_DEFAULT = os.path.join(HERE, "logs")


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

PERSON_COLS = ["histid", "namefrst", "namelast", "age", "birthyr", "sex", "serialp"]
HOUSEHOLD_COLS = ["serial", "city"]
EXPECTED_OUT_COLS = ["histid", "clean_fn", "clean_ln", "county_fips_code",
                     "age", "birthyr", "sex", "city", "potential_match"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Smoke test 0_preclean_census.py on a tiny slice of every census year.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--census-dir", default="/scratch/breschi/census",
                   help="Directory with US{year}P.parquet / US{year}H.parquet datasets.")
    p.add_argument("--crosswalk-dir",
                   default=os.path.join(REPO_ROOT, "data", "geo_census_crosswalk"),
                   help="Directory with {year}_csv.zip geographic crosswalks.")
    p.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS,
                   help="Census years to test.")
    p.add_argument("--n-rows", type=int, default=3000,
                   help="Number of person rows per year in the slice.")
    p.add_argument("--log-dir", default=LOG_DIR_DEFAULT,
                   help="Directory for the timestamped run log.")
    p.add_argument("--workdir", default=None,
                   help="Where to build fixtures (default: a fresh temp dir).")
    p.add_argument("--keep", action="store_true",
                   help="Keep the fixtures/outputs instead of deleting them.")
    p.add_argument("--synthetic-crosswalk", action="store_true",
                   help="Fabricate a tiny crosswalk from the slice histids instead "
                        "of reading the real multi-GB zip (fast mode).")
    return p.parse_args(argv)


def slice_dataset(path, columns, n_rows, out_path):
    """Cheaply take the first `n_rows` rows of a parquet dataset and write them."""
    dataset = ds.dataset(path)
    batch_size = min(max(n_rows, 1024), 65536)
    table = dataset.scanner(columns=columns, batch_size=batch_size).head(n_rows)
    pq.write_table(table, out_path)
    return table.num_rows


def write_synthetic_crosswalk(person_table, out_path):
    """Build a tiny valid crosswalk CSV from the person slice's real histids."""
    histids = person_table.column("histid").to_pylist()
    with open(out_path, "w") as fh:
        fh.write("histid,state_fips_geomatch,county_fips_geomatch,potential_match\n")
        for i, hid in enumerate(histids):
            if hid is None:
                continue
            # Deterministic, valid-looking FIPS so the merge + zfill path runs.
            state = (i % 56) + 1
            county = (i % 199) * 2 + 1
            fh.write(f"{hid},{state},{county},smoketown\n")
    return out_path


def validate_output(out_path):
    """Return a list of failure messages (empty == PASS) plus a stats dict."""
    failures = []
    table = pq.read_table(out_path)
    df = table.to_pandas()
    stats = {"out_rows": len(df)}

    if list(df.columns) != EXPECTED_OUT_COLS:
        failures.append(f"columns {list(df.columns)} != expected {EXPECTED_OUT_COLS}")
        return failures, stats  # schema wrong: skip value checks

    if len(df) == 0:
        failures.append("output is empty")
        return failures, stats

    fn_ok = df["clean_fn"].dropna()
    fn_ok = fn_ok[fn_ok.str.fullmatch(r"[a-z ]+", na=False)]
    if len(fn_ok) == 0:
        failures.append("no cleaned lowercase-alpha clean_fn values")
    stats["clean_fn_nonnull"] = int(df["clean_fn"].notna().sum())

    city_n = int(df["city"].notna().sum())
    stats["city_nonnull"] = city_n
    if city_n == 0:
        failures.append("city never populated (household join failed)")

    fips = df["county_fips_code"].dropna().astype(str)
    fips_ok = fips[fips.str.fullmatch(r"\d{5}", na=False)]
    stats["fips_5digit"] = int(len(fips_ok))
    if len(fips_ok) == 0:
        failures.append("no 5-digit county_fips_code (crosswalk join failed)")

    return failures, stats


def run_year(year, args, workdir):
    """Build the fixture, run the script, validate. Returns a result dict."""
    person_src = os.path.join(args.census_dir, f"US{year}P.parquet")
    household_src = os.path.join(args.census_dir, f"US{year}H.parquet")
    crosswalk_src = os.path.join(args.crosswalk_dir, f"{year}_csv.zip")

    for path in (person_src, household_src):
        if not os.path.exists(path):
            return {"year": year, "status": "FAIL", "secs": 0.0,
                    "error": f"missing input: {path}", "stats": {}}
    if not args.synthetic_crosswalk and not os.path.exists(crosswalk_src):
        return {"year": year, "status": "FAIL", "secs": 0.0,
                "error": f"missing crosswalk: {crosswalk_src}", "stats": {}}

    person_tmp = os.path.join(workdir, f"{year}_person.parquet")
    household_tmp = os.path.join(workdir, f"{year}_household.parquet")
    out_tmp = os.path.join(workdir, f"{year}_out.parquet")

    t0 = time.time()
    try:
        n_person = slice_dataset(person_src, PERSON_COLS, args.n_rows, person_tmp)
        slice_dataset(household_src, HOUSEHOLD_COLS, args.n_rows * 2, household_tmp)
    except Exception as exc:  # noqa: BLE001 - report any slicing failure as a FAIL
        return {"year": year, "status": "FAIL", "secs": round(time.time() - t0, 1),
                "error": f"slicing failed: {exc}", "stats": {}}

    if args.synthetic_crosswalk:
        crosswalk = os.path.join(workdir, f"{year}_cw.csv")
        write_synthetic_crosswalk(pq.read_table(person_tmp, columns=["histid"]), crosswalk)
    else:
        crosswalk = crosswalk_src

    cmd = [sys.executable, SCRIPT,
           "--person", person_tmp,
           "--household", household_tmp,
           "--crosswalk", crosswalk,
           "--output", out_tmp,
           "--n-cores", "2"]
    proc = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
    secs = round(time.time() - t0, 1)

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"year": year, "status": "FAIL", "secs": secs,
                "error": tail[-1] if tail else f"exit {proc.returncode}",
                "stats": {"in_rows": n_person}}

    if not os.path.exists(out_tmp) or os.path.getsize(out_tmp) == 0:
        return {"year": year, "status": "FAIL", "secs": secs,
                "error": "no output written", "stats": {"in_rows": n_person}}

    failures, stats = validate_output(out_tmp)
    stats["in_rows"] = n_person
    return {"year": year,
            "status": "PASS" if not failures else "FAIL",
            "secs": secs,
            "error": "; ".join(failures),
            "stats": stats}


def run_all(args):
    """Run the smoke test for every year and return (results, n_pass)."""
    workdir = args.workdir or tempfile.mkdtemp(prefix="stage0_smoke_")
    os.makedirs(workdir, exist_ok=True)
    mode = "synthetic crosswalk" if args.synthetic_crosswalk else "real crosswalk"
    print(f"Stage-0 smoke test | {len(args.years)} years | {args.n_rows} person rows "
          f"| {mode} | workdir={workdir}\n")

    results = []
    for year in args.years:
        print(f"[{year}] building fixture and running 0_preclean_census.py ...", flush=True)
        res = run_year(year, args, workdir)
        results.append(res)
        line = f"    -> {res['status']}  ({res['secs']}s)"
        if res["stats"]:
            line += "  " + " ".join(f"{k}={v}" for k, v in res["stats"].items())
        print(line, flush=True)
        if res["error"]:
            print(f"       {res['error']}", flush=True)

    print("\n" + "=" * 72)
    print(f"{'YEAR':<6}{'STATUS':<8}{'SECS':>7}  {'OUT_ROWS':>9} {'CITY':>7} {'FIPS5':>7}")
    print("-" * 72)
    for r in results:
        s = r["stats"]
        print(f"{r['year']:<6}{r['status']:<8}{r['secs']:>7}  "
              f"{s.get('out_rows', '-'):>9} {s.get('city_nonnull', '-'):>7} "
              f"{s.get('fips_5digit', '-'):>7}")
    n_pass = sum(r["status"] == "PASS" for r in results)
    print("=" * 72)
    print(f"{n_pass}/{len(results)} passed")

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"fixtures kept in {workdir}")

    return results, n_pass


def main(argv=None):
    args = parse_args(argv)

    if not os.path.exists(SCRIPT):
        sys.exit(f"cannot find {SCRIPT}")

    # Always persist the run to a timestamped log (tee'd from the console).
    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"stage0_smoke_{stamp}.log")

    logfile = open(log_path, "w")
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(real_stdout, logfile)
    sys.stderr = Tee(real_stderr, logfile)
    try:
        print(f"log: {log_path}\n")
        results, n_pass = run_all(args)
        print(f"\nlog written to {log_path}")
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr
        logfile.close()

    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
