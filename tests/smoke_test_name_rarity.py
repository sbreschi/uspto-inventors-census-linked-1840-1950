#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke test for the name-rarity add-on stage 1 (`src/name_rarity/1_surname_rarity.py`)
-- the per-wave census surname-rarity reference table.

SELF-CONTAINED: fabricates tiny fixtures, needs no restricted IPUMS data, runs in a
clean checkout. Per wave it writes a small cleaned-census fixture as a PARTITIONED
parquet *directory* (`US{year}P.parquet/part.*.parquet`, two part files -- so the
glob-dir scan and cross-file surname aggregation are exercised), runs the REAL script
via its CLI (subprocess), and asserts the reference table:

  * schema is exactly (clean_ln, n, surname_logfreq);
  * per-surname counts `n` are correct, aggregated ACROSS the two part files;
  * null / empty `clean_ln` rows are excluded from the count and from N;
  * surname_logfreq == log10(N) - log10(n) for every surname (the negated log
    relative frequency), always >= 0;
  * rarer surnames score strictly HIGHER, and rows are sorted logfreq-descending.

Every run writes a timestamped log to tests/logs/ (tee'd). Exits non-zero on failure.

Usage:
    conda activate ipums_match
    python tests/smoke_test_name_rarity.py            # or: bash tests/run_smoke_test_name_rarity.sh
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SCRIPT = os.path.join(REPO_ROOT, "src", "name_rarity", "1_surname_rarity.py")
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


def run(cmd):
    print("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


# Per wave: {surname: n}. Real surnames counted; the two junk keys (a null and an
# empty string) must be filtered out. Distributions differ by year so each wave's
# numeric check is independent.
FIXTURES = {
    1900: {"SMITH": 5, "JONES": 3, "CATTANEO": 1},
    1910: {"BROWN": 4, "LEE": 2, "ZAMBONI": 1},
}


def build_fixture(work, year, counts):
    """Write US{year}P.parquet/ as a two-part-file directory. Real surname rows are
    split across BOTH parts (so cross-file aggregation is tested); a null and an
    empty-string clean_ln are injected as junk that must be filtered."""
    rows = []
    for surname, n in counts.items():
        rows.extend([surname] * n)
    # junk: one null, one empty string
    rows_part0 = rows[: len(rows) // 2] + [None]
    rows_part1 = rows[len(rows) // 2:] + [""]

    d = os.path.join(work, f"US{year}P.parquet")
    os.makedirs(d, exist_ok=True)
    pl.DataFrame({"clean_ln": rows_part0},
                 schema={"clean_ln": pl.Utf8}).write_parquet(
        os.path.join(d, "part.0.parquet"))
    pl.DataFrame({"clean_ln": rows_part1},
                 schema={"clean_ln": pl.Utf8}).write_parquet(
        os.path.join(d, "part.1.parquet"))
    return d


def validate(out_path, counts, failures, year):
    df = pl.read_parquet(out_path)

    if df.columns != ["clean_ln", "n", "surname_logfreq"]:
        failures.append(f"{year}: schema {df.columns} != "
                        f"['clean_ln', 'n', 'surname_logfreq']")
        return

    # exactly the real surnames (junk null/empty excluded)
    got = {r["clean_ln"]: (r["n"], r["surname_logfreq"])
           for r in df.iter_rows(named=True)}
    if set(got) != set(counts):
        failures.append(f"{year}: surnames {sorted(got)} != {sorted(counts)} "
                        f"(null/empty clean_ln not filtered?)")
        return

    n_total = sum(counts.values())
    for surname, exp_n in counts.items():
        n, lf = got[surname]
        if n != exp_n:
            failures.append(f"{year}/{surname}: n {n} != {exp_n} "
                            f"(cross-part-file aggregation wrong?)")
        exp_lf = math.log10(n_total) - math.log10(exp_n)
        if abs(lf - exp_lf) > 1e-9:
            failures.append(f"{year}/{surname}: surname_logfreq {lf:.6f} != "
                            f"{exp_lf:.6f} (= log10(N)-log10(n))")
        if lf < -1e-12:
            failures.append(f"{year}/{surname}: surname_logfreq {lf} < 0")

    # N == sum of counted n (junk excluded)
    if int(df.select(pl.col("n").sum()).item()) != n_total:
        failures.append(f"{year}: sum(n) {int(df.select(pl.col('n').sum()).item())} "
                        f"!= N {n_total}")

    # rarer -> strictly higher, and rows sorted logfreq-descending
    lfs = df["surname_logfreq"].to_list()
    if lfs != sorted(lfs, reverse=True):
        failures.append(f"{year}: rows not sorted by surname_logfreq descending")
    # the n=1 surname must be the single highest
    rarest = min(counts, key=counts.get)
    if df["clean_ln"][0] != rarest:
        failures.append(f"{year}: highest-logfreq surname {df['clean_ln'][0]!r} "
                        f"!= rarest {rarest!r}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Self-contained smoke test for name_rarity 1_surname_rarity.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--years", type=int, nargs="*", default=sorted(FIXTURES),
                   help="Waves to test (must be keys of the fixture table).")
    p.add_argument("--work-dir", default=None, help="Override the temp working dir.")
    p.add_argument("--keep", action="store_true", help="Keep the temp working dir.")
    p.add_argument("--log-dir", default=LOG_DIR_DEFAULT)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"name_rarity_smoke_{stamp}.log")
    real_stdout, real_stderr = sys.stdout, sys.stderr
    logfile = open(log_path, "w")
    sys.stdout = Tee(real_stdout, logfile)
    sys.stderr = Tee(real_stderr, logfile)

    work = args.work_dir or tempfile.mkdtemp(prefix="name_rarity_smoke_")
    os.makedirs(work, exist_ok=True)
    failures = []
    n_ok = 0
    try:
        print(f"log: {log_path}")
        print(f"work dir: {work}\n")
        for year in args.years:
            counts = FIXTURES.get(year)
            if counts is None:
                failures.append(f"{year}: no fixture defined")
                continue
            print(f"[1] build fixture + run for {year}")
            census = build_fixture(work, year, counts)
            out = os.path.join(work, f"surname_rarity_{year}.parquet")
            run([sys.executable, SCRIPT, "--cleaned-census", census,
                 "--census-year", year, "--output", out])
            print(f"[2] validate {year}")
            before = len(failures)
            validate(out, counts, failures, year)
            if len(failures) == before:
                n_ok += 1
    except Exception:
        import traceback
        failures.append("unhandled exception (see traceback above)")
        traceback.print_exc()
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"\nkept work dir: {work}")
        if n_ok == 0 and not failures:
            failures.append("no wave was validated")
        ok = not failures
        print("\n" + ("PASS" if ok else f"FAIL ({len(failures)} problem(s)):"))
        for m in failures:
            print(f"  - {m}")
        print(f"\nlog written to {log_path}")
        sys.stdout, sys.stderr = real_stdout, real_stderr
        logfile.close()
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
