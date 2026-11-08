#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke test for stage 4 (`src/4_select_ml_columns.py`) -- the publication/ML column
projection.

Stage 4 reads the stage-3 features table (`matched_candidates_features.parquet`) and
writes a slimmed copy keeping ONLY the columns the ML model consumes (the stage-1
`TRAIN_COLUMNS`: keys + feature sources + labels + optional `occ`/`is_hard_negative`),
dropping everything else. All rows are retained; the projection is columns-only.

Like `smoke_test_stage3.py` this is SELF-CONTAINED -- it fabricates a tiny
parquet with every kept column, the two required name columns, PLUS extras that must
be dropped (geo/gender passthroughs, the bulky `occ_desc_*`/`pat_desc_*` strings),
runs the real CLI via subprocess, and asserts:
  * output schema == the keep-list (names + all extras dropped, `occ`/`is_hard_negative` kept);
  * `diff_length` is present and equals abs(len(inventor_name) - len(census_name));
  * `source` == "https://patents.google.com/patent/US<pubnumber> | <IPUMS DOI>", last column;
  * `inventor_name` / `census_name` are ABSENT from the output;
  * row count unchanged (columns-only projection, loss-free);
  * kept values are byte-identical to the input (no reorder/mutation);
  * a table MISSING a required column (incl. a name) exits non-zero with the message;
  * a table lacking the OPTIONAL `occ`/`is_hard_negative` still projects fine.

Every run writes a timestamped log to tests/logs/ (tee'd from the console). Exits
non-zero on any failure.

Usage:
    conda activate ipums_match
    python tests/smoke_test_stage4.py        # or: bash tests/run_smoke_test_stage4.sh
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SRC = os.path.join(REPO_ROOT, "src")
SCRIPT = os.path.join(SRC, "4_select_ml_columns.py")
LOG_DIR_DEFAULT = os.path.join(HERE, "logs")

# The columns stage 4 OUTPUTS (mirrors src/4_select_ml_columns.py OUTPUT_COLUMNS):
# the raw names are dropped and diff_length is precomputed in their place; the
# provenance `source` column is appended last.
KEEP = ["PatInvID", "pubnumber", "histid", "census_year",
        "diff_length", "leven_norm", "tkratio", "tksortratio",
        "surname_logfreq", "n_candidates", "county_distance", "pubyear", "birthyr",
        "ind1950", "occ1950", "occstr", "relatedness_short", "relatedness_long",
        "occ", "is_hard_negative", "true_histid", "true_match", "source"]
# Columns built by stage 4 (absent from the input).
GENERATED = ["diff_length", "source"]
SOURCE_PREFIX = "https://patents.google.com/patent/US"
SOURCE_SUFFIX = " | https://doi.org/10.18128/D014.V4.0R"
# Name columns: required in the INPUT (to derive diff_length) but dropped from OUTPUT.
NAMES = ["inventor_name", "census_name"]
# Extras present in a --keep-all-columns stage-3 table that stage 4 must drop.
EXTRAS = ["county_inv", "state_inv", "county_census", "gender_census",
          "occ_desc_short", "occ_desc_long", "pat_desc_short", "pat_desc_long"]


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


def run(cmd, check=True):
    print("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=check,
                          capture_output=True, text=True)


def build_fixture(path, n=12, drop_cols=()):
    """A tiny features-table parquet with all KEEP columns + EXTRAS, minus drop_cols."""
    vals = {
        "PatInvID": list(range(100, 100 + n)),
        "pubnumber": list(range(1, n + 1)),
        "histid": [f"h{i}" for i in range(n)],
        "census_year": [1900] * n,
        "inventor_name": [f"inv {i}" for i in range(n)],
        "census_name": [f"cen {i}" for i in range(n)],
        "leven_norm": [i / n for i in range(n)],
        "tkratio": [50.0 + i for i in range(n)],
        "tksortratio": [40.0 + i for i in range(n)],
        "surname_logfreq": [1.0 + i * 0.1 for i in range(n)],
        "n_candidates": [3] * n,
        "county_distance": [float(i) for i in range(n)],
        "pubyear": [1895] * n,
        "birthyr": [1860 + i for i in range(n)],
        "ind1950": [100 + i for i in range(n)],
        "occ1950": [200 + i for i in range(n)],
        "occstr": [f"occ {i}" for i in range(n)],
        "relatedness_short": [0.1 * i for i in range(n)],
        "relatedness_long": [0.2 * i for i in range(n)],
        "occ": [300 + i for i in range(n)],
        "is_hard_negative": [0] * n,
        "true_histid": [None] * n,
        "true_match": [None] * n,
        # extras to be dropped
        "county_inv": [f"ci{i}" for i in range(n)],
        "state_inv": [f"si{i}" for i in range(n)],
        "county_census": [f"cc{i}" for i in range(n)],
        "gender_census": ["M"] * n,
        "occ_desc_short": [f"ods {i}" for i in range(n)],
        "occ_desc_long": [f"odl {i}" for i in range(n)],
        "pat_desc_short": [f"pds {i}" for i in range(n)],
        "pat_desc_long": [f"pdl {i}" for i in range(n)],
    }
    for c in drop_cols:
        vals.pop(c, None)
    df = pl.DataFrame(vals).with_columns([
        pl.col("true_histid").cast(pl.Utf8),
        pl.col("true_match").cast(pl.Int64),
    ])
    df.write_parquet(path)
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description="Self-contained smoke test for 4_select_ml_columns.py.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--work-dir", default=None, help="Override the temp working dir.")
    p.add_argument("--keep", action="store_true", help="Keep the temp working dir.")
    p.add_argument("--log-dir", default=LOG_DIR_DEFAULT)
    args = p.parse_args(argv)

    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"stage4_smoke_{stamp}.log")
    real_stdout, real_stderr = sys.stdout, sys.stderr
    logfile = open(log_path, "w")
    sys.stdout = Tee(real_stdout, logfile)
    sys.stderr = Tee(real_stderr, logfile)

    work = args.work_dir or tempfile.mkdtemp(prefix="stage4_smoke_")
    os.makedirs(work, exist_ok=True)
    failures = []
    try:
        print(f"log: {log_path}")
        print(f"work dir: {work}\n")

        # [1] full fixture -> project.
        print("[1] project full features table")
        src = os.path.join(work, "features.parquet")
        out = os.path.join(work, "ml.parquet")
        full = build_fixture(src)
        run([sys.executable, SCRIPT, "--input", src, "--output", out])
        got = pl.read_parquet(out)

        if set(got.columns) != set(KEEP):
            failures.append(f"output columns {sorted(got.columns)} != keep-list {sorted(KEEP)}")
        for c in EXTRAS + NAMES:
            if c in got.columns:
                failures.append(f"column not dropped: {c}")
        if got.height != full.height:
            failures.append(f"row count {got.height} != input {full.height}")
        # diff_length precomputed = abs(len(inventor_name) - len(census_name)).
        if "diff_length" not in got.columns:
            failures.append("diff_length missing from output")
        else:
            exp = (full["inventor_name"].str.len_chars().cast(pl.Int64)
                   - full["census_name"].str.len_chars().cast(pl.Int64)).abs().rename("diff_length")
            if not got["diff_length"].cast(pl.Int64).equals(exp):
                failures.append("diff_length values != abs name-length gap")
        # source = Google Patents URL | IPUMS DOI, derived from pubnumber, no nulls.
        if "source" not in got.columns:
            failures.append("source missing from output")
        else:
            exp_src = (SOURCE_PREFIX + full["pubnumber"].cast(pl.String)
                       + SOURCE_SUFFIX).rename("source")
            if got["source"].null_count() or not got["source"].equals(exp_src):
                failures.append("source values != Google Patents URL | IPUMS DOI")
            if list(got.columns)[-1] != "source":
                failures.append("source is not the last column")
        # Kept pass-through values byte-identical (skip generated cols, not in the input).
        for c in KEEP:
            if c not in GENERATED and c in got.columns and not got[c].equals(full[c]):
                failures.append(f"values changed for kept column {c}")

        # [2] optional columns absent -> still projects.
        print("\n[2] project with optional occ/is_hard_negative absent")
        src2 = os.path.join(work, "features_noopt.parquet")
        out2 = os.path.join(work, "ml_noopt.parquet")
        build_fixture(src2, drop_cols=["occ", "is_hard_negative"])
        run([sys.executable, SCRIPT, "--input", src2, "--output", out2])
        got2 = pl.read_parquet(out2)
        expect2 = [c for c in KEEP if c not in ("occ", "is_hard_negative")]
        if set(got2.columns) != set(expect2):
            failures.append(f"no-optional output {sorted(got2.columns)} != {sorted(expect2)}")

        # [3] missing REQUIRED column -> non-zero exit + message.
        print("\n[3] required-column guard (drop county_distance)")
        src3 = os.path.join(work, "features_bad.parquet")
        out3 = os.path.join(work, "ml_bad.parquet")
        build_fixture(src3, drop_cols=["county_distance"])
        res = run([sys.executable, SCRIPT, "--input", src3, "--output", out3], check=False)
        if res.returncode == 0:
            failures.append("missing-required-column run exited 0 (expected non-zero)")
        if "county_distance" not in (res.stderr + res.stdout):
            failures.append("missing-column message did not name county_distance")
        if os.path.exists(out3):
            failures.append("output written despite missing required column")

    except Exception:
        import traceback
        failures.append("unhandled exception (see traceback above)")
        traceback.print_exc()
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"\nkept work dir: {work}")
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
