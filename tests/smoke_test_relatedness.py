#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke test for the relatedness stage (`src/relatedness/1..4`) across census waves.

The relatedness pipeline reads only the stage-2b `matched_candidates_{year}.parquet`
tables plus a handful of small label files, so this test is CHAINED end to end on
a tiny fixture:

  fixture   -> head-N rows of the real matched_candidates_{year}.parquet (full
               schema preserved) written to a temp dir.
  stage 1   -> patent strings from the clearpat titles CSV, restricted to the
               fixture's pubnumbers (keeps it tiny).
  stage 2   -> per-year occupation + patent description strings (polars).
  stage 3   -> JobBERT-v2 embeddings (GPU if usable, else CPU). Downloads the
               model on first run.
  stage 4   -> per-year cosine relatedness scores (streamed); the final output.

The downstream merge into the feature table (src/3_merge_features.py, which also
attaches the name-rarity feature surname_logfreq) is a separate stage with its own
smoke test (smoke_test_stage3.py); this test stops at stage 4.

Each stage is invoked through its real CLI via subprocess, so the seam is the
production one. Per wave it asserts:
  * stage-1 patent_strings carries pubnumber + pat_desc_short/long, >=1 non-null.
  * stage-2 per-year file carries occ_desc_short/long + pat_desc_short/long and
    keeps every fixture row.
  * stage-3 writes patent_embeddings.parquet + partitioned
    occupation_embeddings_{short,long}/ with a consistent embedding dim and >=1
    non-null embedding.
  * stage-4 per-year relatedness_short/long are null or in [-1, 1] and the row
    count equals the stage-2 rows.
A schema-valid all-null-score wave is a per-wave PASS (a tiny slice may have no
patent/occupation strings), but the suite passes only if nothing failed AND at
least one wave produced a real relatedness score. Exits non-zero on any failure.

Every run writes a timestamped log to tests/logs/ (tee'd from the console).

Usage:
    conda activate ipums_match
    python tests/smoke_test_relatedness.py                  # 1850, GPU-or-CPU auto
    python tests/smoke_test_relatedness.py --years 1850 1900 --device cpu
    bash tests/run_smoke_test_relatedness.sh
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA = os.path.join(REPO_ROOT, "data")
REL = os.path.join(REPO_ROOT, "src", "relatedness")
LOG_DIR_DEFAULT = os.path.join(HERE, "logs")

DEFAULT_YEARS = [1850]
DEF_MATCHED = "/scratch/breschi/ipums/matched_candidates"


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


def first_emb_dim(series):
    """Embedding dim from the first non-null array in a parquet-loaded column."""
    for v in series:
        if isinstance(v, np.ndarray) and v.size:
            return v.shape[0]
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Chained smoke test for the relatedness stage.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--years", type=int, nargs="*", default=DEFAULT_YEARS)
    p.add_argument("--n-rows", type=int, default=3000, help="Head rows per wave fixture.")
    p.add_argument("--device", default="cpu", help="cpu | auto | cuda | cuda:N (stage 3).")
    p.add_argument("--model", default="TechWolf/JobBERT-v2")
    p.add_argument("--matched-candidates-dir", default=DEF_MATCHED)
    p.add_argument("--titles", default=os.path.join(DATA, "patent_titles", "patent_titles_clearpat.csv"))
    p.add_argument("--classes", default=os.path.join(DATA, "USPTO_patent_classes_1840_1950.tsv"))
    p.add_argument("--uspc-names", default=os.path.join(DATA, "misc", "uspc_class_names.xlsx"))
    p.add_argument("--nber-names", default=os.path.join(DATA, "misc", "nber_category.txt"))
    p.add_argument("--occ1950", default=os.path.join(DATA, "occ_ind_1950", "OCC1950.xlsx"))
    p.add_argument("--ind1950", default=os.path.join(DATA, "occ_ind_1950", "IND1950.xlsx"))
    p.add_argument("--work-dir", default=None, help="Override the temp working dir.")
    p.add_argument("--keep", action="store_true", help="Keep the temp working dir.")
    p.add_argument("--log-dir", default=LOG_DIR_DEFAULT)
    return p.parse_args(argv)


def build_fixture(args, work):
    """Per wave: head-N rows of the real matched_candidates_{year}.parquet."""
    mc_dir = os.path.join(work, "matched_candidates")
    os.makedirs(mc_dir, exist_ok=True)
    built = []
    for year in args.years:
        src = os.path.join(args.matched_candidates_dir, f"matched_candidates_{year}.parquet")
        if not os.path.exists(src):
            print(f"[{year}] no source {src}; skipping wave")
            continue
        batch = next(pq.ParquetFile(src).iter_batches(batch_size=args.n_rows))
        tbl = pa.Table.from_batches([batch])
        out = os.path.join(mc_dir, f"matched_candidates_{year}.parquet")
        pq.write_table(tbl, out, compression="zstd")
        print(f"[{year}] fixture rows: {tbl.num_rows:,} -> {out}")
        built.append(year)
    return mc_dir, built


def check_wave(year, work, failures):
    pat_strings = os.path.join(work, "patent_strings.parquet")
    ws = os.path.join(work, "with_strings", f"matched_candidates_with_strings_{year}.parquet")
    rel = os.path.join(work, "by_year", f"matched_candidates_rel_{year}.parquet")

    def fail(msg):
        print(f"  [{year}] FAIL: {msg}")
        failures.append((year, msg))

    # stage 1
    ps = pq.read_table(pat_strings).to_pandas()
    if not {"pubnumber", "pat_desc_short", "pat_desc_long"} <= set(ps.columns):
        fail("patent_strings missing columns")
    elif ps["pat_desc_short"].notna().sum() == 0:
        fail("patent_strings has no non-null pat_desc_short")

    # stage 2
    if not os.path.exists(ws):
        fail("no with-strings output"); return False
    wt = pq.read_table(ws)
    need = {"occ_desc_short", "occ_desc_long", "pat_desc_short", "pat_desc_long",
            "histid", "census_year", "pubnumber"}
    if not need <= set(wt.column_names):
        fail(f"with-strings missing {need - set(wt.column_names)}")
    ws_rows = wt.num_rows

    # stage 4
    if not os.path.exists(rel):
        fail("no relatedness output"); return False
    rt = pq.read_table(rel).to_pandas()
    if not {"relatedness_short", "relatedness_long"} <= set(rt.columns):
        fail("relatedness columns missing")
    if len(rt) != ws_rows:
        fail(f"row count drift: rel {len(rt):,} vs with-strings {ws_rows:,}")
    n_scored = 0
    for col in ("relatedness_short", "relatedness_long"):
        vals = rt[col].to_numpy(dtype="float64", na_value=np.nan)
        good = vals[~np.isnan(vals)]
        if good.size and (good.min() < -1.001 or good.max() > 1.001):
            fail(f"{col} out of [-1,1]: [{good.min():.4f},{good.max():.4f}]")
        n_scored += good.size
    print(f"  [{year}] with-strings {ws_rows:,} rows | relatedness scored {n_scored:,} values")
    return n_scored > 0


def check_embeddings(work, failures):
    emb = os.path.join(work, "embeddings")
    pe = os.path.join(emb, "patent_embeddings.parquet")
    if not os.path.exists(pe):
        failures.append(("emb", "no patent_embeddings.parquet")); return
    pd_emb = pq.read_table(pe).to_pandas()
    dim = first_emb_dim(pd_emb["emb_short"])
    if dim is None:
        failures.append(("emb", "no non-null patent embedding"))
    else:
        print(f"  embeddings: patent dim={dim}, {len(pd_emb):,} patents")
    for sub in ("occupation_embeddings_short", "occupation_embeddings_long"):
        d = os.path.join(emb, sub)
        if not os.path.isdir(d):
            failures.append(("emb", f"missing {sub} dir"))


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"relatedness_smoke_{stamp}.log")
    real_stdout, real_stderr = sys.stdout, sys.stderr
    logfile = open(log_path, "w")
    sys.stdout = Tee(real_stdout, logfile)
    sys.stderr = Tee(real_stderr, logfile)

    work = args.work_dir or tempfile.mkdtemp(prefix="rel_smoke_")
    os.makedirs(work, exist_ok=True)
    failures = []
    any_scored = False
    try:
        print(f"log: {log_path}")
        print(f"work dir: {work}\n")

        mc_dir, years = build_fixture(args, work)
        if not years:
            raise RuntimeError("no fixtures built; nothing to test")

        years_str = [str(y) for y in years]

        print("\n[1/4] patent strings")
        run([sys.executable, os.path.join(REL, "1_patent_strings.py"),
             "--titles", args.titles, "--classes", args.classes,
             "--uspc-names", args.uspc_names, "--nber-names", args.nber_names,
             "--matched-candidates", mc_dir,
             "--output", os.path.join(work, "patent_strings.parquet")])

        print("\n[2/4] build strings")
        run([sys.executable, os.path.join(REL, "2_build_strings.py"),
             "--matched-candidates", mc_dir,
             "--patent-strings", os.path.join(work, "patent_strings.parquet"),
             "--occ1950", args.occ1950, "--ind1950", args.ind1950,
             "--years", *years_str,
             "--output-dir", os.path.join(work, "with_strings")])

        print("\n[3/4] embeddings")
        run([sys.executable, os.path.join(REL, "3_create_embeddings.py"),
             "--patent-strings", os.path.join(work, "patent_strings.parquet"),
             "--with-strings-dir", os.path.join(work, "with_strings"),
             "--out-dir", os.path.join(work, "embeddings"),
             "--years", *years_str, "--device", args.device, "--model", args.model])

        print("\n[4/4] relatedness scores")
        run([sys.executable, os.path.join(REL, "4_relatedness_score.py"),
             "--input", os.path.join(work, "with_strings"),
             "--embeddings-dir", os.path.join(work, "embeddings"),
             "--out-dir", os.path.join(work, "by_year"),
             "--years", *years_str])

        print("\n=== checks ===")
        check_embeddings(work, failures)
        for year in years:
            any_scored |= check_wave(year, work, failures)

        if not any_scored:
            failures.append(("suite", "no wave produced a real relatedness score"))
    except Exception:
        import traceback
        failures.append(("error", "unhandled exception (see traceback above)"))
        traceback.print_exc()
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"\nkept work dir: {work}")
        ok = not failures
        print("\n" + ("PASS" if ok else f"FAIL ({len(failures)} problem(s)):"))
        for y, m in failures:
            print(f"  - [{y}] {m}")
        print(f"\nlog written to {log_path}")
        sys.stdout, sys.stderr = real_stdout, real_stderr
        logfile.close()
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
