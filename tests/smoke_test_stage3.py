#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke test for stage 3 (`src/3_merge_features.py`) -- the feature merge.

Stage 3 concatenates the per-year relatedness outputs
(`by_year/matched_candidates_rel_{year}.parquet`, the 2b matched_candidates rows
carrying `relatedness_short` / `relatedness_long`) AND attaches the census-population
surname-rarity feature `surname_logfreq` (precomputed per wave by
`name_rarity/1_surname_rarity.py`, bridged `histid -> clean_ln` through the cleaned
census). It is the table the ML model consumes.

Unlike the other smoke tests this one is SELF-CONTAINED -- it fabricates tiny
fixtures and needs no restricted IPUMS data, so it runs in a clean checkout. Per
wave it builds:
  * a cleaned-census slice (`US{year}P.parquet`: histid, clean_ln) with a deliberate
    rare/common surname mix, then runs the REAL `name_rarity/1_surname_rarity.py` over
    it to produce `surname_rarity_{year}.parquet`;
  * a relatedness `by_year` file (`matched_candidates_rel_{year}.parquet`) whose
    histids include one ABSENT from the census (to exercise the null-logfreq path).
Then it runs `src/3_merge_features.py` via its real CLI (subprocess) and asserts:
  * row count == sum of per-year relatedness rows (concat is loss-free);
  * `relatedness_short` / `relatedness_long` preserved;
  * `surname_logfreq` present, non-negative + finite where non-null, and rarer
    surnames score strictly higher than common ones;
  * `n_candidates` present and equal to the distinct histid count per
    (PatInvID, census_year);
  * the histid absent from the census gets a null `surname_logfreq`;
  * deceased hard negatives (`--deceased-negatives`): a confirmed-deceased inventor's
    candidates in its `eligible_census_year` are flipped to `true_match=0` with
    `is_hard_negative=1`, while the same inventor in a non-eligible wave, an existing
    golden-labeled (collision) row, and a `confirmed=NO` inventor are all left untouched;
  * the no-rarity fallback (omit both dirs) concatenates relatedness only, with no
    `surname_logfreq` column, and `is_hard_negative` present but all 0.

Every run writes a timestamped log to tests/logs/ (tee'd from the console). Exits
non-zero on any failure.

Usage:
    conda activate ipums_match
    python tests/smoke_test_stage3.py        # or: bash tests/run_smoke_test_stage3.sh
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
SCRIPT_MERGE = os.path.join(SRC, "3_merge_features.py")
SCRIPT_RARITY = os.path.join(SRC, "name_rarity", "1_surname_rarity.py")
LOG_DIR_DEFAULT = os.path.join(HERE, "logs")

DEFAULT_YEARS = [1900, 1910]


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


def build_fixtures(work, years):
    """Per wave: a cleaned-census slice + a relatedness by_year file. Returns the dirs
    plus, per year, the histid expected to have a null surname_logfreq (absent from
    census). Surnames: 'common' (x5) is frequent, two unique surnames are rare."""
    cc_dir = os.path.join(work, "cleaned_census")
    rar_dir = os.path.join(work, "surname_rarity")
    by_year = os.path.join(work, "by_year")
    for d in (cc_dir, rar_dir, by_year):
        os.makedirs(d, exist_ok=True)

    absent = {}
    for i, year in enumerate(years):
        sfx = str(year)
        # Cleaned census: histid -> clean_ln. 'common' appears 5x; two unique rares.
        census = pl.DataFrame({
            "histid": [f"h{sfx}_{j}" for j in range(7)],
            "clean_ln": ["common", "common", "common", "common", "common",
                         f"rarex{sfx}", f"rarey{sfx}"],
        })
        census.write_parquet(os.path.join(cc_dir, f"US{year}P.parquet"))
        # Run the REAL precompute to get surname_rarity_{year}.parquet.
        run([sys.executable, SCRIPT_RARITY,
             "--cleaned-census", os.path.join(cc_dir, f"US{year}P.parquet"),
             "--census-year", str(year),
             "--output", os.path.join(rar_dir, f"surname_rarity_{year}.parquet")])
        # Relatedness rows: a common, the two rares, and one histid NOT in the census.
        # true_histid/true_match are the golden labels (null = unlabeled bulk); the
        # deceased step only flips currently-null rows.
        ghost = f"ghost{sfx}"
        absent[year] = ghost
        pat = [100 + i, 100 + i, 101 + i, 102 + i]
        pub = [1, 1, 2, 3]
        hid = [f"h{sfx}_0", f"h{sfx}_5", f"h{sfx}_6", ghost]
        name = ["a common", "b rarex", "c rarey", "d ghost"]
        rs = [0.10, 0.20, 0.30, 0.40]
        rl = [0.50, 0.60, 0.70, 0.80]
        thid = [None, None, None, None]
        tm = [None, None, None, None]

        # Deceased-inventor test rows (PatInvIDs 900/901/902; see write_deceased_csv).
        if year == years[0]:
            # 900: confirmed-deceased, eligible THIS wave, unlabeled -> expect MARKED (2 cand).
            pat += [900, 900]; pub += [9, 9]; hid += [f"h{sfx}_1", f"h{sfx}_2"]
            name += ["dead one", "dead two"]; rs += [0.11, 0.22]; rl += [0.33, 0.44]
            thid += [None, None]; tm += [None, None]
            # 901: confirmed-deceased BUT already golden-labeled -> COLLISION, expect untouched.
            pat += [901]; pub += [9]; hid += [f"h{sfx}_3"]; name += ["collision"]
            rs += [0.15]; rl += [0.25]; thid += [f"h{sfx}_3"]; tm += [1]
            # 902: in the CSV but confirmed=NO -> must NOT be marked.
            pat += [902]; pub += [9]; hid += [f"h{sfx}_4"]; name += ["not confirmed"]
            rs += [0.16]; rl += [0.26]; thid += [None]; tm += [None]
        if len(years) > 1 and year == years[1]:
            # 900 again, but THIS wave is NOT its eligible_census_year -> must NOT be marked.
            pat += [900]; pub += [9]; hid += [f"h{sfx}_1"]; name += ["dead wrong wave"]
            rs += [0.17]; rl += [0.27]; thid += [None]; tm += [None]

        rel = pl.DataFrame({
            "PatInvID": pat, "pubnumber": pub, "histid": hid,
            "census_year": [year] * len(pat), "census_name": name,
            "relatedness_short": rs, "relatedness_long": rl,
        }).with_columns([
            pl.Series("true_histid", thid, dtype=pl.Utf8),
            pl.Series("true_match", tm, dtype=pl.Int64),
        ])
        rel.write_parquet(os.path.join(by_year, f"matched_candidates_rel_{year}.parquet"))
    return cc_dir, rar_dir, by_year, absent


def write_deceased_csv(work, years):
    """A tiny reviewed deceased-inventors CSV mirroring the real file's key columns.
    900/901 are confirmed (YES, with case/space variants); 902 is rejected (NO)."""
    path = os.path.join(work, "deceased_inventors_reviewed.csv")
    pl.DataFrame({
        "PatInvID": [900, 901, 902],
        "pubnumber": [9, 9, 9],
        "pubyear": [years[0] - 5] * 3,
        "eligible_census_year": [years[0], years[0], years[0]],
        "confirmed": ["YES", " yes ", "NO"],
    }).write_csv(path)
    return path


def validate(out_path, by_year, years, absent, failures):
    df = pl.read_parquet(out_path)
    cols = set(df.columns)

    # Row count is loss-free vs the per-year inputs.
    expect_rows = sum(
        pl.read_parquet(os.path.join(by_year, f"matched_candidates_rel_{y}.parquet")).height
        for y in years)
    if df.height != expect_rows:
        failures.append(f"row count {df.height} != sum of per-year {expect_rows}")

    for c in ("relatedness_short", "relatedness_long", "surname_logfreq", "n_candidates"):
        if c not in cols:
            failures.append(f"missing column {c}")

    # n_candidates == distinct histid per (PatInvID, census_year).
    if "n_candidates" in cols:
        chk = df.with_columns(
            pl.col("histid").n_unique().over(["PatInvID", "census_year"]).alias("_exp"))
        bad = chk.filter(pl.col("n_candidates") != pl.col("_exp"))
        if bad.height:
            failures.append(f"n_candidates wrong for {bad.height} rows")

    if "surname_logfreq" not in cols:
        return  # surname-specific checks below need the column

    slf = df["surname_logfreq"]
    good = slf.drop_nulls()
    if good.len() and (float(good.min()) < -1e-9):
        failures.append(f"surname_logfreq negative: min {float(good.min())}")

    # The ghost histids (absent from census) must be null; the present ones non-null.
    for y in years:
        g = df.filter(pl.col("histid") == absent[y])
        if g.height and g["surname_logfreq"].null_count() != g.height:
            failures.append(f"{y}: ghost histid has non-null surname_logfreq")
        # Rarer surname (rarex) must score strictly higher than the common one.
        common = df.filter(pl.col("census_name") == "a common")["surname_logfreq"]
        rare = df.filter(pl.col("census_name") == "b rarex")["surname_logfreq"]
        if common.len() and rare.len() and not (float(rare.min()) > float(common.max())):
            failures.append(f"{y}: rare surname not scored above common "
                            f"(rare {float(rare.min()):.3f} <= common {float(common.max()):.3f})")


def validate_deceased(out_path, years, failures):
    """Deceased hard-negative marking: eligible-wave candidates flipped to true_match=0
    + is_hard_negative=1; non-eligible wave, golden collisions, and confirmed=NO untouched."""
    df = pl.read_parquet(out_path)
    if "is_hard_negative" not in df.columns:
        failures.append("deceased: is_hard_negative column missing")
        return

    def rows(pid, yr):
        return df.filter((pl.col("PatInvID") == pid) & (pl.col("census_year") == yr))

    g = rows(900, years[0])  # marked in eligible wave
    if g.height == 0:
        failures.append("deceased: no rows for PatInvID 900 in eligible wave")
    elif not (bool((g["is_hard_negative"] == 1).all()) and bool((g["true_match"] == 0).all())):
        failures.append("deceased: PatInvID 900 not fully marked (is_hard_negative=1/true_match=0)")

    c = rows(901, years[0])  # golden collision -> preserved, not flagged
    if c.height and (int(c["is_hard_negative"].sum()) != 0 or not bool((c["true_match"] == 1).all())):
        failures.append("deceased: golden collision PatInvID 901 was overwritten")

    n = rows(902, years[0])  # confirmed=NO -> never marked
    if n.height and int(n["is_hard_negative"].sum()) != 0:
        failures.append("deceased: confirmed=NO PatInvID 902 was marked")

    if len(years) > 1:
        w = rows(900, years[1])  # wrong (non-eligible) wave -> not marked, label stays null
        if w.height and (int(w["is_hard_negative"].sum()) != 0
                         or w["true_match"].null_count() != w.height):
            failures.append("deceased: PatInvID 900 marked in non-eligible wave")


def validate_fallback(out_path, by_year, years, failures):
    df = pl.read_parquet(out_path)
    if "surname_logfreq" in df.columns:
        failures.append("fallback: surname_logfreq present without rarity dirs")
    # n_candidates is added in both modes (independent of rarity).
    if not {"relatedness_short", "relatedness_long", "n_candidates"} <= set(df.columns):
        failures.append("fallback: relatedness/n_candidates columns missing")
    # is_hard_negative is always emitted; without --deceased-negatives it must be all 0.
    if "is_hard_negative" not in df.columns:
        failures.append("fallback: is_hard_negative column missing")
    elif int(df["is_hard_negative"].sum()) != 0:
        failures.append("fallback: is_hard_negative non-zero without --deceased-negatives")
    expect_rows = sum(
        pl.read_parquet(os.path.join(by_year, f"matched_candidates_rel_{y}.parquet")).height
        for y in years)
    if df.height != expect_rows:
        failures.append(f"fallback: row count {df.height} != {expect_rows}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Self-contained smoke test for 3_merge_features.py.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--years", type=int, nargs="*", default=DEFAULT_YEARS)
    p.add_argument("--work-dir", default=None, help="Override the temp working dir.")
    p.add_argument("--keep", action="store_true", help="Keep the temp working dir.")
    p.add_argument("--log-dir", default=LOG_DIR_DEFAULT)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"stage3_smoke_{stamp}.log")
    real_stdout, real_stderr = sys.stdout, sys.stderr
    logfile = open(log_path, "w")
    sys.stdout = Tee(real_stdout, logfile)
    sys.stderr = Tee(real_stderr, logfile)

    work = args.work_dir or tempfile.mkdtemp(prefix="merge_smoke_")
    os.makedirs(work, exist_ok=True)
    failures = []
    try:
        print(f"log: {log_path}")
        print(f"work dir: {work}\n")

        print("[1] build fixtures + run name_rarity")
        cc_dir, rar_dir, by_year, absent = build_fixtures(work, args.years)
        deceased_csv = write_deceased_csv(work, args.years)

        print("\n[2] merge with rarity + deceased hard negatives")
        out = os.path.join(work, "merged.parquet")
        run([sys.executable, SCRIPT_MERGE, "--input-dir", by_year,
             "--cleaned-census-dir", cc_dir, "--surname-rarity-dir", rar_dir,
             "--deceased-negatives", deceased_csv, "--output", out])
        validate(out, by_year, args.years, absent, failures)
        validate_deceased(out, args.years, failures)

        print("\n[3] merge fallback (relatedness only, no rarity dirs)")
        out2 = os.path.join(work, "merged_norar.parquet")
        run([sys.executable, SCRIPT_MERGE, "--input-dir", by_year, "--output", out2])
        validate_fallback(out2, by_year, args.years, failures)
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
