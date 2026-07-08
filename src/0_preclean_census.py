#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pre-clean one census year into the input format consumed by the blocking step.

This is stage 0 of the linkage pipeline. For a single census year it:
  1. reads the raw IPUMS person file (first/last name, age, sex, ...);
  2. attaches the city of residence from the matching household file;
  3. normalises first and last names into `clean_fn` / `clean_ln`;
  4. attaches the 5-digit county FIPS from a geographic crosswalk;
  5. writes a compact parquet with exactly the columns the blocking scripts
     expect:
     histid, clean_fn, clean_ln, county_fips_code, age, birthyr, sex, city,
     potential_match

Inputs are restricted IPUMS USA full-count extracts and are NOT redistributed
with this code (see README). Run this once per census year.

Requirements: pyarrow, pandas, Unidecode (see environment.yml).

"""

import argparse
import gc
import os

import pandas as pd
from unidecode import unidecode

from blocking_common import parallelize_dataframe


# ---------------------------------------------------------------------------#
# Name cleaning
# ---------------------------------------------------------------------------#

def clean_first(df):
    df["clean_fn"] = df["namefrst"].map(lambda x: unidecode(x))
    df["clean_fn"] = df["clean_fn"].str.lower()
    df["clean_fn"] = df["clean_fn"].str.replace(r"[^a-z\s]+", "", regex=True)
    df["clean_fn"] = df["clean_fn"].str.replace(r"^\s+", "", regex=True)
    df["clean_fn"] = df["clean_fn"].str.replace(r"\s+$", "", regex=True)
    return df


def clean_last(df):
    df["clean_ln"] = df["namelast"].map(lambda x: unidecode(x))
    df["clean_ln"] = df["clean_ln"].str.lower()
    df["clean_ln"] = df["clean_ln"].str.replace(r"[^a-z\s]+", "", regex=True)
    df["clean_ln"] = df["clean_ln"].str.replace(r"^\s+", "", regex=True)
    df["clean_ln"] = df["clean_ln"].str.replace(r"\s+$", "", regex=True)
    return df


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pre-clean one census year for the blocking step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--person", required=True,
                   help="Raw IPUMS person parquet (e.g. US1900P.parquet) with columns "
                        "histid, namefrst, namelast, age, birthyr, sex, serialp.")
    p.add_argument("--household", required=True,
                   help="Raw IPUMS household parquet (e.g. US1900H.parquet) with columns serial, city.")
    p.add_argument("--crosswalk", required=True,
                   help="Geographic crosswalk CSV/zip with columns histid, "
                        "state_fips_geomatch, county_fips_geomatch, potential_match.")
    p.add_argument("--output", required=True,
                   help="Output cleaned parquet path (e.g. cleaned_census/US1900P.parquet).")
    p.add_argument("--n-cores", type=int, default=8,
                   help="Process count for parallel name cleaning.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Person records.
    census = pd.read_parquet(
        args.person,
        columns=["histid", "namefrst", "namelast", "age", "birthyr", "sex", "serialp"])
    census["serialp"] = census["serialp"].astype(float)
    census.rename(columns={"serialp": "serial"}, inplace=True)
    print(f"Person records: {census.shape[0]}")

    # City of residence from the household file.
    household = pd.read_parquet(args.household, columns=["serial", "city"])
    household["serial"] = household["serial"].astype(float)
    census = pd.merge(census, household, how="left", on="serial")

    # Clean first / last names on the unique sets, then merge back.
    fn = census[census["namefrst"].notnull()][["namefrst"]].drop_duplicates()
    ln = census[census["namelast"].notnull()][["namelast"]].drop_duplicates()
    print(f"Unique first names: {len(fn)} | unique last names: {len(ln)}")

    cleaned_fn = parallelize_dataframe(fn, clean_first, n_cores=args.n_cores)
    cleaned_ln = parallelize_dataframe(ln, clean_last, n_cores=args.n_cores)

    census = pd.merge(census, cleaned_fn, how="left", on="namefrst")
    census = pd.merge(census, cleaned_ln, how="left", on="namelast")
    census = census[["histid", "serial", "clean_fn", "clean_ln",
                     "age", "birthyr", "sex", "city"]].drop_duplicates()

    # County FIPS from the geographic crosswalk.
    cross = pd.read_csv(
        args.crosswalk,
        usecols=["histid", "state_fips_geomatch", "county_fips_geomatch", "potential_match"])
    cross["state_fips_geomatch"] = cross["state_fips_geomatch"].astype(str).str.split(".").str[0].str.zfill(2)
    cross["county_fips_geomatch"] = cross["county_fips_geomatch"].astype(str).str.split(".").str[0].str.zfill(3)
    cross["county_fips_code"] = cross["state_fips_geomatch"] + cross["county_fips_geomatch"]
    census = pd.merge(census, cross, on="histid", how="left")
    census = census[["histid", "clean_fn", "clean_ln", "county_fips_code",
                     "age", "birthyr", "sex", "city", "potential_match"]].drop_duplicates()

    del cross
    gc.collect()

    census.to_parquet(args.output, index=False, engine="pyarrow", compression="GZIP")
    print(f"Wrote {census.shape[0]} cleaned records -> {args.output}")


if __name__ == "__main__":
    main()
