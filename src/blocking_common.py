#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared helpers for the blocking stage of the patent-inventor / census record
linkage pipeline.

The three blocking entry points (`1a_block_faiss_cosine.py`,
`1b_block_rapidfuzz_levenshtein.py` and `1c_block_faiss_cosine_country.py`)
generate candidate matches between USPTO patent inventors and U.S. census
individuals. This module holds the logic they share: name cleaning, data loading
(with the published-table column mapping), geographic blocking-key derivation and
a few small utilities.

"""

import gc
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from unidecode import unidecode
from multiprocessing import Pool

# ---------------------------------------------------------------------------#
# Gender mapping
#
#   Census `sex` codes:  1 = Male, 2 = Female, 8 = Illegible, 9 = Missing.
#   Inventor `gender`:   'M', 'F', 'U' (unknown).
#
# A request for gender 'U' (unknown) matches inventors whose gender could not
# be determined against census individuals of either sex.
# ---------------------------------------------------------------------------#

GENDER_TO_CENSUS_SEX = {
    "M": [1],
    "F": [2],
    "U": [1, 2],
}

GEO_LEVELS = ("county", "state")


# ---------------------------------------------------------------------------#
# Small utilities
# ---------------------------------------------------------------------------#

def timer(start, end):
    """Print an elapsed interval as HH:MM:SS.ss."""
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def format_elapsed(elapsed):
    """Return a human-readable string for a timedelta."""
    days = elapsed.days
    hours, remainder = divmod(elapsed.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{days} days, {hours} hours, {minutes} minutes"


def parallelize_dataframe(df, func, n_cores=8):
    """Apply `func` to `df` in `n_cores` chunks using a process pool."""
    if len(df) == 0:
        return func(df)
    n_cores = max(1, min(n_cores, len(df)))
    # Split by row position via an index array; np.array_split on a DataFrame
    # triggers a deprecated DataFrame.swapaxes call, so split the positions.
    df_split = [df.iloc[idx] for idx in np.array_split(np.arange(len(df)), n_cores)]
    pool = Pool(n_cores)
    try:
        df = pd.concat(pool.map(func, df_split))
    finally:
        pool.close()
        pool.join()
    return df


def cleanother(df):
    """Normalise the inventor `name` column into a lowercase `clean_name`.

    Transliterates to ASCII, lowercases, strips honorifics/`deceased`, removes
    any non-alphabetic characters and trims whitespace.
    """
    df = df[df["name"].notnull()].copy()
    df["name"] = df["name"].map(lambda x: unidecode(x))
    df["clean_name"] = df["name"].str.lower()
    df["clean_name"] = df["clean_name"].str.replace(r"\s+deceased$", "", regex=True)
    df["clean_name"] = df["clean_name"].str.replace(r"^prof\.?\s+dr\.?\s+", "", regex=True)
    df["clean_name"] = df["clean_name"].str.replace(r"^dr\.?\s+prof\.?\s+", "", regex=True)
    df["clean_name"] = df["clean_name"].str.replace(r"^dr\.?\s+", "", regex=True)
    df["clean_name"] = df["clean_name"].str.replace(r"[^a-z\s]+", "", regex=True)
    df["clean_name"] = df["clean_name"].str.replace(r"^\s+", "", regex=True)
    df["clean_name"] = df["clean_name"].str.replace(r"\s+$", "", regex=True)
    return df


# ---------------------------------------------------------------------------#
# Geographic blocking key
# ---------------------------------------------------------------------------#

def normalize_fips(series):
    """Coerce a county-FIPS column to a zero-padded 5-digit object string.

    County FIPS = 2-digit state + 3-digit county. Both the inventor and census
    sides must use the same representation or the geographic `==` join silently
    fails. This accepts string OR numeric input (an integer/float FIPS would
    have dropped its leading zero, e.g. 1001 -> "01001") and returns plain
    object strings with `None` for missing values.
    """
    if pd.api.types.is_numeric_dtype(series):
        s = series.astype("Int64").astype("string")
    else:
        s = series.astype("string")
    s = s.str.zfill(5)
    return s.astype(object).where(s.notna(), None)


def add_geo_key(df, geo_level, county_col="county_fips_code"):
    """Add a `geo_key` column used to block candidate pairs.

    county  -> 5-digit county FIPS
    state   -> first 2 digits of the county FIPS (state FIPS)
    """
    if geo_level == "county":
        df["geo_key"] = df[county_col]
    elif geo_level == "state":
        df["geo_key"] = df[county_col].str[:2]
    else:
        raise ValueError(f"Unknown geo_level: {geo_level!r}; expected one of {GEO_LEVELS}")
    return df


# ---------------------------------------------------------------------------#
# Data loaders
# ---------------------------------------------------------------------------#

def load_inventors(path, gender, geo_level):
    """Load and filter the published patent-inventor table.

    Expects the published deposit schema (tab-separated), with columns:
        PatInvID, pubnumber, progr, pubdate, pubyear, name, citizen,
        country, state, county, city, FIPS, gender

    Keeps the requested gender, U.S. residents with a known county FIPS, derives
    a numeric `year`, the 5-digit `county_fips_code` and the `geo_key`.
    """
    patents = pd.read_csv(path, sep="\t", dtype={"FIPS": object})

    # Publication year: prefer the explicit column, else parse from the date.
    if "pubyear" in patents.columns:
        patents["year"] = pd.to_numeric(patents["pubyear"], errors="coerce").astype("Int64")
    else:
        patents["year"] = pd.to_numeric(patents["pubdate"].str[:4], errors="coerce").astype("Int64")

    # Keep requested gender (for 'U', inventor gender is literally 'U').
    patents = patents[patents["gender"] == gender]

    # Keep U.S. inventors with a usable county FIPS.
    patents = patents[patents["country"] == "UNITED STATES"]
    patents = patents[patents["FIPS"].notnull()]
    patents["county_fips_code"] = normalize_fips(patents["FIPS"])

    patents = add_geo_key(patents, geo_level, county_col="county_fips_code")
    return patents


def load_census(path, gender, geo_level, age_min=8, age_max=80):
    """Load a single pre-cleaned census parquet, filtered by age and sex.

    Expects the cleaned-census schema with columns:
        histid, clean_fn, clean_ln, county_fips_code, age, birthyr, sex, city

    Age/sex filters are pushed down to the parquet reader. NOTE: census files
    are restricted IPUMS USA extracts and are not redistributable.
    """
    census_sex = GENDER_TO_CENSUS_SEX[gender]
    filters = [
        ("age", ">=", age_min),
        ("age", "<=", age_max),
        ("sex", "in", census_sex),
    ]
    census = pq.read_table(path, filters=filters).to_pandas()
    # Align the census FIPS representation with the inventor side (both must be
    # zero-padded 5-digit strings) so the geographic join cannot silently fail.
    census["county_fips_code"] = normalize_fips(census["county_fips_code"])
    census = add_geo_key(census, geo_level, county_col="county_fips_code")
    return census


def census_full_names(census, geo_value):
    """Build unique, indexed census full names within one geographic bucket.

    Returns (census_names, unique_names) where `unique_names` has a `cenidx`
    surrogate key and `census_names` carries `histid` + `cenidx` per record.
    """
    census_names = census[census["geo_key"] == geo_value].copy()
    census_names = census_names[census_names["clean_fn"].notnull()]
    census_names = census_names[census_names["clean_ln"].notnull()]
    census_names = census_names[census_names["clean_fn"].str.len() > 0]
    census_names = census_names[census_names["clean_ln"].str.len() > 0]
    census_names["clean_name"] = census_names["clean_fn"] + " " + census_names["clean_ln"]
    census_names = census_names[census_names["clean_name"].str.len() > 3]

    unique_names = census_names[census_names["clean_name"].notnull()][["clean_name"]].drop_duplicates()
    unique_names = unique_names.reset_index(drop=True)
    unique_names["cenidx"] = unique_names.index + 1

    census_names = pd.merge(census_names, unique_names, how="left", on="clean_name")
    return census_names, unique_names


def iter_geo_values(census, max_buckets=None):
    """Sorted list of non-null geographic buckets present in the census slice.

    `max_buckets` bounds the number processed (useful for quick smoke tests).
    """
    values = census[census["geo_key"].notnull()]["geo_key"].drop_duplicates().tolist()
    values.sort()
    if max_buckets is not None:
        values = values[:max_buckets]
    return values
