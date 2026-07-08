#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Relatedness stage 2: attach occupation + patent description strings to
matched_candidates, processing one census year at a time.

The big candidate table is handled in **polars** (lazy scan + streaming
sink) so memory stays bounded to a single year.

Produces, per candidate row:
  - occ_desc_short = occstr
  - occ_desc_long  = occstr + IND1950 name + OCC1950 name
  - pat_desc_short / pat_desc_long (merged from stage 1 on pubnumber)

Inputs:
  --matched-candidates  directory of 2b per-year `matched_candidates_{year}.parquet`
                        (has occstr/ind1950/occ1950/pubnumber)
  --patent-strings      stage-1 patent_strings parquet
  --occ1950 / --ind1950 misc occupation/industry label xlsx (code col + 'value')
  --output-dir          directory for per-year `matched_candidates_with_strings_{year}.parquet`
"""

import argparse
import os

import numpy as np
import pandas as pd
import polars as pl

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
DEF_OCC1950 = os.path.join(DATA, "occ_ind_1950", "OCC1950.xlsx")
DEF_IND1950 = os.path.join(DATA, "occ_ind_1950", "IND1950.xlsx")

ALL_YEARS = [1850, 1860, 1870, 1880, 1900, 1910, 1920, 1930, 1940]


def remove_duplicate_words(text):
    """Simple lowercase + trailing-'s' dedup."""
    if not isinstance(text, str):
        return text
    seen = set()
    result = []
    for word in text.split():
        norm = word.lower().rstrip("s")
        if norm not in seen:
            seen.add(norm)
            result.append(word)
    return " ".join(result)


def clean_ind1950(ind1950):
    s = ind1950["value"].astype("string")
    s = (s.str.replace("^Miscellaneous ", "", regex=True)
          .str.replace(r"\:|etc\.|\(|\)|\.", "", regex=True)
          .str.replace(r"\, except .*", "", regex=True)
          .str.replace(r"\-\-", " ", regex=True)
          .str.replace(r"\, level not specified", "", regex=True)
          .str.replace("^Not specified ", "", regex=True)
          .str.replace("^Other and ", "", regex=True)
          .replace("Professional equipment and supplies", np.nan)
          .str.upper())
    ind1950 = ind1950.copy()
    ind1950["value"] = s
    return ind1950.rename(columns={"value": "ind1950_str"})


def clean_occ1950(occ1950):
    s = occ1950["value"].astype("string")
    s = (s.str.replace(r"\:|etc\.|\(|\)|\.", "", regex=True)
          .str.replace(r"\, except.*", "", regex=True)
          .str.replace(r"\/", " ", regex=True)
          .str.replace(", occupation undetermined", "", regex=False)
          .str.replace(", other specified trades", "", regex=False)
          .str.replace(" trades", "", regex=False)
          .str.replace(", self-employed", "", regex=False)
          .str.replace("professional", "", regex=False)
          .str.replace(", private household", "", regex=False)
          .str.replace("^Farm.*", "Farm", regex=True)
          .str.replace("^Apprentices, ", "", regex=True)
          .str.replace("^Apprentice ", "", regex=True)
          .str.replace("^Imputed ", "", regex=True)
          .str.replace("^Attendents, ", "", regex=True)
          .replace("Works", np.nan).replace("Laborers", np.nan)
          .replace("Apprentices", np.nan).replace("Service workers", np.nan)
          .str.upper())
    occ1950 = occ1950.copy()
    occ1950["value"] = s
    return occ1950.rename(columns={"value": "occ1950_str"})


def label_frame(df, code_col, str_col):
    """pandas label table -> polars (Int64 code, str). Drops null/dup codes."""
    out = pl.from_pandas(df[[code_col, str_col]])
    return (out.with_columns(pl.col(code_col).cast(pl.Int64))
               .drop_nulls(code_col).unique(subset=[code_col]))


def build_occ_strings(year_lf, ind_pl, occ_pl):
    """Per-year occupation strings on the unique (histid, census_year) set.

    Returns an in-memory polars DataFrame keyed on (histid, census_year) with
    occ_desc_short / occ_desc_long. Memory ~ the deduped person set, not the
    full patent x person candidate table."""
    uniq = (year_lf.select(["histid", "census_year", "occstr", "ind1950", "occ1950"])
                   .unique()
                   .with_columns([pl.col("ind1950").cast(pl.Int64),
                                  pl.col("occ1950").cast(pl.Int64)])
                   .collect(engine="streaming"))

    uniq = (uniq.join(ind_pl, on="ind1950", how="left")
                .join(occ_pl, on="occ1950", how="left"))

    long = (pl.concat_str([pl.col("occstr").fill_null(""),
                           pl.col("ind1950_str").fill_null(""),
                           pl.col("occ1950_str").fill_null("")], separator=" ")
              .str.strip_chars())
    uniq = uniq.with_columns(
        pl.when(long == "").then(None).otherwise(long).alias("occ_desc_long"))

    # remove_duplicate_words on the DISTINCT long strings, then map back.
    longs = uniq.get_column("occ_desc_long").drop_nulls().unique().to_list()
    mapping = {s: remove_duplicate_words(s) for s in longs}
    uniq = uniq.with_columns(pl.col("occ_desc_long").replace(mapping))

    return (uniq.rename({"occstr": "occ_desc_short"})
                .select(["histid", "census_year", "occ_desc_short", "occ_desc_long"])
                .unique(subset=["histid", "census_year"]))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Attach occupation + patent strings to matched_candidates (per year).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--matched-candidates", required=True,
                   help="Directory of per-year matched_candidates_{year}.parquet.")
    p.add_argument("--patent-strings", required=True, help="Stage-1 patent_strings parquet.")
    p.add_argument("--occ1950", default=DEF_OCC1950)
    p.add_argument("--ind1950", default=DEF_IND1950)
    p.add_argument("--years", type=int, nargs="*", default=ALL_YEARS)
    p.add_argument("--output-dir", required=True,
                   help="Directory for per-year matched_candidates_with_strings_{year}.parquet.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    ind_pl = label_frame(clean_ind1950(pd.read_excel(args.ind1950)), "ind1950", "ind1950_str")
    occ_pl = label_frame(clean_occ1950(pd.read_excel(args.occ1950)), "occ1950", "occ1950_str")

    patents = pl.scan_parquet(args.patent_strings).select(
        ["pubnumber", "pat_desc_short", "pat_desc_long"])

    for year in args.years:
        in_file = os.path.join(args.matched_candidates, f"matched_candidates_{year}.parquet")
        if not os.path.exists(in_file):
            print(f"[{year}] no input ({in_file}); skipping")
            continue

        year_lf = pl.scan_parquet(in_file)
        occ_strings = build_occ_strings(year_lf, ind_pl, occ_pl)
        print(f"[{year}] unique occupations: {occ_strings.height:,}")

        out_file = os.path.join(args.output_dir, f"matched_candidates_with_strings_{year}.parquet")
        (year_lf.join(occ_strings.lazy(), on=["histid", "census_year"], how="left")
                .join(patents, on="pubnumber", how="left")
                .sink_parquet(out_file, compression="zstd"))
        n = pl.scan_parquet(out_file).select(pl.len()).collect().item()
        print(f"[{year}] wrote {n:,} rows with strings -> {out_file}")


if __name__ == "__main__":
    main()
