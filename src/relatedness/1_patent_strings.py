#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Relatedness stage 1: build per-patent text strings for semantic scoring.

For every patent it produces two strings later embedded with JobBERT-v2:
  - pat_desc_short = patent title
  - pat_desc_long  = title + USPC class name + NBER category + NBER subcategory

Inputs (all under ipums_match/data, override via CLI):
  --titles      USPTO_patent_titles_1840_1950.tsv   (pubnumber, title)
  --classes     USPTO_patent_classes_1840_1950.tsv  (pubnumber, uspc_class, SubCat, Cat)
  --uspc-names  misc/uspc_class_names.xlsx           (uspc_class, title)
  --nber-names  misc/nber_category.txt               (Cat, SubCat, SubCatName, CatNameShort, CatNameLong)

Optionally restrict the output to patents that actually appear in a
matched_candidates table (--matched-candidates) to keep the file small.

Nothing is run automatically; this is a plain CLI script.
"""

import argparse
import os

import numpy as np
import pandas as pd
import polars as pl
from nltk.stem import PorterStemmer

# Default paths (relative to ipums_match/). Override on the command line.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
DEF_TITLES = os.path.join(DATA, "patent_titles", "patent_titles_clearpat.csv")
DEF_CLASSES = os.path.join(DATA, "USPTO_patent_classes_1840_1950.tsv")
DEF_USPC_NAMES = os.path.join(DATA, "misc", "uspc_class_names.xlsx")
DEF_NBER_NAMES = os.path.join(DATA, "misc", "nber_category.txt")


def read_titles(path):
    """Read the patent-titles table, picking the separator by extension.

    The clearpat deposit (`patent_titles_clearpat.csv`) is a properly-quoted
    comma CSV (titles with embedded commas are double-quoted); the legacy
    `USPTO_patent_titles_1840_1950.tsv` is tab-separated. Both expose
    `pubnumber, title`."""
    sep = "," if path.lower().endswith(".csv") else "\t"
    return pd.read_csv(path, sep=sep)


def remove_duplicate_words(text):
    """Drop repeated words (Porter-stem aware), preferring the plural form."""
    if not isinstance(text, str):
        return text
    ps = PorterStemmer()
    seen = {}
    result = []
    for word in text.split():
        stem = ps.stem(word)
        if stem not in seen:
            seen[stem] = word
            result.append(word)
        elif word.endswith("s") and seen[stem] != word:
            result.remove(seen[stem])
            result.append(word)
            seen[stem] = word
    return " ".join(result)


def clean_nber(nber_names):
    nber_names = nber_names.drop(columns=["CatNameShort"])
    nber_names = nber_names.rename(columns={"SubCatName": "nber_subcat", "CatNameLong": "nber_cat"})
    s = nber_names["nber_subcat"].astype("string")
    s = (s.str.replace(",", " ", regex=False)
          .str.replace("Miscellaneous-chemical", "Chemicals", regex=False)
          .str.replace(r"Mat\.", "Materials", regex=True)
          .str.replace("Proc", "Proces", regex=False)
          .str.replace(r"Med Inst\.", "Medical Instruments", regex=True)
          .str.replace(r"Miscellaneous\-Drgs\&Med", "Drugs Medical", regex=True)
          .str.replace("Miscellaneous-Mechanical", "Mechanical", regex=False)
          .str.replace("Miscellaneous-Elec", "Electronic", regex=False)
          .str.replace(r"\+\s", "", regex=True)
          .str.replace(r"\&\s", "", regex=True)
          .replace("Miscellaneous-Others", np.nan)
          .str.upper())
    nber_names["nber_subcat"] = s
    c = nber_names["nber_cat"].astype("string")
    c = c.replace("Others", np.nan).str.replace(r"\&\s", "", regex=True).str.upper()
    nber_names["nber_cat"] = c
    return nber_names


def clean_uspc(uspc_names):
    uspc_names = uspc_names.rename(columns={"title": "uspc_name"})
    s = uspc_names["uspc_name"].astype("string")
    for pat in [r"\,", r"\:", r"\;", r"\(", r"\)", r"\[", r"\]",
                r"e\.g\.", r"etc\.", r"or\s", "thereof", "therefor"]:
        s = s.str.replace(pat, "", regex=True)
    s = (s.str.replace(r"\/", " ", regex=True)
          .str.replace("Organic compounds containing.*", "Organic compounds", regex=True)
          .str.replace(r"\s\-\-\spart\sof\sthe\sclass\s.*", "", regex=True)
          .str.upper())
    uspc_names["uspc_name"] = s
    return uspc_names


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build per-patent description strings.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--titles", default=DEF_TITLES)
    p.add_argument("--classes", default=DEF_CLASSES, help="uspc_class/SubCat/Cat per pubnumber.")
    p.add_argument("--uspc-names", default=DEF_USPC_NAMES)
    p.add_argument("--nber-names", default=DEF_NBER_NAMES)
    p.add_argument("--matched-candidates", default=None,
                   help="Optional matched_candidates parquet (file or dir); restrict to its pubnumbers.")
    p.add_argument("--output", required=True, help="Output patent_strings parquet path.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    titles = read_titles(args.titles)                      # pubnumber, title

    # Optionally keep only patents that were actually matched.
    if args.matched_candidates:
        src = args.matched_candidates
        scan = src.rstrip("/\\") + "/**/*.parquet" if os.path.isdir(src) else src
        pubs = (pl.scan_parquet(scan).select("pubnumber").unique().collect()
                  .to_series().to_list())
        titles = titles[titles["pubnumber"].isin(set(pubs))]
        print(f"Restricted to {len(titles):,} matched patents")

    pct = round(titles["title"].isna().sum() / max(len(titles), 1) * 100, 2)
    print(f"{pct}% of patents have no title")

    uspc_nber = pd.read_csv(args.classes, sep="\t")        # pubnumber, uspc_class, SubCat, Cat
    nber_names = clean_nber(pd.read_table(args.nber_names))
    uspc_names = clean_uspc(pd.read_excel(args.uspc_names))

    patents = titles.merge(uspc_nber, how="left", on="pubnumber")
    patents = patents.merge(uspc_names, how="left", on="uspc_class")
    patents = patents.merge(nber_names, how="left", on=["SubCat", "Cat"])
    patents = patents.drop(columns=["uspc_class", "SubCat", "Cat"])

    patents["pat_desc_short"] = patents["title"]
    patents["pat_desc_long"] = patents[["title", "uspc_name", "nber_cat", "nber_subcat"]].apply(
        lambda x: " ".join(x.dropna()) if x.notna().any() else None, axis=1)
    patents = patents.drop(columns=["title", "uspc_name", "nber_cat", "nber_subcat"])

    patents["pat_desc_short"] = patents["pat_desc_short"].apply(remove_duplicate_words)
    patents["pat_desc_long"] = patents["pat_desc_long"].apply(remove_duplicate_words)

    patents.to_parquet(args.output, index=False, engine="pyarrow", compression="zstd")
    print(f"Wrote {len(patents):,} patent strings -> {args.output}")


if __name__ == "__main__":
    main()
