#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Technical validation 4: attach census household relatives to the deceased-inventor
validation roster links.

Paper-analysis add-on (Scientific Data descriptor) -- NOT a pipeline stage.

A patent filed by an executor/administrator proves the inventor died before `pubyear`,
so in the census wave BEFORE the patent (`previous_census_year`) the inventor was
alive and the model's link there should be a real person. When the estate
representative shares the deceased inventor's surname, he or she is almost surely a
close relative -- feminine role words (executrix/administratrix/widow) the widow,
masculine ones (executor/administrator) a son, brother or father. This stage joins the
validation roster (`2c_accept_verdicts.py` output; the CLEAN `deceased_names` /
`representatives` columns are used, never the raw OCR `name`) to the model's link in
each inventor's `previous_census_year` (release-table rows with `from_ctp == False`,
verified identical to the raw `3_predict.py` output) and, from the RAW IPUMS person
file, classifies every household co-resident's relationship TO the linked inventor --
pointer variables (`sploc`/`momloc`/`poploc`) first, a conservative `relate`-code
fallback via the household head otherwise (see `_relative_common`). Eligible
comparators are ALL classified relatives of either sex, under sex-refined names
(spouse; son/daughter; brother/sister; father/mother); the role word gates which
comparators a representative may match BY THE RELATIVE'S SEX (feminine roles ->
female relatives: widow/daughter/sister/mother; masculine -> male relatives:
son/brother/father; unknown-gender roles -> either).

Outputs under --output-dir (console mirrors the attrition table):
  relative_attach_pairs.csv   one row per (link x same-surname rep x eligible
                              relative), incl. sex_gate_pass=False rows for audit
  relative_attach_rows.csv    one row per roster row -- stage 5's unit of analysis
  relative_attrition.csv      roster -> checkable attrition counts

Stage 5 (`5_relative_report.py`) scores the widow--spouse comparison against a
sex-stratified permutation null (the wife is the one relative almost always
co-resident with the inventor, so the spouse is the test's one high-sensitivity
comparison); this stage classifies and records ALL household relatives as the
audit record behind it. The `pair_concordant`/`concordant`/`checkable` columns
here are informational all-relatives previews at this run's --cutoff, so
`checkable` is a superset of stage 5's spouse-checkable set. Heavy census collects use the polars streaming
engine: on census parquet stored as partitioned directories (thousands of part files,
some zero-length), polars' in-memory hash join can panic non-deterministically, while
the streaming join path is unaffected.
All paths are CLI arguments; nothing is hardcoded.
"""

import argparse
import os
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Attach census household relatives (spouse/son/brother/father) to "
                    "the deceased-inventor validation roster's pre-patent model links.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--roster", required=True,
                   help="validation_sample_deceased.csv (the 2c validation roster).")
    p.add_argument("--links", required=True,
                   help="Released links parquet (CTP-annotated table; only "
                        "from_ctp==False model rows are used).")
    p.add_argument("--raw-census-dir", required=True,
                   help="Dir of raw IPUMS person files (US{year}P.parquet per wave).")
    p.add_argument("--person-template", default="US{year}P.parquet",
                   help="Per-year person-file name under --raw-census-dir.")
    p.add_argument("--output-dir", required=True,
                   help="Directory for the three output CSVs.")
    p.add_argument("--years", type=int, nargs="*", default=None,
                   help="Restrict to these previous_census_year waves (default: all).")
    p.add_argument("--surname-cutoff", type=float, default=90.0,
                   help="rapidfuzz fallback cutoff for the rep-vs-deceased surname "
                        "gate (exact match always passes).")
    p.add_argument("--cutoff", type=float, default=85.0,
                   help="rapidfuzz cutoff for the informational given-name "
                        "concordance columns (stage 5 recomputes).")
    p.add_argument("--nicknames", nargs="+",
                   default=["data/misc/given_name_nicknames.json",
                            "data/misc/given_name_diminutives.json"],
                   help="Nickname/diminutive cluster JSON(s), merged (earlier "
                        "files win on conflicts). Curated, independently attested "
                        "clusters ONLY -- see _relative_common.load_nickmaps.")
    p.add_argument("--given-sex-ref", default=None,
                   help="given-name sex reference parquet (_given_sex_ref.py "
                        "output). When set, a representative's role-word gender "
                        "is corrected when the given name decisively contradicts "
                        "it (the Eugenia-the-'executor' pattern). Omit to trust "
                        "role words as-is.")
    p.add_argument("--sex-hi", type=float, default=0.9,
                   help="female_share at/above which a masculine-role rep name "
                        "is treated as female.")
    p.add_argument("--sex-lo", type=float, default=0.1,
                   help="female_share at/below which a feminine-role rep name "
                        "is treated as male.")
    p.add_argument("--spill-dir", default=None,
                   help="Directory for polars streaming spill (sets POLARS_TEMP_DIR/"
                        "TMPDIR). Default <output-dir>/.polars_tmp. MUST be real "
                        "disk, never a tmpfs /tmp.")
    return p.parse_args(argv)


REQUIRED_CENSUS_COLS = ["histid", "namefrst", "namelast", "sex", "serialp",
                        "relate", "pernum"]
POINTER_COLS = ["sploc", "momloc", "poploc"]

PAIR_COLS = [
    "PatInvID", "pubnumber", "progr", "pubyear", "previous_census_year", "histid",
    "probability", "inventor_name", "deceased_name", "deceased_surname",
    "rep_index", "rep_full", "rep_given", "rep_surname", "rep_role", "rep_gender",
    "rep_role_gender", "surname_match_type", "surname_score",
    "relative_histid", "relative_pernum", "relationship", "method",
    "relative_given", "relative_surname", "relative_sex", "relative_birthyr",
    "sex_gate_pass", "pair_concordant",
]

ROW_COLS = [
    "PatInvID", "pubnumber", "progr", "pubyear", "previous_census_year",
    "deceased_name", "deceased_given", "deceased_surname",
    "n_reps", "n_reps_same_surname",
    "rep_givens_F", "rep_givens_M", "rep_givens_U",
    "has_model_link", "histid", "probability", "inventor_name",
    "household_found", "n_household", "pointers_available",
    "n_rel_spouse", "n_rel_son", "n_rel_daughter", "n_rel_brother", "n_rel_sister",
    "n_rel_father", "n_rel_mother",
    "rel_givens_spouse", "rel_givens_son", "rel_givens_daughter",
    "rel_givens_brother", "rel_givens_sister", "rel_givens_father",
    "rel_givens_mother", "rel_givens_F", "rel_givens_M",
    "rel_methods", "n_pairs_gated", "checkable", "concordant",
    "concordant_relationships", "concordant_method",
]


def load_roster(pl, path, surname_cutoff, rc):
    """Parse the validation roster into one dict per row: cleaned deceased name,
    representatives split/gated/gendered. Every roster row is kept (attrition needs
    the failures too); only an unparseable previous_census_year drops a row."""
    df = pl.read_csv(path, infer_schema_length=0)
    rows = []
    for r in df.iter_rows(named=True):
        try:
            pubyear = int(float(str(r["pubyear"]).strip()))
            prev_year = int(float(str(r["previous_census_year"]).strip()))
        except (ValueError, TypeError, KeyError):
            continue
        dec_given, dec_surname, _ = rc.split_person_name(r.get("deceased_names", ""))
        reps_raw = [x.strip() for x in str(r.get("representatives", "") or "").split(";")
                    if x.strip()]
        reps = []
        for i, (full, role, gender) in enumerate(
                rc.assign_rep_roles(reps_raw, r.get("roles", ""))):
            given, surname, _ = rc.split_person_name(full)
            mtype, score = rc.surname_gate(surname, dec_surname, surname_cutoff)
            reps.append({"index": i, "full": full, "given": given,
                         "surname": surname, "role": role, "gender": gender,
                         "role_gender": gender,
                         "surname_match_type": mtype, "surname_score": score})
        rows.append({
            "PatInvID": str(r["PatInvID"]).strip(),
            "pubnumber": str(r.get("pubnumber", "")).strip(),
            "progr": str(r.get("progr", "")).strip(),
            "pubyear": pubyear,
            "previous_census_year": prev_year,
            "deceased_name": str(r.get("deceased_names", "") or "").strip(),
            "deceased_given": dec_given,
            "deceased_surname": dec_surname,
            "reps": reps,
        })
    return rows


def load_model_links(pl, path, patinv_ids):
    """(PatInvID, census_year) -> (histid, probability, inventor_name) for the model
    rows (from_ctp==False) of the release table, restricted to the roster inventors.
    Exits non-zero if the restricted subset is not unique on the key."""
    lf = pl.scan_parquet(path)
    cols = lf.collect_schema().names()
    sel = [pl.col("PatInvID").cast(pl.Utf8).str.strip_chars(),
           pl.col("census_year").cast(pl.Int64),
           pl.col("histid").cast(pl.Utf8),
           pl.col("probability").cast(pl.Float64)]
    sel.append(pl.col("inventor_name").cast(pl.Utf8) if "inventor_name" in cols
               else pl.lit(None, dtype=pl.Utf8).alias("inventor_name"))
    lf = lf.select(sel + ([pl.col("from_ctp")] if "from_ctp" in cols else []))
    if "from_ctp" in cols:
        lf = lf.filter(~pl.col("from_ctp")).drop("from_ctp")
    sub = lf.filter(pl.col("PatInvID").is_in(sorted(patinv_ids))).collect()
    if sub.select("PatInvID", "census_year").is_duplicated().any():
        raise SystemExit("[6] model links are not unique on (PatInvID, census_year)")
    return {(r["PatInvID"], r["census_year"]):
            (r["histid"], r["probability"], r["inventor_name"])
            for r in sub.iter_rows(named=True)}


def load_wave_households(pl, person_path, histids):
    """Scan one wave's raw person file for the matched histids' households. Returns
    (person_by_histid, members_by_serialp, pointers_available). Missing pointer
    columns are WARNed and null-injected (relate-code fallback only)."""
    if os.path.isdir(person_path):
        scan = pl.scan_parquet(person_path.rstrip("/\\") + "/**/*.parquet")
    else:
        scan = pl.scan_parquet(person_path)
    names = scan.collect_schema().names()
    missing = [c for c in REQUIRED_CENSUS_COLS if c not in names]
    if missing:
        raise SystemExit(f"[6] {person_path}: required census columns missing: "
                         f"{missing}")
    missing_ptrs = [c for c in POINTER_COLS if c not in names]
    pointers_available = not missing_ptrs
    if missing_ptrs:
        print(f"[6] WARN {person_path}: pointer column(s) {missing_ptrs} absent; "
              f"relate-code fallback only", flush=True)
    sel = [pl.col("histid").cast(pl.Utf8),
           pl.col("namefrst").cast(pl.Utf8),
           pl.col("namelast").cast(pl.Utf8),
           pl.col("sex").cast(pl.Int64, strict=False),
           pl.col("serialp").cast(pl.Int64, strict=False),
           pl.col("relate").cast(pl.Int64, strict=False),
           pl.col("pernum").cast(pl.Int64, strict=False)]
    for c in POINTER_COLS:
        sel.append(pl.col(c).cast(pl.Int64, strict=False) if c in names
                   else pl.lit(None, dtype=pl.Int64).alias(c))
    sel.append(pl.col("birthyr").cast(pl.Int64, strict=False) if "birthyr" in names
               else pl.lit(None, dtype=pl.Int64).alias("birthyr"))
    person = scan.select(sel)

    hist = pl.DataFrame({"histid": sorted(histids)})
    matched = (person.join(hist.lazy(), on="histid", how="semi")
               .collect(engine="streaming"))
    if matched.height == 0:
        return {}, {}, pointers_available
    serialps = matched.select("serialp").drop_nulls().unique().to_series().to_list()
    hh = (person.filter(pl.col("serialp").is_in(serialps))
          .collect(engine="streaming"))
    by_serial = {}
    for row in hh.iter_rows(named=True):
        by_serial.setdefault(row["serialp"], []).append(row)
    by_histid = {row["histid"]: row for row in matched.iter_rows(named=True)}
    return by_histid, by_serial, pointers_available


def eligible_relatives(rc, inv, members):
    """Classify every co-resident vs the inventor and keep ALL classified relatives
    of either sex as comparators, under sex-refined names (spouse; son/daughter;
    brother/sister; father/mother). Returns (relatives, counts) where each relative
    dict carries the relationship name, method, given/surname, sex, birthyr."""
    out = []
    counts = {r: 0 for r in rc.ELIGIBLE_RELATIONSHIPS}
    for m in members:
        if m["histid"] == inv["histid"]:
            continue
        rel, method = rc.classify_relationship(inv, m)
        if rel is None:
            continue
        name = rc.refine_relationship(rel, m["sex"])
        if name is None:
            continue
        counts[name] += 1
        out.append({"histid": m["histid"], "pernum": m["pernum"],
                    "relationship": name, "method": method,
                    "given": rc.first_token(m["namefrst"]),
                    "surname": rc.norm_surname(m["namelast"]),
                    "sex": m["sex"], "birthyr": m["birthyr"]})
    return out, counts


def gate_pass(rep_gender, relative_sex):
    """The role-word sex gate, on the RELATIVE'S sex: feminine-role reps may match
    only female relatives (widow, daughter, sister, mother), masculine-role reps
    only male ones (son, brother, father), unknown-gender reps may match either."""
    if rep_gender == "F":
        return relative_sex == 2
    if rep_gender == "M":
        return relative_sex == 1
    return True


def main(argv=None):
    args = parse_args(argv)
    t0 = time.perf_counter()

    spill_dir = args.spill_dir or os.path.join(args.output_dir, ".polars_tmp")
    os.makedirs(spill_dir, exist_ok=True)
    os.environ["POLARS_TEMP_DIR"] = spill_dir
    os.environ["TMPDIR"] = spill_dir
    import polars as pl
    import _relative_common as rc

    os.makedirs(args.output_dir, exist_ok=True)
    nickmap = rc.load_nickmaps(args.nicknames, tag="[6]")

    roster = load_roster(pl, args.roster, args.surname_cutoff, rc)
    t0 = rc._lap(f"roster: {len(roster):,} rows", t0)

    fshare = rc.load_given_sex_ref(args.given_sex_ref, tag="[6]")
    if fshare:
        n_corr = 0
        for row in roster:
            for p in row["reps"]:
                p["gender"] = rc.correct_role_gender(
                    p["given"], p["role_gender"], fshare, args.sex_hi, args.sex_lo)
                n_corr += p["gender"] != p["role_gender"]
        print(f"[6] given-sex reference: {len(fshare):,} names; corrected the "
              f"role-word gender of {n_corr} representative(s)", flush=True)

    links = load_model_links(pl, args.links, {r["PatInvID"] for r in roster})
    t0 = rc._lap(f"model links for roster inventors: {len(links):,}", t0)

    for row in roster:
        hit = links.get((row["PatInvID"], row["previous_census_year"]))
        row["has_model_link"] = hit is not None
        row["histid"], row["probability"], row["inventor_name"] = \
            hit if hit else (None, None, None)
        row["household_found"] = False
        row["n_household"] = 0
        row["pointers_available"] = None
        row["relatives"] = []
        row["rel_counts"] = {r: 0 for r in rc.ELIGIBLE_RELATIONSHIPS}

    years = sorted({r["previous_census_year"] for r in roster if r["has_model_link"]})
    if args.years:
        years = [y for y in years if y in set(args.years)]
    match = rc._MatchCache(args.cutoff, nickmap)

    for year in years:
        sub = [r for r in roster
               if r["has_model_link"] and r["previous_census_year"] == year]
        person_path = os.path.join(args.raw_census_dir,
                                   args.person_template.format(year=year))
        if not os.path.exists(person_path):
            print(f"[6] WARN {year}: person file missing ({person_path}); skipping "
                  f"{len(sub)} links", flush=True)
            continue
        by_histid, by_serial, ptrs = load_wave_households(
            pl, person_path, {r["histid"] for r in sub})
        n_hh = 0
        for row in sub:
            row["pointers_available"] = ptrs
            inv = by_histid.get(row["histid"])
            if inv is None:
                continue
            members = by_serial.get(inv["serialp"], [])
            row["household_found"] = True
            row["n_household"] = len(members)
            row["relatives"], row["rel_counts"] = eligible_relatives(rc, inv, members)
            n_hh += 1
        print(f"[6] {year}: {len(sub)} linked roster rows, {n_hh} households "
              f"resolved (pointers={'yes' if ptrs else 'NO'})", flush=True)
        t0 = rc._lap(f"wave {year}", t0)

    # ---- pair- and row-level outputs + attrition -------------------------------
    roster.sort(key=lambda r: (r["previous_census_year"], r["PatInvID"]))
    pair_rows, row_rows = [], []
    att = {k: 0 for k in ("roster", "parsed_surname", "any_rep", "same_surname_rep",
                          "model_link", "household", "checkable", "concordant")}
    for row in roster:
        att["roster"] += 1
        ss_reps = [p for p in row["reps"]
                   if p["surname_match_type"] and p["given"]]
        rep_givens = {g: sorted({p["given"] for p in ss_reps if p["gender"] == g})
                      for g in rc.GENDER_STRATA}
        rel_givens = {r: sorted({x["given"] for x in row["relatives"]
                                 if x["relationship"] == r and x["given"]})
                      for r in rc.ELIGIBLE_RELATIONSHIPS}
        rel_by_sex = {sx: sorted({x["given"] for x in row["relatives"]
                                  if x["sex"] == sx and x["given"]})
                      for sx in (1, 2)}
        methods = sorted({x["method"] for x in row["relatives"]})

        n_pairs_gated = 0
        concordant_rels, concordant_methods = set(), set()
        for p in ss_reps:
            for x in row["relatives"]:
                gp = gate_pass(p["gender"], x["sex"])
                pc = match(p["given"], x["given"]) if x["given"] else False
                if gp and x["given"]:
                    n_pairs_gated += 1
                    if pc:
                        concordant_rels.add(x["relationship"])
                        concordant_methods.add(x["method"])
                pair_rows.append({
                    "PatInvID": row["PatInvID"], "pubnumber": row["pubnumber"],
                    "progr": row["progr"], "pubyear": row["pubyear"],
                    "previous_census_year": row["previous_census_year"],
                    "histid": row["histid"], "probability": row["probability"],
                    "inventor_name": row["inventor_name"],
                    "deceased_name": row["deceased_name"],
                    "deceased_surname": row["deceased_surname"],
                    "rep_index": p["index"], "rep_full": p["full"],
                    "rep_given": p["given"], "rep_surname": p["surname"],
                    "rep_role": p["role"], "rep_gender": p["gender"],
                    "rep_role_gender": p["role_gender"],
                    "surname_match_type": p["surname_match_type"],
                    "surname_score": p["surname_score"],
                    "relative_histid": x["histid"], "relative_pernum": x["pernum"],
                    "relationship": x["relationship"], "method": x["method"],
                    "relative_given": x["given"], "relative_surname": x["surname"],
                    "relative_sex": x["sex"], "relative_birthyr": x["birthyr"],
                    "sex_gate_pass": gp, "pair_concordant": gp and pc,
                })
        checkable = bool(ss_reps) and row["has_model_link"] \
            and row["household_found"] and n_pairs_gated > 0
        is_conc = bool(concordant_rels)

        if row["deceased_surname"]:
            att["parsed_surname"] += 1
            if row["reps"]:
                att["any_rep"] += 1
                if ss_reps:
                    att["same_surname_rep"] += 1
                    if row["has_model_link"]:
                        att["model_link"] += 1
                        if row["household_found"]:
                            att["household"] += 1
                            if checkable:
                                att["checkable"] += 1
                                if is_conc:
                                    att["concordant"] += 1

        row_rows.append({
            "PatInvID": row["PatInvID"], "pubnumber": row["pubnumber"],
            "progr": row["progr"], "pubyear": row["pubyear"],
            "previous_census_year": row["previous_census_year"],
            "deceased_name": row["deceased_name"],
            "deceased_given": row["deceased_given"],
            "deceased_surname": row["deceased_surname"],
            "n_reps": len(row["reps"]), "n_reps_same_surname": len(ss_reps),
            "rep_givens_F": "|".join(rep_givens["F"]),
            "rep_givens_M": "|".join(rep_givens["M"]),
            "rep_givens_U": "|".join(rep_givens["U"]),
            "has_model_link": row["has_model_link"], "histid": row["histid"],
            "probability": row["probability"],
            "inventor_name": row["inventor_name"],
            "household_found": row["household_found"],
            "n_household": row["n_household"],
            "pointers_available": row["pointers_available"],
            "n_rel_spouse": row["rel_counts"]["spouse"],
            "n_rel_son": row["rel_counts"]["son"],
            "n_rel_daughter": row["rel_counts"]["daughter"],
            "n_rel_brother": row["rel_counts"]["brother"],
            "n_rel_sister": row["rel_counts"]["sister"],
            "n_rel_father": row["rel_counts"]["father"],
            "n_rel_mother": row["rel_counts"]["mother"],
            "rel_givens_spouse": "|".join(rel_givens["spouse"]),
            "rel_givens_son": "|".join(rel_givens["son"]),
            "rel_givens_daughter": "|".join(rel_givens["daughter"]),
            "rel_givens_brother": "|".join(rel_givens["brother"]),
            "rel_givens_sister": "|".join(rel_givens["sister"]),
            "rel_givens_father": "|".join(rel_givens["father"]),
            "rel_givens_mother": "|".join(rel_givens["mother"]),
            "rel_givens_F": "|".join(rel_by_sex[2]),
            "rel_givens_M": "|".join(rel_by_sex[1]),
            "rel_methods": "|".join(methods), "n_pairs_gated": n_pairs_gated,
            "checkable": checkable, "concordant": is_conc,
            "concordant_relationships": "|".join(sorted(concordant_rels)),
            "concordant_method": "|".join(sorted(concordant_methods)),
        })

    pair_rows.sort(key=lambda r: (r["previous_census_year"], r["PatInvID"],
                                  r["rep_index"], r["relative_pernum"] or 0))
    pairs_df = (pl.DataFrame(pair_rows).select(PAIR_COLS) if pair_rows
                else pl.DataFrame(schema={c: pl.Utf8 for c in PAIR_COLS}))
    rows_df = pl.DataFrame(row_rows).select(ROW_COLS)

    steps = [("0", "roster rows", att["roster"]),
             ("1", "parsed deceased surname", att["parsed_surname"]),
             ("2", ">=1 representative", att["any_rep"]),
             ("3", ">=1 same-surname rep with a given name", att["same_surname_rep"]),
             ("4", "model link in previous_census_year (from_ctp=False)",
              att["model_link"]),
             ("5", "household resolved in raw census", att["household"]),
             ("6", ">=1 sex-gated eligible relative -> CHECKABLE", att["checkable"]),
             ("7", f"concordant at cutoff {args.cutoff:g} (informational)",
              att["concordant"])]
    attr_df = pl.DataFrame({"step": [s[0] for s in steps],
                            "label": [s[1] for s in steps],
                            "n": [s[2] for s in steps]})
    print("\n== attrition ==")
    for s, lab, n in steps:
        print(f"  [{s}] {lab}: {n:,}", flush=True)

    pairs_path = os.path.join(args.output_dir, "relative_attach_pairs.csv")
    rows_path = os.path.join(args.output_dir, "relative_attach_rows.csv")
    attr_path = os.path.join(args.output_dir, "relative_attrition.csv")
    pairs_df.write_csv(pairs_path)
    rows_df.write_csv(rows_path)
    attr_df.write_csv(attr_path)
    print(f"\n[6] wrote {pairs_df.height:,} pair rows -> {pairs_path}", flush=True)
    print(f"[6] wrote {rows_df.height:,} roster rows -> {rows_path}", flush=True)
    print(f"[6] wrote attrition -> {attr_path}", flush=True)
    rc._lap("done", t0)


if __name__ == "__main__":
    main()
