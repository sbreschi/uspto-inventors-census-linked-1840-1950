#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML model stage 4: post-match CTP consistency annotation + supplementation.

Runs directly after 3_predict.py. The classifier scores each census wave
independently, so one patent-inventor (PatInvID) can be linked to different census
people across waves with no guarantee they are the same individual. This stage uses
the Census Tree Project (CTP) adjacent-census crosswalks to cross-check that.

**NON-DESTRUCTIVE (by design):** unlike the manuscript's Step 6, this stage never
removes a link or overwrites a predicted histid. Every 3_predict row is kept exactly
as-is (its `histid` stays the model's prediction). CTP is used only to

  (a) ANNOTATE each model link with
        - `ctp_flag`   : 1 if the PatInvID is matched to >1 wave and the matched
                        histids do NOT form one CTP-connected chain (i.e. they are not
                        the same individual per CTP). Anchor-free; broadcast to every
                        row of the PatInvID.
        - `histid_ctp`: CTP's own view of this wave's histid, propagated from the
                        inventor's single highest-probability matched link (the
                        "anchor"; tie-break: closest to pubyear, then earliest wave).
                        On the anchor row `== histid`; on a matched wave where CTP
                        disagrees it holds CTP's alternative (the model histid is STILL
                        kept in `histid`); null where CTP offers no unambiguous link.
        - `ctp_action`: human-readable per-row label — `single` (lone matched wave),
                        `anchor`, `consistent` (`histid_ctp==histid`), `replaced`
                        (`histid_ctp != histid`, row kept — label only), `unverified`
                        (`histid_ctp` null), or `supplement` (an appended CTP row).
  (b) SUPPLEMENT — for each eligible wave (within the +/-10yr window of the patent's
      publication year) that the model did NOT match, retrieve a histid from CTP by
      tracing from the directly-CTP-adjacent matched wave(s). If several matched
      neighbours trace there they must AGREE on a single histid; a disagreement, or a
      CTP one-to-many fan-out, is skipped (conservative). Supplements are appended as
      new rows with `from_ctp=True`; the CTP-retrieved id is written to BOTH `histid`
      (no ML prediction exists for that wave, so the CTP id is the record's identifier —
      a valid IPUMS histid, keeping the (PatInvID, histid, census_year) key usable) and
      `histid_ctp`, with `probability` set by `--supplement-prob` (null by default;
      `from_ctp=True` marks the row's origin). Inventor-level columns (inventor_name,
      pubyear, county_inv, state_inv) are propagated; candidate-level features
      (leven_simil, county_distance, age_at_pat, relatedness_*) are null — no
      candidate comparison exists for a supplemented wave.

A downstream user reconstructs a "CTP-corrected" histid as
`coalesce(histid_ctp, histid)` (or by comparing the two columns); raw-ML users just
ignore the added columns. The output row set is EXACTLY the 3_predict rows (model
histid preserved) plus the appended `from_ctp=True` supplements. It is written in
both parquet (`--output`) and CSV (`--csv-output`, default: `--output` with a .csv
extension) form.

CTP crosswalks: `--ctp-dir`/<pair>/<pair>.csv for the adjacent census-year pairs
(header `histid{y1},histid{y2},clp,mlp,xgb,family_tree,direct_hint,profile_hint,
implied`). Only the two histid columns are used (a link = the row exists; expose a
stricter rule via --ctp-link-cols/--ctp-link-mode). GOTCHA: census histid case varies
by year in this project, so BOTH sides are upper-cased before joining (CTP ids are
upper-case UUIDs).

All inputs/outputs are CLI arguments; nothing is hardcoded.
"""

import argparse
import json
import os
import time

# Census waves in scope (1890 destroyed; the 1880_1900 crosswalk bridges that gap).
DEFAULT_CENSUS_YEARS = [1850, 1860, 1870, 1880, 1900, 1910, 1920, 1930, 1940]

# The 3_predict output columns (publication schema), preserved verbatim and in order.
ORIG_COLS = ["histid", "PatInvID", "pubnumber", "pubyear", "census_year",
             "inventor_name", "leven_simil", "county_inv", "state_inv",
             "county_distance", "age_at_pat", "relatedness_short",
             "relatedness_long", "probability"]
NEW_COLS = ["histid_ctp", "ctp_flag", "ctp_action", "from_ctp"]


def _lap(label, t0):
    now = time.perf_counter()
    print(f"[timing] {label}: {now - t0:.1f}s", flush=True)
    return now


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Non-destructive CTP consistency annotation + supplementation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--predicted-links", required=True,
                   help="Accepted-links parquet from 3_predict.py.")
    p.add_argument("--ctp-dir", required=True,
                   help="Directory holding <pair>/<pair>.csv CTP crosswalks "
                        "(e.g. 1860_1870/1860_1870.csv).")
    p.add_argument("--output", required=True, help="Annotated + supplemented parquet.")
    p.add_argument("--csv-output", default=None,
                   help="CSV copy of --output (default: --output with a .csv extension). "
                        "A CSV is always written alongside the parquet.")
    p.add_argument("--report-json", default=None,
                   help="Optional path to dump the correction-composition report as JSON.")
    p.add_argument("--census-years", type=int, nargs="*", default=DEFAULT_CENSUS_YEARS,
                   help="Census waves in scope (sorted; consecutive years define the "
                        "CTP-adjacent pairs, so 1880/1900 bridge the missing 1890).")
    p.add_argument("--window", type=int, default=10,
                   help="Eligibility half-window: a wave is eligible if "
                        "|census_year - pubyear| <= window.")
    p.add_argument("--ctp-link-cols", default=None,
                   help="Comma-separated indicator columns that must be set for a CTP "
                        "row to count as a link (default: row existence = a link).")
    p.add_argument("--ctp-link-mode", choices=["any", "all"], default="any",
                   help="Combine --ctp-link-cols with any(OR) or all(AND).")
    p.add_argument("--no-bridge-gap", dest="bridge_gap", action="store_false",
                   help="For ctp_flag, do NOT bridge two matched waves through an "
                        "unmatched middle wave (treat the gap as an unverifiable break).")
    p.add_argument("--no-supplement", dest="supplement", action="store_false",
                   help="Skip CTP supplementation (annotate only).")
    p.add_argument("--supplement-prob", type=float, default=None,
                   help="probability for supplemented rows (default: null).")
    p.add_argument("--spill-dir", default=None,
                   help="Directory for polars streaming spill (sets POLARS_TEMP_DIR/"
                        "TMPDIR). Default <output dir>/.polars_tmp. MUST be real disk: "
                        "the OS default /tmp is often tmpfs (RAM), which OOM-kills big joins.")
    return p.parse_args(argv)


def load_ctp_pairs(pl, ctp_dir, census_years, universe, link_cols, link_mode):
    """Load the adjacent-pair crosswalks, upper-cased and restricted to rows touching a
    matched histid (semi-join to `universe` on EITHER endpoint). Returns an eager frame
    (y_lo, y_hi, h_lo, h_hi). `pairs` = consecutive census years, so file <lo>_<hi>."""
    pairs = list(zip(census_years[:-1], census_years[1:]))
    frames = []
    for lo, hi in pairs:
        name = f"{lo}_{hi}"
        path = os.path.join(ctp_dir, name, f"{name}.csv")
        if not os.path.isfile(path):
            raise SystemExit(f"[4] missing CTP crosswalk: {path}")
        base = pl.scan_csv(path)
        sel = [pl.col(f"histid{lo}").str.to_uppercase().alias("h_lo"),
               pl.col(f"histid{hi}").str.to_uppercase().alias("h_hi")]
        if link_cols:
            sel += [pl.col(c) for c in link_cols]
        lf = base.select(sel)
        if link_cols:
            checks = [pl.col(c) == 1 for c in link_cols]
            cond = pl.all_horizontal(checks) if link_mode == "all" else pl.any_horizontal(checks)
            lf = lf.filter(cond).select("h_lo", "h_hi")
        uni = universe.lazy()
        keep = pl.concat([
            lf.join(uni, left_on="h_lo", right_on="histid_uc", how="semi"),
            lf.join(uni, left_on="h_hi", right_on="histid_uc", how="semi"),
        ]).unique().with_columns([
            pl.lit(lo, dtype=pl.Int64).alias("y_lo"),
            pl.lit(hi, dtype=pl.Int64).alias("y_hi"),
        ])
        kept = keep.collect(engine="streaming")
        print(f"[4]   pair {name}: {kept.height:,} universe-touching links", flush=True)
        frames.append(kept)
    return pl.concat(frames)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.perf_counter()

    # Spill to real disk BEFORE importing polars (see 3_predict.py) so big joins don't
    # OOM against a tmpfs /tmp.
    spill_dir = args.spill_dir or os.path.join(os.path.dirname(args.output) or ".",
                                               ".polars_tmp")
    os.makedirs(spill_dir, exist_ok=True)
    os.environ["POLARS_TEMP_DIR"] = spill_dir
    os.environ["TMPDIR"] = spill_dir
    import polars as pl

    census_years = sorted(args.census_years)
    window = args.window
    link_cols = [c.strip() for c in args.ctp_link_cols.split(",")] if args.ctp_link_cols else None
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ---- links: preserve every column; add upper-cased histid + tie-break distance ----
    links = pl.read_parquet(args.predicted_links)
    missing_cols = [c for c in ORIG_COLS if c not in links.columns]
    if missing_cols:
        raise SystemExit(f"[4] --predicted-links missing columns: {missing_cols}")
    links = links.with_columns([
        pl.col("histid").str.to_uppercase().alias("histid_uc"),
        (pl.col("census_year").cast(pl.Float64) - pl.col("pubyear")).abs().alias("year_dist"),
    ])
    n_links = links.height
    print(f"[4] loaded {n_links:,} predicted links "
          f"({links.select('PatInvID').n_unique():,} PatInvIDs)", flush=True)
    t0 = _lap("load links", t0)

    universe = links.select("histid_uc").unique()

    # ---- CTP crosswalks -> directed edges + 2-hop bridge edges ----
    ctp_pairs = load_ctp_pairs(pl, args.ctp_dir, census_years, universe, link_cols,
                               args.ctp_link_mode)
    edges = pl.concat([
        ctp_pairs.select(pl.col("y_lo").alias("y_from"), pl.col("h_lo").alias("h_from"),
                         pl.col("y_hi").alias("y_to"), pl.col("h_hi").alias("h_to")),
        ctp_pairs.select(pl.col("y_hi").alias("y_from"), pl.col("h_hi").alias("h_from"),
                         pl.col("y_lo").alias("y_to"), pl.col("h_lo").alias("h_to")),
    ]).unique()
    # 2-hop (a -> mid -> c) edges, used to bridge a matched pair separated by one
    # unmatched wave; drop trivial returns to the start.
    e2 = edges.rename({"y_from": "y_mid", "h_from": "h_mid",
                       "y_to": "y_to2", "h_to": "h_to2"})
    bridge_edges = (edges.join(e2, left_on=["y_to", "h_to"], right_on=["y_mid", "h_mid"])
                    .select("y_from", "h_from",
                            pl.col("y_to2").alias("y_to"), pl.col("h_to2").alias("h_to"))
                    .filter(pl.col("y_from") != pl.col("y_to")).unique())
    t0 = _lap("build CTP edges", t0)

    # ---- eligible waves + per-PatInvID anchor / n_matched ----
    pat = links.select("PatInvID", "pubyear").unique()
    cy = pl.DataFrame({"census_year": pl.Series(census_years, dtype=pl.Int64)})
    elig = (pat.filter(pl.col("pubyear").is_not_null())
            .join(cy, how="cross")
            .filter((pl.col("census_year") - pl.col("pubyear")).abs() <= window)
            .select("PatInvID", "census_year"))

    nmatch = links.group_by("PatInvID").agg(pl.len().alias("n_matched"))
    order_keys = ["probability", "year_dist", "census_year"]
    order_desc = [True, False, False]
    anchor = links.group_by("PatInvID").agg([
        pl.col("census_year").sort_by(order_keys, descending=order_desc).first().alias("anchor_year"),
        pl.col("histid_uc").sort_by(order_keys, descending=order_desc).first().alias("anchor_hist"),
    ])

    # ---- histid_ctp: propagate the anchor histid through CTP into each eligible wave ----
    asrc = anchor.select("PatInvID", pl.col("anchor_year").alias("y"),
                         pl.col("anchor_hist").alias("h"))
    av = pl.concat([
        asrc.join(edges, left_on=["y", "h"], right_on=["y_from", "h_from"])
            .select("PatInvID", pl.col("y_to").alias("census_year"), pl.col("h_to").alias("cand")),
        asrc.join(bridge_edges, left_on=["y", "h"], right_on=["y_from", "h_from"])
            .select("PatInvID", pl.col("y_to").alias("census_year"), pl.col("h_to").alias("cand")),
    ]).join(elig, on=["PatInvID", "census_year"], how="semi")
    av_view = (av.group_by(["PatInvID", "census_year"])
               .agg([pl.col("cand").n_unique().alias("n"), pl.col("cand").first().alias("cand")])
               .with_columns(pl.when(pl.col("n") == 1).then(pl.col("cand"))
                             .otherwise(None).alias("histid_ctp"))
               .select("PatInvID", "census_year", "histid_ctp"))

    # ---- ctp_flag: every consecutive matched-wave pair must be CTP-linked ----
    m = (links.select("PatInvID", "census_year", "histid_uc")
         .sort(["PatInvID", "census_year"])
         .with_columns([
             pl.col("census_year").shift(-1).over("PatInvID").alias("next_year"),
             pl.col("histid_uc").shift(-1).over("PatInvID").alias("next_hist"),
         ]).filter(pl.col("next_year").is_not_null()))
    direct_key = edges.select("y_from", "h_from", "y_to", "h_to").with_columns(
        pl.lit(True).alias("_direct"))
    m = m.join(direct_key, left_on=["census_year", "histid_uc", "next_year", "next_hist"],
               right_on=["y_from", "h_from", "y_to", "h_to"], how="left")
    if args.bridge_gap:
        bridge_key = bridge_edges.with_columns(pl.lit(True).alias("_bridge"))
        m = m.join(bridge_key, left_on=["census_year", "histid_uc", "next_year", "next_hist"],
                   right_on=["y_from", "h_from", "y_to", "h_to"], how="left")
        linked = pl.col("_direct").fill_null(False) | pl.col("_bridge").fill_null(False)
    else:
        linked = pl.col("_direct").fill_null(False)
    m = m.with_columns(linked.alias("linked"))
    tp = (m.group_by("PatInvID").agg((~pl.col("linked")).any().alias("_incons"))
          .with_columns(pl.when(pl.col("_incons")).then(1).otherwise(0)
                        .cast(pl.Int8).alias("ctp_flag"))
          .select("PatInvID", "ctp_flag"))
    t0 = _lap("consistency + histid_ctp", t0)

    # ---- assemble model rows (all preserved) ----
    model = (links
             .join(av_view, on=["PatInvID", "census_year"], how="left")
             .join(anchor, on="PatInvID", how="left")
             .join(nmatch, on="PatInvID", how="left")
             .join(tp, on="PatInvID", how="left")
             .with_columns([
                 pl.col("ctp_flag").fill_null(0).cast(pl.Int8),
                 pl.when(pl.col("census_year") == pl.col("anchor_year"))
                   .then(pl.col("anchor_hist")).otherwise(pl.col("histid_ctp")).alias("histid_ctp"),
             ])
             .with_columns([
                 pl.when(pl.col("n_matched") == 1).then(pl.lit("single"))
                   .when(pl.col("census_year") == pl.col("anchor_year")).then(pl.lit("anchor"))
                   .when(pl.col("histid_ctp").is_null()).then(pl.lit("unverified"))
                   .when(pl.col("histid_ctp") == pl.col("histid_uc")).then(pl.lit("consistent"))
                   .otherwise(pl.lit("replaced")).alias("ctp_action"),
                 pl.lit(False).alias("from_ctp"),
             ]))
    model_out = model.select(ORIG_COLS + NEW_COLS)

    # ---- supplementation: fill eligible-but-unmatched waves from CTP-adjacent matches ----
    n_missing = n_supp = n_skip_ambig = n_skip_none = 0
    supp_out = None
    if args.supplement:
        matched_keys = links.select("PatInvID", "census_year").unique()
        missing = elig.join(matched_keys, on=["PatInvID", "census_year"], how="anti")
        n_missing = missing.height
        msrc = links.select("PatInvID", pl.col("census_year").alias("y"),
                            pl.col("histid_uc").alias("h"))
        supp = (msrc.join(edges, left_on=["y", "h"], right_on=["y_from", "h_from"])
                .select("PatInvID", pl.col("y_to").alias("census_year"),
                        pl.col("h_to").alias("cand"))
                .join(missing, on=["PatInvID", "census_year"], how="semi"))
        supp_grp = supp.group_by(["PatInvID", "census_year"]).agg([
            pl.col("cand").n_unique().alias("n"), pl.col("cand").first().alias("cand")])
        supp_final = supp_grp.filter(pl.col("n") == 1).select(
            "PatInvID", "census_year", pl.col("cand").alias("histid_ctp"))
        n_supp = supp_final.height
        n_skip_ambig = supp_grp.filter(pl.col("n") > 1).height
        n_skip_none = n_missing - n_supp - n_skip_ambig

        if n_supp:
            # Inventor-level columns propagate to the supplement row; candidate-level
            # features stay null (no candidate comparison exists for that wave).
            pat_info = links.group_by("PatInvID").agg([
                pl.col("pubnumber").first(), pl.col("pubyear").first(),
                pl.col("inventor_name").first(),
                pl.col("county_inv").first(), pl.col("state_inv").first()])
            prob_lit = (pl.lit(None, dtype=pl.Float64) if args.supplement_prob is None
                        else pl.lit(args.supplement_prob, dtype=pl.Float64))
            supp_out = (supp_final
                        .join(pat_info, on="PatInvID", how="left")
                        .join(tp, on="PatInvID", how="left")
                        .with_columns([
                            # No ML prediction exists for a supplemented wave, so the
                            # CTP-supplied id IS the record's histid (a valid IPUMS histid,
                            # keeping (PatInvID, histid, census_year) a usable key); from_ctp
                            # flags its origin. histid_ctp mirrors it.
                            pl.col("histid_ctp").alias("histid"),
                            pl.lit(None, dtype=pl.Float64).alias("leven_simil"),
                            pl.lit(None, dtype=pl.Float64).alias("county_distance"),
                            pl.lit(None, dtype=pl.Int64).alias("age_at_pat"),
                            pl.lit(None, dtype=pl.Float64).alias("relatedness_short"),
                            pl.lit(None, dtype=pl.Float64).alias("relatedness_long"),
                            prob_lit.alias("probability"),
                            pl.col("ctp_flag").fill_null(0).cast(pl.Int8),
                            pl.lit("supplement").alias("ctp_action"),
                            pl.lit(True).alias("from_ctp"),
                        ])
                        .select(ORIG_COLS + NEW_COLS))

    out = model_out if supp_out is None else pl.concat([model_out, supp_out],
                                                       how="vertical_relaxed")
    out.write_parquet(args.output)
    csv_path = args.csv_output or (
        args.output[:-len(".parquet")] + ".csv" if args.output.endswith(".parquet")
        else args.output + ".csv")
    out.write_csv(csv_path)
    t0 = _lap("assemble + write", t0)

    # ---- report (Figure fig:ctp_corrections composition) ----
    action_counts = {r["ctp_action"]: r["len"]
                     for r in model.group_by("ctp_action").len().iter_rows(named=True)}
    n_multi_inv = int(nmatch.filter(pl.col("n_matched") >= 2).height)
    n_incons_inv = int(tp.filter(pl.col("ctp_flag") == 1).height)
    report = {
        "predicted_links": n_links,
        "patinv_total": int(links.select("PatInvID").n_unique()),
        "multi_wave_inventors": n_multi_inv,
        "inconsistent_inventors_ctp_flag": n_incons_inv,
        "consistent_inventors": n_multi_inv - n_incons_inv,
        "action_counts": action_counts,
        "eligible_unmatched_waves": n_missing,
        "supplemented": n_supp,
        "supplements_skipped_ambiguous": n_skip_ambig,
        "supplements_skipped_no_ctp": n_skip_none,
        "output_rows": out.height,
    }
    print(f"[4] links={n_links:,} | multi-wave inventors={n_multi_inv:,} "
          f"(inconsistent ctp_flag=1: {n_incons_inv:,})", flush=True)
    print(f"[4] per-row action: " + ", ".join(f"{k}={v:,}" for k, v in sorted(action_counts.items())),
          flush=True)
    print(f"[4] supplements: added={n_supp:,} / eligible-unmatched={n_missing:,} "
          f"(skipped ambiguous={n_skip_ambig:,}, no-CTP={n_skip_none:,})", flush=True)
    print(f"[4] wrote {out.height:,} rows -> {args.output}", flush=True)
    print(f"[4] wrote CSV copy -> {csv_path}", flush=True)
    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[4] wrote report -> {args.report_json}", flush=True)


if __name__ == "__main__":
    main()
