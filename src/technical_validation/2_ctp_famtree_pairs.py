#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Technical validation 2: pairwise validation of the model links against the Census Tree
Project Family Tree crosswalks (the manuscript's `tab:ctp_validation` +
`fig:ctp_famtree_sweep`).

Paper-analysis add-on (Scientific Data descriptor) -- NOT a pipeline stage. Reads the
RAW predicted-links parquet (the 14-column `3_predict.py` output -- NOT the
CTP-annotated table, which contains CTP-derived supplement rows that would validate
tautologically) plus the CTP crosswalk directory, and scores each adjacent census wave
pair independently:

  * N linked       -- inventors (PatInvID) the model matched in BOTH waves of the pair.
                      Predicted links are top-1 per (PatInvID, census_year), so each
                      such inventor contributes exactly one (histid_t, histid_t') pair.
  * validatable    -- (--rule source, the paper's definition) the earlier-wave histid_t
                      appears as a source node (histid{y1}) in the link-col-filtered
                      crosswalk for that pair, i.e. the CTP has an independent link for
                      the same census record. Pairs with no CTP link are excluded as
                      uninformative (CTP coverage, not evidence either way).
  * consistent     -- validatable AND one of histid_t's CTP targets equals histid_t'.
                      Fan-out rule: if the crosswalk links histid_t to several targets,
                      the pair counts as consistent when ANY of them is ours -- the CTP
                      then does not contradict the match. (4_ctp_correction.py skips
                      fan-outs only because it must WRITE a single id; here we only test
                      membership.) `n_fanout_sources` in the CSV reports the footprint.

  --rule symmetric additionally counts a pair validatable when histid_t' appears as a
  target node (and inconsistent if either direction contradicts) -- a robustness
  variant; the manuscript uses `source`.

Adjacent pairs are consecutive entries of the sorted --census-years list (so 1880/1900
pair across the destroyed 1890 census); an inventor matched only in, say, 1900 and 1920
contributes to no pair. Default --link-cols family_tree restricts the crosswalks to the
CTP's crowdsourced-genealogy ground truth links (same semantics as 4_ctp_correction.py's
--ctp-link-cols). Histids are upper-cased on BOTH sides of every join (census histid
case varies by wave; CTP ids are upper-case UUIDs).

Because the release is no-threshold, the pooled rates are dominated by the mass of
low-probability links most users would discard, so the headline outputs stratify by
min(probability_t, probability_t') -- a cross-wave pair is only as reliable as its
weaker link:

  * the manuscript table shows, per wave pair, the consistency rate among validatable
    pairs at a few user-relevant cutoffs (--table-thresholds, default: all / >=0.4 /
    >=0.8);
  * a fine-grained threshold sweep (--sweep-step/--sweep-max) feeds a line figure of
    consistency vs threshold, one line per wave pair.

Chance benchmark (quoted in the table notes): under the null of an erroneous
second-wave link, the probability that the Family Tree connects histid_t to precisely
that record is essentially zero, so chance-level consistency is ~0%.

Outputs under --output-dir (console mirrors every table):
  ctp_famtree_pairs.csv               per-pair counts + threshold columns + Overall row
  ctp_famtree_pairs_by_prob.csv       pooled counts by min-probability bin (+ Overall)
  ctp_famtree_consistency_sweep.csv   per-pair + pooled consistency vs threshold grid
  ctp_famtree_consistency_sweep.pdf   the fig:ctp_famtree_sweep line figure
  ctp_val_tab_famtree.tex             the manuscript table (label tab:ctp_validation;
                                      dynamic numbers in the notes come from the same
                                      frames as the body, so they cannot drift)
  ctp_val_tab_famtree_by_prob.tex     binned variant (NOT included in the manuscript;
                                      kept as a robustness artifact)

All paths are CLI arguments; nothing is hardcoded.
"""

import argparse
import os
import time

DEFAULT_CENSUS_YEARS = [1850, 1860, 1870, 1880, 1900, 1910, 1920, 1930, 1940]
DEFAULT_PROB_BINS = [0.2, 0.4, 0.6, 0.8]
DEFAULT_TABLE_THRESHOLDS = [0.0, 0.4, 0.8]

# Qualitative, colorblind-safe hues for the wave pairs (Paul Tol "muted"). A single-hue
# ramp made the eight time-ordered lines indistinguishable where they converge in the
# 80-100% band, so we trade the ordinal encoding for maximal separability and pair each
# line with a distinct marker as a redundant channel.
# Ordered so consecutive wave pairs (which cluster tightly in value at high thresholds)
# alternate warm/cool and never repeat a hue family back-to-back.
SWEEP_COLORS = ["#332288", "#44AA99", "#882255", "#DDCC77", "#117733", "#CC6677",
                "#88CCEE", "#AA4499"]
SWEEP_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
INK, INK_2, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _lap(label, t0):
    now = time.perf_counter()
    print(f"[timing] {label}: {now - t0:.1f}s", flush=True)
    return now


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pairwise validation of model links against CTP Family Tree "
                    "crosswalks (per adjacent wave pair, threshold columns, sweep "
                    "figure).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--predicted-links", required=True,
                   help="Accepted-links parquet from 3_predict.py (the raw model "
                        "links, NOT the CTP-annotated table).")
    p.add_argument("--ctp-dir", required=True,
                   help="Directory holding <pair>/<pair>.csv CTP crosswalks.")
    p.add_argument("--output-dir", required=True,
                   help="Directory for the CSV/tex/PDF outputs.")
    p.add_argument("--link-cols", default="family_tree",
                   help="Comma-separated crosswalk indicator columns that must be set "
                        "for a row to count as a CTP link (empty string: row "
                        "existence = a link).")
    p.add_argument("--link-mode", choices=["any", "all"], default="any",
                   help="Combine --link-cols with any(OR) or all(AND).")
    p.add_argument("--census-years", type=int, nargs="*", default=DEFAULT_CENSUS_YEARS,
                   help="Census waves (sorted; consecutive entries define the pairs, "
                        "so 1880/1900 bridge the missing 1890).")
    p.add_argument("--rule", choices=["source", "symmetric"], default="source",
                   help="Validatability rule: 'source' (earlier-wave histid appears as "
                        "a crosswalk source node -- the manuscript definition) or "
                        "'symmetric' (either endpoint appears; robustness variant).")
    p.add_argument("--prob-bins", type=float, nargs="+", default=DEFAULT_PROB_BINS,
                   help="Cut points for the binned by-prob variant CSV/tex.")
    p.add_argument("--table-thresholds", type=float, nargs="+",
                   default=DEFAULT_TABLE_THRESHOLDS,
                   help="min(p_t, p_t') >= threshold columns of the manuscript table "
                        "(0 = all links).")
    p.add_argument("--sweep-step", type=float, default=0.05,
                   help="Threshold grid step of the consistency sweep.")
    p.add_argument("--sweep-max", type=float, default=0.9,
                   help="Largest threshold of the consistency sweep.")
    p.add_argument("--spill-dir", default=None,
                   help="Directory for polars streaming spill (sets POLARS_TEMP_DIR/"
                        "TMPDIR). Default <output-dir>/.polars_tmp. MUST be real disk, "
                        "never a tmpfs /tmp.")
    return p.parse_args(argv)


def load_pair_crosswalk(pl, ctp_dir, lo, hi, universe, link_cols, link_mode):
    """The (x_lo, x_hi) crosswalk links of pair lo_hi, link-col-filtered, upper-cased,
    restricted to rows touching the pair's model histids (semi-join on EITHER
    endpoint), unique. `universe` is a small eager frame with column `histid_uc`."""
    name = f"{lo}_{hi}"
    path = os.path.join(ctp_dir, name, f"{name}.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"[2] missing CTP crosswalk: {path}")
    base = pl.scan_csv(path)
    sel = [pl.col(f"histid{lo}").str.to_uppercase().alias("x_lo"),
           pl.col(f"histid{hi}").str.to_uppercase().alias("x_hi")]
    if link_cols:
        sel += [pl.col(c) for c in link_cols]
    lf = base.select(sel)
    if link_cols:
        checks = [pl.col(c) == 1 for c in link_cols]
        cond = pl.all_horizontal(checks) if link_mode == "all" else pl.any_horizontal(checks)
        lf = lf.filter(cond).select("x_lo", "x_hi")
    uni = universe.lazy()
    keep = pl.concat([
        lf.join(uni, left_on="x_lo", right_on="histid_uc", how="semi"),
        lf.join(uni, left_on="x_hi", right_on="histid_uc", how="semi"),
    ]).unique()
    return keep.collect(engine="streaming")


def score_pair(pl, pairs, xw, rule):
    """Per-inventor validatable/consistent verdicts for one wave pair.

    `pairs`: one row per inventor matched in both waves (PatInvID, h_lo, h_hi,
    p_lo, p_hi). `xw`: the pair's crosswalk links (x_lo, x_hi), unique. Returns
    one row per inventor with boolean `validatable`/`inconsistent`, Int `n_targets`
    (source-side fan-out size) and `min_prob` = min(p_lo, p_hi)."""
    src = (pairs.join(xw, left_on="h_lo", right_on="x_lo", how="left")
           .group_by("PatInvID")
           .agg([
               pl.col("x_hi").is_not_null().any().alias("val_src"),
               (pl.col("x_hi") == pl.col("h_hi")).any().alias("cons_src"),
               pl.col("x_hi").drop_nulls().n_unique().alias("n_targets"),
           ]))
    out = pairs.join(src, on="PatInvID", how="left")
    if rule == "symmetric":
        tgt = (pairs.join(xw, left_on="h_hi", right_on="x_hi", how="left")
               .group_by("PatInvID")
               .agg([
                   pl.col("x_lo").is_not_null().any().alias("val_tgt"),
                   (pl.col("x_lo") == pl.col("h_lo")).any().alias("cons_tgt"),
               ]))
        out = (out.join(tgt, on="PatInvID", how="left")
               .with_columns([
                   (pl.col("val_src") | pl.col("val_tgt")).alias("validatable"),
                   ((pl.col("val_src") & ~pl.col("cons_src"))
                    | (pl.col("val_tgt") & ~pl.col("cons_tgt"))).alias("inconsistent"),
               ]))
    else:
        out = out.with_columns([
            pl.col("val_src").alias("validatable"),
            (pl.col("val_src") & ~pl.col("cons_src")).alias("inconsistent"),
        ])
    return out.select("PatInvID", "p_lo", "p_hi",
                      pl.min_horizontal("p_lo", "p_hi").alias("min_prob"),
                      "validatable", "inconsistent", "n_targets")


def _bin_labels(breaks):
    labels = [f"<={breaks[0]:g}"]
    labels += [f"{lo:g}-{hi:g}" for lo, hi in zip(breaks[:-1], breaks[1:])]
    labels.append(f">{breaks[-1]:g}")
    return labels


def _bin_exprs(pl, col, breaks):
    labels = _bin_labels(breaks)
    v = pl.col(col)
    lab = pl.when(v.is_null()).then(pl.lit("n/a"))
    ix = pl.when(v.is_null()).then(pl.lit(len(labels)))
    for i, b in enumerate(breaks):
        lab = lab.when(v <= b).then(pl.lit(labels[i]))
        ix = ix.when(v <= b).then(pl.lit(i))
    lab = lab.otherwise(pl.lit(labels[-1]))
    ix = ix.otherwise(pl.lit(len(breaks)))
    return lab.alias("prob_bin"), ix.cast(pl.Int32).alias("_bin_ix")


def _counts(pl, df, key_out, key_val):
    """One tabulated row: linked/validatable/(in)consistent counts + rates."""
    n_linked = df.height
    n_val = df.filter(pl.col("validatable")).height
    n_inc = df.filter(pl.col("inconsistent")).height
    n_fan = df.filter(pl.col("validatable") & (pl.col("n_targets") > 1)).height
    return {
        key_out: key_val,
        "n_linked": n_linked,
        "n_validatable": n_val,
        "n_inconsistent": n_inc,
        "pct_inconsistent": 100.0 * n_inc / n_val if n_val else None,
        "pct_consistent": 100.0 * (n_val - n_inc) / n_val if n_val else None,
        "n_fanout_sources": n_fan,
    }


def _threshold_stats(pl, df, t):
    """(n_validatable, pct_consistent) among pairs with min_prob >= t."""
    sub = df.filter(pl.col("min_prob") >= t) if t > 0 else df
    n_val = sub.filter(pl.col("validatable")).height
    n_con = sub.filter(pl.col("validatable") & ~pl.col("inconsistent")).height
    return n_val, (100.0 * n_con / n_val if n_val else None)


def _tex_bin(label):
    if label.startswith("<="):
        return f"$\\leq${label[2:]}"
    if label.startswith(">"):
        return f"$>${label[1:]}"
    return label.replace("-", "--")


def _fmt_rate(x, decimals=1):
    return f"{x:.{decimals}f}\\%" if x is not None else "--"


def _fmt_pair_table(rows, total, thresholds, link_desc):
    """The manuscript table (label tab:ctp_validation): one row per wave pair,
    consistency among validatable pairs at each --table-threshold; pooled numbers in
    the notes come from the same rows as the body."""
    def _th_head(t):
        return "\\textbf{All links}" if t == 0 else f"\\textbf{{$p \\geq {t:g}$}}"

    ncols = 3 + len(thresholds)
    head_rates = " & ".join(_th_head(t) for t in thresholds)
    body = []
    for r in rows:
        rates = " & ".join(_fmt_rate(r["th"][t][1]) for t in thresholds)
        body.append(f"{r['wave_pair'].replace('_', '--')} & {r['n_linked']:,} & "
                    f"{r['n_validatable']:,} & {rates} \\\\")
    body_str = "\n".join(body)
    tot_rates = " & ".join(_fmt_rate(total["th"][t][1]) for t in thresholds)
    th_note = "; ".join(
        f"at $p \\geq {t:g}$: {total['th'][t][0]:,} validatable pairs pooled, "
        f"{_fmt_rate(total['th'][t][1])} consistent" for t in thresholds if t > 0)
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Consistency of patent--census matches with Family Tree matches from the Census Tree Project}}
\label{{tab:ctp_validation}}
\begin{{threeparttable}}
\begin{{tabular}}{{l{'c' * (ncols - 1)}}}
\toprule
 & & & \multicolumn{{{len(thresholds)}}}{{c}}{{\textbf{{\% consistent among validatable}}}} \\
\cmidrule(lr){{4-{ncols}}}
\textbf{{Wave pair}} & \textbf{{N linked}} & \textbf{{N validatable}} & {head_rates} \\
\midrule
{body_str}
\midrule
All pairs & {total['n_linked']:,} & {total['n_validatable']:,} & {tot_rates} \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} This table reports the consistency of the patent--census matches with the {link_desc} links of individuals across census waves, derived from the Census Tree Project (CTP) crosswalks~\citep{{censustreeproject}}. For each adjacent census wave pair $(t, t')$, \textit{{N linked}} counts the patent-inventors in our matched data linked to both census waves of the pair ({total['n_linked']:,} pooled across pairs). A patent-inventor is \textit{{validatable}} if the histid of its earlier-wave match appears as a source node in the crosswalk for that pair, meaning the CTP has an independent link for the same individual ({total['n_validatable']:,} pooled); inventors without such a link are excluded as uninformative, since their absence may reflect CTP coverage rather than evidence on the match. A validatable pair is \textit{{consistent}} if the CTP links $histid_t$ to the same record we match in $t'$; where the crosswalk links $histid_t$ to several records, the pair counts as consistent if any of them coincides with ours. The dataset is released without a probability threshold, so the \textit{{All links}} column includes every top-ranked link however low its predicted probability; the remaining columns restrict to pairs whose weaker link clears the threshold, $\min(p_t, p_{{t'}}) \geq p$ ({th_note}). Under the null of an erroneous match, the probability that the Family Tree connects $histid_t$ to precisely the record we link is essentially zero, so chance-level consistency is approximately 0\%.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""


def _fmt_prob_table(rows, total, link_desc):
    """Binned by-prob variant (robustness artifact; NOT included in the manuscript)."""
    body = []
    for r in rows:
        body.append(f"{_tex_bin(r['prob_bin'])} & {r['n_linked']:,} & "
                    f"{r['n_validatable']:,} & {r['n_inconsistent']:,} & "
                    f"{_fmt_rate(r['pct_consistent'], 2)} \\\\")
    body_str = "\n".join(body)
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Consistency with Family Tree matches from the Census Tree Project, by predicted-probability bin}}
\label{{tab:ctp_validation_prob}}
\begin{{threeparttable}}
\begin{{tabular}}{{lcccc}}
\toprule
\textbf{{Probability bin}} & \textbf{{N linked}} & \textbf{{N validatable}} & \textbf{{N inconsistent}} & \textbf{{\% consistent}} \\
\midrule
{body_str}
\midrule
Overall & {total['n_linked']:,} & {total['n_validatable']:,} & {total['n_inconsistent']:,} & {_fmt_rate(total['pct_consistent'], 2)} \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Same construction as Table~\ref{{tab:ctp_validation}}, pooled across all adjacent wave pairs and binned by the smaller of the two linked waves' predicted match probabilities, $\min(p_t, p_{{t'}})$. Consistency is measured against the {link_desc} links and computed over validatable pairs only.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""


def make_sweep_figure(pl, sweep, pair_names, pdf_path):
    """Consistency vs probability threshold: one line per wave pair in a colorblind-safe
    qualitative palette, each with a distinct staggered marker, plus a thick dashed
    pooled line. Quiet chrome, legend below."""
    from _viz_common import configure_matplotlib
    plt, _ = configure_matplotlib(ink=INK, baseline=BASELINE, label=INK_2,
                                  xtick=INK_2, ytick=MUTED)
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    for i, name in enumerate(pair_names):
        sub = sweep.filter((pl.col("wave_pair") == name)
                           & pl.col("pct_consistent").is_not_null())
        ax.plot(sub["threshold"].to_list(), sub["pct_consistent"].to_list(),
                color=SWEEP_COLORS[i % len(SWEEP_COLORS)], linewidth=1.5,
                marker=SWEEP_MARKERS[i % len(SWEEP_MARKERS)], markersize=4.5,
                markevery=(i % 3, 3), markeredgecolor="white", markeredgewidth=0.4,
                solid_capstyle="round", solid_joinstyle="round",
                label=name.replace("_", "-"))
    pooled = sweep.filter((pl.col("wave_pair") == "All pairs")
                          & pl.col("pct_consistent").is_not_null())
    ax.plot(pooled["threshold"].to_list(), pooled["pct_consistent"].to_list(),
            color=INK, linewidth=2.2, linestyle=(0, (4, 2)), label="All pairs",
            zorder=10)

    ax.set_xlabel("Probability threshold ($\\min(p_t, p_{t'}) \\geq$ threshold)")
    ax.set_ylabel("Consistent with Family Tree links")
    # Truncated baseline (fine for a line chart, unlike bars): the curves live in
    # 40-100%, so a zero baseline wastes half the panel and compresses the spread.
    ax.set_ylim(40, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.4)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.perf_counter()

    spill_dir = args.spill_dir or os.path.join(args.output_dir, ".polars_tmp")
    os.makedirs(spill_dir, exist_ok=True)
    os.environ["POLARS_TEMP_DIR"] = spill_dir
    os.environ["TMPDIR"] = spill_dir
    import polars as pl

    os.makedirs(args.output_dir, exist_ok=True)
    census_years = sorted(args.census_years)
    breaks = sorted(args.prob_bins)
    thresholds = sorted(args.table_thresholds)
    link_cols = [c.strip() for c in args.link_cols.split(",") if c.strip()] or None
    link_desc = " + ".join(c.replace("_", " ").title() for c in link_cols) \
        if link_cols else "any-type"

    links = (pl.read_parquet(args.predicted_links,
                             columns=["PatInvID", "census_year", "histid", "probability"])
             .with_columns(pl.col("histid").str.to_uppercase().alias("histid_uc")))
    if links.select("PatInvID", "census_year").is_duplicated().any():
        raise SystemExit("[2] predicted links are not unique on (PatInvID, census_year)")
    print(f"[2] loaded {links.height:,} predicted links "
          f"({links.select('PatInvID').n_unique():,} PatInvIDs) | rule={args.rule} | "
          f"link cols={link_cols or 'row existence'}", flush=True)
    t0 = _lap("load links", t0)

    pair_names = [f"{lo}_{hi}" for lo, hi in zip(census_years[:-1], census_years[1:])]
    empty_verdicts = pl.DataFrame(schema={
        "PatInvID": pl.Utf8, "p_lo": pl.Float64, "p_hi": pl.Float64,
        "min_prob": pl.Float64, "validatable": pl.Boolean,
        "inconsistent": pl.Boolean, "n_targets": pl.UInt32})
    per_pair_verdicts = {}
    for lo, hi in zip(census_years[:-1], census_years[1:]):
        name = f"{lo}_{hi}"
        L = (links.filter(pl.col("census_year") == lo)
             .select("PatInvID", pl.col("histid_uc").alias("h_lo"),
                     pl.col("probability").alias("p_lo")))
        R = (links.filter(pl.col("census_year") == hi)
             .select("PatInvID", pl.col("histid_uc").alias("h_hi"),
                     pl.col("probability").alias("p_hi")))
        pairs = L.join(R, on="PatInvID", how="inner")
        if not pairs.height:
            per_pair_verdicts[name] = empty_verdicts
            print(f"[2]   pair {name}: 0 linked inventors", flush=True)
            continue
        universe = pl.concat([pairs.select(pl.col("h_lo").alias("histid_uc")),
                              pairs.select(pl.col("h_hi").alias("histid_uc"))]).unique()
        xw = load_pair_crosswalk(pl, args.ctp_dir, lo, hi, universe,
                                 link_cols, args.link_mode)
        verdicts = score_pair(pl, pairs, xw, args.rule)
        per_pair_verdicts[name] = verdicts
        row = _counts(pl, verdicts, "wave_pair", name)
        pct = (f"{row['pct_consistent']:.2f}%" if row["pct_consistent"] is not None
               else "n/a")
        print(f"[2]   pair {name}: linked={row['n_linked']:,} | "
              f"CTP links kept={xw.height:,} | validatable={row['n_validatable']:,} | "
              f"consistent={row['n_validatable'] - row['n_inconsistent']:,} ({pct}) | "
              f"fan-out sources={row['n_fanout_sources']:,}", flush=True)
        t0 = _lap(f"pair {name}", t0)

    all_verdicts = pl.concat(
        [v.with_columns(pl.lit(k).alias("wave_pair"))
         for k, v in per_pair_verdicts.items()])

    # ---- per-pair table rows + threshold columns ----
    def _row_with_thresholds(df, key_val):
        r = _counts(pl, df, "wave_pair", key_val)
        r["th"] = {t: _threshold_stats(pl, df, t) for t in thresholds}
        return r

    pair_rows = [_row_with_thresholds(per_pair_verdicts[n], n) for n in pair_names]
    total = _row_with_thresholds(all_verdicts, "Overall")

    flat_rows = []
    for r in pair_rows + [total]:
        flat = {k: v for k, v in r.items() if k != "th"}
        for t in thresholds:
            n_val, pct = r["th"][t]
            flat[f"n_validatable_ge{t:g}"] = n_val
            flat[f"pct_consistent_ge{t:g}"] = pct
        flat_rows.append(flat)
    per_pair = pl.DataFrame(flat_rows)
    print("\n== Family Tree pairwise validation (per wave pair) ==")
    with pl.Config(tbl_rows=per_pair.height, tbl_cols=len(per_pair.columns),
                   tbl_width_chars=220):
        print(per_pair)
    per_pair.write_csv(os.path.join(args.output_dir, "ctp_famtree_pairs.csv"))

    # ---- binned by-prob variant (robustness artifact) ----
    prob_rows = []
    if all_verdicts.height:
        lab_e, ix_e = _bin_exprs(pl, "min_prob", breaks)
        strat = all_verdicts.with_columns([lab_e, ix_e])
        for key in strat.select("_bin_ix", "prob_bin").unique().sort("_bin_ix").iter_rows():
            sub = strat.filter(pl.col("_bin_ix") == key[0])
            prob_rows.append(_counts(pl, sub, "prob_bin", key[1]))
    by_prob = pl.DataFrame(prob_rows
                           + [{k: v for k, v in total.items() if k != "th"} | {
                               "prob_bin": "Overall"}]).drop("wave_pair", strict=False)
    by_prob = by_prob.select(["prob_bin"] + [c for c in by_prob.columns if c != "prob_bin"])
    print("\n== Family Tree pairwise validation (by min link probability, pooled) ==")
    with pl.Config(tbl_rows=by_prob.height, tbl_cols=len(by_prob.columns)):
        print(by_prob)
    by_prob.write_csv(os.path.join(args.output_dir, "ctp_famtree_pairs_by_prob.csv"))

    # ---- consistency-vs-threshold sweep (per pair + pooled) ----
    grid, t = [], 0.0
    while t <= args.sweep_max + 1e-9:
        grid.append(round(t, 4))
        t += args.sweep_step
    sweep_rows = []
    for name in pair_names + ["All pairs"]:
        df = all_verdicts if name == "All pairs" else per_pair_verdicts[name]
        for t in grid:
            n_val, pct = _threshold_stats(pl, df, t)
            sweep_rows.append({"wave_pair": name, "threshold": t,
                               "n_validatable": n_val, "pct_consistent": pct})
    sweep = pl.DataFrame(sweep_rows)
    sweep.write_csv(os.path.join(args.output_dir, "ctp_famtree_consistency_sweep.csv"))
    t0 = _lap("tabulate + sweep", t0)

    # ---- artifacts: manuscript table, by-prob variant, sweep figure ----
    tex_path = os.path.join(args.output_dir, "ctp_val_tab_famtree.tex")
    with open(tex_path, "w") as fh:
        fh.write(_fmt_pair_table(pair_rows, total, thresholds, link_desc))
    print(f"\n[2] wrote {tex_path}")
    tex_prob_path = os.path.join(args.output_dir, "ctp_val_tab_famtree_by_prob.tex")
    with open(tex_prob_path, "w") as fh:
        fh.write(_fmt_prob_table(prob_rows, total, link_desc))
    print(f"[2] wrote {tex_prob_path}")
    pdf_path = os.path.join(args.output_dir, "ctp_famtree_consistency_sweep.pdf")
    make_sweep_figure(pl, sweep, pair_names, pdf_path)
    print(f"[2] wrote {pdf_path}")
    print(f"[2] wrote CSVs -> {args.output_dir}", flush=True)
    t0 = _lap("artifacts", t0)


if __name__ == "__main__":
    main()
