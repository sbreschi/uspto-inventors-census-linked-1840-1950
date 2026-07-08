#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Technical validation 5: widow--spouse concordance report (the manuscript's
`tab:relative_validation` + `fig:relative_sweep`).

Paper-analysis add-on (Scientific Data descriptor) -- NOT a pipeline stage. Reads the
stage-4 (`4_relative_attach.py`) row/pair CSVs -- no census access, re-runs in
seconds -- and scores how often a deceased inventor's same-surname estate
representative carries the given name of the SPOUSE (in practice the widow) of the
model-linked census person in the pre-patent wave.

The test is the widow--spouse comparison alone, by design: the wife is the one
person almost always co-resident with the inventor, so the spouse is the only
comparison whose sensitivity (the probability that a CORRECT link produces a name
match) is high and interpretable -- other household relatives routinely live
elsewhere (an adult child or sibling heading their own household), which would
confound co-residence patterns with link quality. Stage 4 classifies and records
ALL household relatives (its CSVs are the audit record); this stage scores the
spouse.

Spouse semantics: the comparator set is every spouse given name, treated as
female-side -- the deceased-inventor roster is effectively all male, so the spouse
is the wife -- and is tested against the feminine-role representatives
(executrix/administratrix/widow, after the census given-name sex correction) and
the unknown/sex-neutral-role ones; masculine-role representatives
(executor/administrator -- typically a son or brother) are gated out. A checkable
row therefore needs >= 1 spouse given name AND >= 1 F- or U-gender same-surname
representative, so the stage-4 `checkable` flag (which admits any relative) is a
superset of this stage's checkable set.

What it reports
---------------
* N_checkable and the roster->checkable attrition (echoed from stage 4).
* p_obs: observed concordance (nickname clusters + fuzzy >= cutoff) + Wilson 95% CI.
* p_null: chance concordance from a seeded SEX-STRATIFIED permutation (representative
  names re-paired across households within each role-gender stratum). The lift
  p_obs - p_null is the signal; the permutation IS the chance level, reported
  explicitly (unlike the CTP table's analytic ~0%).
* Implied SUBGROUP precision from the Rogan--Gladen mixture over a band of assumed
  sensitivities s < 1 (a correct link can still miss: a wife who died, remarried or
  lives elsewhere between the census and the inventor's death, or a garbled given
  name). Where observed concordance reaches the assumed sensitivity the point
  estimate saturates at one and is uninformative (near-tautological -- s is
  anchored on the top of the probability range), so the table quotes a conservative
  lower bound instead: the Wilson lower confidence limit of concordance under the
  UPPER sensitivity, rendered as ">=".
* A by-probability-floor breakdown + threshold sweep (CSV/figure), per-year and
  pointer-vs-relate splits, and a unique-inventor dedup sensitivity line (roster rows
  are per-patent; the same deceased inventor can repeat).

HONEST FRAMING: this estimates precision on the deceased-with-same-surname-
representative SUBPOPULATION only, but the representative's given name comes from the
patent side alone -- it is never a model feature -- so the test is fully independent
of both the training labels and the Census Tree Project.

Outputs under --output-dir (console mirrors every table):
  relative_summary.csv             long-format metric/group/n/k/rate/ci rows
  relative_sweep.csv               concordance + null vs probability threshold
  relative_concordance_sweep.pdf   the fig:relative_sweep line figure
  relative_val_tab.tex             the manuscript table (label tab:relative_validation;
                                   one row per probability floor; dynamic numbers in
                                   the notes come from the same frames as the body,
                                   so they cannot drift)

All paths are CLI arguments; nothing is hardcoded.
"""

import argparse
import csv
import math
import os
import time

import polars as pl

import _relative_common as rc

INK, INK_2, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OBS_COLOR = "#332288"          # Tol muted indigo, as in 2_ctp_famtree_pairs.py

# implied-precision band for the manuscript table: two-sided under s in the band,
# except where the point estimate saturates (p_obs >= S_BAND[0]) -- then a
# conservative lower bound under S_BAND[-1] (see _prec_cell).
S_BAND = (0.80, 0.90)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Widow--spouse concordance report (summary CSV, permutation "
                    "null, sweep figure, manuscript table).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pairs", required=True,
                   help="Stage-4 relative_attach_pairs.csv (cutoff cross-check only).")
    p.add_argument("--rows", required=True,
                   help="Stage-4 relative_attach_rows.csv (the unit of analysis).")
    p.add_argument("--attrition", default=None,
                   help="Stage-4 relative_attrition.csv (echoed into the summary; "
                        "default: relative_attrition.csv next to --rows).")
    p.add_argument("--output-dir", required=True,
                   help="Directory for the CSV/tex/PDF outputs.")
    p.add_argument("--cutoff", type=float, default=85.0,
                   help="rapidfuzz cutoff for a given-name concordance (0-100).")
    p.add_argument("--nicknames", nargs="+",
                   default=["data/misc/given_name_nicknames.json",
                            "data/misc/given_name_diminutives.json"],
                   help="Nickname/diminutive cluster JSON(s), merged (earlier "
                        "files win) -- MUST match stage 4's.")
    p.add_argument("--n-perm", type=int, default=2000, help="Permutation replicates.")
    p.add_argument("--seed", type=int, default=39, help="Permutation seed.")
    p.add_argument("--sensitivities", type=float, nargs="*",
                   default=[0.70, 0.80, 0.90, 0.95],
                   help="Assumed true-match concordance rates for implied precision.")
    p.add_argument("--prob-thresholds", type=float, nargs="*",
                   default=[0.0, 0.2, 0.25, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
                   help="Probability floors for the by-confidence breakdown.")
    p.add_argument("--table-thresholds", type=float, nargs="+",
                   default=[0.0, 0.2, 0.4, 0.8],
                   help="Probability floors shown as manuscript-table rows "
                        "(0 = all links).")
    p.add_argument("--sweep-step", type=float, default=0.05,
                   help="Threshold grid step of the concordance sweep.")
    p.add_argument("--sweep-max", type=float, default=0.9,
                   help="Largest threshold of the concordance sweep.")
    return p.parse_args(argv)


def _split(v):
    return [x for x in str(v or "").split("|") if x]


def build_rows(df):
    """Per-row structures for `observed_concordance` / `permutation_null_gated` from
    the stage-4 rows CSV, spouse-only: the comparator set is every spouse given
    name, placed in the F and U strata (female-side semantics: the roster's
    deceased inventors are effectively all male, so the spouse is the wife) and
    never in M -- masculine-role representatives are gated out. Only rows with a
    spouse comparator AND >= 1 F- or U-gender representative are kept; rows whose
    only representatives are masculine-role, or whose linked household has no
    spouse, fall out of the checkable set."""
    out = []
    for r in df.iter_rows(named=True):
        spouse = _split(r["rel_givens_spouse"])
        row = {
            "PatInvID": r["PatInvID"],
            "previous_census_year": r["previous_census_year"],
            "probability": float(r["probability"] or 0.0),
            "deceased_surname": r["deceased_surname"],
            "histid": r["histid"],
            "rel_methods": _split(r["rel_methods"]),
            "reps_by_gender": {"F": _split(r["rep_givens_F"]),
                               "M": _split(r["rep_givens_M"]),
                               "U": _split(r["rep_givens_U"])},
            "rels_by_gender": {"F": spouse, "M": [], "U": spouse},
        }
        gated = any(row["reps_by_gender"][g] and row["rels_by_gender"][g]
                    for g in rc.GENDER_STRATA)
        if gated:
            out.append(row)
    out.sort(key=lambda x: (x["previous_census_year"], x["PatInvID"]))
    return out


def _fmt_rate(x, decimals=1):
    return f"{100 * x:.{decimals}f}\\%" if x is not None and not math.isnan(x) else "--"


def _th_label(t):
    return "All links" if t == 0 else f"$p \\geq {t:g}$"


def _prec_cell(p_obs, ci_lo, p_null):
    """Implied-precision table cell. Two-sided band over S_BAND when the point
    estimate is interior; where p_obs >= min(S_BAND) the estimate saturates at one
    (uninformative -- and near-tautological, since the sensitivity band is anchored
    on the top of the probability range), so quote a conservative lower bound
    instead: the Wilson lower CI of concordance under the UPPER sensitivity. A
    saturated cell therefore reads ">= x%", never 100%."""
    if math.isnan(p_null):
        return "--"
    if p_obs >= S_BAND[0]:
        lb = rc.implied_precision(ci_lo, p_null, S_BAND[-1])
        return "n/a" if math.isnan(lb) else f"$\\geq$ {100 * lb:.1f}\\%"
    vals = sorted(rc.implied_precision(p_obs, p_null, s) for s in S_BAND)
    if any(math.isnan(v) for v in vals):
        return "n/a"
    return f"{100 * vals[0]:.1f}--{100 * vals[1]:.1f}\\%"


def fmt_table(table_rows, args, n_perm, n_unique):
    """The manuscript table: one row per probability floor (All links first), with
    the checkable N, concordance + Wilson 95% CI, that subset's own seeded
    permutation chance level, and the implied precision under s = 0.8--0.9 (lower
    bound where saturated -- see _prec_cell). `table_rows` dicts carry
    t/n/k/p/lo/hi/null."""
    body = []
    for row in table_rows:
        prec = _prec_cell(row["p"], row["lo"], row["null"])
        body.append(
            f"{_th_label(row['t'])} & {row['n']:,} & "
            f"{_fmt_rate(row['p'])} ({100 * row['lo']:.1f}--{100 * row['hi']:.1f}) & "
            f"{_fmt_rate(row['null'])} & {prec} \\\\")
    body = "\n".join(body)
    n_all = table_rows[0]["n"]
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Concordance of estate-representative (widow) given names with the linked census spouse, deceased-inventor patents}}
\label{{tab:relative_validation}}
\begin{{threeparttable}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lcccc@{{}}}}
\toprule
\textbf{{Links}} & \makecell{{\textbf{{N}}\\\textbf{{checkable}}}} & \makecell{{\textbf{{Concordant}}\\\textbf{{(95\% CI)}}}} & \makecell{{\textbf{{Chance}}\\\textbf{{level}}}} & \makecell{{\textbf{{Implied precision}}\\($s=0.8$--$0.9$)}} \\
\midrule
{body}
\bottomrule
\end{{tabular*}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Each observation is a patent granted to the estate of a deceased inventor whose representative --- extracted from the patent text --- shares the inventor's surname and carries a feminine role word (\textit{{executrix}}, \textit{{administratrix}}, \textit{{widow}}) or a sex-neutral one (trustee, guardian, \ldots); masculine-role representatives (\textit{{executor}}, \textit{{administrator}} --- typically a son or brother) are excluded from the comparison. The representative's sex is taken from the role word, corrected when the given name's census sex distribution decisively contradicts it (patent role words are noisy: plural forms cover mixed-sex representative pairs and the masculine form is sometimes used for women). The row is \textit{{checkable}} when the model links the inventor in the census wave preceding the patent and the linked household contains a spouse ({n_all:,} checkable rows, {n_unique:,} unique linked inventors). A row is \textit{{concordant}} when the representative's given name matches the census spouse's given name, allowing nickname and period-diminutive variants from standard genealogical lists and minor spelling differences (one edit, two edits on names of six or more letters, or rapidfuzz ratio $\geq$ {args.cutoff:g}). The representative's name is never a model feature, so concordance is independent evidence the link is correct. The dataset is released without a probability threshold; the \textit{{All links}} row includes every link however low its predicted probability, and the remaining rows restrict to links with probability $\geq p$. The chance level is the average concordance when the representative names are randomly re-assigned across the checkable patents while each household's spouse stays fixed (seeded permutation, {n_perm:,} replicates; names are re-assigned only among representatives with the same role-word gender, so the pool of names entering the spouse comparison is unchanged); each table row's chance level is computed by permuting within that row's own subset of links. Implied precision follows the mixture $p_{{\mathrm{{obs}}}} = \pi s + (1 - \pi)\,p_0$ with assumed sensitivity $s = 0.8$--$0.9$; a correct link can still fail the test (a wife who died, remarried or lived elsewhere between the census and the inventor's death, or a garbled given name), so $s < 1$. Where observed concordance reaches the assumed sensitivity the point estimate saturates at one and is uninformative; those rows quote a conservative lower bound instead --- the Wilson lower confidence limit of the concordance under $s = 0.9$ --- marked ``$\geq$''.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""


def make_sweep_figure(sweep_rows, pdf_path):
    """Observed widow--spouse concordance (line + Wilson band) vs the permutation
    chance level (dashed), by probability threshold. Quiet chrome, style of the
    famtree sweep."""
    from _viz_common import configure_matplotlib
    plt, using_tex = configure_matplotlib(ink=INK, baseline=BASELINE, label=INK_2,
                                          xtick=INK_2, ytick=MUTED)
    from matplotlib.ticker import PercentFormatter

    xs = [r["threshold"] for r in sweep_rows]
    obs = [100 * r["p_obs"] for r in sweep_rows]
    lo = [100 * r["ci_lo"] for r in sweep_rows]
    hi = [100 * r["ci_hi"] for r in sweep_rows]
    null = [100 * r["p_null"] for r in sweep_rows]

    pct = "\\%" if using_tex else "%"
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.fill_between(xs, lo, hi, color=OBS_COLOR, alpha=0.15, linewidth=0)
    ax.plot(xs, obs, color=OBS_COLOR, linewidth=1.8, marker="o", markersize=4.0,
            markeredgecolor="white", markeredgewidth=0.4, solid_capstyle="round",
            label=f"Widow--spouse concordance (95{pct} CI)" if using_tex
            else f"Widow-spouse concordance (95{pct} CI)")
    ax.plot(xs, null, color=INK, linewidth=1.6, linestyle=(0, (4, 2)),
            label="Permutation chance level")
    ax.set_xlabel("Probability threshold (links with $p \\geq$ threshold)")
    ax.set_ylabel("Representative name concordant")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              frameon=False, fontsize=8, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.perf_counter()
    nickmap = rc.load_nickmaps(args.nicknames, tag="[5]")
    cache = rc._MatchCache(args.cutoff, nickmap)

    rows_df = pl.read_csv(args.rows, infer_schema_length=0).with_columns([
        pl.col("previous_census_year").cast(pl.Int64),
        pl.col("probability").cast(pl.Float64, strict=False),
        pl.col("checkable").str.to_lowercase() == "true",
    ])
    chk_df = rows_df.filter(pl.col("checkable"))
    n_total = rows_df.height
    if chk_df.height == 0:
        raise SystemExit(f"[5] no checkable rows (of {n_total} total); nothing to score.")
    rows = build_rows(chk_df)
    if not rows:
        raise SystemExit(f"[5] no spouse-checkable rows (of {chk_df.height} stage-4 "
                         f"checkable); nothing to score.")
    t0 = rc._lap(f"loaded {n_total:,} roster rows, {chk_df.height:,} stage-4 "
                 f"checkable, {len(rows):,} spouse-checkable", t0)

    # Cross-check the stage-4 informational pair flags at this cutoff (spouse
    # pairs only -- the scored universe).
    pairs_df = pl.read_csv(args.pairs, infer_schema_length=0)
    gated = pairs_df.filter((pl.col("sex_gate_pass").str.to_lowercase() == "true")
                            & (pl.col("relationship") == "spouse")
                            & (pl.col("rep_given") != "") & (pl.col("relative_given") != ""))
    mism = sum(1 for r in gated.iter_rows(named=True)
               if cache(r["rep_given"], r["relative_given"])
               != (str(r["pair_concordant"]).lower() == "true"))
    if mism:
        print(f"[5] WARN {mism:,} gated spouse pairs flip concordance at cutoff "
              f"{args.cutoff:g} vs the stage-4 preview (different --cutoff/"
              f"--nicknames between stages -- this stage's numbers win)", flush=True)

    # ---- headline ----
    flags = rc.observed_concordance(rows, args.cutoff, nickmap)
    fuzzy_only = rc.observed_concordance(rows, args.cutoff, None)
    n_nick_added = int(sum(flags)) - int(sum(fuzzy_only))
    k = int(sum(flags))
    p_obs, lo, hi = rc.wilson(k, len(rows))
    p_null, null_lo, null_hi = rc.permutation_null_gated(
        rows, args.cutoff, args.n_perm, args.seed, nickmap)
    prec = [(s, rc.implied_precision(p_obs, p_null, s)) for s in args.sensitivities]
    t0 = rc._lap(f"headline + permutation null ({args.n_perm} reps)", t0)

    # ---- by probability floor ----
    prob_rows = []
    for t in sorted(set(args.prob_thresholds)):
        idx = [j for j, r in enumerate(rows) if r["probability"] >= t]
        if not idx:
            continue
        sub = [rows[j] for j in idx]
        sk = int(sum(flags[j] for j in idx))
        sp, slo, shi = rc.wilson(sk, len(idx))
        spn, _, _ = rc.permutation_null_gated(sub, args.cutoff, args.n_perm,
                                              args.seed, nickmap)
        prob_rows.append((t, len(idx), sk, sp, slo, shi, spn))
    t0 = rc._lap("by-probability breakdown", t0)

    # ---- sweep grid (figure input) ----
    grid, t = [], 0.0
    while t <= args.sweep_max + 1e-9:
        grid.append(round(t, 4))
        t += args.sweep_step
    sweep_rows = []
    for t in grid:
        idx = [j for j, r in enumerate(rows) if r["probability"] >= t]
        if not idx:
            continue
        sub = [rows[j] for j in idx]
        sk = int(sum(flags[j] for j in idx))
        sp, slo, shi = rc.wilson(sk, len(idx))
        spn, _, _ = rc.permutation_null_gated(sub, args.cutoff, args.n_perm,
                                              args.seed, nickmap)
        sweep_rows.append({"threshold": t, "n": len(idx), "k": sk, "p_obs": sp,
                           "ci_lo": slo, "ci_hi": shi, "p_null": spn})
    t0 = rc._lap("threshold sweep", t0)

    # ---- by census year / method split / unique-inventor dedup ----
    by_year = {}
    for r, f in zip(rows, flags):
        by_year.setdefault(r["previous_census_year"], []).append(f)
    year_rows = [(y, len(v), int(sum(v))) for y, v in sorted(by_year.items())]

    ptr_flags = [f for r, f in zip(rows, flags) if "relate" not in r["rel_methods"]]
    rel_flags = [f for r, f in zip(rows, flags) if "relate" in r["rel_methods"]]
    method_rows = [("pointer_only", len(ptr_flags), int(sum(ptr_flags))),
                   ("any_relate", len(rel_flags), int(sum(rel_flags)))]

    seen, dedup_flags = set(), []
    for r, f in zip(rows, flags):
        key = (r["deceased_surname"], r["histid"])
        if key in seen:
            continue
        seen.add(key)
        dedup_flags.append(f)
    n_unique = len(dedup_flags)

    # ---- console report ----
    print("\n===== widow--spouse concordance =====", flush=True)
    print(f"roster rows ............... {n_total:,}", flush=True)
    print(f"checkable (spouse) ........ {len(rows):,} "
          f"({n_unique:,} unique linked inventors)", flush=True)
    print(f"p_obs (concordance) ....... {p_obs:.3f}  [{lo:.3f}, {hi:.3f}]  "
          f"(cutoff {args.cutoff:g}; +{n_nick_added} via nickname map)", flush=True)
    print(f"p_null (permutation) ...... {p_null:.3f}  [{null_lo:.3f}, {null_hi:.3f}]",
          flush=True)
    print(f"lift (p_obs - p_null) ..... {p_obs - p_null:+.3f}", flush=True)
    print("implied subgroup precision by assumed sensitivity s:", flush=True)
    for s, pr in prec:
        print(f"   s={s:.2f} -> precision {pr:.3f}"
              if not math.isnan(pr) else f"   s={s:.2f} -> n/a (s<=p_null)", flush=True)
    print("by probability floor (n | concordance | p_null | implied precision "
          "@ s=.80/.90):", flush=True)
    for t, sn, sk, sp, slo, shi, spn in prob_rows:
        pr80 = rc.implied_precision(sp, spn, 0.80)
        pr90 = rc.implied_precision(sp, spn, 0.90)
        print(f"   >={t:.2f}: n={sn:<5d} {sk}/{sn} = {sp:.3f} [{slo:.3f}, {shi:.3f}]  "
              f"p_null={spn:.3f}  prec={pr80:.3f}/{pr90:.3f}", flush=True)
    print("by census year (concordance):", flush=True)
    for y, n, kk in year_rows:
        yp, ylo, yhi = rc.wilson(kk, n)
        print(f"   {y}: {kk}/{n} = {yp:.3f} [{ylo:.3f}, {yhi:.3f}]", flush=True)
    print("by classification method:", flush=True)
    for lab, n, kk in method_rows:
        mp, mlo, mhi = rc.wilson(kk, n)
        print(f"   {lab}: {kk}/{n} = {mp:.3f} [{mlo:.3f}, {mhi:.3f}]"
              if n else f"   {lab}: n=0", flush=True)
    dk = int(sum(dedup_flags))
    dp, dlo, dhi = rc.wilson(dk, n_unique)
    print(f"unique-inventor dedup ..... {dk}/{n_unique} = {dp:.3f} "
          f"[{dlo:.3f}, {dhi:.3f}]", flush=True)

    # ---- summary CSV ----
    summary_path = os.path.join(args.output_dir, "relative_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "group", "n", "k", "rate", "ci_lo", "ci_hi"])
        attr_path = args.attrition or os.path.join(
            os.path.dirname(os.path.abspath(args.rows)), "relative_attrition.csv")
        if os.path.exists(attr_path):
            for r in pl.read_csv(attr_path).iter_rows(named=True):
                w.writerow(["attrition", r["label"], r["n"], "", "", "", ""])
        w.writerow(["checkable_spouse", "overall", len(rows), "", "", "", ""])
        w.writerow(["p_obs", "overall", len(rows), k, p_obs, lo, hi])
        w.writerow(["p_null", "permutation", len(rows), "", p_null, null_lo, null_hi])
        for s, pr in prec:
            w.writerow(["implied_precision", f"s={s:.2f}", len(rows), "", pr, "", ""])
        for t, sn, sk, sp, slo, shi, spn in prob_rows:
            w.writerow(["p_obs_by_prob", f">={t:.2f}", sn, sk, sp, slo, shi])
            w.writerow(["p_null_by_prob", f">={t:.2f}", sn, "", spn, "", ""])
            for s in args.sensitivities:
                w.writerow(["implied_precision_by_prob", f">={t:.2f} s={s:.2f}", sn,
                            "", rc.implied_precision(sp, spn, s), "", ""])
        for y, n, kk in year_rows:
            yp, ylo, yhi = rc.wilson(kk, n)
            w.writerow(["p_obs_by_year", str(y), n, kk, yp, ylo, yhi])
        for lab, n, kk in method_rows:
            mp, mlo, mhi = rc.wilson(kk, n)
            w.writerow(["p_obs_by_method", lab, n, kk, mp, mlo, mhi])
        w.writerow(["p_obs_dedup", "unique (deceased_surname, histid)",
                    n_unique, dk, dp, dlo, dhi])
        w.writerow(["nickname_added", "beyond fuzzy cutoff", len(rows),
                    n_nick_added, "", "", ""])
    print(f"\n[5] wrote summary -> {summary_path}", flush=True)

    # ---- sweep CSV + figure ----
    sweep_path = os.path.join(args.output_dir, "relative_sweep.csv")
    with open(sweep_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["threshold", "n", "k", "p_obs", "ci_lo", "ci_hi", "p_null"]
        w.writerow(cols)
        for r in sweep_rows:
            w.writerow([r[c] for c in cols])
    print(f"[5] wrote sweep -> {sweep_path}", flush=True)
    pdf_path = os.path.join(args.output_dir, "relative_concordance_sweep.pdf")
    make_sweep_figure(sweep_rows, pdf_path)
    print(f"[5] wrote {pdf_path}", flush=True)

    # ---- manuscript table (one row per probability floor, each against its own
    # subset permutation null; t=0 reuses the headline null) ----
    table_rows = []
    for t in sorted(set(args.table_thresholds)):
        idx = [j for j, r in enumerate(rows) if r["probability"] >= t]
        if not idx:
            continue
        sub = [rows[j] for j in idx]
        tk = int(sum(flags[j] for j in idx))
        tp, tlo, thi = rc.wilson(tk, len(idx))
        if t <= 0:
            tnull = p_null
        else:
            tnull, _, _ = rc.permutation_null_gated(sub, args.cutoff, args.n_perm,
                                                    args.seed, nickmap)
        table_rows.append({"t": t, "n": len(idx), "k": tk, "p": tp, "lo": tlo,
                           "hi": thi, "null": tnull})
    tex = fmt_table(table_rows, args, args.n_perm, n_unique)
    tex_path = os.path.join(args.output_dir, "relative_val_tab.tex")
    with open(tex_path, "w") as fh:
        fh.write(tex)
    print(f"[5] wrote {tex_path}", flush=True)
    rc._lap("done", t0)


if __name__ == "__main__":
    main()
