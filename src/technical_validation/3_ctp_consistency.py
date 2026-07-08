#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Technical validation 3: CTP agreement with the model links, by predicted probability.

Paper-analysis add-on (Scientific Data descriptor) -- NOT a pipeline stage. Reads ONE
CTP-annotated links parquet (the `ml_model/4_ctp_correction.py` output: the 14
publication columns + histid_ctp / ctp_flag / ctp_action / from_ctp) and tabulates how
often the model's cross-wave links agree with the Census Tree Project, overall, by
predicted-probability bin, and by census wave.

Definitions (all on MODEL rows only -- `from_ctp==True` supplements are excluded, they
are CTP-derived and would agree tautologically):

  * comparable  -- rows with ctp_action in {consistent, replaced, unverified}: the
                   non-anchor rows of multi-wave inventors, i.e. the links CTP could in
                   principle check. `anchor` rows are excluded because histid_ctp is
                   propagated FROM the anchor (anchor-relative -- it agrees with itself
                   by construction); `single` rows have no second wave to check.
  * definite    -- the comparable rows where CTP takes a definite view (consistent or
                   replaced). `unverified` rows are CTP silence (no unambiguous CTP
                   link traced from the anchor): coverage, not contradiction.
  * agreement % -- consistent / (consistent + replaced), i.e. the share of
                   definite-view links where CTP names the SAME census person the
                   model picked.

Because the release is no-threshold (every group's top-1 candidate is published with
its probability), raw agreement pools high- and low-confidence links; the by-probability
table is the headline view. Run it twice -- on the full-CTP annotation and on a
Family-Tree-only re-annotation (`4_ctp_correction.py --ctp-link-cols family_tree`) --
distinguished by `--label`.

Outputs under --output-dir (console mirrors every table):
  {label}_action_counts.csv         sanity block: per-row ctp_action counts
  {label}_consistency_by_prob.csv   agreement by probability bin (+ Overall row)
  {label}_consistency_by_wave.csv   agreement by census wave (+ Total row)
  {label}_consistency_by_prob.tex   ready-to-include booktabs/threeparttable table
                                    (dynamic numbers in the notes come from the same
                                    frame as the body, so they cannot drift)

All paths are CLI arguments; nothing is hardcoded.
"""

import argparse
import os
import time

DEFAULT_PROB_BINS = [0.2, 0.4, 0.6, 0.8]
COMPARABLE_ACTIONS = ["consistent", "replaced", "unverified"]


def _lap(label, t0):
    now = time.perf_counter()
    print(f"[timing] {label}: {now - t0:.1f}s", flush=True)
    return now


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CTP agreement with model links, overall / by probability bin / "
                    "by census wave.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotated", required=True,
                   help="CTP-annotated links parquet (4_ctp_correction.py output).")
    p.add_argument("--output-dir", required=True,
                   help="Directory for the CSV/tex outputs.")
    p.add_argument("--label", default="full",
                   help="Run label prefixed to every artifact (e.g. full / famtree).")
    p.add_argument("--prob-bins", type=float, nargs="+", default=DEFAULT_PROB_BINS,
                   help="Probability cut points (bins are [<b1), [b1,b2), ..., [bk,1]).")
    p.add_argument("--tex-label", default="tab:ctp_consistency_prob",
                   help="LaTeX \\label of the emitted by-probability table.")
    p.add_argument("--spill-dir", default=None,
                   help="Directory for polars streaming spill (sets POLARS_TEMP_DIR/"
                        "TMPDIR). Default <output-dir>/.polars_tmp. MUST be real disk, "
                        "never a tmpfs /tmp.")
    return p.parse_args(argv)


def _bin_labels(breaks):
    """['<0.2', '0.2-0.4', ..., '>=0.8'] for sorted cut points."""
    labels = [f"<={breaks[0]:g}"]
    labels += [f"{lo:g}-{hi:g}" for lo, hi in zip(breaks[:-1], breaks[1:])]
    labels.append(f">{breaks[-1]:g}")
    return labels


def _bin_exprs(pl, breaks):
    """(label_expr, index_expr) assigning each probability its bin; nulls -> 'n/a'."""
    labels = _bin_labels(breaks)
    prob = pl.col("probability")
    lab = pl.when(prob.is_null()).then(pl.lit("n/a"))
    ix = pl.when(prob.is_null()).then(pl.lit(len(labels)))
    for i, b in enumerate(breaks):
        lab = lab.when(prob <= b).then(pl.lit(labels[i]))
        ix = ix.when(prob <= b).then(pl.lit(i))
    lab = lab.otherwise(pl.lit(labels[-1]))
    ix = ix.otherwise(pl.lit(len(breaks)))
    return lab.alias("prob_bin"), ix.cast(pl.Int32).alias("_bin_ix")


def consistency_table(pl, comparable, key_cols, total_label):
    """Agreement tabulation of the comparable rows grouped by `key_cols`, with a
    pooled total row appended. Returns an eager frame ordered by the group key."""
    act = pl.col("ctp_action")
    aggs = [
        pl.len().alias("n_comparable"),
        (act == "consistent").sum().alias("n_consistent"),
        (act == "replaced").sum().alias("n_replaced"),
        (act == "unverified").sum().alias("n_unverified"),
    ]
    per = comparable.group_by(key_cols).agg(aggs).sort(key_cols)
    total = comparable.select(aggs)
    key_out = key_cols[0] if key_cols[0] != "_bin_ix" else "prob_bin"
    per = per.select([key_out] + [c for c in per.columns if c not in key_cols])
    total = total.with_columns(pl.lit(total_label).alias(key_out)).select(per.columns)
    out = pl.concat([per.with_columns(pl.col(key_out).cast(pl.Utf8)), total])
    return out.with_columns([
        (pl.col("n_consistent") + pl.col("n_replaced")).alias("n_definite"),
        pl.when(pl.col("n_consistent") + pl.col("n_replaced") > 0)
          .then(100.0 * pl.col("n_consistent")
                / (pl.col("n_consistent") + pl.col("n_replaced")))
          .otherwise(None).alias("agreement_pct"),
    ])


def _tex_bin(label):
    """CSV bin label -> LaTeX ('<0.2' -> '$<$0.2', '0.2-0.4' -> '0.2--0.4')."""
    if label.startswith("<="):
        return f"$\\leq${label[2:]}"
    if label.startswith(">"):
        return f"$>${label[1:]}"
    return label.replace("-", "--")


def _fmt_latex_table(rows, tex_label, ctp_desc):
    """The by-probability agreement table (booktabs + threeparttable, house style).
    `rows` = list of dicts incl. the Overall row (last); every number in the notes is
    interpolated from the same rows as the body."""
    def _pct(r):
        return f"{r['agreement_pct']:.1f}\\%" if r["agreement_pct"] is not None else "--"

    body = []
    for r in rows[:-1]:
        body.append(f"{_tex_bin(r['prob_bin'])} & {r['n_comparable']:,} & "
                    f"{r['n_definite']:,} & {_pct(r)} \\\\")
    body_str = "\n".join(body)
    tot = rows[-1]
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Agreement of cross-wave model links with the Census Tree Project, by predicted probability}}
\label{{{tex_label}}}
\begin{{threeparttable}}
\begin{{tabular}}{{lccc}}
\toprule
\textbf{{Probability bin}} & \textbf{{N comparable}} & \textbf{{N with definite CTP view}} & \textbf{{\% agreement}} \\
\midrule
{body_str}
\midrule
Overall & {tot['n_comparable']:,} & {tot['n_definite']:,} & {_pct(tot)} \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Model-predicted links only (CTP-supplemented rows excluded). \textit{{Comparable}} links are the non-anchor observations of inventors matched in more than one census wave -- the links the {ctp_desc} crosswalks could in principle check ({tot['n_comparable']:,} in total); the anchor wave itself is excluded because \textit{{histid\_ctp}} is propagated from it. The CTP takes a \textit{{definite view}} on {tot['n_definite']:,} of these (the remainder are CTP-silent: no unambiguous CTP link traced from the anchor, reflecting coverage rather than contradiction). \textit{{\% agreement}} is the share of definite-view links where the CTP names the same census record the model selected; it is {_pct(tot)} pooled across all probabilities. The dataset is released without a probability threshold, so low-confidence links are included by design; users applying their own threshold retain the upper bins.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""


def main(argv=None):
    args = parse_args(argv)
    t0 = time.perf_counter()

    spill_dir = args.spill_dir or os.path.join(args.output_dir, ".polars_tmp")
    os.makedirs(spill_dir, exist_ok=True)
    os.environ["POLARS_TEMP_DIR"] = spill_dir
    os.environ["TMPDIR"] = spill_dir
    import polars as pl

    os.makedirs(args.output_dir, exist_ok=True)
    breaks = sorted(args.prob_bins)
    lab = args.label

    df = pl.read_parquet(args.annotated,
                         columns=["census_year", "probability", "ctp_action", "from_ctp"])
    model = df.filter(~pl.col("from_ctp"))
    n_supp = df.height - model.height
    print(f"[1] {args.annotated}")
    print(f"[1] rows={df.height:,} | model rows={model.height:,} | "
          f"excluded from_ctp supplements={n_supp:,}", flush=True)
    t0 = _lap("load annotated links", t0)

    # ---- sanity block: per-row ctp_action counts on model rows ----
    actions = model.group_by("ctp_action").len().sort("ctp_action")
    print("\n== ctp_action counts (model rows) ==")
    print(actions)
    actions.write_csv(os.path.join(args.output_dir, f"{lab}_action_counts.csv"))

    # ---- comparable / definite sets ----
    bin_lab, bin_ix = _bin_exprs(pl, breaks)
    comparable = (model.filter(pl.col("ctp_action").is_in(COMPARABLE_ACTIONS))
                  .with_columns([bin_lab, bin_ix]))
    print(f"\n[1] comparable rows (consistent+replaced+unverified): "
          f"{comparable.height:,}", flush=True)

    by_prob = consistency_table(pl, comparable, ["_bin_ix", "prob_bin"], "Overall")
    print("\n== CTP agreement by probability bin ==")
    with pl.Config(tbl_rows=by_prob.height, tbl_cols=len(by_prob.columns)):
        print(by_prob)
    by_prob.write_csv(os.path.join(args.output_dir, f"{lab}_consistency_by_prob.csv"))

    by_wave = consistency_table(pl, comparable, ["census_year"], "Total")
    print("\n== CTP agreement by census wave ==")
    with pl.Config(tbl_rows=by_wave.height, tbl_cols=len(by_wave.columns)):
        print(by_wave)
    by_wave.write_csv(os.path.join(args.output_dir, f"{lab}_consistency_by_wave.csv"))
    t0 = _lap("tabulate", t0)

    # ---- LaTeX table (by-probability view) ----
    ctp_desc = ("Family-Tree-restricted Census Tree Project (CTP)"
                if lab == "famtree" else "Census Tree Project (CTP)")
    tex = _fmt_latex_table(by_prob.to_dicts(), args.tex_label, ctp_desc)
    tex_path = os.path.join(args.output_dir, f"{lab}_consistency_by_prob.tex")
    with open(tex_path, "w") as fh:
        fh.write(tex)
    print(f"\n[1] wrote {tex_path}")
    print(f"[1] wrote CSVs -> {args.output_dir} (prefix '{lab}_')", flush=True)


if __name__ == "__main__":
    main()
