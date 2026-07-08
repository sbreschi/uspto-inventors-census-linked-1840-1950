#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Technical validation 1: the Data-Overview figure for the manuscript
(`fig:links_by_threshold`).

Paper-analysis add-on (Scientific Data descriptor) -- NOT a pipeline stage. The
released dataset is a *no-threshold* deposit: `3_predict.py` keeps the top-1 census
candidate for every (patent-inventor, census-year) that survived blocking + filtering,
each carrying its classifier `probability`, and applies NO acceptance cut. The single
descriptive question a downstream user asks is therefore "how much of the file survives
at the probability threshold I choose?".

This script reads the prediction parquet's `probability` (+ `PatInvID`, `census_year`)
columns and, over a fine threshold grid, computes two purely descriptive series:

  n_links      # released links with probability >= tau
  n_inventors  # distinct PatInvID with at least one link >= tau
               (per-inventor max-probability, then count)

It writes a small CSV of both curves and a vector PDF (two step series vs tau, log
y-axis) styled to match the manuscript body font. It also prints headline totals and a
per-census-year distinct-inventor breakdown at tau in {0, --illustrative-tau} -- the
numbers needed to state coverage in the text without pulling a whole table into the
paper.

Input is the ML prediction parquet (3_predict.py output, `probability` non-null), NOT
the CTP-annotated deposit -- supplemented rows have a null probability and are not part
of a probability-threshold description. The two headline counts (ML links vs total
deposited rows incl. CTP supplements) are quoted from the CTP report in the text.

All paths are CLI arguments; nothing is hardcoded.
"""

import argparse
import os
import time

# Palette mirrors the sibling technical-validation figures.
INK, INK_2, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LINKS_COLOR, INV_COLOR = "#2a78d6", "#e34948"

# Probability bands for the by-wave composition figure (highest first, drawn bottom-up),
# each an interval [lo, hi). Sequential blue ramp: darker = higher confidence.
PROB_BANDS = [
    (0.8, 1.01, r"$p \geq 0.8$", "#123c72"),
    (0.6, 0.8, r"$0.6 \leq p < 0.8$", "#2a78d6"),
    (0.4, 0.6, r"$0.4 \leq p < 0.6$", "#7fb0e6"),
    (0.0, 0.4, r"$p < 0.4$", "#c9ddf3"),
]


def _lap(label, t0):
    now = time.perf_counter()
    print(f"[timing] {label}: {now - t0:.1f}s", flush=True)
    return now


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Data-Overview figure: released links and distinct patent-inventors "
                    "at or above each match-probability threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prediction-parquet", required=True,
                   help="ML prediction parquet (3_predict.py output) with a non-null "
                        "`probability` column, one row per released link.")
    p.add_argument("--fig-out", required=True,
                   help="Output PDF path (e.g. the Overleaf Images/links_by_threshold.pdf).")
    p.add_argument("--csv-out", required=True,
                   help="Output CSV path for the (tau, n_links, n_inventors) curve.")
    p.add_argument("--year-fig-out", default=None,
                   help="Optional output PDF for the by-census-wave composition figure "
                        "(distinct patent-inventors per wave, stacked by strongest-link "
                        "probability band). Omit to skip.")
    p.add_argument("--year-csv-out", default=None,
                   help="Optional output CSV for the by-wave composition "
                        "(census_year x probability band -> n_inventors). Omit to skip.")
    p.add_argument("--tau-step", type=float, default=0.01,
                   help="Threshold grid step over [0, 1].")
    p.add_argument("--illustrative-tau", type=float, default=0.5,
                   help="A second threshold (besides 0) for the per-wave inventor "
                        "breakdown printed to stdout.")
    p.add_argument("--inventor-total", type=int, default=2_471_008,
                   help="Total patent-inventor universe; the figure's right axis "
                        "expresses the distinct-inventor count as a share of this.")
    p.add_argument("--spill-dir", default=None,
                   help="Directory for polars streaming spill (sets POLARS_TEMP_DIR/"
                        "TMPDIR). Default <fig-out dir>/.polars_tmp. MUST be real disk, "
                        "never a tmpfs /tmp.")
    return p.parse_args(argv)


def _grid(tau_step):
    """Inclusive [0, 1] grid at tau_step, integer-scaled to avoid float drift."""
    n = int(round(1.0 / tau_step))
    return [round(i * tau_step, 10) for i in range(n + 1)]


def compute_curves(pl, parquet_path, taus):
    """Return (curve_df, per_inv_df).

    curve_df: one row per tau with n_links (links with prob >= tau) and n_inventors
    (distinct PatInvID whose max prob >= tau). Both are computed from cheap aggregates
    -- a probability histogram for the link counts, and the per-inventor max
    probability for the inventor counts -- so the parquet is scanned once, streaming,
    two columns only.
    """
    lf = pl.scan_parquet(parquet_path).select("PatInvID", "census_year", "probability")

    # Per-inventor max probability (drives the distinct-inventor curve + per-wave table).
    per_inv = (lf.group_by("PatInvID")
               .agg(pl.col("probability").max().alias("max_prob"),
                    # keep the census_year of the strongest link for the per-wave table
                    pl.col("census_year").sort_by("probability").last().alias("best_year"))
               .collect(engine="streaming"))

    probs = lf.select("probability").collect(engine="streaming")["probability"]
    inv_max = per_inv["max_prob"]

    # For each tau, count via a single sort+searchsorted rather than |taus| passes.
    import numpy as np
    probs_sorted = np.sort(probs.to_numpy())
    inv_sorted = np.sort(inv_max.to_numpy())
    tau_arr = np.asarray(taus, dtype=float)
    # n with value >= tau  ==  N - searchsorted(sorted, tau, side="left")
    n_links = len(probs_sorted) - np.searchsorted(probs_sorted, tau_arr, side="left")
    n_inv = len(inv_sorted) - np.searchsorted(inv_sorted, tau_arr, side="left")

    curve = pl.DataFrame({
        "tau": tau_arr,
        "n_links": n_links.astype("int64"),
        "n_inventors": n_inv.astype("int64"),
    })
    return curve, per_inv


def per_wave_inventors(pl, per_inv, tau):
    """Distinct inventors (by strongest-link census wave) with max_prob >= tau."""
    return (per_inv.filter(pl.col("max_prob") >= tau)
            .group_by("best_year").len()
            .sort("best_year"))


def per_wave_composition(pl, per_inv):
    """Distinct patent-inventors per strongest-link census wave, split by the
    probability band of that strongest link (PROB_BANDS). Returns a wide DataFrame:
    one row per census_year, one integer column per band label, sorted by year.

    Each inventor is counted once, in the band of its max-probability link -- so the
    band columns sum to the same per-wave distinct-inventor totals the tau=0 table
    prints, exactly (the two views must reconcile; the verification step checks this).
    """
    band_label = pl.col("max_prob")
    # Assign a band label per inventor (bands are contiguous and cover [0, 1]).
    expr = pl.when(pl.col("max_prob") >= PROB_BANDS[0][0]).then(pl.lit(PROB_BANDS[0][2]))
    for lo, hi, lab, _ in PROB_BANDS[1:]:
        expr = expr.when(pl.col("max_prob") >= lo).then(pl.lit(lab))
    expr = expr.otherwise(pl.lit(PROB_BANDS[-1][2]))
    del band_label

    labels = [b[2] for b in PROB_BANDS]
    tagged = per_inv.with_columns(expr.alias("band"))
    wide = (tagged.group_by("best_year", "band").len()
            .collect() if hasattr(tagged, "collect") else
            tagged.group_by("best_year", "band").len())
    wide = (wide.pivot(values="len", index="best_year", on="band", aggregate_function="sum")
            .rename({"best_year": "census_year"})
            .fill_null(0)
            .sort("census_year"))
    # Ensure every band column exists (a band may be empty in a tiny fixture).
    for lab in labels:
        if lab not in wide.columns:
            wide = wide.with_columns(pl.lit(0).cast(pl.Int64).alias(lab))
    return wide.select(["census_year"] + labels)


def make_year_figure(comp_df, pdf_path):
    """Stacked-bar composition: distinct patent-inventors per census wave, coloured by
    the strongest-link probability band. Highest-confidence band on the bottom so the
    growth of the confident mass over time reads directly off the axis."""
    from _viz_common import configure_matplotlib
    from matplotlib.ticker import FuncFormatter
    plt, _ = configure_matplotlib(ink=INK, baseline=BASELINE, label=INK_2,
                                  xtick=INK_2, ytick=MUTED)

    years = [str(y) for y in comp_df["census_year"].to_list()]
    x = list(range(len(years)))
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    bottom = [0.0] * len(years)
    for lo, hi, lab, color in PROB_BANDS:
        vals = comp_df[lab].to_list()
        ax.bar(x, vals, bottom=bottom, width=0.72, color=color, label=lab,
               edgecolor="white", linewidth=0.4)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("Census wave")
    ax.set_ylabel("Distinct patent-inventors")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1e3:g}k"))
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Legend in probability order (highest band first) despite bottom-up stacking.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=8,
              handlelength=1.2, ncol=2)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def make_figure(curve_rows, pdf_path, inventor_total):
    from _viz_common import configure_matplotlib
    from matplotlib.ticker import FuncFormatter
    plt, using_tex = configure_matplotlib(ink=INK, baseline=BASELINE, label=INK_2,
                                          xtick=INK_2, ytick=MUTED)
    pct = r"\%" if using_tex else "%"

    taus = [r["tau"] for r in curve_rows]
    links = [r["n_links"] for r in curve_rows]
    shares = [100.0 * r["n_inventors"] / inventor_total for r in curve_rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    # Left axis (linear, red): share of ALL patent-inventors matched at >= tau -- its
    # own explicit curve, distinct in shape from the log-count line on the right.
    (line_share,) = ax.plot(taus, shares, color=INV_COLOR, linewidth=2.0,
                            label="Distinct patent-inventors (share)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Match-probability threshold")
    ax.set_ylabel("Share of all patent-inventors")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}{pct}"))
    ax.yaxis.label.set_color(INV_COLOR)
    ax.tick_params(axis="y", colors=INV_COLOR, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_color(LINKS_COLOR)

    # Right axis (linear, blue): number of released links at >= tau, in millions.
    ax2 = ax.twinx()
    (line_links,) = ax2.plot(taus, links, color=LINKS_COLOR, linewidth=2.0,
                            label="Released links (count)")
    ax2.set_ylim(0, 4_000_000)
    ax2.set_ylabel("Released links (millions)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1e6:g}"))
    ax2.yaxis.label.set_color(LINKS_COLOR)
    ax2.tick_params(axis="y", colors=LINKS_COLOR, length=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(LINKS_COLOR)

    ax.legend(handles=[line_share, line_links], loc="lower left",
              frameon=False, fontsize=8, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.perf_counter()

    fig_dir = os.path.dirname(os.path.abspath(args.fig_out))
    spill_dir = args.spill_dir or os.path.join(fig_dir, ".polars_tmp")
    os.makedirs(spill_dir, exist_ok=True)
    os.environ["POLARS_TEMP_DIR"] = spill_dir
    os.environ["TMPDIR"] = spill_dir
    import polars as pl

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)), exist_ok=True)

    taus = _grid(args.tau_step)
    curve, per_inv = compute_curves(pl, args.prediction_parquet, taus)
    t0 = _lap("compute curves", t0)

    n_links_total = int(curve.filter(pl.col("tau") == 0.0)["n_links"][0])
    n_inv_total = int(curve.filter(pl.col("tau") == 0.0)["n_inventors"][0])
    print(f"[5] released ML links (prob>=0): {n_links_total:,}")
    print(f"[5] distinct patent-inventors:   {n_inv_total:,}")

    # Headline threshold readout (the numbers most likely quoted in the text).
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        row = curve.filter((pl.col("tau") - tau).abs() < 1e-9)
        if row.height:
            print(f"[5]   tau>={tau:.1f}: links={int(row['n_links'][0]):>10,}  "
                  f"inventors={int(row['n_inventors'][0]):>10,}")

    curve.write_csv(args.csv_out)
    print(f"[5] wrote {args.csv_out}")

    # Per-wave distinct-inventor breakdown at tau in {0, illustrative}.
    for tau in sorted({0.0, args.illustrative_tau}):
        pw = per_wave_inventors(pl, per_inv, tau)
        total = int(pw["len"].sum()) if pw.height else 0
        print(f"\n== distinct patent-inventors by strongest-link wave, tau>={tau:g} "
              f"(total {total:,}) ==")
        with pl.Config(tbl_rows=pw.height):
            print(pw)
    t0 = _lap("per-wave tables", t0)

    make_figure(curve.to_dicts(), args.fig_out, args.inventor_total)
    print(f"[5] wrote {args.fig_out}", flush=True)
    t0 = _lap("figure", t0)

    # Optional by-census-wave composition (distinct inventors per wave x prob band).
    if args.year_fig_out or args.year_csv_out:
        comp = per_wave_composition(pl, per_inv)
        band_cols = [b[2] for b in PROB_BANDS]
        comp = comp.with_columns(pl.sum_horizontal(band_cols).alias("total"))
        print("\n== distinct patent-inventors by wave x probability band ==")
        with pl.Config(tbl_rows=comp.height, tbl_cols=comp.width):
            print(comp)
        if args.year_csv_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.year_csv_out)), exist_ok=True)
            comp.write_csv(args.year_csv_out)
            print(f"[5] wrote {args.year_csv_out}")
        if args.year_fig_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.year_fig_out)), exist_ok=True)
            make_year_figure(comp, args.year_fig_out)
            print(f"[5] wrote {args.year_fig_out}", flush=True)
        _lap("by-wave figure", t0)


if __name__ == "__main__":
    main()
