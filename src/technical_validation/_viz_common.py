"""Shared figure styling for the technical-validation plots.

The manuscript is a plain ``\\documentclass[12pt]{article}`` with
``\\usepackage[T1]{fontenc}`` and no font package, so its body font is the LaTeX
default Computer Modern. To make figure text match the manuscript exactly --- the
same glyphs for ``%``, en-dashes and ``\\geq`` that matplotlib's bundled ``cmr10``
does not carry and would otherwise silently substitute from a fallback face --- we
typeset all figure text through LaTeX itself via the **pgf** backend driven by
``pdflatex`` (which needs no ghostscript, unlike the usetex PDF backend). When no
``pdflatex`` is on PATH (e.g. a bare CI checkout) we fall back to matplotlib's
bundled Computer Modern so the scripts still run anywhere.

Kept a plain ``_``-prefixed helper (not a digit-prefixed pipeline stage) so both
``2_ctp_famtree_pairs.py`` and ``5_relative_report.py`` (and scratchpad scripts) can import it.
"""

import shutil


def configure_matplotlib(ink, baseline, label, xtick, ytick, usetex=None):
    """Pick backend + fonts so figure text matches the manuscript's Computer Modern.

    Parameters mirror the scripts' palette constants (all hex strings). ``usetex``:
    ``None`` auto-detects a ``pdflatex`` toolchain; ``True``/``False`` force it.
    Returns ``(plt, using_tex)``; the backend is selected before ``pyplot`` import,
    so call this once, before creating any figure.
    """
    import matplotlib

    if usetex is None:
        usetex = shutil.which("pdflatex") is not None

    matplotlib.use("pgf" if usetex else "Agg")
    import matplotlib.pyplot as plt

    params = {
        "font.size": 9,
        "pdf.fonttype": 42,
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "text.color": ink,
        "axes.edgecolor": baseline,
        "axes.labelcolor": label,
        "xtick.color": xtick,
        "ytick.color": ytick,
    }
    if usetex:
        params.update({
            "text.usetex": True,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "font.family": "serif",
            # mirror the manuscript preamble so metrics/glyphs match the body font
            "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}",
        })
    else:
        params.update({
            "font.family": "serif",
            "font.serif": ["cmr10", "CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        })
    plt.rcParams.update(params)
    return plt, usetex
