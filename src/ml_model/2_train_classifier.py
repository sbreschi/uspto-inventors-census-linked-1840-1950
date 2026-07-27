#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML model stage 2: train and evaluate the candidate-scoring classifier.

Reads the labeled training table from stage 1, builds the model features, trains a
classifier, and evaluates it with the group-wise ranking rule (within each
(PatInvID, census_year) group the highest-probability candidate is taken as the
predicted match).

Training-set composition is decided HERE, not in stage 1 (which keeps every labeled
group and tags it `patinv_type` in {matched, orphan, hard_negative}). `--exclude-orphans`
/ `--exclude-hard-negatives` drop those types from the model FIT ONLY: the GroupShuffleSplit
runs on the full tagged set first, so the held-out evaluation split is identical regardless
of these flags (excluded types still appear in eval as answer-absent precision stress-tests).
`patinv_type` is not a feature (not in BASE_FEATURES), so it never leaks into the model.

Features (24):
  - diff_length = |len(inventor_name) - len(census_name)|
  - leven_simil = 1 - leven_norm
  - inter       = leven_simil * tksortratio
  - tkratio, tksortratio, county_distance
  - surname_logfreq = census-population surname rarity (-log10(n/N), rarer -> higher;
    attached at stage 3 (3_merge_features.py) from the name_rarity reference table)
  - age_at_pat  = pubyear - birthyr
  - 14 occupation/industry dummies built from the occupation dictionary applied to
    ind1950 / occ1950 / occstr (categories: agriculture, apparel, chemistry,
    civil_eng, comm_comp, electricity, mechanical, medical, engineering, physics,
    science, inventors, managers, manufacturing)
  - relatedness_short, relatedness_long

The occupation dictionary lives in data (`--occ-dict`, default
data/misc/occupation_dict.json); only the `broad_new_v2` / `strict_new_v2`
variants carry the `_ind1950`/`_occ1950`/`_occstr` triples this stage consumes.

Hyperparameters are explicit CLI arguments (`--rfc-estimators` / `--rfc-criterion`
/ `--rfc-max-features` / `--rfc-max-depth` / `--xgb-min-child-weight` /
`--xgb-gamma`); their defaults are the production values. All predicted matches are
published with their probabilities and the threshold is chosen downstream, so no
operating point is imposed here. `--prob-filter` (default 0.4) is a plain reporting
knob for the r2/r3 summary; the threshold-sweep report is the canonical full curve.
The classifier feature build uses pandas `.isin` / `.str.contains` semantics;
`random_state` is fixed for reproducibility.

`--diagnostics` appends a report-only overfitting section to the xlsx report
(train-vs-test gap, forest OOB, split-seed stability, inventor-level leakage
sensitivity, complexity curves, permutation importance, calibration). It runs
AFTER the production model is fitted, evaluated, and pickled, and never mutates
it; `--n-jobs` parallelizes only the diagnostics fits/predicts (the production
fit keeps n_jobs=1 for bit-reproducibility).

All inputs/outputs are CLI arguments; nothing is hardcoded.

"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

# The 14 occupation/industry categories the model uses.
OCC_CATEGORIES = ["agriculture", "apparel", "chemistry", "civil_eng", "comm_comp",
                  "electricity", "mechanical", "medical", "engineering", "physics",
                  "science", "inventors", "managers", "manufacturing"]

# Base model feature order. The optional `n_candidates` group-size feature (from the
# stage-3 merge) is appended at runtime unless --no-group-size.
BASE_FEATURES = (["diff_length", "leven_simil", "inter", "tkratio", "tksortratio",
                  "surname_logfreq", "county_distance", "age_at_pat"]
                 + OCC_CATEGORIES
                 + ["relatedness_short", "relatedness_long"])

# Split seeds for the --diagnostics seed-stability sweep (includes the production
# split seed 39 so the production point appears in the table).
DIAG_SEEDS = [7, 13, 21, 39, 42]

DEF_OCC_DICT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "data", "misc", "occupation_dict.json")


def _lap(label, t0):
    """Print elapsed seconds since t0 for a pipeline stage; return a fresh timestamp."""
    now = time.perf_counter()
    print(f"[timing] {label}: {now - t0:.1f}s", flush=True)
    return now


def split_dict(dictionary):
    """Split a v2 occupation dict into per-variable lookups by key suffix."""
    def by_suffix(suffix):
        return {k[: -len(suffix)]: v for k, v in dictionary.items() if k.endswith(suffix)}
    return by_suffix("_ind1950"), by_suffix("_occ1950"), by_suffix("_occstr")


def build_features(training, dictionary):
    """Build the 14 occupation dummies + derived features in place."""
    ind1950_list, occ1950_list, occstr_list = split_dict(dictionary)
    missing = [c for c in OCC_CATEGORIES
               if c not in ind1950_list or c not in occ1950_list or c not in occstr_list]
    if missing:
        raise SystemExit(
            f"[2] occupation dictionary variant is missing _ind1950/_occ1950/_occstr "
            f"entries for: {missing}. Use a v2 variant (broad_new_v2 / strict_new_v2).")

    # Occupation/industry dummies. occstr nulls -> "" so `.str.contains` returns False
    # (rather than NaN propagating through the boolean OR).
    occstr = training["occstr"].fillna("") if "occstr" in training else ""
    for occ in OCC_CATEGORIES:
        cond = (training["ind1950"].isin(ind1950_list[occ])
                | training["occ1950"].isin(occ1950_list[occ])
                | occstr.str.contains(occstr_list[occ], regex=True, case=False))
        training[occ] = np.where(cond, 1, 0)

    # Other features. diff_length may already be precomputed upstream (the stage-4
    # publication table drops the raw name strings and ships
    # diff_length in their place); only compute it here when the names are present and it
    # is not. The value is identical either way, so results are unchanged.
    if "diff_length" not in training.columns:
        training["diff_length"] = abs(training["inventor_name"].str.len()
                                      - training["census_name"].str.len())
    training["leven_simil"] = 1 - training["leven_norm"]
    training["inter"] = training["leven_simil"] * training["tksortratio"]
    training["age_at_pat"] = training["pubyear"] - training["birthyr"]
    return training


def group_ranking_eval(df_test, prob_filter):
    """Group-wise ranking precision/recall/F1 in three reporting variants. Returns a
    dict of metrics plus confusion/classification."""
    from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                                 precision_score, recall_score)

    df = df_test.copy()
    df["max_group_probability"] = df.groupby(["PatInvID", "census_year"])[
        "predicted_probability"].transform("max")
    df["rank_in_group"] = df.groupby(["PatInvID", "census_year"])[
        "predicted_probability"].rank(method="first", ascending=False)
    df["highest_probability_match_unique"] = (df["rank_in_group"] == 1).astype(int)

    out = {}
    ranked = df[df["rank_in_group"].notnull()]

    # r1: unique highest-probability match per group.
    y_true = ranked["true_match"]
    y_pred = ranked["highest_probability_match_unique"]
    out["r1"] = {"precision": precision_score(y_true, y_pred, zero_division=0),
                 "recall": recall_score(y_true, y_pred, zero_division=0),
                 "f1": f1_score(y_true, y_pred, zero_division=0)}

    # r2: same, but require max group probability above prob_filter.
    y_pred2 = np.where((ranked["max_group_probability"] > prob_filter)
                       & (ranked["highest_probability_match_unique"] == 1), 1, 0)
    out["r2"] = {"precision": precision_score(y_true, y_pred2, zero_division=0),
                 "recall": recall_score(y_true, y_pred2, zero_division=0),
                 "f1": f1_score(y_true, y_pred2, zero_division=0)}
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred2)
    out["classification_report"] = pd.DataFrame(
        classification_report(y_true, y_pred2, output_dict=True, zero_division=0)).transpose()

    # r3: filtered unique match, evaluated only on the true-match rows.
    true_rows = ranked[ranked["true_match"] == 1]
    y_true3 = true_rows["true_match"]
    y_pred3 = np.where((true_rows["max_group_probability"] > prob_filter)
                       & (true_rows["highest_probability_match_unique"] == 1), 1, 0)
    out["r3"] = {"precision": precision_score(y_true3, y_pred3, zero_division=0),
                 "recall": recall_score(y_true3, y_pred3, zero_division=0),
                 "f1": f1_score(y_true3, y_pred3, zero_division=0)}
    return out


# Column order for the threshold-sweep report.
SWEEP_COLUMNS = ["precision", "recall", "F1", "TP", "FP", "FN",
                 "RFC_estimators", "RFC_criterion", "RFC_features", "RFC_depth",
                 "xgb_min_child_weight", "xgb_gamma",
                 "occdict_name", "test_size", "random_state", "model_random_state"]


def model_hyperparams(model_name, cfg):
    """The hyperparameters actually consumed by each model in `build_model`, mapped
    to the SWEEP_COLUMNS names. Params a model does not use are left None (blank in
    the report) so the per-model sheets reflect the real config rather than the
    shared cfg dict: xgb ignores criterion/max_features; forest ignores
    min_child_weight/gamma."""
    shared = {"RFC_estimators": cfg["RFC_estimators"], "RFC_depth": cfg["RFC_depth"]}
    if model_name == "xgb":
        return {**shared, "RFC_criterion": None, "RFC_features": None,
                "xgb_min_child_weight": cfg["xgb_min_child_weight"], "xgb_gamma": cfg["xgb_gamma"]}
    if model_name == "forest":
        return {**shared, "RFC_criterion": cfg["RFC_criterion"], "RFC_features": cfg["RFC_features"],
                "xgb_min_child_weight": None, "xgb_gamma": None}
    # svc / other: none of these tree-booster params apply.
    return {"RFC_estimators": None, "RFC_depth": None, "RFC_criterion": None,
            "RFC_features": None, "xgb_min_child_weight": None, "xgb_gamma": None}


def threshold_sweep(df_test, model_name, cfg, occdict_name, test_size, random_state,
                    model_random_state=None, step=0.05):
    """Sweep the r2 group-ranking rule across probability thresholds (0.00..0.95 in
    `step`). At each threshold the prediction is the single top-ranked row per
    (PatInvID, census_year) group whose `max_group_probability` exceeds the
    threshold (same rule as `group_ranking_eval`'s r2). Returns a DataFrame indexed
    by `prob_threshold` with the precision/recall/F1/TP/FP/FN curve plus the
    `model_name`'s actual hyperparameters, in `SWEEP_COLUMNS` order. `random_state`
    is the GroupShuffleSplit seed (shared); `model_random_state` is the estimator's
    own RNG seed (e.g. forest uses 0, independent of the split seed)."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    df = df_test.copy()
    df["max_group_probability"] = df.groupby(["PatInvID", "census_year"])[
        "predicted_probability"].transform("max")
    df["rank_in_group"] = df.groupby(["PatInvID", "census_year"])[
        "predicted_probability"].rank(method="first", ascending=False)
    df["top_in_group"] = (df["rank_in_group"] == 1).astype(int)

    ranked = df[df["rank_in_group"].notnull()]
    y_true = ranked["true_match"].astype(int).to_numpy()
    const = {**model_hyperparams(model_name, cfg),
             "occdict_name": occdict_name, "test_size": test_size, "random_state": random_state,
             "model_random_state": model_random_state}

    rows = {}
    for thr in np.round(np.arange(0.0, 1.0, step), 2):
        y_pred = np.where((ranked["max_group_probability"] > thr)
                          & (ranked["top_in_group"] == 1), 1, 0)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        rows[float(thr)] = {"precision": precision_score(y_true, y_pred, zero_division=0),
                            "recall": recall_score(y_true, y_pred, zero_division=0),
                            "F1": f1_score(y_true, y_pred, zero_division=0),
                            "TP": tp, "FP": fp, "FN": fn, **const}

    out = pd.DataFrame.from_dict(rows, orient="index")[SWEEP_COLUMNS]
    out.index.name = "prob_threshold"
    return out


def build_model(model_name, cfg, random_state, n_jobs=1):
    # n_jobs=1 + a fixed seed make training bit-reproducible: multi-threaded float
    # reductions are the other run-to-run nondeterminism source besides row order.
    # The production/sweep fits always use the default n_jobs=1; only the
    # --diagnostics refits pass a higher value (forest fits are n_jobs-invariant,
    # but threaded predict_proba accumulation can jitter at ~1e-16).
    if model_name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(
            eval_metric="logloss", max_depth=cfg["RFC_depth"],
            min_child_weight=cfg["xgb_min_child_weight"], n_estimators=cfg["RFC_estimators"],
            learning_rate=0.1, objective="binary:logistic", booster="gbtree",
            tree_method="hist", grow_policy="lossguide", max_bin=256,
            scale_pos_weight=1, gamma=cfg["xgb_gamma"],
            random_state=random_state, n_jobs=1)
    if model_name == "forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            random_state=0, n_estimators=cfg["RFC_estimators"], criterion=cfg["RFC_criterion"],
            max_features=cfg["RFC_features"], max_depth=cfg["RFC_depth"], n_jobs=n_jobs)
    if model_name == "svc":
        from sklearn import svm
        return svm.SVC(probability=True, random_state=random_state)
    raise SystemExit(f"[2] unknown --model '{model_name}' (xgb | forest | svc)")


def _eval_headline(training, idx, proba, prob_filter):
    """Row-level ROC-AUC / PR-AUC plus the r1/r2/r3 group-ranking metrics for the
    training rows at positional `idx` scored with `proba`. Single-class subsets
    (tiny fixtures) yield NaN AUCs instead of raising."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    sub = training.iloc[idx]
    y_true = sub["true_match"].to_numpy()
    out = {}
    try:
        out["roc_auc"] = roc_auc_score(y_true, proba)
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = average_precision_score(y_true, proba)
    except ValueError:
        out["pr_auc"] = float("nan")
    df = sub[["PatInvID", "census_year", "true_match"]].copy()
    df["predicted_probability"] = proba
    metrics = group_ranking_eval(df, prob_filter)
    for tag in ("r1", "r2", "r3"):
        for k, v in metrics[tag].items():
            out[f"{tag}_{k}"] = v
    return out


# Complexity-curve grids for --diagnostics (one axis at a time around the CLI values;
# the point equal to the CLI hyperparameters reuses the production fit).
DIAG_DEPTHS = [10, 20, 30, 50, None]
DIAG_MAX_FEATURES = [4, 6, 8, 10, 12]


def run_diagnostics(args, model, training, X, y, groups, ml_features,
                    train_idx, test_idx, y_pred_proba, prob_filter, cfg):
    """Report-only overfitting diagnostics (--diagnostics).

    Runs AFTER the production model is fitted, evaluated, and saved; never mutates
    `model` -- every refit is a fresh build_model() instance sharing the exact split
    and fit exclusions. Appends diag_* sheets to args.report and prints each table.
    `args.n_jobs` parallelizes only these refits/predicts (forest fits are
    n_jobs-invariant); the production numbers above are untouched.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    t_start = t0 = time.perf_counter()
    model_name = args.model
    print("\n[2][diag] === overfitting diagnostics (report-only; the saved model is "
          "the plain single fit above) ===", flush=True)
    sheets = {}
    y_train = y.iloc[train_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    # --- 1. train-vs-test gap (production model, no refit) -----------------------
    proba_train = model.predict_proba(X.iloc[train_idx])[:, 1]
    head_train = _eval_headline(training, train_idx, proba_train, prob_filter)
    head_test = _eval_headline(training, test_idx, y_pred_proba, prob_filter)
    gap = pd.DataFrame({"train": head_train, "test": head_test})
    gap["gap_train_minus_test"] = gap["train"] - gap["test"]
    sheets["diag_train_test_gap"] = gap
    print("[2][diag] train-vs-test gap -- raw train metrics of a deep forest are "
          "expected inflated;\n          read jointly with the OOB and complexity "
          "sheets before calling it memorization:\n" + gap.to_string(), flush=True)
    t0 = _lap("diag: train-vs-test gap", t0)

    if model_name != "forest":
        print(f"[2][diag] refit-based diagnostics (oob / seed stability / leakage / "
              f"complexity) are forest-only; skipped for --model {model_name}.", flush=True)
    else:
        # --- 2. forest OOB (separate fit; the saved pickle stays oob-free) -------
        oob_model = build_model("forest", cfg, args.random_state, n_jobs=args.n_jobs)
        oob_model.set_params(oob_score=True)
        oob_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oob_proba = oob_model.oob_decision_function_[:, 1]
        oob_mask = ~np.isnan(oob_proba)
        try:
            oob_roc = roc_auc_score(y_train[oob_mask], oob_proba[oob_mask])
            oob_pr = average_precision_score(y_train[oob_mask], oob_proba[oob_mask])
        except ValueError:
            oob_roc = oob_pr = float("nan")
        oob = pd.DataFrame.from_dict({
            "oob_accuracy": oob_model.oob_score_,
            "oob_roc_auc": oob_roc, "oob_pr_auc": oob_pr,
            "test_roc_auc": head_test["roc_auc"], "test_pr_auc": head_test["pr_auc"],
            "oob_minus_test_roc_auc": oob_roc - head_test["roc_auc"],
            "oob_minus_test_pr_auc": oob_pr - head_test["pr_auc"],
            "n_train_rows_without_oob": int((~oob_mask).sum()),
        }, orient="index", columns=["Value"])
        sheets["diag_oob"] = oob
        print("[2][diag] forest OOB vs held-out test:\n" + oob.to_string(), flush=True)
        t0 = _lap("diag: oob", t0)

        # --- 3. split-seed stability ---------------------------------------------
        seed_rows = {}
        for seed in DIAG_SEEDS:
            gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
            tr_i, te_i = next(gss.split(X, y, groups))
            tr_i = _apply_fit_exclusions(tr_i, training, args, verbose=False)
            m = build_model("forest", cfg, args.random_state, n_jobs=args.n_jobs)
            m.fit(X.iloc[tr_i], y.iloc[tr_i])
            seed_rows[str(seed)] = _eval_headline(
                training, te_i, m.predict_proba(X.iloc[te_i])[:, 1], prob_filter)
        seeds_df = pd.DataFrame.from_dict(seed_rows, orient="index")
        seeds_df.index.name = "split_seed"
        stats = pd.DataFrame({"mean": seeds_df.mean(), "sd": seeds_df.std()}).T
        seeds_df = pd.concat([seeds_df, stats])
        sheets["diag_seed_stability"] = seeds_df
        print(f"[2][diag] split-seed stability ({len(DIAG_SEEDS)} GroupShuffleSplit "
              f"seeds, fit+eval each):\n" + seeds_df.to_string(), flush=True)
        t0 = _lap("diag: seed stability", t0)

        # --- 4. inventor-level leakage sensitivity -------------------------------
        # The production split groups by PatInvID+census_year, so the SAME inventor
        # in different waves can straddle train/test. Re-split by PatInvID alone:
        # a material drop means the published numbers are optimistic.
        groups_inv = training["PatInvID"].astype(str)
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                random_state=args.random_state)
        tr_i, te_i = next(gss.split(X, y, groups_inv))
        tr_i = _apply_fit_exclusions(tr_i, training, args, verbose=False)
        m = build_model("forest", cfg, args.random_state, n_jobs=args.n_jobs)
        m.fit(X.iloc[tr_i], y.iloc[tr_i])
        head_inv = _eval_headline(
            training, te_i, m.predict_proba(X.iloc[te_i])[:, 1], prob_filter)
        leak = pd.DataFrame({"group_id_split": head_test, "patinvid_split": head_inv})
        leak["delta"] = leak["patinvid_split"] - leak["group_id_split"]
        sheets["diag_leakage_patinvid"] = leak
        print("[2][diag] inventor-level leakage (split by PatInvID alone vs the "
              "production PatInvID+census_year split, same seed):\n"
              + leak.to_string(), flush=True)
        t0 = _lap("diag: leakage", t0)

        # --- 5. complexity curves (one axis at a time around the CLI values) -----
        def _complexity_row(cfg_mod, reuse_production):
            if reuse_production:
                head, tr_pr = head_test, head_train["pr_auc"]
            else:
                m = build_model("forest", cfg_mod, args.random_state, n_jobs=args.n_jobs)
                m.fit(X.iloc[train_idx], y.iloc[train_idx])
                head = _eval_headline(
                    training, test_idx, m.predict_proba(X.iloc[test_idx])[:, 1], prob_filter)
                try:
                    tr_pr = average_precision_score(
                        y_train, m.predict_proba(X.iloc[train_idx])[:, 1])
                except ValueError:
                    tr_pr = float("nan")
            return {"test_roc_auc": head["roc_auc"], "test_pr_auc": head["pr_auc"],
                    "r2_precision": head["r2_precision"], "r2_recall": head["r2_recall"],
                    "r2_f1": head["r2_f1"], "train_pr_auc": tr_pr,
                    "train_minus_test_pr_auc": tr_pr - head["pr_auc"]}

        depth_rows = {}
        for d in DIAG_DEPTHS:
            depth_rows[str(d)] = _complexity_row({**cfg, "RFC_depth": d},
                                                 reuse_production=(d == cfg["RFC_depth"]))
        depth_df = pd.DataFrame.from_dict(depth_rows, orient="index")
        depth_df.index.name = f"max_depth (max_features={cfg['RFC_features']})"
        sheets["diag_complexity_depth"] = depth_df
        print("[2][diag] complexity curve, max_depth axis:\n"
              + depth_df.to_string(), flush=True)

        feat_rows = {}
        for f in DIAG_MAX_FEATURES:
            feat_rows[str(f)] = _complexity_row({**cfg, "RFC_features": f},
                                                reuse_production=(f == cfg["RFC_features"]))
        feat_df = pd.DataFrame.from_dict(feat_rows, orient="index")
        feat_df.index.name = f"max_features (max_depth={cfg['RFC_depth']})"
        sheets["diag_complexity_max_features"] = feat_df
        print("[2][diag] complexity curve, max_features axis:\n"
              + feat_df.to_string(), flush=True)
        t0 = _lap("diag: complexity curves", t0)

    # --- 6. permutation feature importance on the held-out split -----------------
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(model, X.iloc[test_idx], y.iloc[test_idx],
                                  scoring="average_precision", n_repeats=5,
                                  random_state=0, n_jobs=args.n_jobs)
    perm_df = pd.DataFrame({"importance_mean": perm.importances_mean,
                            "importance_sd": perm.importances_std},
                           index=ml_features).sort_values("importance_mean",
                                                          ascending=False)
    perm_df["rank"] = range(1, len(perm_df) + 1)
    sheets["diag_permutation_importance"] = perm_df
    print("[2][diag] permutation importance (held-out, average_precision, top 15):\n"
          + perm_df.head(15).to_string(), flush=True)
    t0 = _lap("diag: permutation importance", t0)

    # --- 7. calibration (reliability table, 10 bins, held-out) -------------------
    edges = np.linspace(0.0, 1.0, 11)
    cal_src = pd.DataFrame({"proba": y_pred_proba, "y": y_test})
    cal_src["bin"] = pd.cut(cal_src["proba"], bins=edges, include_lowest=True)
    grp = cal_src.groupby("bin", observed=False)
    cal = pd.DataFrame({"n": grp.size(), "mean_predicted": grp["proba"].mean(),
                        "observed_positive_rate": grp["y"].mean()})
    cal["gap_predicted_minus_observed"] = (cal["mean_predicted"]
                                           - cal["observed_positive_rate"])
    cal.index = cal.index.astype(str)
    cal.index.name = "probability_bin"
    sheets["diag_calibration"] = cal
    print("[2][diag] calibration (held-out reliability table):\n"
          + cal.to_string(), flush=True)

    with pd.ExcelWriter(args.report, mode="a", engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name)
    print(f"[2][diag] appended {len(sheets)} diag_* sheets -> {args.report}", flush=True)
    _lap("diag: total", t_start)


def _int_or_none(s):
    """argparse type for --rfc-max-depth: 'none'/'null' -> None (unbounded forest)."""
    return None if s.lower() in ("none", "null") else int(s)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train and evaluate the candidate-scoring classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--training", required=True, help="Training parquet from stage 1.")
    p.add_argument("--occ-dict", default=DEF_OCC_DICT,
                   help="Occupation dictionary JSON (data/misc/occupation_dict.json).")
    p.add_argument("--occdict-name", default="broad_new_v2",
                   choices=["broad_new_v2", "strict_new_v2"],
                   help="Which dictionary variant to use for the occupation dummies.")
    p.add_argument("--model", default="forest", choices=["xgb", "forest", "svc"],
                   help="Classifier. Default 'forest' (random forest -- higher precision at "
                        "the operating thresholds on the deceased-hard-negative training set; "
                        "'xgb' selectable). 'forest' and 'xgb' both handle the NaN feature "
                        "values present in county_distance / relatedness (sklearn >=1.4 trees "
                        "are NaN-native); 'svc' requires NaN-free features.")
    p.add_argument("--no-group-size", action="store_true",
                   help="Exclude the n_candidates group-size feature (24-feature baseline).")
    p.add_argument("--exclude-orphans", action="store_true",
                   help="Drop orphan groups (golden, true match lost downstream) from the "
                        "training FIT only; they remain in the held-out evaluation split. "
                        "Needs patinv_type (stage-1 output). Default: include them.")
    p.add_argument("--exclude-hard-negatives", action="store_true",
                   help="Drop deceased hard-negative groups from the training FIT only; "
                        "they remain in the evaluation split. Needs patinv_type. "
                        "Default: include them.")
    # Hyperparameters. Defaults are the production values.
    p.add_argument("--rfc-estimators", type=int, default=300,
                   help="Number of trees (forest n_estimators; also xgb n_estimators).")
    p.add_argument("--rfc-criterion", default="entropy",
                   choices=["gini", "entropy", "log_loss"],
                   help="Forest split criterion (xgb ignores it).")
    p.add_argument("--rfc-max-features", type=int, default=8,
                   help="Forest max_features (xgb ignores it).")
    p.add_argument("--rfc-max-depth", type=_int_or_none, default=20,
                   help="Forest/xgb max_depth; 'none' = unbounded forest.")
    p.add_argument("--xgb-min-child-weight", type=int, default=1,
                   help="xgb min_child_weight (forest ignores it).")
    p.add_argument("--xgb-gamma", type=float, default=0.05,
                   help="xgb gamma (forest ignores it).")
    p.add_argument("--prob-filter", type=float, default=0.4,
                   help="Min group probability for the r2/r3 reporting summary. A reporting "
                        "knob only -- the threshold-sweep report carries the full curve.")
    p.add_argument("--test-size", type=float, default=0.2, help="GroupShuffleSplit test size.")
    p.add_argument("--random-state", type=int, default=39, help="GroupShuffleSplit seed.")
    p.add_argument("--model-out", default=None,
                   help="If set, pickle the trained model to this path.")
    p.add_argument("--report", default=None,
                   help="If set, write the xlsx performance report to this path.")
    p.add_argument("--threshold-report", default=None,
                   help="If set, write the per-threshold precision/recall sweep xlsx "
                        "to this path (one sheet per model in --threshold-models).")
    p.add_argument("--threshold-models", default="xgb,forest",
                   help="Comma-separated models to include as sheets in the threshold "
                        "report (each trained on the same split; sheet named per model).")
    p.add_argument("--diagnostics", action="store_true",
                   help="Append the report-only overfitting diagnostics (extra diag_* sheets "
                        "in --report + stdout tables). Runs after the production model is "
                        "fitted and saved; never changes it. Requires --report. Forest only "
                        "for the refit-based diagnostics.")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallelism for the --diagnostics forest refits/predicts only. The "
                        "production and threshold-sweep fits always run n_jobs=1 "
                        "(bit-reproducibility).")
    return p.parse_args(argv)


def _apply_fit_exclusions(train_idx, training, args, verbose=True):
    """Drop --exclude-orphans / --exclude-hard-negatives groups from the FIT indices only.

    Composition control applies to the TRAINING FIT ONLY. The split runs on the full tagged
    set first, so the held-out test set (and thus the evaluation metrics) is identical
    regardless of these flags -- excluded types still stress-test precision as answer-absent
    groups. patinv_type is per-group and GroupShuffleSplit keeps groups whole, so this drops
    whole groups from train; test groups stay disjoint (no leakage). Shared by the production
    fit and every --diagnostics refit so training composition is identical everywhere.
    """
    if (args.exclude_orphans or args.exclude_hard_negatives) and "patinv_type" in training.columns:
        ttype = training["patinv_type"].to_numpy()
        drop = np.zeros(len(ttype), dtype=bool)
        if args.exclude_orphans:
            drop |= (ttype == "orphan")
        if args.exclude_hard_negatives:
            drop |= (ttype == "hard_negative")
        n_before = len(train_idx)
        train_idx = train_idx[~drop[train_idx]]
        if verbose:
            excl = [t for t, on in (("orphan", args.exclude_orphans),
                                    ("hard_negative", args.exclude_hard_negatives)) if on]
            print(f"[2] train-fit exclusion {excl}: {n_before:,} -> {len(train_idx):,} "
                  f"train rows (test set unchanged)", flush=True)
    return train_idx


def main(argv=None):
    args = parse_args(argv)
    # cfg maps the CLI hyperparameters to the RFC_*/xgb_* key names build_model /
    # model_hyperparams / threshold_sweep / SWEEP_COLUMNS expect.
    cfg = {"RFC_estimators": args.rfc_estimators, "RFC_criterion": args.rfc_criterion,
           "RFC_features": args.rfc_max_features, "RFC_depth": args.rfc_max_depth,
           "xgb_min_child_weight": args.xgb_min_child_weight, "xgb_gamma": args.xgb_gamma}
    model_name = args.model
    prob_filter = args.prob_filter
    if args.diagnostics and not args.report:
        raise SystemExit("[2] --diagnostics requires --report (the diag_* sheets are "
                         "appended to the xlsx report).")
    t0 = time.perf_counter()

    with open(args.occ_dict) as f:
        occ_dict = json.load(f)
    if args.occdict_name not in occ_dict:
        raise SystemExit(f"[2] '{args.occdict_name}' not in {args.occ_dict} "
                         f"(have: {list(occ_dict)})")
    dictionary = occ_dict[args.occdict_name]

    training = pd.read_parquet(args.training)
    print(f"[2] loaded {len(training):,} training rows from {args.training}", flush=True)
    # Stage-1 tags each group `patinv_type` in {matched, orphan, hard_negative}; neither it
    # nor is_hard_negative is a feature (not in BASE_FEATURES), so no label leakage. Logged
    # for transparency.
    if "patinv_type" in training.columns:
        comp = training["patinv_type"].value_counts().to_dict()
        print(f"[2] loaded composition by patinv_type (rows): {comp}", flush=True)
    elif args.exclude_orphans or args.exclude_hard_negatives:
        raise SystemExit("[2] --exclude-orphans/--exclude-hard-negatives need the "
                         "patinv_type column; rebuild the training table with stage 1.")
    training["group_id"] = (training["PatInvID"].astype(str) + "_"
                            + training["census_year"].astype(str))

    training = build_features(training, dictionary)
    t0 = _lap("feature build", t0)

    # Active feature set: base 24 + optional n_candidates group-size feature.
    ml_features = list(BASE_FEATURES)
    if not args.no_group_size and "n_candidates" in training.columns:
        ml_features.append("n_candidates")
    print(f"[2] features ({len(ml_features)}): group_size="
          f"{'n_candidates' in ml_features}", flush=True)

    # Feature-presence + null diagnostic before fitting.
    missing = [c for c in ml_features if c not in training.columns]
    if missing:
        raise SystemExit(f"[2] features failed to build: {missing}")
    nulls = {c: int(training[c].isna().sum()) for c in ml_features if training[c].isna().any()}
    if nulls:
        print(f"[2] nulls in features (kept as-is for reproducibility): {nulls}", flush=True)
        if model_name == "svc":
            raise SystemExit(
                f"[2] --model svc cannot fit on NaN features {list(nulls)}. "
                f"Use --model xgb or forest (both NaN-native here), or impute/drop upstream.")

    from sklearn.model_selection import GroupShuffleSplit

    X = training[ml_features]
    y = training["true_match"]
    groups = training["group_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    train_idx = _apply_fit_exclusions(train_idx, training, args)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    if "patinv_type" in training.columns:
        tr_comp = training["patinv_type"].to_numpy()[train_idx]
        te_comp = training["patinv_type"].to_numpy()[test_idx]
        import collections
        print(f"[2] train composition: {dict(collections.Counter(tr_comp))}", flush=True)
        print(f"[2] test  composition: {dict(collections.Counter(te_comp))}", flush=True)
    print(f"[2] split: {len(X_train):,} train / {len(X_test):,} test rows "
          f"({model_name})", flush=True)

    model = build_model(model_name, cfg, args.random_state)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    t0 = _lap(f"fit + predict ({model_name})", t0)

    feature_scores = None
    if model_name != "svc":
        feature_scores = pd.Series(model.feature_importances_,
                                   index=X_train.columns).sort_values(ascending=False)
        print("[2] top feature importances:\n"
              + feature_scores.head(10).to_string(), flush=True)

    # Name columns are display-only here (absent from the stage-4 publication table);
    # keep whichever exist so evaluation works on both the full and published tables.
    id_cols = [c for c in ["PatInvID", "histid", "inventor_name", "census_name",
                           "census_year"] if c in training.columns]
    df_test = pd.concat([X_test, y_test], axis=1)
    df_test = df_test.merge(training[id_cols],
                            how="left", left_index=True, right_index=True)
    df_test["predicted_probability"] = y_pred_proba

    # --- Plot Precision/Recall/F1 vs probability threshold (held-out test set only) ---
    # Report-only: the model is already fitted and frozen above, so any failure here
    # (missing _viz_common, missing plotting dep, plotting error) MUST degrade to a
    # warning and never abort the run before the model is pickled.
    if args.report:
      try:
        # _viz_common lives in the sibling src/technical_validation/ package.
        _tv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "technical_validation")
        if _tv_dir not in sys.path:
            sys.path.insert(0, _tv_dir)
        from _viz_common import configure_matplotlib
        from matplotlib.ticker import FuncFormatter
        from scipy.interpolate import make_interp_spline

        # Palette mirrors the sibling technical-validation figures.
        INK, INK_2, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
        PRECISION_COLOR, RECALL_COLOR, F1_COLOR = "#0072b2", "#e69f00", "#009e73"

        curve_df = threshold_sweep(df_test, model_name, cfg, args.occdict_name,
                                   args.test_size, args.random_state,
                                   model_random_state=getattr(model, "random_state", None),
                                   step=0.01)

        plt, using_tex = configure_matplotlib(ink=INK, baseline=BASELINE, label=INK_2,
                                              xtick=INK_2, ytick=MUTED)
        pct = r"\%" if using_tex else "%"

        def _smooth(x, y, n=400):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            order = np.argsort(x)
            x, y = x[order], y[order]
            x_uniq, idx = np.unique(x, return_index=True)
            y_uniq = y[idx]
            spline = make_interp_spline(x_uniq, y_uniq, k=3)
            x_smooth = np.linspace(x_uniq.min(), x_uniq.max(), n)
            y_smooth = np.clip(spline(x_smooth), 0.0, 1.0)
            return x_smooth, y_smooth

        taus = curve_df.index.to_numpy()
        x_p, y_p = _smooth(taus, curve_df["precision"])
        x_r, y_r = _smooth(taus, curve_df["recall"])
        x_f, y_f = _smooth(taus, curve_df["F1"])

        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        (line_prec,) = ax.plot(x_p, y_p * 100, color=PRECISION_COLOR, linewidth=2.0, label="Precision")
        (line_rec,) = ax.plot(x_r, y_r * 100, color=RECALL_COLOR, linewidth=2.0, label="Recall")
        (line_f1,) = ax.plot(x_f, y_f * 100, color=F1_COLOR, linewidth=2.2, label="F1 score")

        ax.set_ylim(0, 100)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Match-probability threshold")
        ax.set_ylabel("Precision / Recall / F1")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}{pct}"))
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=0)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(handles=[line_prec, line_rec, line_f1], loc="best",
                  frameon=False, fontsize=8, handlelength=1.6)

        fig.tight_layout()
        curve_pdf_path = os.path.join(os.path.dirname(args.report) or ".", "precision_recall_f1_by_threshold.pdf")
        os.makedirs(os.path.dirname(curve_pdf_path) or ".", exist_ok=True)
        fig.savefig(curve_pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[2] wrote threshold curve plot -> {curve_pdf_path}", flush=True)
      except Exception as _plot_err:
        print(f"[2] skipping threshold curve plot: {type(_plot_err).__name__}: {_plot_err}",
              flush=True)
    else:
        print("[2] skipping threshold curve plot: --report not set (needed for output path)",
              flush=True)

    metrics = group_ranking_eval(df_test, prob_filter)
    print(f"\n[2] group-ranking results ({model_name}, prob_filter={prob_filter}):", flush=True)
    for tag, label in [("r1", "unique highest prob"),
                       ("r2", f"filtered (>{prob_filter})"),
                       ("r3", "filtered, true rows only")]:
        m = metrics[tag]
        print(f"  {tag} {label:28s} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f}", flush=True)
    print("\n[2] confusion matrix (r2):\n", metrics["confusion_matrix"], flush=True)

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        params = {"model": model_name, **cfg,
                  "prob_filter": prob_filter, "occdict_name": args.occdict_name,
                  "group_size": "n_candidates" in ml_features, "n_features": len(ml_features),
                  "test_size": args.test_size, "random_state": args.random_state,
                  "n_jobs": args.n_jobs}
        flat = {}
        for tag in ("r1", "r2", "r3"):
            for k, v in metrics[tag].items():
                flat[f"{tag}_{k}"] = v
        with pd.ExcelWriter(args.report) as writer:
            pd.DataFrame.from_dict(params, orient="index", columns=["Value"]).to_excel(
                writer, sheet_name="parameters")
            pd.DataFrame.from_dict(flat, orient="index", columns=["Value"]).to_excel(
                writer, sheet_name="performance")
            if feature_scores is not None:
                feature_scores.to_excel(writer, sheet_name="feature_scores")
            metrics["classification_report"].to_excel(writer, sheet_name="classification_report")
            pd.DataFrame(metrics["confusion_matrix"]).to_excel(writer, sheet_name="confusion_matrix")
        print(f"[2] wrote report -> {args.report}", flush=True)

    if args.threshold_report:
        os.makedirs(os.path.dirname(args.threshold_report) or ".", exist_ok=True)
        group_cols = training[[c for c in ["PatInvID", "histid", "inventor_name",
                                           "census_name", "census_year"]
                               if c in training.columns]]
        sweep_models = [m.strip() for m in args.threshold_models.split(",") if m.strip()]
        sweeps = {}
        for m in sweep_models:
            # Reuse the already-fitted main model when it matches; else train on the
            # same split so every sheet is comparable.
            if m == model_name:
                mdl = model
                proba = y_pred_proba
            else:
                mdl = build_model(m, cfg, args.random_state)
                mdl.fit(X_train, y_train)
                proba = mdl.predict_proba(X_test)[:, 1]
            df_m = pd.concat([X_test, y_test], axis=1).merge(
                group_cols, how="left", left_index=True, right_index=True)
            df_m["predicted_probability"] = proba
            sweeps[m] = threshold_sweep(df_m, m, cfg, args.occdict_name,
                                        args.test_size, args.random_state,
                                        model_random_state=getattr(mdl, "random_state", None))
        with pd.ExcelWriter(args.threshold_report) as writer:
            for m, sweep in sweeps.items():
                sweep.to_excel(writer, sheet_name=m)
        print(f"[2] wrote threshold sweep ({', '.join(sweeps)}) -> "
              f"{args.threshold_report}", flush=True)

    if args.model_out:
        os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
        with open(args.model_out, "wb") as f:
            pickle.dump(model, f)
        print(f"[2] saved model -> {args.model_out}", flush=True)

    # Overfitting diagnostics run LAST, after the production model is fitted,
    # reported, and pickled -- they are extra evaluation, never model selection.
    if args.diagnostics:
        run_diagnostics(args, model, training, X, y, groups, ml_features,
                        train_idx, test_idx, y_pred_proba, prob_filter, cfg)

    _lap("done", t0)


if __name__ == "__main__":
    main()
