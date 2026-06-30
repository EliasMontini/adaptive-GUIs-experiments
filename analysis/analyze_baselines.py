#!/usr/bin/env python3
"""
analyze_baselines.py
====================================================================
Offline evaluation of threshold-based adaptive engines for the LEGO
forklift assembly-training AUI, computed entirely from the recorded
interaction logs (``all_20_experiments.csv``).

It implements TWO engines under an identical Leave-One-Out (LOO)
cross-validation protocol, matching the methodology of the manuscript:

  1. DETERMINISTIC  -- the published threshold baseline.
       For step s and format f, show f iff the proportion of the N-1
       calibration users who accessed f at step s is >= theta (0.5).
       Uses POPULATION statistics only. Serves as the validation anchor
       (should reproduce the paper's 74.9% / SD 11.5% / per-format and
       per-step-type tables).

  2. DET+HIST  -- a NEW within-session-aware threshold baseline.
       Blends the population proportion with the held-out user's OWN
       within-session access history (over prior steps of the SAME type)
       via evidence-weighted (Dirichlet/empirical-Bayes) shrinkage:

           p_blend(s,f) = ( n * p_self(s,f) + k * p_pop(s,f) ) / (n + k)

       where  n        = number of prior same-type steps completed by the
                         held-out user (the count behind p_self),
              p_self   = that user's mean access rate for f over those
                         prior same-type steps (access-based),
              p_pop    = population proportion for f at THIS step s
                         (identical to the deterministic baseline),
              k        = pseudo-count = strength of the population prior.

       EXPLORATION WARM-UP (--warmup, default 3): for the first `warmup`
       occurrences of a step type (n < warmup) the user is assumed to still
       be exploring the available formats, so the prediction uses p_pop ONLY;
       the user's own preference becomes relevant only from occurrence
       warmup+1 onward (n >= warmup), and then grows via the shrinkage above.
       This mirrors the rule given to Gemini in the LLM prompt.

       Properties (by design):
         * n < warmup  =>  p_blend == p_pop  => Det+Hist reduces EXACTLY to
           the deterministic baseline during the exploration window (and
           always on QC, which has a single step).
         * Mirrors the LLM's information use (population proportions PLUS
           accumulating within-session per-step-type history, gated by the
           same exploration warm-up), which is what makes it a fair foil for
           isolating the *information* effect (Det -> Det+Hist) from the
           *reasoning* effect (Det+Hist -> LLM).

CONSTRAINTS (manuscript Table 1), enforced during inference:
  * Format availability by step type (structurally-absent formats are
    excluded from prediction AND from the accuracy denominator).
  * Short text and Long text are mutually exclusive: when both are
    available and both clear the threshold, only the higher-scoring one
    is shown; on an exact tie, SHORT is preferred (the same rule used in
    the Gemini prompt). Fully deterministic.

ACCURACY (manuscript Eq. 2):
  acc(s,u) = mean over available formats F_s of 1[ y_hat(s,f) == y(s,u,f) ]
  per-user accuracy = mean of acc(s,u) over the 16 steps
  overall accuracy  = mean of acc(s,u) over all users and steps

OUTPUTS:
  * Console summary (overall table, per-format, per-step-type, stats,
    tie diagnostics, validation vs the paper).
  * CSVs under --outdir (default ./baseline_results/).

USAGE:
  python3 analyze_baselines.py
  python3 analyze_baselines.py --warmup 3                      # exploration window
  python3 analyze_baselines.py --k 1 2 4 8 --k-primary 4
  python3 analyze_baselines.py --llm-csv per_user_llm.csv      # adds Det+Hist vs LLM tests
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Fixed task definition (manuscript Table 1)
# --------------------------------------------------------------------------
SEP = ";"

# CSV column -> manuscript format name
FORMAT_COLS = {
    "short_text_viewed":    "Short text",
    "long_text_viewed":     "Long text",
    "single_pieces_viewed": "Component image",
    "assembly_viewed":      "Assembly image",
    "video_viewed":         "Video",
}
FORMATS = list(FORMAT_COLS.keys())                  # canonical column order
SHORT, LONG = "short_text_viewed", "long_text_viewed"

# Format availability by step type (the set F_s)
AVAILABILITY = {
    "Picking":  {"short_text_viewed", "single_pieces_viewed"},
    "Assembly": set(FORMATS),
    "QC":       {"short_text_viewed", "long_text_viewed",
                 "single_pieces_viewed", "video_viewed"},
}

THETA = 0.5

# Paper reference values for the validation anchor (deterministic baseline)
PAPER_DET = {
    "overall_mean": 74.9, "overall_sd": 11.5, "min": 57.5, "max": 96.3,
    "per_format": {  # Acc, FP, FN (%)
        "Short text":     (69.7, 17.5, 12.8),
        "Long text":      (82.2,  0.0, 17.8),
        "Component image":(75.0, 18.1,  6.9),
        "Assembly image": (93.1,  6.9,  0.0),
        "Video":          (77.2,  7.2, 15.6),
    },
    "per_step_type": {"Picking": 68.2, "Assembly": 82.2, "QC": 62.5},
}
PAPER_VOLATILITY = {"mean": 16.2, "sd": 7.4, "min": 9, "max": 32}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def step_type(name: str) -> str:
    if name.startswith("Withdraw"):
        return "Picking"
    if name.startswith("Assembly"):
        return "Assembly"
    if name.startswith("Final"):
        return "QC"
    raise ValueError(f"Unrecognised step_name: {name!r}")


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=SEP)
    missing = [c for c in ["experiment_id", "step_id", "step_name", *FORMATS]
               if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: missing expected columns: {missing}")
    df["stype"] = df["step_name"].map(step_type)
    # sanity: 0/1 only, 16 steps per participant
    for f in FORMATS:
        bad = set(df[f].unique()) - {0, 1}
        if bad:
            sys.exit(f"ERROR: column {f} has non-binary values {bad}")
    counts = df.groupby("experiment_id").size()
    if not (counts == 16).all():
        sys.exit(f"ERROR: not all participants have 16 steps:\n{counts}")
    # constraint sanity: structurally-absent formats must be all zero
    for st, sub in df.groupby("stype"):
        for f in FORMATS:
            if f not in AVAILABILITY[st] and (sub[f] == 1).any():
                sys.exit(f"ERROR: absent format {f} viewed on {st} step(s)")
    return df


# --------------------------------------------------------------------------
# Prediction primitives
# --------------------------------------------------------------------------
def population_proportions(train: pd.DataFrame) -> dict:
    """p_pop[step_id][col] over the calibration users (per specific step)."""
    p = {}
    for sid, sub in train.groupby("step_id"):
        p[sid] = {f: sub[f].mean() for f in FORMATS}
    return p


def predict_step(scores: dict, available: set, diag: dict):
    """
    Turn per-format real-valued scores into a binary visibility dict over the
    AVAILABLE formats, applying the threshold and short/long mutual exclusivity.

    short/long tie-break: PREFER SHORT (identical to the rule given to Gemini in
    the LLM prompt). When both clear the threshold and scores are exactly equal,
    keep short text and hide long text. Fully deterministic.

    diag accumulates tie diagnostics.
    """
    pred = {f: int(scores[f] >= THETA) for f in available}

    # short/long mutual exclusivity (only when both are available)
    if SHORT in available and LONG in available and pred[SHORT] == 1 and pred[LONG] == 1:
        diag["mutex_conflicts"] += 1
        s_sc, l_sc = scores[SHORT], scores[LONG]
        if l_sc > s_sc:
            pred[SHORT] = 0
        else:                                   # s_sc > l_sc, or exact tie -> prefer short
            if abs(s_sc - l_sc) < 1e-12:
                diag["mutex_exact_ties"] += 1
            pred[LONG] = 0
    return pred


# --------------------------------------------------------------------------
# Engines (per held-out user, returns dict: step_id -> {col: pred})
# --------------------------------------------------------------------------
def run_deterministic(test_user_df, p_pop, diag):
    preds = {}
    for _, row in test_user_df.sort_values("step_id").iterrows():
        sid, st = row["step_id"], row["stype"]
        avail = AVAILABILITY[st]
        scores = {f: p_pop[sid][f] for f in avail}
        preds[sid] = predict_step(scores, avail, diag)
    return preds


def run_det_hist(test_user_df, p_pop, k, warmup, diag):
    """
    Sequential pass through the 16 steps, accumulating the held-out user's own
    same-type access history. p_self is the user's mean access over PRIOR
    same-type steps (ground-truth accesses, i.e. what is observable at runtime).

    EXPLORATION WARM-UP: for the first `warmup` occurrences of a step type
    (i.e. while n < warmup), the user is assumed to still be exploring the
    available formats, so the prediction relies on the population proportion
    ONLY. From occurrence `warmup`+1 onward (n >= warmup) the user's own
    preference becomes relevant and is blended in via shrinkage, growing with
    accumulated evidence. This mirrors the rule given to Gemini in the LLM
    prompt (first three same-type steps treated as exploratory).
    """
    preds = {}
    # accumulator: per step type, list of access vectors seen SO FAR
    hist = {st: [] for st in AVAILABILITY}     # each entry: dict col->0/1
    for _, row in test_user_df.sort_values("step_id").iterrows():
        sid, st = row["step_id"], row["stype"]
        avail = AVAILABILITY[st]
        n = len(hist[st])
        scores = {}
        for f in avail:
            p_pop_f = p_pop[sid][f]
            if n < warmup:
                blended = p_pop_f                    # exploration: population only
            else:
                p_self_f = np.mean([h[f] for h in hist[st]])
                blended = (n * p_self_f + k * p_pop_f) / (n + k)
                if abs(blended - THETA) < 1e-12:
                    diag["score_exact_half"] += 1
            scores[f] = blended
        preds[sid] = predict_step(scores, avail, diag)
        # update history AFTER predicting (the step's actual accesses)
        hist[st].append({f: int(row[f]) for f in FORMATS})
    return preds


# --------------------------------------------------------------------------
# Accuracy / error decomposition
# --------------------------------------------------------------------------
def step_accuracy(pred_step, truth_row, available):
    return np.mean([int(pred_step[f] == int(truth_row[f])) for f in available])


def evaluate(df, engine_fn, **engine_kwargs):
    """
    LOO over participants. Returns:
      per_user      : DataFrame [experiment_id, accuracy]
      cell_records  : list of dicts (one per (user, step, format)) for per-format
                      and per-step-type breakdowns
      diag          : tie diagnostics
    """
    diag = {"mutex_conflicts": 0, "mutex_exact_ties": 0, "score_exact_half": 0}
    users = sorted(df["experiment_id"].unique())
    per_user_rows, cell_records = [], []

    for u in users:
        train = df[df["experiment_id"] != u]
        test = df[df["experiment_id"] == u]
        p_pop = population_proportions(train)
        preds = engine_fn(test, p_pop, diag=diag, **engine_kwargs)

        step_accs = []
        for _, row in test.iterrows():
            sid, st = row["step_id"], row["stype"]
            avail = AVAILABILITY[st]
            step_accs.append(step_accuracy(preds[sid], row, avail))
            for f in avail:
                yhat, y = preds[sid][f], int(row[f])
                cell_records.append({
                    "experiment_id": u, "step_id": sid, "stype": st,
                    "format": f, "pred": yhat, "truth": y,
                    "correct": int(yhat == y),
                    "fp": int(yhat == 1 and y == 0),
                    "fn": int(yhat == 0 and y == 1),
                })
        per_user_rows.append({"experiment_id": u, "accuracy": np.mean(step_accs)})

    return (pd.DataFrame(per_user_rows),
            pd.DataFrame(cell_records),
            diag)


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------
def overall_stats(per_user):
    a = per_user["accuracy"].values * 100
    return dict(mean=a.mean(), sd=a.std(ddof=1), min=a.min(), max=a.max(),
                vals=a)


def per_format_table(cells):
    rows = []
    for col, name in FORMAT_COLS.items():
        sub = cells[cells["format"] == col]
        if len(sub) == 0:
            continue
        rows.append({
            "Format": name, "n": len(sub),
            "Acc": 100 * sub["correct"].mean(),
            "FP":  100 * sub["fp"].mean(),
            "FN":  100 * sub["fn"].mean(),
        })
    return pd.DataFrame(rows)


def per_step_type_table(cells):
    rows = []
    for st in ["Picking", "Assembly", "QC"]:
        sub = cells[cells["stype"] == st]
        # accuracy is per (user, step) averaged over available formats, then
        # averaged over cells of this step type -> equivalently mean of 'correct'
        # weighted per cell. To match the paper's per-step-type accuracy we
        # average acc(s,u) over (user, step). Recover acc(s,u) by grouping.
        g = sub.groupby(["experiment_id", "step_id"])["correct"].mean()
        rows.append({"Step type": st, "n_cells": len(sub),
                     "n_us": g.size, "Acc": 100 * g.mean()})
    return pd.DataFrame(rows)


def volatility_index(df):
    """Per-user behavioural volatility: the mean normalised Hamming distance
    between the format-access vectors of consecutive steps of the same type.
    For each consecutive same-type pair, the distance is the number of formats
    whose access status differs, divided by the number of formats available at
    that step type; the index averages this over all consecutive same-type
    pairs. Graded: a single flipped format contributes less than a wholesale
    change of the accessed set. (QC has a single step, so it contributes no
    transitions.)"""
    out = {}
    for u, sub in df.groupby("experiment_id"):
        num = den = 0.0
        for st, ss in sub.groupby("stype"):
            ss = ss.sort_values("step_id")
            cols = sorted(AVAILABILITY[st])
            k = len(cols)
            vecs = ss[cols].values
            for i in range(1, len(vecs)):
                num += (vecs[i] != vecs[i - 1]).sum() / k
                den += 1
        out[u] = num / den if den else np.nan
    return pd.Series(out, name="volatility")


def f_test_var(a, b):
    """Two-sided F-test for equality of variances. Reports F = larger/smaller."""
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    if va >= vb:
        F, dfn, dfd, hi = va / vb, len(a) - 1, len(b) - 1, "a"
    else:
        F, dfn, dfd, hi = vb / va, len(b) - 1, len(a) - 1, "b"
    p = 2 * min(stats.f.cdf(F, dfn, dfd), 1 - stats.f.cdf(F, dfn, dfd))
    return F, dfn, dfd, p, hi


def paired_tests(a, b):
    """a, b are paired per-user accuracy vectors (fractions or %)."""
    w_stat, w_p = stats.wilcoxon(a, b)
    t_stat, t_p = stats.ttest_rel(a, b)
    return (w_stat, w_p), (t_stat, t_p)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="all_20_experiments.csv")
    ap.add_argument("--outdir", default="baseline_results")
    ap.add_argument("--k", type=float, nargs="+", default=[1, 2, 4, 8],
                    help="pseudo-count sweep for Det+Hist")
    ap.add_argument("--k-primary", type=float, default=4,
                    help="k used for the detailed (per-format / per-step-type) tables")
    ap.add_argument("--warmup", type=int, default=3,
                    help="exploration warm-up: own-history is ignored for the first "
                         "WARMUP occurrences of each step type (population only); "
                         "user preference becomes relevant from occurrence WARMUP+1")
    ap.add_argument("--llm-csv", default=None,
                    help="optional CSV with columns experiment_id,accuracy "
                         "(LLM per-user accuracy, fraction or %) to enable "
                         "Det+Hist vs LLM tests")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load(args.csv)
    print(f"Loaded {len(df)} rows / {df['experiment_id'].nunique()} participants "
          f"from {args.csv}\n")

    # ---- volatility ----
    vol = volatility_index(df)
    print("=== Behavioural volatility index ===")
    print(f"  mean {100*vol.mean():.1f}%  SD {100*vol.std(ddof=1):.1f}%  "
          f"range {100*vol.min():.1f}-{100*vol.max():.1f}%   "
          f"(paper: mean {PAPER_VOLATILITY['mean']}%, SD {PAPER_VOLATILITY['sd']}%, "
          f"range {PAPER_VOLATILITY['min']}-{PAPER_VOLATILITY['max']}%)\n")

    # ---- deterministic baseline (validation anchor) ----
    det_user, det_cells, det_diag = evaluate(df, run_deterministic)
    det = overall_stats(det_user)
    print("=== DETERMINISTIC baseline (validation anchor) ===")
    print(f"  overall {det['mean']:.1f}%  SD {det['sd']:.1f}%  "
          f"min {det['min']:.1f}%  max {det['max']:.1f}%")
    print(f"  paper :  overall {PAPER_DET['overall_mean']}%  "
          f"SD {PAPER_DET['overall_sd']}%  min {PAPER_DET['min']}%  "
          f"max {PAPER_DET['max']}%")
    dmm = det['mean'] - PAPER_DET['overall_mean']
    print(f"  -> reproduction delta on overall mean: {dmm:+.1f} pp "
          f"({'OK' if abs(dmm) < 1.0 else 'CHECK conventions'})\n")

    pf_det = per_format_table(det_cells)
    print("  Per-format (Acc / FP / FN %), paper values in [...]:")
    for _, r in pf_det.iterrows():
        pa, pfp, pfn = PAPER_DET["per_format"][r["Format"]]
        print(f"    {r['Format']:16s} n={int(r['n']):4d}  "
              f"Acc {r['Acc']:5.1f} [{pa:4.1f}]  "
              f"FP {r['FP']:5.1f} [{pfp:4.1f}]  FN {r['FN']:5.1f} [{pfn:4.1f}]")
    pst_det = per_step_type_table(det_cells)
    print("  Per-step-type accuracy, paper values in [...]:")
    for _, r in pst_det.iterrows():
        print(f"    {r['Step type']:9s} n={int(r['n_cells']):4d}  "
              f"Acc {r['Acc']:5.1f} [{PAPER_DET['per_step_type'][r['Step type']]:.1f}]")
    print(f"  tie diagnostics: {det_diag}\n")

    # ---- Det+Hist sweep ----
    print("=== DET+HIST (within-session shrinkage) — k sweep ===")
    print(f"  tie-break: prefer-short (matches LLM prompt);  "
          f"exploration warm-up: {args.warmup} occurrences\n")
    sweep_rows = []
    dethist_user_by_k = {}
    for k in args.k:
        dh_user, dh_cells, dh_diag = evaluate(
            df, run_det_hist, k=k, warmup=args.warmup)
        dethist_user_by_k[k] = (dh_user, dh_cells, dh_diag)
        dh = overall_stats(dh_user)
        # variance comparison vs deterministic
        F, dfn, dfd, p_F, hi = f_test_var(det["vals"], dh["vals"])
        (w, w_p), (t, t_p) = paired_tests(det_user["accuracy"].values,
                                          dh_user["accuracy"].values)
        sweep_rows.append({
            "k": k, "mean": dh["mean"], "sd": dh["sd"],
            "min": dh["min"], "max": dh["max"],
            "F_vs_det": F, "F_p": p_F,
            "wilcoxon_W": w, "wilcoxon_p": w_p,
            "ttest_t": t, "ttest_p": t_p,
            "mutex_ties": dh_diag["mutex_exact_ties"],
            "score_half": dh_diag["score_exact_half"],
        })
        print(f"  k={k:>4}:  mean {dh['mean']:5.1f}%  SD {dh['sd']:5.1f}%  "
              f"[{dh['min']:.1f}-{dh['max']:.1f}]   "
              f"F(det/dh)={F:.2f} p={p_F:.3f}   "
              f"Wilcoxon(det vs dh) p={w_p:.3f}   "
              f"exact-ties={dh_diag['mutex_exact_ties']}, "
              f"score=0.5 events={dh_diag['score_exact_half']}")
    sweep = pd.DataFrame(sweep_rows)
    print()

    # ---- detailed tables for primary k ----
    kp = args.k_primary
    if kp not in dethist_user_by_k:
        dh_user, dh_cells, dh_diag = evaluate(
            df, run_det_hist, k=kp, warmup=args.warmup)
        dethist_user_by_k[kp] = (dh_user, dh_cells, dh_diag)
    dh_user, dh_cells, dh_diag = dethist_user_by_k[kp]
    print(f"=== DET+HIST detailed tables (primary k={kp}) ===")
    print("  Per-format (Acc / FP / FN %):")
    pf_dh = per_format_table(dh_cells)
    for _, r in pf_dh.iterrows():
        print(f"    {r['Format']:16s} n={int(r['n']):4d}  "
              f"Acc {r['Acc']:5.1f}  FP {r['FP']:5.1f}  FN {r['FN']:5.1f}")
    print("  Per-step-type accuracy:")
    pst_dh = per_step_type_table(dh_cells)
    for _, r in pst_dh.iterrows():
        print(f"    {r['Step type']:9s} n={int(r['n_cells']):4d}  Acc {r['Acc']:5.1f}")
    print()

    # ---- volatility correlations ----
    print("=== Volatility vs per-user accuracy (Pearson r) ===")
    merged = (det_user.rename(columns={"accuracy": "det"})
              .merge(dh_user.rename(columns={"accuracy": "dethist"}),
                     on="experiment_id")
              .merge(vol.reset_index().rename(columns={"index": "experiment_id"}),
                     on="experiment_id"))
    for label in ["det", "dethist"]:
        r, p = stats.pearsonr(merged["volatility"], merged[label])
        n = len(merged)
        t = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else float("inf")
        print(f"  volatility vs {label:8s} accuracy: r={r:+.2f} "
              f"t({n-2})={t:.2f} p={p:.4f}")
    print(f"  (paper: LLM r=-0.67 p<0.001; deterministic r=-0.08 p=0.73)\n")

    # ---- optional LLM comparison ----
    if args.llm_csv and os.path.exists(args.llm_csv):
        llm = pd.read_csv(args.llm_csv)
        if llm["accuracy"].max() > 1.5:
            llm["accuracy"] = llm["accuracy"] / 100.0
        cmp = dh_user.rename(columns={"accuracy": "dethist"}).merge(
            llm.rename(columns={"accuracy": "llm"}), on="experiment_id")
        F, dfn, dfd, p_F, hi = f_test_var(cmp["dethist"].values, cmp["llm"].values)
        (w, w_p), (t, t_p) = paired_tests(cmp["dethist"].values, cmp["llm"].values)
        print("=== DET+HIST vs LLM (isolates the *reasoning* effect) ===")
        print(f"  Det+Hist mean {100*cmp['dethist'].mean():.1f}%  "
              f"LLM mean {100*cmp['llm'].mean():.1f}%")
        print(f"  Wilcoxon p={w_p:.3f}  paired-t p={t_p:.3f}  "
              f"F(var)={F:.2f} p={p_F:.3f}\n")

    # ---- write CSVs ----
    sweep.to_csv(os.path.join(args.outdir, "overall_sweep.csv"), index=False)
    merged.to_csv(os.path.join(args.outdir, "per_user_accuracy.csv"), index=False)
    pf_det.to_csv(os.path.join(args.outdir, "per_format_deterministic.csv"), index=False)
    pf_dh.to_csv(os.path.join(args.outdir, f"per_format_dethist_k{kp}.csv"), index=False)
    pst_det.to_csv(os.path.join(args.outdir, "per_step_type_deterministic.csv"), index=False)
    pst_dh.to_csv(os.path.join(args.outdir, f"per_step_type_dethist_k{kp}.csv"), index=False)
    print(f"Wrote CSV outputs to {args.outdir}/")


if __name__ == "__main__":
    main()
