#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Maximize Recall for tree methods subject to per-dataset latency / build / memory budgets
and draw bar charts (three constraint profiles per dataset).

Source sheet: "tree_results"

Expected columns:
    model
    dataset
    construction_time (parallel)
    construction_time (sequential)
    search_time
    search_timex_refined (parallel)
    search_timex_refined (sequential)
    recall
    refined_recall

Unified to:
    dataset, combo, qps, recall, construction_time, memory

Then we reuse the same logic as the quantization script:
    - objective function
    - per-dataset profiles (supports AUTO_PROFILE)
    - 3-row bar charts
    - per dataset/profile CSV outputs
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# ----- Configuration -----
# =========================

EXCEL_PATH = "../Polyvector results.xlsx"   # change to your real path

# Toggle memory constraint globally
USE_MEMORY = False          # tree side: default no memory constraint
PARALLEL_SETTING = False     # True  -> use parallel construction + refined time
                            # False -> use sequential construction + refined time
AUTO_PROFILE = True         # whether to auto-generate profiles from data

# Three (Tq, Tc, Tm) default profiles (seconds/query, seconds, GB)
DEFAULT_PROFILES: List[Tuple[float, float, float]] = [
    (0.010,  120.0, 64.0),
    (0.005,  300.0, 64.0),
    (0.0025, 600.0, 128.0),
]

# Same per-dataset overrides as your quantization script
PROFILES: Dict[str, List[Tuple[float, float, float]]] = {
    "audio": [
        (1/20000,   5,   8),
        (1/15000,  10,  16),
        (1/10000,  15,  32),
    ],
    "notre": [
        (1/30000,   20,   8),
        (1/20000,  100,  16),
        (1/10000, 500,  32),
    ],
    "sun": [
        (1/10000,  10,  12),
        (1/7500,  50,  24),
        (1/5000, 200,  48),
    ],
    "nuswide": [
        (1/7500,  500,  12),
        (1/5000,  1000,  24),
        (1/2500, 2000,  48),
    ],
    "MNIST": [
        (1/15000,  200,  32),
        (1/10000,  80,  64),
        (1/5000,  200,  96),
    ],
    "sift_twenty": [
        (1/7500,  100,  24),
        (1/5000, 500,  48),
        (1/2500,1500,  96),
    ],
    "deep": [
        (1/15000,  200,  32),
        (1/10000, 800,  64),
        (1/5000, 2000,  96),
    ],
    "glove": [
        (1/20000,  500,  32),
        (1/10000, 2000,  64),
        (1/5000, 4000,  96),
    ],
    "imageNet": [
        (1/15000,  200,  24),
        (1/10000, 1000,  48),
        (1/5000, 2000,  96),
    ],
    "millionSong": [
        (1/15000,  600,  24),
        (1/10000, 1800,  48),
        (1/5000, 3600,  96),
    ],
}

# number of queries per dataset (normalized to lower-case keys)
NQ_LOWER = {
    "audio":        200,
    "deep":         200,
    "glove":        200,
    "imagenet":     200,
    "millionsong":  200,
    "mnist":        200,
    "notre":        200,
    "nuswide":      200,
    "sift_twenty":         10000,
    "sun":          200,
}

# Output directory
if PARALLEL_SETTING:
    OUT_DIR = Path("fig/max_rec_tree_parallel")
else:
    OUT_DIR = Path("fig/max_rec_tree_sequential")


# =========================
# ----- Helpers / IO ------
# =========================

def _clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _coerce_numeric(series: pd.Series, default=None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if default is not None:
        out = out.fillna(default)
    return out


def _find_col(df: pd.DataFrame, target: str) -> str:
    """
    Find a column whose normalized name matches `target`.
    Normalization: strip + lower.
    """
    target_norm = target.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == target_norm:
            return c
    raise KeyError(f"Could not find column matching '{target}' in {list(df.columns)}")


def build_profiles_per_dataset_auto(
    df: pd.DataFrame,
    base_profiles: Dict[str, List[Tuple[float, float, float]]],
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Automatically generate 3 profiles per dataset:
        - Tq: quantiles of latency (sec/query)
        - Tc: quantiles of construction_time
        - Tm: taken from base_profiles or DEFAULT_PROFILES

    Same logic as your hash / quant scripts.
    """
    profiles_per_dataset: Dict[str, List[Tuple[float, float, float]]] = {}

    for dataset, group in df.groupby("dataset"):
        g = group.copy()
        g = g[pd.notna(g["qps"]) & (g["qps"] > 0)]
        g = g[pd.notna(g["construction_time"]) & (g["construction_time"] > 0)]
        if g.empty:
            profiles_per_dataset[dataset] = base_profiles.get(dataset, DEFAULT_PROFILES)
            continue

        latencies = (1.0 / g["qps"]).astype(float)
        ctimes = g["construction_time"].astype(float)

        try:
            lat_q = latencies.quantile([0.25, 0.4, 0.5]).tolist()
            cst_q = ctimes.quantile([0.25, 0.4, 0.5]).tolist()

            # if dataset == "audio":
            #     lat_q = latencies.quantile([0.2, 0.25, 0.3]).tolist()
            #     cst_q = ctimes.quantile([0.2, 0.25, 0.3]).tolist()
            # if dataset == "deep":
            #     lat_q = latencies.quantile([0.2, 0.25, 0.5]).tolist()
            #     cst_q = ctimes.quantile([0.2, 0.25, 0.5]).tolist()
            # if dataset == "imageNet":
            #     lat_q = latencies.quantile([0.3, 0.4, 0.5]).tolist()
            #     cst_q = ctimes.quantile([0.3, 0.4, 0.5]).tolist()
            # if dataset == "millionSong":
            #     lat_q = latencies.quantile([0.2, 0.3, 0.4]).tolist()
            #     cst_q = ctimes.quantile([0.2, 0.3, 0.4]).tolist()
            # if dataset == "MNIST":
            #     lat_q = latencies.quantile([0.15, 0.17, 0.2]).tolist()
            #     cst_q = ctimes.quantile([0.15, 0.17, 0.2]).tolist()
            # if dataset == "notre":
            #     lat_q = latencies.quantile([0.2, 0.3, 0.5]).tolist()
            #     cst_q = ctimes.quantile([0.2, 0.3, 0.5]).tolist()
            # if dataset == "nuswide":
            #     lat_q = latencies.quantile([0.3, 0.5, 0.7]).tolist()
            #     cst_q = ctimes.quantile([0.3, 0.5, 0.7]).tolist()
            # if dataset == "sift_twenty":
            #     lat_q = latencies.quantile([0.4, 0.45, 0.5]).tolist()
            #     cst_q = ctimes.quantile([0.4, 0.45, 0.5]).tolist()

        except Exception:
            profiles_per_dataset[dataset] = base_profiles.get(dataset, DEFAULT_PROFILES)
            continue

        base = base_profiles.get(dataset, DEFAULT_PROFILES)
        if len(base) < 3:
            base = (base + DEFAULT_PROFILES)[:3]
        Tm_list = [p[2] for p in base]

        profiles = []
        for (Tq, Tc, Tm) in zip(lat_q, cst_q, Tm_list):
            profiles.append((float(Tq), float(Tc), float(Tm)))

        profiles_per_dataset[dataset] = profiles

    return profiles_per_dataset


# ---------- tree_results loader ----------

def load_all_unified_tree(path: str) -> pd.DataFrame:
    """
    Load tree results from sheet 'tree_results' and unify into:
        dataset, combo, qps, recall, construction_time, memory

    Logic:
        - combo  = model
        - recall = refined_recall
        - construction_time: choose (parallel / sequential) based on PARALLEL_SETTING
        - search time (for qps):
              PARALLEL_SETTING = True  -> use 'search_timex_refined (parallel)'
              PARALLEL_SETTING = False -> use 'search_timex_refined (sequential)'
          Then:
              sec_total = search_time_refined
              sec_per_query = sec_total / NQ[dataset]
              qps = 1.0 / sec_per_query = NQ[dataset] / sec_total
        - memory = 0.0 (no memory col for trees)
    """
    df = pd.read_excel(path, sheet_name="tree_new_datasets_results")
    df = _clean_colnames(df)

    # basic cols
    col_ds    = _find_col(df, "dataset")
    col_model = _find_col(df, "model")
    col_rc    = _find_col(df, "refined_recall")

    # construction time
    if PARALLEL_SETTING:
        col_cst = _find_col(df, "construction_time (parallel)")
        col_st  = _find_col(df, "search_timex_refined (parallel)")
    else:
        col_cst = _find_col(df, "construction_time (sequential)")
        col_st  = _find_col(df, "search_timex_refined (sequential)")

    ds_raw = df[col_ds].astype(str).str.strip()
    ds_norm = ds_raw.str.lower()

    # map NQ per dataset (lower-case)
    NQ_series = ds_norm.map(NQ_LOWER).fillna(200)

    search_total = _coerce_numeric(df[col_st])
    cst = _coerce_numeric(df[col_cst])
    rc  = _coerce_numeric(df[col_rc])

    # sec_per_query = total_time / NQ
    sec_per_query = search_total / NQ_series
    # qps = 1 / sec_per_query
    qps = 1.0 / sec_per_query

    out = pd.DataFrame({
        "dataset": ds_raw,
        "combo":   df[col_model].astype(str),
        "qps":     qps,
        "recall":  rc,
        "construction_time": cst,
        "memory":  0.0,  # no memory info for tree_results
    })

    # filter valid rows
    out = out[
        out["qps"].notna() &
        (out["qps"] > 0) &
        out["construction_time"].notna() &
        (out["construction_time"] > 0) &
        out["recall"].notna()
    ].reset_index(drop=True)

    return out


# =========================
# ----- Core compute ------
# =========================

def _best_recall_for_rows(
    rows: pd.DataFrame,
    Tq: float, Tc: float, Tm: float,
    use_memory: bool
) -> tuple[float, bool]:
    """
    For multiple configurations (rows) of one (dataset, combo),
    pick the max recall under:
        Q(A) = 1 / qps <= Tq   <=> qps >= 1/Tq
        C(A) <= Tc
        M(A) <= Tm (if use_memory)

    Returns (best_recall, feasible_flag).
    """
    best_rec = 0.0
    feasible = False

    min_qps = (1.0 / Tq) if Tq > 0 else float("inf")

    for _, r in rows.iterrows():
        qps = r.get("qps")
        cst = r.get("construction_time")
        rcl = r.get("recall")
        mem = r.get("memory", 0.0) if use_memory else 0.0

        if not (pd.notna(qps) and pd.notna(cst) and pd.notna(rcl)):
            continue
        if qps <= 0 or cst < 0:
            continue
        if qps < min_qps:
            continue
        if cst > Tc:
            continue
        if use_memory and pd.notna(mem) and mem > Tm:
            continue

        feasible = True
        if rcl > best_rec:
            best_rec = float(rcl)

    return (best_rec if feasible else 0.0), feasible


def compute_max_rec_per_dataset(
    df: pd.DataFrame,
    profiles_per_dataset: Dict[str, List[Tuple[float, float, float]]],
    use_memory: bool
) -> Dict[str, List[pd.DataFrame]]:
    """
    For each dataset and each profile, compute best recall per combo.
    Returns: dataset -> [DataFrame per profile]
    Each DataFrame has columns:
        combo, best_recall, feasible
    """
    results: Dict[str, List[pd.DataFrame]] = {}

    for dataset, group in df.groupby("dataset"):
        profiles = profiles_per_dataset.get(dataset, DEFAULT_PROFILES)
        per_profiles: List[pd.DataFrame] = []

        by_combo = group.groupby("combo", sort=False)
        all_combos = list(by_combo.groups.keys())

        for (Tq, Tc, Tm) in profiles:
            rows_out = []
            for combo in all_combos:
                combo_rows = by_combo.get_group(combo)
                rec, feas = _best_recall_for_rows(combo_rows, Tq, Tc, Tm, use_memory)
                rows_out.append((combo, rec, feas))

            dd = pd.DataFrame(rows_out, columns=["combo", "best_recall", "feasible"])
            dd = dd.sort_values(
                by=["feasible", "best_recall"],
                ascending=[False, False],
                kind="mergesort"
            )
            per_profiles.append(dd)

        results[dataset] = per_profiles

    return results


# =========================
# ------- Plotting --------
# =========================

def plot_bars_for_dataset(
    dataset: str,
    per_profiles: List[pd.DataFrame],
    profiles_cfg: List[Tuple[float, float, float]],
    outdir: Path
):
    """
    For one dataset, draw a figure with 3 subplots (one per profile):
        x-axis: combo
        y-axis: best_recall
    Infeasible combos are grey and labeled "N.A.".
    """
    n_sub = 3
    profiles_cfg = (profiles_cfg + DEFAULT_PROFILES)[:n_sub]
    per_profiles = (
        per_profiles + [pd.DataFrame(columns=["combo", "best_recall", "feasible"])] * n_sub
    )[:n_sub]

    fig_h = 4.0 * n_sub
    fig, axes = plt.subplots(n_sub, 1, figsize=(12, fig_h), constrained_layout=True)
    if n_sub == 1:
        axes = [axes]

    for idx, (ax, df_sub, (Tq, Tc, Tm)) in enumerate(
        zip(axes, per_profiles, profiles_cfg), start=1
    ):
        if df_sub.empty:
            title = f"[{dataset}] Profile {idx}: Tq≤{Tq*1000:.2f}ms, Tc≤{Tc:.2f}s"
            if USE_MEMORY:
                title += f", Tm≤{Tm}GB"
            ax.set_title(title + " (no data)")
            ax.set_ylabel("best recall")
            ax.set_xticks([])
            ax.set_ylim(0, 1)
            continue

        combos = df_sub["combo"].tolist()
        values = df_sub["best_recall"].tolist()
        feas   = df_sub["feasible"].tolist()

        bars = ax.bar(range(len(values)), values)
        for b, ok in zip(bars, feas):
            if not ok:
                b.set_color("lightgray")

        title = f"[{dataset}] Profile {idx}: Tq≤{Tq*1000:.2f}ms, Tc≤{Tc:.2f}s"
        if USE_MEMORY:
            title += f", Tm≤{Tm}GB"
        ax.set_title(title)
        ax.set_ylabel("best recall")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(combos, rotation=30, ha="right")

        ymax = max(values) if values else 1.0
        ymax = max(ymax, 0.1)
        ytxt = ymax * 0.05
        for x, (v, ok) in enumerate(zip(values, feas)):
            if not ok:
                ax.text(x, ytxt, "N.A.", ha="center", va="bottom", fontsize=9)

        ax.set_ylim(0, min(1.0, ymax * 1.15))

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{dataset}.png"
    fig.suptitle(
        f"Max Recall under Budgets — {dataset}   (Memory: {'ON' if USE_MEMORY else 'OFF'})",
        y=1.02,
        fontsize=13,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def filter_out_models(df: pd.DataFrame, banned_models: List[str]) -> pd.DataFrame:
    """
    Filter out rows whose combo contains any banned model token.
    combo is split on '+'.
    """
    if not banned_models:
        return df

    banned_set = set(m.strip() for m in banned_models)

    def is_banned(combo: str) -> bool:
        tokens = [t.strip() for t in combo.split("+")]
        return any(t in banned_set for t in tokens)

    mask = df["combo"].apply(lambda x: not is_banned(str(x)))
    return df[mask].reset_index(drop=True)


# =========================
# ---------- Run ----------
# =========================

def main():
    df = load_all_unified_tree(EXCEL_PATH)
    if df.empty:
        print("No usable rows from tree_results sheet.")
        return
    

    # Example: filter some models if needed
    # df = filter_out_models(df, ["KD-Tree"])

    datasets = sorted(df["dataset"].unique().tolist())

    base_profiles: Dict[str, List[Tuple[float, float, float]]] = {}
    for ds in datasets:
        base_profiles[ds] = PROFILES.get(ds, DEFAULT_PROFILES)

    if AUTO_PROFILE:
        profiles_per_dataset = build_profiles_per_dataset_auto(df, base_profiles)
    else:
        profiles_per_dataset = base_profiles

    results = compute_max_rec_per_dataset(df, profiles_per_dataset, use_memory=USE_MEMORY)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        per_profiles = results.get(ds, [])
        profiles = profiles_per_dataset.get(ds, DEFAULT_PROFILES)
        plot_bars_for_dataset(ds, per_profiles, profiles, OUT_DIR)

    # dump CSV per dataset/profile
    for ds, per_profiles in results.items():
        for i, dd in enumerate(per_profiles, start=1):
            if not dd.empty:
                dd.to_csv(OUT_DIR / f"{ds}_profile{i}.csv", index=False)

    print(f"Done. Figures saved under: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
