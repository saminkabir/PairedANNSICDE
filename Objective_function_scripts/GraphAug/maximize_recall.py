#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Maximize Recall subject to per-dataset latency / build / memory budgets and
optionally compute Pareto-based domination rates.

Objective:
    maximize  Rec(A)
subject to
    Q(A) <= Tq,     # query latency (sec/query)
    C(A) <= Tc,     # construction time (sec)
    M(A) <= Tm      # memory footprint (GB)  [optional via USE_MEMORY]

Where:
    Rec(A):  recall@k of method A
    Q(A):    query latency (sec/query)  -> computed as 1 / qps
    C(A):    index construction time (sec)
    M(A):    memory footprint (GB)

Input Excel: "../Polyvector results.xlsx"

Sheets (current setup; adjust sheet names if needed):
- Sheet10 (augmented graph configs):
    dataset
    model1, model2, model3  (use '-' if missing)
    "augmented model's qps with refinement in parallal settings"
    "augmented model's recall with refinement"
    "augmented model's construction time in parallal settings"
    optional: memory

- Sheet9 (standalone graph configs):
    dataset
    model (or algo)
    "search_time without 2k refienement"
    "recall with 2k refinement" (or "recall")
    "construction time" (or "construction_time")
    optional: memory

- all_possible_vaq_hnsw_combo (optional, only used when HETEROGENEOUS=True):
    dataset
    model (optional; otherwise we label "VAQ+HNSW")
    "{parallal,sequential} qps"
    "refined recall" (or "recall")
    "{parallal,sequential} indexing time" (or "construction_time")
    optional: memory

Outputs:
    - Pareto domination CSVs (if RUN_DOMINATION=True):
        fig/max_rec_parallel/ or fig/max_rec_sequential/:
            domination_search_time.csv
            domination_index_time.csv

    - Per-dataset figures:
        fig/max_rec_parallel/{dataset}.png  (or max_rec_sequential)
        each with 3 bar charts (one per budget profile)

    - Per-dataset profile CSVs:
        {OUT_DIR}/{dataset}_profile{i}.csv
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# =========================
# ----- Configuration -----
# =========================

#EXCEL_PATH = "../Polyvector results merged.xlsx"
EXCEL_PATH = "../Polyvector results.xlsx"

# Global toggles
USE_MEMORY = False        # whether to enforce memory budget
PARALLEL_SETTING = True   # affects column names & output directory
HETEROGENEOUS = False     # include VAQ+HNSW sheet when True
AUTO_PROFILE = True       # automatically build profiles per dataset from data
RUN_DOMINATION = False     # compute Pareto-based domination rates

RC_THRESHOLD = 1.5

# Default profiles if no per-dataset override and/or AUTO_PROFILE is False.
# (Tq, Tc, Tm) where Tq is latency (sec/query), Tc construction time (sec), Tm memory (GB).
DEFAULT_PROFILES: List[Tuple[float, float, float]] = [
    (0.010,  120.0, 64.0),   # Tq=10ms,  Tc=120s, Tm=64
    (0.005,  300.0, 64.0),   # Tq=5ms,   Tc=300s, Tm=64
    (0.0025, 600.0, 128.0),  # Tq=2.5ms, Tc=600s, Tm=128
]

''''''
# Per-dataset manual profiles (used as base; AUTO_PROFILE adapts Tq/Tc from data).
PROFILES: Dict[str, List[Tuple[float, float, float]]] = {
    # Small-to-mid datasets
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

    # Large 1M-scale datasets
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
''''''

# Output directory
postifx = ""
if QUERY_HARDNESS is not None:
    postifx = "_hard" if QUERY_HARDNESS else "_easy" 
if PARALLEL_SETTING:
    OUT_DIR = Path(f"fig/max_rec_parallel{postifx}")
else:
    OUT_DIR = Path(f"fig/max_rec_sequential{postifx}")


# =========================
# ----- Helpers / IO ------
# =========================

def _clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _synthesize_combo_name(models: List[str]) -> str:
    parts = [m.strip() for m in models
             if isinstance(m, str) and m.strip() and m.strip() != "-"]
    return " + ".join(parts) if parts else "UNKNOWN"


def _coerce_numeric(series: pd.Series, default=None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if default is not None:
        out = out.fillna(default)
    return out


def load_augmented(path: str) -> pd.DataFrame:
    """Return columns: [dataset, combo, qps, recall, construction_time, memory]."""
    df = pd.read_excel(path, sheet_name="graph_new_datasets_augmented_re")
    df = _clean_colnames(df)

    aug_way = "parallal" if PARALLEL_SETTING else "sequential"

    col_qps = f"augmented model's qps with refinement in {aug_way} settings"
    col_rc  = "augmented model's recall with refinement"
    col_cst = f"augmented model's construction time in {aug_way} settings"
    col_mem = "memory" if "memory" in df.columns else None

    combo = df.apply(
        lambda r: _synthesize_combo_name(
            [r.get("model1", ""), r.get("model2", ""), r.get("model3", "")]
        ),
        axis=1,
    )

    out = pd.DataFrame({
        "source": "augmented",
        "dataset": df["dataset"].astype(str),
        "combo": combo,
        "qps": _coerce_numeric(df[col_qps]) if col_qps in df.columns else float("nan"),
        "recall": _coerce_numeric(df[col_rc]) if col_rc in df.columns else float("nan"),
        "construction_time": _coerce_numeric(df[col_cst]) if col_cst in df.columns else float("nan"),
        "memory": _coerce_numeric(df[col_mem]) if col_mem else 0.0,
    })
    return out


# nq map for deriving qps in standalone sheet (Sheet9)
STANDALONE_NQ = {
    "audio": 200,
    "deep": 200,
    "glove": 200,
    "imageNet": 200,
    "millionSong": 200,
    "MNIST": 200,
    "notre": 200,
    "nuswide": 200,
    "sift_twenty": 10000,
    "sun": 200,
}


def _lookup_standalone_nq(dataset_name: str) -> int | None:
    key = dataset_name.strip().lower()
    aliases = {
        "imagenet": "imageNet",
        "milionsong": "millionSong",
        "millionsong": "millionSong",
        "mnist": "MNIST",
    }
    for canon in STANDALONE_NQ.keys():
        c = canon.lower()
        if key == c or c in key or aliases.get(key, "").lower() == c:
            return STANDALONE_NQ[canon]
    return None


def load_standalone(path: str) -> pd.DataFrame:
    """
    Return columns: [dataset, combo, qps, recall, construction_time, memory].
    qps is derived via: qps = nq / "search_time without 2k refienement".
    """
    xls = pd.ExcelFile(path)
    # Keep this check as in your original code; adjust if your sheet names change
    if "graph_new_datasets_standalone_r" not in xls.sheet_names:
        assert False, "No sheet found for standalone"
        return pd.DataFrame(columns=[
            "source", "dataset", "combo", "qps", "recall", "construction_time", "memory"
        ])

    df = pd.read_excel(path, sheet_name="graph_new_datasets_standalone_r")
    df = _clean_colnames(df)


    # time_col = "search_time with 2k refienement"
    # if time_col not in df.columns:
    #     return pd.DataFrame(columns=[
    #         "source", "dataset", "combo", "qps", "recall", "construction_time", "memory"
    #     ])

    col_rc = (
        "recall with 2k refinement"
        if "recall with 2k refinement" in df.columns
        else ("recall" if "recall" in df.columns else None)
    )
    col_cst = (
        "construction time"
        if "construction time" in df.columns
        else ("construction_time" if "construction_time" in df.columns else None)
    )
    col_mem = "memory" if "memory" in df.columns else None

    # df[time_col] = _coerce_numeric(df[time_col])
    # df = df[pd.notna(df[time_col]) & (df[time_col] > 0)].copy()

    # nq_vals = []
    # for ds in df["dataset"].astype(str):
    #     nq = _lookup_standalone_nq(ds)
    #     nq_vals.append(nq if nq is not None else float("nan"))
    # df["nq_measured"] = nq_vals
    # df = df[pd.notna(df["nq_measured"]) & (df["nq_measured"] > 0)].copy()

    #df["qps"] = df["nq_measured"] / df[time_col]

    out = pd.DataFrame({
        "source": "standalone",
        "dataset": df["dataset"].astype(str),
        "combo": (
            df["model"].astype(str)
            if "model" in df.columns
            else df.get("algo", pd.Series(["standalone"] * len(df))).astype(str)
        ),
        "qps": _coerce_numeric(df["qps with 2k refienement"]),
        "recall": _coerce_numeric(df[col_rc]) if col_rc else float("nan"),
        "construction_time": _coerce_numeric(df[col_cst]) if col_cst else float("nan"),
        "memory": _coerce_numeric(df[col_mem]) if col_mem else 0.0,
    })

    out = out[pd.notna(out["qps"]) & (out["qps"] > 0)]
    return out.reset_index(drop=True)


def load_vaq_hnsw(path: str) -> pd.DataFrame:
    """Return columns: [dataset, combo, qps, recall, construction_time, memory]."""
    name = "all_possible_vaq_hnsw_combo"
    if name not in pd.ExcelFile(path).sheet_names:
        return pd.DataFrame(columns=[
            "source", "dataset", "combo", "qps", "recall", "construction_time", "memory"
        ])

    df = pd.read_excel(path, sheet_name=name)
    df = _clean_colnames(df)

    aug_way = "parallal" if PARALLEL_SETTING else "sequential"

    col_rc  = "refined recall" if "refined recall" in df.columns else "recall"
    col_cst = f"{aug_way} indexing time"
    col_mem = "memory" if "memory" in df.columns else None

    combo = df["model"].astype(str) if "model" in df.columns else pd.Series(["VAQ+HNSW"] * len(df))

    out = pd.DataFrame({
        "source": "vaq_hnsw",
        "dataset": df["dataset"].astype(str),
        "combo": combo,
        "qps": _coerce_numeric(df[f"{aug_way} qps"]),
        "recall": _coerce_numeric(df[col_rc]) if col_rc in df.columns else float("nan"),
        "construction_time": _coerce_numeric(df[col_cst]) if col_cst in df.columns else float("nan"),
        "memory": _coerce_numeric(df[col_mem]) if col_mem else 0.0,
    })
    return out


def load_all_unified(path: str) -> pd.DataFrame:
    """
    Load augmented + standalone (+ optional VAQ+HNSW) and unify schema:
        [source, dataset, combo, qps, recall, construction_time, memory]
    """
    ua = load_augmented(path)
    us = load_standalone(path)

    if HETEROGENEOUS:
        uv = load_vaq_hnsw(path)
        df = pd.concat([ua, us, uv], ignore_index=True)
    else:
        df = pd.concat([ua, us], ignore_index=True)

    # Normalize & filter
    df = df[pd.notna(df["qps"]) & pd.notna(df["construction_time"]) & pd.notna(df["recall"])]
    df["qps"] = pd.to_numeric(df["qps"], errors="coerce")
    df["construction_time"] = pd.to_numeric(df["construction_time"], errors="coerce")
    df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
    df["memory"] = pd.to_numeric(df["memory"], errors="coerce").fillna(0.0)
    df["combo"] = df["combo"].astype(str).str.replace(r"\s*\+\s*", " + ", regex=True)
    df = df[
        df["qps"].notna() &
        (df["qps"] > 0) &
        df["construction_time"].notna() &
        (df["construction_time"] > 0) &
        df["recall"].notna()
    ].reset_index(drop=True)

    return df


def build_profiles_per_dataset_auto(
    df: pd.DataFrame,
    base_profiles: Dict[str, List[Tuple[float, float, float]]],
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Automatically generate 3 profiles per dataset based on data:

        - Tq: latency (sec/query) = 1 / qps, using quantiles [0.2, 0.3, 0.5]
        - Tc: construction_time, using quantiles [0.2, 0.3, 0.5]
        - Tm: taken from base_profiles (or DEFAULT_PROFILES) for that dataset

    For each dataset:
        profiles[i] = (Tq_quantile[i], Tc_quantile[i], Tm_base[i])
    """
    profiles_per_dataset: Dict[str, List[Tuple[float, float, float]]] = {}

    for dataset, group in df.groupby("dataset"):
        g = group.copy()
        g = g[pd.notna(g["qps"]) & (g["qps"] > 0)]
        g = g[pd.notna(g["construction_time"]) & (g["construction_time"] > 0)]

        if g.empty:
            # Fallback to manual profiles
            profiles_per_dataset[dataset] = base_profiles.get(dataset, DEFAULT_PROFILES)
            continue

        latencies = (1.0 / g["qps"]).astype(float)
        ctimes = g["construction_time"].astype(float)

        try:
            # Use 0.2, 0.3, 0.5 quantiles as requested
            lat_q = latencies.quantile([0.25, 0.3, 0.5]).tolist()
            cst_q = ctimes.quantile([0.25, 0.3, 0.5]).tolist()
        except Exception:
            profiles_per_dataset[dataset] = base_profiles.get(dataset, DEFAULT_PROFILES)
            continue

        base = base_profiles.get(dataset, DEFAULT_PROFILES)
        if len(base) < 3:
            base = (base + DEFAULT_PROFILES)[:3]
        Tm_list = [p[2] for p in base]

        profiles: List[Tuple[float, float, float]] = []
        for (Tq, Tc, Tm) in zip(lat_q, cst_q, Tm_list):
            profiles.append((float(Tq), float(Tc), float(Tm)))

        profiles_per_dataset[dataset] = profiles

    return profiles_per_dataset


# =========================
# ----- Core compute ------
# =========================

def _best_recall_for_rows(
    rows: pd.DataFrame,
    Tq: float,
    Tc: float,
    Tm: float,
    use_memory: bool,
) -> tuple[float, bool]:
    """
    For all rows of a (dataset, combo), pick the maximum recall that satisfies:
        1/qps <= Tq, construction_time <= Tc, [memory <= Tm if enabled].
    Returns (best_recall, feasible_flag). If infeasible, (0.0, False).
    """
    best_rec = 0.0
    feasible = False

    # qps >= 1/Tq  <=>  1/qps <= Tq.
    min_qps = (1.0 / Tq) if Tq > 0 else float("inf")

    for _, r in rows.iterrows():
        qps = r.get("qps", float("nan"))
        cst = r.get("construction_time", float("nan"))
        rcl = r.get("recall", float("nan"))
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
    use_memory: bool,
) -> Dict[str, List[pd.DataFrame]]:
    """
    For each dataset and each (Tq, Tc, Tm) profile, compute the best achievable
    recall per combo.

    Returns:
        dataset -> [DataFrame per profile], each DataFrame has:
            ['combo', 'best_recall', 'feasible']
        sorted by feasible desc, best_recall desc.
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
                rec, feas = _best_recall_for_rows(
                    combo_rows, Tq=Tq, Tc=Tc, Tm=Tm, use_memory=use_memory
                )
                rows_out.append((combo, rec, feas))

            dd = pd.DataFrame(rows_out, columns=["combo", "best_recall", "feasible"])
            dd = dd.sort_values(
                by=["feasible", "best_recall"],
                ascending=[False, False],
                kind="mergesort",
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
    outdir: Path,
):
    """
    One tall figure with three bar charts (one per profile).
    Gray bars at 0 with "N.A." mean no feasible configuration met the budgets.
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


# =========================
# --- Pareto domination ---
# =========================

def _domination_ratio_recall_x(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric_col: str,
    num_grid: int = 200,
    larger_is_better: bool = False,
) -> float:
    """
    For the same dataset, compare two curves A and B using recall as x-axis and
    metric_col as y-axis (e.g., search_time or index_time), and compute the
    fraction of recall range where A dominates B.

    If larger_is_better=False (typical for time metrics), smaller y is better,
    and domination condition is y_A < y_B.

    Returns:
        domination rate in [0, 1], or np.nan if ranges do not overlap.
    """
    df_a = df_a[["recall", metric_col]].dropna().copy()
    df_b = df_b[["recall", metric_col]].dropna().copy()

    if df_a.empty or df_b.empty:
        return np.nan

    agg_func = "max" if larger_is_better else "min"
    df_a = (
        df_a.groupby("recall", as_index=False)[metric_col]
        .agg(agg_func)
        .sort_values("recall")
    )
    df_b = (
        df_b.groupby("recall", as_index=False)[metric_col]
        .agg(agg_func)
        .sort_values("recall")
    )

    xa = df_a["recall"].to_numpy()
    ya = df_a[metric_col].to_numpy()
    xb = df_b["recall"].to_numpy()
    yb = df_b[metric_col].to_numpy()

    left = max(xa.min(), xb.min())
    right = min(xa.max(), xb.max())

    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        return np.nan

    grid = np.linspace(left, right, num_grid)

    ya_grid = np.interp(grid, xa, ya)
    yb_grid = np.interp(grid, xb, yb)

    if larger_is_better:
        dom = np.mean(ya_grid > yb_grid)
    else:
        dom = np.mean(ya_grid < yb_grid)

    return float(dom)


def compute_pareto_domination_rates(
    df: pd.DataFrame,
    metric: str = "search_time",
    num_grid: int = 200,
) -> pd.DataFrame:
    """
    Compute Pareto-based domination rates across all datasets for each pair
    (combo_A, combo_B), using recall as x-axis and the chosen metric as y-axis.

    Parameters:
        df: unified DataFrame from load_all_unified(), with columns:
            ['dataset', 'combo', 'qps', 'recall', 'construction_time', ...]
        metric:
            - "search_time": y = 1 / qps (query latency per query, smaller is better)
            - "index_time" or "construction_time": y = construction_time (smaller is better)
        num_grid: number of interpolation points over recall range.

    Returns:
        DataFrame with columns:
            ['combo_A', 'combo_B',
             'mean_domination', 'std_domination',
             'n_datasets', 'metric']
    """
    df = df.copy()

    if metric == "search_time":
        df = df[pd.notna(df["qps"]) & (df["qps"] > 0)].copy()
        df["search_time"] = 1.0 / df["qps"]
        metric_col = "search_time"
        larger_is_better = False
    elif metric in ("index_time", "construction_time"):
        df = df[pd.notna(df["construction_time"]) & (df["construction_time"] > 0)].copy()
        df["index_time"] = df["construction_time"]
        metric_col = "index_time"
        larger_is_better = False
    else:
        raise ValueError(f"Unknown metric '{metric}', expected 'search_time' or 'index_time'")

    df = df[pd.notna(df["recall"])].copy()

    pair_to_vals: dict[tuple[str, str], list[float]] = {}

    for dataset, df_ds in df.groupby("dataset"):
        combos = sorted(df_ds["combo"].unique())
        if len(combos) < 2:
            continue

        combo_to_df = {
            c: df_ds[df_ds["combo"] == c][["combo", "recall", metric_col]].copy()
            for c in combos
        }

        for ca in combos:
            for cb in combos:
                if ca == cb:
                    continue

                da = combo_to_df[ca]
                db = combo_to_df[cb]
                dom = _domination_ratio_recall_x(
                    da,
                    db,
                    metric_col=metric_col,
                    num_grid=num_grid,
                    larger_is_better=larger_is_better,
                )
                if np.isnan(dom):
                    continue
                pair_to_vals.setdefault((ca, cb), []).append(dom)

    rows = []
    for (ca, cb), vals in pair_to_vals.items():
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        mean_dom = float(np.mean(vals))
        std_dom = float(np.std(vals))
        n = len(vals)
        rows.append(
            {
                "combo_A": ca,
                "combo_B": cb,
                "mean_domination": mean_dom,
                "std_domination": std_dom,
                "n_datasets": n,
                "metric": metric,
            }
        )

    if not rows:
        print("No valid domination pairs found for metric:", metric)
        return pd.DataFrame(
            columns=[
                "combo_A",
                "combo_B",
                "mean_domination",
                "std_domination",
                "n_datasets",
                "metric",
            ]
        )

    result = pd.DataFrame(rows).sort_values(
        ["mean_domination", "n_datasets"],
        ascending=[False, False],
    )


    print(f"\n=== Pareto-based domination rates (metric = {metric}) ===")
    for _, row in result.iterrows():
        print(
            f"{row['combo_A']}  ≻  {row['combo_B']}: "
            f"{row['mean_domination']:.3f}  (over {int(row['n_datasets'])} datasets)"
        )

    return result


# =========================
# ---------- Run ----------
# =========================
import re
def main():
    
    ds_property = pd.read_csv("../dataset property.csv")
    RC_THRESHOLD = ds_property['rc'].median()
    print(RC_THRESHOLD)
    df = load_all_unified(EXCEL_PATH)
    # pattern = "^(" + "|".join(map(re.escape, ds_property["Dataset"])) + ")"
    # df = df[df["dataset"].str.contains(pattern, regex=True)]
    print(df["dataset"].unique())

    if QUERY_HARDNESS is not None:
        if QUERY_HARDNESS:
            ds_property = ds_property[ds_property['rc'] <= RC_THRESHOLD]
            print(f"Hard Datasets with RC >= {RC_THRESHOLD}, we have {len(ds_property['Dataset'].unique())} in total")
        else:
            ds_property = ds_property[ds_property['rc'] > RC_THRESHOLD]
            print(f"Hard Datasets with RC <= {RC_THRESHOLD}, we have {len(ds_property['Dataset'].unique())} in total")



    if df.empty:
        print("No usable rows (missing qps/recall/construction_time). "
              "Check your Excel sheets & column names.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Optional: Pareto domination analysis
    if RUN_DOMINATION:
        dom_search = compute_pareto_domination_rates(df, metric="search_time")
        dom_search.to_csv(OUT_DIR / "domination_search_time.csv", index=False)

        dom_index = compute_pareto_domination_rates(df, metric="index_time")
        dom_index.to_csv(OUT_DIR / "domination_index_time.csv", index=False)

    # 2) Profile-based max recall & bar plots
    datasets = sorted(df["dataset"].unique().tolist())

    # Base profiles from manual PROFILES / DEFAULT_PROFILES
    base_profiles: Dict[str, List[Tuple[float, float, float]]] = {}

    # Optionally adapt Tq/Tc per dataset using auto profiles
    if AUTO_PROFILE:
        profiles_per_dataset = build_profiles_per_dataset_auto(df, base_profiles)
    else:
        for ds in datasets:
            base_profiles[ds] = PROFILES.get(ds, DEFAULT_PROFILES)
        profiles_per_dataset = base_profiles

    results = compute_max_rec_per_dataset(df, profiles_per_dataset, use_memory=USE_MEMORY)

    for ds in datasets:
        per_profiles = results.get(ds, [])
        profiles = profiles_per_dataset.get(ds, DEFAULT_PROFILES)
        plot_bars_for_dataset(ds, per_profiles, profiles, OUT_DIR)

    # Save CSVs per dataset/profile
    for ds, per_profiles in results.items():
        for i, dd in enumerate(per_profiles, start=1):
            if not dd.empty:
                dd.to_csv(OUT_DIR / f"{ds}_profile{i}.csv", index=False)

    print(f"Done. Outputs saved under: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
