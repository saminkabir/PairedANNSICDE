#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Maximize Recall subject to per-dataset latency / build / memory budgets and
draw bar charts (three constraint profiles per dataset).

现在数据全部来自一个 sheet: "hash_results"

Sheet: hash_results
    Method
    Dataset
    Refined Recall
    QPS
    Construction Time
    [可选: memory]

其中：
- standalone 的 Method 就是单模型名，例如 "PM-LSH"
- augmented 的 Method 会是多个 base model 拼起来，例如 "DB-LSH+PM-LSH"

其它逻辑（目标函数、profile、画图、输出 CSV）和原来一样。
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# ----- Configuration -----
# =========================

EXCEL_PATH = "../Polyvector results.xlsx"   # 换成你的真实路径

# Toggle memory constraint globally
USE_MEMORY = False          # 新 sheet 一般没有 memory，就先关掉
PARALLEL_SETTING = False     # 只是决定输出目录名字而已
HETEROGENEOUS = False       # 对新 sheet 没意义，但保留不影响
AUTO_PROFILE = True 

# 三个 (Tq, Tc, Tm) profile 默认值
DEFAULT_PROFILES: List[Tuple[float, float, float]] = [
    (0.010,  120.0, 64.0),   # Tq=10ms,  Tc=120s, Tm=64
    (0.005,  300.0, 64.0),   # Tq=5ms,   Tc=300s, Tm=64
    (0.0025, 600.0, 128.0),  # Tq=2.5ms, Tc=600s, Tm=128
]

# Per-dataset overrides (和你原来的一样直接复用)
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

# 输出目录（你也可以改个名字，比如 max_rec_hash）
if PARALLEL_SETTING:
    OUT_DIR = Path("fig/max_rec_parallel")
else:
    OUT_DIR = Path("fig/max_rec_sequential")


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

def build_profiles_per_dataset_auto(
    df: pd.DataFrame,
    base_profiles: Dict[str, List[Tuple[float, float, float]]],
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    基于数据自动为每个 dataset 生成 3 个 profile:
        - Tq: latency(秒/query) 的 75/50/25 percentile
        - Tc: construction_time 的 75/50/25 percentile
        - Tm: 沿用 base_profiles 中该 dataset (或 DEFAULT_PROFILES) 的 Tm

    注意：percentile 是“从松到紧”，即：
        Profile 1: 75th (宽松)
        Profile 2: 50th
        Profile 3: 25th (最紧)
    """
    profiles_per_dataset: Dict[str, List[Tuple[float, float, float]]] = {}

    for dataset, group in df.groupby("dataset"):
        # 只保留合法的 qps 和 construction_time
        g = group.copy()
        g = g[pd.notna(g["qps"]) & (g["qps"] > 0)]
        g = g[pd.notna(g["construction_time"]) & (g["construction_time"] > 0)]
        if g.empty:
            # 没数据，回退到手动配置
            profiles_per_dataset[dataset] = base_profiles.get(dataset, DEFAULT_PROFILES)
            continue

        # latency = 1 / qps (秒/query)
        latencies = (1.0 / g["qps"]).astype(float)
        ctimes = g["construction_time"].astype(float)

        # 计算 percentile
        try:
            lat_q = latencies.quantile([0.25, 0.4, 0.5]).tolist()
            cst_q = ctimes.quantile([0.25, 0.4, 0.5]).tolist()
            # if dataset == "audio":
            #     lat_q = latencies.quantile([0.3, 0.4, 0.5]).tolist()
            #     cst_q = ctimes.quantile([0.3, 0.4, 0.5]).tolist()
            # if dataset == "imageNet":
            #     lat_q = latencies.quantile([0.3, 0.4, 0.5]).tolist()
            #     cst_q = ctimes.quantile([0.3, 0.4, 0.5]).tolist()
            # if dataset == "millionSong":
            #     lat_q = latencies.quantile([0.05, 0.1, 0.15]).tolist()
            #     cst_q = ctimes.quantile([0.05, 0.1, 0.15]).tolist()                
            # if dataset == 'sift':
            #     lat_q = latencies.quantile([0.1, 0.1, 0.3]).tolist()
            #     cst_q = ctimes.quantile([0.1, 0.1, 0.3]).tolist()                              
        except Exception:
            # quantile 出问题就退回手动配置
            profiles_per_dataset[dataset] = base_profiles.get(dataset, DEFAULT_PROFILES)
            continue

        # 拿一份“基准” Tm（memory budget），只用它的第三个元素
        base = base_profiles.get(dataset, DEFAULT_PROFILES)
        if len(base) < 3:
            # 不足 3 个，就用 DEFAULT_PROFILES 填满
            base = (base + DEFAULT_PROFILES)[:3]
        Tm_list = [p[2] for p in base]  # 只取第三个元素（memory）

        # 组合成 3 个 profile: (Tq, Tc, Tm)
        profiles = []
        for (Tq, Tc, Tm) in zip(lat_q, cst_q, Tm_list):
            profiles.append((float(Tq), float(Tc), float(Tm)))

        profiles_per_dataset[dataset] = profiles

    return profiles_per_dataset


def load_hash_results(path: str) -> pd.DataFrame:
    """
    从 sheet: hash_results 统一读出数据，返回列:
        dataset, combo, qps, recall, construction_time, memory
    """
    df = pd.read_excel(path, sheet_name="hash_results_new_datasets")
    df = _clean_colnames(df)

    # 列名对应你给的例子：
    # Method, Dataset, Refined Recall, QPS, Construction Time
    mark = "parallel" if PARALLEL_SETTING else "sequential"

    col_method = "Method"
    col_ds     = "Dataset"
    col_rc     = "Refined Recall"
    col_qps    = f"QPS ({mark})"
    col_cst    = f"Construction Time ({mark})"
    col_mem    = "memory" if "memory" in df.columns else None

    out = pd.DataFrame({
        "dataset": df[col_ds].astype(str),
        "combo":   df[col_method].astype(str),           # standalone & augmented 一视同仁
        "qps":     _coerce_numeric(df[col_qps]),
        "recall":  _coerce_numeric(df[col_rc]),
        "construction_time": _coerce_numeric(df[col_cst]),
        "memory": _coerce_numeric(df[col_mem]) if col_mem else 0.0,
    })

    # 过滤掉缺失 / 非法的行
    out = out[
        out["qps"].notna() &
        out["construction_time"].notna() &
        out["recall"].notna()
    ].reset_index(drop=True)

    return out


def load_all_unified(path: str) -> pd.DataFrame:
    """
    对新数据，只需要从 hash_results 读一次即可。
    """
    df = load_hash_results(path)
    # 正规化一下，和老 pipeline 保持一致
    df["qps"] = pd.to_numeric(df["qps"], errors="coerce")
    df["construction_time"] = pd.to_numeric(df["construction_time"], errors="coerce")
    df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
    df["memory"] = pd.to_numeric(df["memory"], errors="coerce").fillna(0.0)
    df["combo"] = df["combo"].astype(str).str.replace(r"\s*\+\s*", " + ", regex=True)
    return df


# =========================
# ----- Core compute ------
# =========================

def _best_recall_for_rows(rows: pd.DataFrame,
                          Tq: float, Tc: float, Tm: float,
                          use_memory: bool) -> tuple[float, bool]:
    """
    对一个 (dataset, combo) 的多行配置，选出满足约束的最大 recall:
        Q(A) = 1 / qps <= Tq
        C(A) <= Tc
        M(A) <= Tm (如果启用)
    返回 (best_recall, feasible_flag)。不可行则返回 (0.0, False)。
    """
    best_rec = 0.0
    feasible = False

    # qps >= 1/Tq  <=>  1/qps <= Tq
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
        if qps < min_qps:      # query 太慢
            continue
        if cst > Tc:           # build 时间超预算
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
    对每个 dataset / 每个 profile，计算每个 combo 的 best_recall 和可行性。
    返回: dataset -> [DataFrame per profile]
    每个 DataFrame 列:
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
    为单个 dataset 画一张图，三行子图，每行对应一个 profile 的 bar chart。
    不可行的配置画成 0 高度的灰色柱，并标 "N.A."。
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

        # 不可行标 "N.A."
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
# ---------- Run ----------
# =========================

def main():
    df = load_all_unified(EXCEL_PATH)
    if df.empty:
        print("No usable rows from hash_results (missing QPS/Recall/Construction Time?).")
        return

    datasets = sorted(df["dataset"].unique().tolist())

    # 为每个 dataset 准备 3 个 profiles
    # profiles_per_dataset: Dict[str, List[Tuple[float, float, float]]] = {}
    # for ds in datasets:
    #     profiles_per_dataset[ds] = PROFILES.get(ds, DEFAULT_PROFILES)

    base_profiles: Dict[str, List[Tuple[float, float, float]]] = {}
    for ds in datasets:
        base_profiles[ds] = PROFILES.get(ds, DEFAULT_PROFILES)

    # 再根据 AUTO_PROFILE 决定是否启用自动 profile
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

    # 输出每个 dataset / profile 的 CSV
    for ds, per_profiles in results.items():
        for i, dd in enumerate(per_profiles, start=1):
            if not dd.empty:
                dd.to_csv(OUT_DIR / f"{ds}_profile{i}.csv", index=False)

    print(f"Done. Figures saved under: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
