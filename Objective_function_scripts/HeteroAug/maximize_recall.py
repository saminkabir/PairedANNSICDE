#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Maximize Recall subject to per-dataset latency / build / memory budgets and
draw bar charts (three constraint profiles per dataset).

本脚本针对 VAQ / HNSW / VAQ+HNSW augmentation 数据：

1) HNSW 指标来自 Sheet9：
   - 使用 model == "HNSW" 的行
   - 使用 "recall with 2k refinement"
   - 使用 "search_time without 2k refienement"（总 search time），
     再用每个 dataset 对应的 NQ[dataset] 换算成 qps

2) VAQ standalone 指标来自 sheet: "all_possible_standalone_vaq"

3) VAQ+HNSW augmentation 指标来自 sheet: "all_possible_vaq_hnsw_combo"

统一转换成下列列：
    dataset, combo, qps, recall, construction_time, memory

后续 pipeline（AUTO_PROFILE / 最大 recall / 画图 / 输出 CSV）
和你原来 hash_results 脚本的逻辑保持一致。
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

# 每个 dataset 的查询数（用于把 HNSW total search time 转成 per-query qps）
NQ: Dict[str, int] = {
    "audio": 200, "deep": 200, "glove": 200, "imageNet": 200, "millionSong": 200,
    "MNIST": 200, "notre": 200, "nuswide": 200, "sift_twenty": 10000, "sun": 200
}

# Toggle memory constraint globally
USE_MEMORY = False
PARALLEL_SETTING = True     # 决定 augmentation 的 qps / construction time 取 parallel 还是 sequential
HETEROGENEOUS = True        # 这里没特别含义，保留不影响
AUTO_PROFILE = True
NO_CONSTRAINT = True

if NO_CONSTRAINT:
    assert PARALLEL_SETTING

# 三个 (Tq, Tc, Tm) profile 默认值
DEFAULT_PROFILES: List[Tuple[float, float, float]] = [
    (0.010,  120.0, 64.0),   # Tq=10ms,  Tc=120s, Tm=64
    (0.005,  300.0, 64.0),   # Tq=5ms,   Tc=300s, Tm=64
    (0.0025, 600.0, 128.0),  # Tq=2.5ms, Tc=600s, Tm=128
]

# Per-dataset overrides
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

# 输出目录
if PARALLEL_SETTING:
    if NO_CONSTRAINT:
        OUT_DIR = Path("fig/max_rec_vaq_hnsw_no_constraints")
    else:
        OUT_DIR = Path("fig/max_rec_vaq_hnsw_parallel")
else:
    OUT_DIR = Path("fig/max_rec_vaq_hnsw_sequential")



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
    基于数据自动为每个 dataset 生成 3 个 profile。
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
        #latencies = latencies[latencies < latencies.quantile(0.8)]
        #ctimes = ctimes[ctimes > ctimes.quantile(0.1)]

        try:
            lat_q = latencies.quantile([0.25, 0.4, 0.5]).tolist()
            cst_q = ctimes.quantile([0.25, 0.4, 0.5]).tolist()       
        except Exception:
            assert False
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


# ============ 读取三张表，统一成 dataset / combo / qps / recall / construction_time / memory ============

def load_hnsw_from_sheet9(path: str) -> pd.DataFrame:
    """
    从 Sheet9 读取 HNSW 指标：
        - 只要 model == "HNSW"
        - 使用 recall with 2k refinement
        - 使用 search_time without 2k refienement (总时间)，用每个 dataset 的 NQ[dataset] 换算 qps
    """
    df = pd.read_excel(path, sheet_name="graph_new_datasets_standalone_r")
    df = _clean_colnames(df)

    col_model    = "model"
    col_dataset  = "dataset"
    col_rec_2k   = "recall with 2k refinement"
    #col_t_2k     = "search_time without 2k refienement"
    col_cst      = "construction time"
    col_mem      = "memory" if "memory" in df.columns else None

    df = df[df[col_model].astype(str) == "HNSW"].copy()

    # dataset -> NQ
    ds_series = df[col_dataset].astype(str)
    #num_queries = ds_series.map(NQ)  # 如果某个 dataset 不在 NQ，会得到 NaN

    #total_time = _coerce_numeric(df[col_t_2k])
    # 避免除以 0 或 NaN
    #total_time = total_time.replace(0, pd.NA)

    # qps = NQ[dataset] / total_time
    #qps = num_queries / total_time

    out = pd.DataFrame({
        "dataset": ds_series,
        "combo":   "HNSW",  # 如需区分不同 HNSW 配置可以在这里加上 fileName 等
        "qps":     _coerce_numeric(df["qps with 2k refienement"]),
        "recall":  _coerce_numeric(df[col_rec_2k]),
        "construction_time": _coerce_numeric(df[col_cst]),
        "memory": _coerce_numeric(df[col_mem]) if col_mem else 0.0,
    })


    out = out[
        out["qps"].notna() &
        out["construction_time"].notna() &
        out["recall"].notna()
    ].reset_index(drop=True)

    #out = out[out['recall'] >= out.groupby('dataset')['recall'].transform(lambda x: x.quantile(0.15))]
    return out


def load_vaq_standalone(path: str) -> pd.DataFrame:
    """
    从 all_possible_standalone_vaq 读取 VAQ standalone：
    combo 名称设为：VAQ[b<bits>,r<refinementx>]
    """
    df = pd.read_excel(path, sheet_name="all_possible_standalone_vaq_new")
    df = _clean_colnames(df)

    col_ds   = "dataset"
    col_rec  = "recall"
    col_qps  = "qps"
    col_cst  = "construction_time"
    col_mem  = "memory" if "memory" in df.columns else None
    col_refx = "refinementx"
    col_bits = "bits"

    bits = df[col_bits].astype(str)
    refx = df[col_refx].astype(str)
    #combo_name = "VAQ[" + "b" + bits + ",r" + refx + "]"
    combo_name = "VAQ"

    out = pd.DataFrame({
        "dataset": df[col_ds].astype(str),
        "combo":   combo_name,
        "qps":     _coerce_numeric(df[col_qps]),
        "recall":  _coerce_numeric(df[col_rec]),
        "construction_time": _coerce_numeric(df[col_cst]),
        "memory": _coerce_numeric(df[col_mem]) if col_mem else 0.0,
    })

    out = out[
        out["qps"].notna() &
        out["construction_time"].notna() &
        out["recall"].notna()
    ].reset_index(drop=True)

    #out = out[out['recall'] >= out.groupby('dataset')['recall'].transform(lambda x: x.quantile(0.15))]
    return out


def load_vaq_hnsw_combo(path: str) -> pd.DataFrame:
    """
    从 all_possible_vaq_hnsw_combo 读取 VAQ+HNSW augmentation：
    使用 refined recall，
    qps / construction time 根据 PARALLEL_SETTING 选择 parallel 或 sequential。
    """
    df = pd.read_excel(path, sheet_name="all_possible_vaq_hnsw_combo_new")
    df = _clean_colnames(df)

    col_ds    = "dataset"
    col_vaqfp = "vaq_file_path"
    col_qps_p = "parallal qps"
    col_qps_s = "sequential qps"
    col_cst_p = "parallel indexing time"
    col_cst_s = "sequential indexing time"
    col_mem   = "memory"
    col_rec   = "refined recall"

    if PARALLEL_SETTING:
        col_qps = col_qps_p
        col_cst = col_cst_p
        setting_mark = "P"
    else:
        col_qps = col_qps_s
        col_cst = col_cst_s
        setting_mark = "S"

    # vaq_base = df[col_vaqfp].astype(str).apply(
    #     lambda p: Path(p).name if isinstance(p, str) else str(p)
    # )

    #combo_name = "HNSW+VAQ[" + setting_mark + ":" + vaq_base + "]"
    combo_name = "HNSW+VAQ"

    out = pd.DataFrame({
        "dataset": df[col_ds].astype(str),
        "combo":   combo_name,
        "qps":     _coerce_numeric(df[col_qps]),
        "recall":  _coerce_numeric(df[col_rec]),
        "construction_time": _coerce_numeric(df[col_cst]),
        #"memory": _coerce_numeric(df[col_mem]),
    })

    out = out[
        out["qps"].notna() &
        out["construction_time"].notna() &
        out["recall"].notna()
    ].reset_index(drop=True)

    #out = out[out['recall'] >= out.groupby('dataset')['recall'].transform(lambda x: x.quantile(0.3))]

    return out


def load_all_unified(path: str) -> pd.DataFrame:
    """
    合并 HNSW / VAQ standalone / VAQ+HNSW combo 成统一表。
    """
    df_hnsw  = load_hnsw_from_sheet9(path)
    df_vaq   = load_vaq_standalone(path)
    df_combo = load_vaq_hnsw_combo(path)

    df = pd.concat([df_hnsw, df_vaq, df_combo], ignore_index=True)

    df["qps"] = pd.to_numeric(df["qps"], errors="coerce")
    df["construction_time"] = pd.to_numeric(df["construction_time"], errors="coerce")
    df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
    df["memory"] = pd.to_numeric(df["memory"], errors="coerce").fillna(0.0)
    df["combo"] = df["combo"].astype(str).str.replace(r"\s*\+\s*", " + ", regex=True)

    df = df[
        df["qps"].notna() &
        df["construction_time"].notna() &
        df["recall"].notna()
    ].reset_index(drop=True)

    return df


# =========================
# ----- Core compute ------
# =========================

def _best_recall_for_rows(
    rows: pd.DataFrame,
    Tq: float,
    Tc: float,
    Tm: float,
    use_memory: bool
) -> tuple[float, bool]:
    best_rec = 0.0
    feasible = False

    min_qps = (1.0 / Tq) if Tq > 0 else float("inf")

    for _, r in rows.iterrows():
        qps = r.get("qps")
        cst = r.get("construction_time")
        rcl = r.get("recall")
        mem = r.get("memory", 0.0) if use_memory else 0.0

        if not NO_CONSTRAINT:
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
        f"Max Recall under Budgets — {dataset} (Memory: {'ON' if USE_MEMORY else 'OFF'})",
        y=1.02,
        fontsize=13,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


from pathlib import Path
import pandas as pd
import numpy as np


def _prepare_curves_per_combo(df_dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    对单个 dataset 的数据，根据 combo 分组，返回:
        combo -> df_curve

    要求 df_dataset 至少包含:
        combo, recall, search_time, construction_time
    """
    curves: dict[str, pd.DataFrame] = {}
    for combo, g in df_dataset.groupby("combo"):
        # 只保留我们关心的列，避免后面 groupby / dropna 出错
        tmp = g[["recall", "search_time", "construction_time"]].dropna(subset=["recall"])
        if tmp.empty:
            continue
        curves[str(combo)] = tmp.copy()
    return curves


def _domination_ratio_recall_x(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric_col: str,
    num_grid: int = 200,
    larger_is_better: bool = False,
) -> float:
    """
    对同一个 dataset 中的两条曲线 A 和 B，使用 recall 作为 x 轴，
    对 metric_col 进行线性插值，计算 A dominates B 的比例。

    - x: recall
    - y: metric_col (例如 search_time 或 construction_time)
    - 如果 larger_is_better=False，则 y 越小越好（时间越短越好）
      此时 A dominates B 的条件是 y_A < y_B

    返回值：A dominates B 的比例 in [0, 1]。
    若两条曲线在 recall 上无交集，则 recall 整体更高者视为完全 dominate。
    """
    # 只保留 recall 和目标 metric
    df_a = df_a[["recall", metric_col]].dropna().copy()
    df_b = df_b[["recall", metric_col]].dropna().copy()

    if df_a.empty or df_b.empty:
        return np.nan

    # 按 recall 排序，并在相同 recall 时保留“更好”的点
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

    # 只在 recall 交集区间上比较
    left = max(xa.min(), xb.min())
    right = min(xa.max(), xb.max())

    # 无 overlap：recall 整体更高者视为完全 dominate
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        if xa.min() > xb.max():
            return 1.0
        elif xa.max() < xb.min():
            return 0.0
        else:
            return np.nan

    grid = np.linspace(left, right, num_grid)

    ya_grid = np.interp(grid, xa, ya)
    yb_grid = np.interp(grid, xb, yb)

    if larger_is_better:
        dom = np.mean(ya_grid > yb_grid)
    else:
        dom = np.mean(ya_grid < yb_grid)
    return float(dom)

def _compute_domination_matrix_for_dataset(
    df_dataset: pd.DataFrame,
    metric_col: str,
    num_grid: int = 200,
    larger_is_better: bool = False,
) -> pd.DataFrame:
    curves = _prepare_curves_per_combo(df_dataset)
    combos = sorted(curves.keys())
    n = len(combos)
    if n == 0:
        return pd.DataFrame()

    mat = np.full((n, n), np.nan, dtype=float)

    for i, ci in enumerate(combos):
        df_i = curves[ci]
        for j, cj in enumerate(combos):
            df_j = curves[cj]

            dom_ij = _domination_ratio_recall_x(
                df_a=df_i,
                df_b=df_j,
                metric_col=metric_col,
                num_grid=num_grid,
                larger_is_better=larger_is_better,
            )

            if pd.isna(dom_ij):
                print(f"[NaN] {metric_col}: {ci} vs {cj}")
                print(f"  {ci}: {len(df_i)} rows, recall range = "
                      f"{df_i['recall'].min()} ~ {df_i['recall'].max()}")
                print(f"  {cj}: {len(df_j)} rows, recall range = "
                      f"{df_j['recall'].min()} ~ {df_j['recall'].max()}")

            mat[i, j] = dom_ij

    return pd.DataFrame(mat, index=combos, columns=combos)

def compute_and_save_domination_tables(
    df_all: pd.DataFrame,
    out_dir: Path | str = Path("domination"),
    num_grid: int = 200,
):
    """
    给定一个包含所有 dataset 的 DataFrame（列至少包含:
        dataset, combo, recall, qps, construction_time
    ）：

    1) 先在副本中构造 search_time = 1 / qps （秒/query）
    2) 对每个 dataset，分别生成两张 domination 矩阵：
        - 基于 search_time   →  <dataset>_domination_search_time.csv
        - 基于 construction_time → <dataset>_domination_construction_time.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df_all.copy()

    # 计算 search_time = 1 / qps
    df["qps"] = pd.to_numeric(df["qps"], errors="coerce")
    df["search_time"] = np.where(
        (df["qps"] > 0) & df["qps"].notna(),
        1.0 / df["qps"],
        np.nan,
    )

    # 确保 construction_time 是数值
    df["construction_time"] = pd.to_numeric(df["construction_time"], errors="coerce")

    for dataset, g in df.groupby("dataset"):
        print(f"[Domination] Processing dataset: {dataset}")
        # ---- 1) 基于 search_time 的 domination ----
        dom_search = _compute_domination_matrix_for_dataset(
            df_dataset=g,
            metric_col="search_time",
            num_grid=num_grid,
            larger_is_better=False,  # 时间越小越好
        )
        if not dom_search.empty:
            dom_search.to_csv(
                out_dir / f"{dataset}_domination_search_time.csv",
                index=True,
            )

        # ---- 2) 基于 construction_time 的 domination ----
        dom_build = _compute_domination_matrix_for_dataset(
            df_dataset=g,
            metric_col="construction_time",
            num_grid=num_grid,
            larger_is_better=False,  # build 时间越小越好
        )
        if not dom_build.empty:
            dom_build.to_csv(
                out_dir / f"{dataset}_domination_construction_time.csv",
                index=True,
            )


def compute_average_domination_matrix(
    domination_dir: Path | str,
    metric: str = "search_time",
    output_path: str | Path = "domination_avg.csv"
):
    """
    从 domination_dir 中读取所有 dataset 的 domination CSV，
    对齐后求平均，得到一个 overall domination matrix。

    metric:
        - "search_time"
        - "construction_time"

    输出:
        domination_avg.csv
    """
    domination_dir = Path(domination_dir)
    files = sorted(domination_dir.glob(f"*domination_{metric}.csv"))
    if not files:
        print(f"No domination files found for metric={metric}")
        return

    dom_mats = []
    combo_index = ['HNSW', "VAQ"]

    for f in files:
        df = pd.read_csv(f, index_col=0)
        df = df.astype(float)

        # 初次建立 row/col 顺序
        if combo_index is None:
            combo_index = df.index.tolist()
        else:
            # 对齐：如果矩阵维度不同，补全缺失 combo
            df = df.reindex(index=combo_index, columns=combo_index)

        if df.isna().values.any():
            print(f"Skip {f} due to NaN")
            continue

        dom_mats.append(df)

    # 求平均
    avg_mat = sum(dom_mats) / len(dom_mats)
    #print(dom_mats)

    avg_mat.to_csv(output_path, index=True)
    print(f"[Average Domination] saved → {output_path}")


# =========================
# ---------- Run ----------
# =========================

def main():
    df = load_all_unified(EXCEL_PATH)
    if df.empty:
        print("No usable rows from VAQ/HNSW sheets.")
        return
    
    suffix = "_parallel" if PARALLEL_SETTING else "_sequential"
    compute_and_save_domination_tables(df, out_dir=f"domination_vaq_hnsw/details{suffix}")
    compute_average_domination_matrix(
        domination_dir=f"domination_vaq_hnsw/details{suffix}",
        metric="search_time",
        output_path=f"domination_vaq_hnsw/avg_domination_search_time{suffix}.csv"
    )

    compute_average_domination_matrix(
        domination_dir=f"domination_vaq_hnsw/details{suffix}",
        metric="construction_time",
        output_path=f"domination_vaq_hnsw/avg_domination_construction_time{suffix}.csv"
    )
    #exit(0)

    datasets = sorted(df["dataset"].unique().tolist())

    base_profiles: Dict[str, List[Tuple[float, float, float]]] = {}
    for ds in datasets:
        base_profiles[ds] = PROFILES.get(ds, DEFAULT_PROFILES)

    if AUTO_PROFILE:
        profiles_per_dataset = build_profiles_per_dataset_auto(df, base_profiles)
    else:
        profiles_per_dataset = base_profiles

    print(profiles_per_dataset)

    results = compute_max_rec_per_dataset(df, profiles_per_dataset, use_memory=USE_MEMORY)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        per_profiles = results.get(ds, [])
        profiles = profiles_per_dataset.get(ds, DEFAULT_PROFILES)
        plot_bars_for_dataset(ds, per_profiles, profiles, OUT_DIR)

    for ds, per_profiles in results.items():
        for i, dd in enumerate(per_profiles, start=1):
            if not dd.empty:
                dd.to_csv(OUT_DIR / f"{ds}_profile{i}.csv", index=False)

    print(f"Done. Figures saved under: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
