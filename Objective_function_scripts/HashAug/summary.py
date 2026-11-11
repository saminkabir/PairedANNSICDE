import os
import pandas as pd
import re
from collections import defaultdict

# ==============================
# 配置
# ==============================
DIM_THRESHOLD = 512


for PARALLEL_SETTING in [True, False]:
    #for QUERY_HARDNESS, OOD in [(True, None), (False, None), (None, True), (None, False), (None, None)]:
    for QUERY_HARDNESS, OOD, HIGH_DIM in [(None, None, True), (None, None, False)]:

        postfix = ""
        if QUERY_HARDNESS is not None:
            postfix = "_hard" if QUERY_HARDNESS else "_easy"
        assert not(OOD is not None and QUERY_HARDNESS is not None)
        if OOD is not None:
            postfix = "_ood" if OOD else "_iid"
        if HIGH_DIM is not None:
            postfix = "_highdim" if HIGH_DIM else "_lowdim" 

        if PARALLEL_SETTING:
            data_dir = "./fig/max_rec_parallel"
            out_path = f"./fig/max_rec_parallel/summary{postfix}.csv"
        else:
            data_dir = "./fig/max_rec_sequential"
            out_path = f"./fig/max_rec_sequential/summary{postfix}.csv"

        pattern = re.compile(r"(.+)_profile_?(\d+)\.csv")

        # ==============================
        # 存储结构
        # dataset_best[constraint][dataset] -> {"standalone": ..., "pairing": ...}
        # constraint ∈ {"tight", "relaxed"}
        # ==============================
        dataset_best = defaultdict(
            lambda: defaultdict(lambda: {"standalone": -1.0, "pairing": -1.0})
        )

        ds_property = pd.read_csv("../dataset property.csv")
        if QUERY_HARDNESS is not None:
            rc_median = ds_property['rc'].median()
            if QUERY_HARDNESS:
                ds_property = ds_property[ds_property['rc'] <= rc_median ]
            else:
                ds_property = ds_property[ds_property['rc'] > rc_median ]
        if OOD is not None:
            if OOD:
                ds_property = ds_property[ds_property['Distribution'] == "OOD" ]
            else:
                ds_property = ds_property[ds_property['Distribution'] == "ID" ]
        if HIGH_DIM is not None:
            if HIGH_DIM:
                ds_property = ds_property[ds_property['Dimensionality'] >= DIM_THRESHOLD ]
            else:
                ds_property = ds_property[ds_property['Dimensionality'] < DIM_THRESHOLD ]

        print(f"We have {len(ds_property['Dataset'].unique())}")


        # ==============================
        # 处理所有 CSV
        # ==============================
        for fname in os.listdir(data_dir):
            m = pattern.match(fname)
            if not m:
                continue

            dataset = m.group(1)
            profile_id = m.group(2)
            if dataset not in ds_property["Dataset"].unique():
                continue


            # 映射 profile -> constraint
            if profile_id == "1":
                constraint = "tight"
            elif profile_id == "3":
                constraint = "relaxed"
            else:
                # 只关心 profile1 (tight) 和 profile3 (relaxed)
                continue

            df = pd.read_csv(os.path.join(data_dir, fname))

            for _, row in df.iterrows():
                combo = row["combo"]
                acc = row["best_recall"]
                feasible = row["feasible"]

                if not feasible:
                    continue

                # 统计组合里有多少个 model
                num_models = combo.count("+") + 1

                # 只要 standalone (1 个模型) 和 pairing (2 个模型)
                if num_models == 1:
                    category = "standalone"
                elif num_models == 2:
                    category = "pairing"
                else:
                    # 跳过三元组合（如 NSG + FLATNAV + HNSW）
                    continue

                # 更新该 dataset 在该 constraint 下该 category 的 best accuracy
                dataset_best[constraint][dataset][category] = max(
                    dataset_best[constraint][dataset][category], acc
                )

        # ==============================
        # Compute summary
        # ==============================
        results = {
            "constraint": [],
            "model": [],
            "average_best_recall": [],
            "win": [],
        }

        categories = ["standalone", "pairing"]
        constraints = ["tight", "relaxed"]

        for constraint in constraints:
            datasets_for_constraint = dataset_best.get(constraint, {})

            # average best recall (exclude -1)
            for cat in categories:
                recalls = [
                    v[cat]
                    for v in datasets_for_constraint.values()
                    if v[cat] >= 0
                ]
                avg = sum(recalls) / len(recalls) if recalls else 0.0

                results["constraint"].append(constraint)
                results["model"].append(cat)
                results["average_best_recall"].append(avg)

            # compute wins (在该 constraint 下 standalone vs pairing 的胜出次数)
            standalone_win = 0
            pairing_win = 0

            for ds, vals in datasets_for_constraint.items():
                s = vals["standalone"]
                p = vals["pairing"]
                if s < 0 or p < 0:
                    continue
                if s > p:
                    standalone_win += 1
                elif p > s:
                    pairing_win += 1
                # tie 的情况不加分

            # 把 win 写到对应两行里
            # 行顺序是：standalone, pairing
            results["win"].append(standalone_win)  # 对应 standalone
            results["win"].append(pairing_win)     # 对应 pairing

        # ==============================
        # Save summary.csv
        # ==============================
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(out_path, index=False)

        print(f"{out_path} generated!")
        print(summary_df)
