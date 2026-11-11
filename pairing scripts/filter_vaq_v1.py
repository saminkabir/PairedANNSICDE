import pandas as pd

# Load your CSV
df = pd.read_csv("all_possible_standalone_vaq_v1.csv")

def best_file_by_recall(given_dataset_name, prefix):
    """
    Returns the fileName with highest recall (refinementx=1)
    for a given dataset and prefix.
    """
    filtered = df[
        (df["refinementx"] == 1) &
        (df["dataset"] == given_dataset_name) &
        (df["fileName"].str.contains(prefix, na=False))
    ]
    
    if filtered.empty:
        return None
    
    idx = filtered["recall"].idxmax()
    return filtered.loc[idx, "fileName"]


def get_info_by_peram(file_name, refinement):
    """
    Given an exact fileName and refinement level,
    return recall, qps, and construction_time.
    """
    filtered = df[
        (df["refinementx"] == refinement) &
        (df["fileName"] == file_name)
    ]
    
    if filtered.empty:
        return None, None, None
    
    row = filtered.iloc[0]
    return row["recall"], row["qps"], row["construction_time"]


# Dataset list
datasets = [
    'deep_hundred', 'glove_hundred', 'sun_hundred', 'audio_hundred',
    'millionSong_hundred', 'nuswide_hundred', 'MNIST_hundred',
    'notre_hundred', 'sift_twenty', 'imageNet_hundred'
]

# Parameter prefixes
prefixes = ['64m8', '64m16', '128m32', '128m16', '256m32', '256m64']

results = []

for dataset in datasets:
    # query size 20 for sift, otherwise 100
    qs = 10000 if dataset == 'sift_twenty' else 200
    
    for prfx in prefixes:
        
        best_peram = best_file_by_recall(dataset, prfx)
        if best_peram is None:
            print(dataset,prfx)
            continue
        
        # Refinement levels 1..4
        recall1, qps1, ct1 = get_info_by_peram(best_peram, 1)
        recall2, qps2, ct2 = get_info_by_peram(best_peram, 2)
        recall3, qps3, ct3 = get_info_by_peram(best_peram, 3)
        recall4, qps4, ct4 = get_info_by_peram(best_peram, 4)

        result = [
            "VAQ", qs, dataset, best_peram,
            ct1, recall1, qs/(qps1 if qps1 else 1), qps1,
            recall2, qs/(qps2 if qps2 else 1), qps2,
            recall3, qs/(qps3 if qps3 else 1), qps3,
            recall4, qs/(qps4 if qps4 else 1), qps4,
            best_peram
        ]
        
        results.append(result)


# Final CSV column order
columns = [
    "model", "q", "dataset", "parameter", "construction-time",
    "recallx1", "searchx1", "qpsx1",
    "recallx2", "searchx2", "qpsx2",
    "recallx3", "searchx3", "qpsx3",
    "recallx4", "searchx4", "qpsx4",
    "file"
]

# Create dataframe & save
df_out = pd.DataFrame(results, columns=columns)
df_out.to_csv("vaq_best_params_summary.csv", index=False)

print("Saved → vaq_best_params_summary.csv")
