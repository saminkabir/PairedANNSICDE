import os
import re
import pysvs as ps
import csv

dataset_path = '/data/[username]/similarity-search/dataset/'

peram_mp=dict()

params = {
    "log8":  {"lowDim": 10, "c": 1.5,  "T": 0.2, "R_min": 1.0},
    "log9":  {"lowDim": 10, "c": 1.75, "T": 0.2, "R_min": 1.0},
    "log10": {"lowDim": 10, "c": 1.25, "T": 0.2, "R_min": 1.0},

    "log11": {"lowDim": 10, "c": 1.5,  "T": 0.2, "R_min": 20.0},
    "log12": {"lowDim": 10, "c": 1.5,  "T": 0.2, "R_min": 300},

    "log13": {"lowDim": 10, "c": 1.5,  "T": 0.1, "R_min": 1.0},
    "log14": {"lowDim": 10, "c": 1.5,  "T": 0.4, "R_min": 1.0},

    "log15": {"lowDim": 5, "c": 1.5,  "T": 0.2, "R_min": 1.0},
    "log16": {"lowDim": 5, "c": 1.75, "T": 0.2, "R_min": 1.0},
    "log17": {"lowDim": 5, "c": 1.25, "T": 0.2, "R_min": 1.0},

    "log18": {"lowDim": 5, "c": 1.5,  "T": 0.2, "R_min": 20.0},
    "log19": {"lowDim": 5, "c": 1.5,  "T": 0.2, "R_min": 300},

    "log20": {"lowDim": 5, "c": 1.5,  "T": 0.1, "R_min": 1.0},
    "log21": {"lowDim": 5, "c": 1.5,  "T": 0.4, "R_min": 1.0},

    "log31": {"lowDim": 5, "c": 1.2,  "T": 0.2, "R_min": 1.0},
    "log32": {"lowDim": 5, "c": 1.15, "T": 0.2, "R_min": 1.0},
    "log30": {"lowDim": 5, "c": 1.1,  "T": 0.2, "R_min": 1.0},

    "log33": {"lowDim": 4, "c": 1.25, "T": 0.2, "R_min": 1.0},
    "log34": {"lowDim": 3, "c": 1.25, "T": 0.2, "R_min": 1.0},
    "log35": {"lowDim": 2, "c": 1.25, "T": 0.2, "R_min": 1.0},
}

for log, p in params.items():
    peram_mp[log] = f"{p['lowDim']}-{p['c']}-{p['T']}-{p['R_min']}"
    
params = {
    0:  {"c": 1.5,  "L": 5,  "K": 10, "beta": 0.1, "R_min": 0.50},
    1:  {"c": 1.75, "L": 5,  "K": 10, "beta": 0.1, "R_min": 0.50},
    2:  {"c": 1.25, "L": 5,  "K": 10, "beta": 0.1, "R_min": 0.50},

    3:  {"c": 1.5,  "L": 10, "K": 10, "beta": 0.1, "R_min": 0.50},
    4:  {"c": 1.5,  "L": 15, "K": 10, "beta": 0.1, "R_min": 0.50},

    5:  {"c": 1.5,  "L": 5,  "K": 12, "beta": 0.1, "R_min": 0.50},
    6:  {"c": 1.5,  "L": 5,  "K": 14, "beta": 0.1, "R_min": 0.50},

    7:  {"c": 1.5,  "L": 5,  "K": 10, "beta": 0.1, "R_min": 6.50},
    8:  {"c": 1.75, "L": 5,  "K": 10, "beta": 0.1, "R_min": 6.50},
    9:  {"c": 1.25, "L": 5,  "K": 10, "beta": 0.1, "R_min": 6.50},

    10: {"c": 1.5,  "L": 5,  "K": 10, "beta": 0.1, "R_min": 300},
    11: {"c": 1.5,  "L": 5,  "K": 10, "beta": 0.1, "R_min": 600},

    12: {"c": 1.5,  "L": 10, "K": 10, "beta": 0.1, "R_min": 6.50},
    13: {"c": 1.5,  "L": 15, "K": 10, "beta": 0.1, "R_min": 6.50},

    14: {"c": 1.5,  "L": 5,  "K": 12, "beta": 0.1, "R_min": 6.50},
    15: {"c": 1.5,  "L": 5,  "K": 14, "beta": 0.1, "R_min": 6.50},
}

for logId, p in params.items():
    peram_mp[logId] = f"{p['c']}-{p['L']}-{p['K']}-{p['beta']}-{p['R_min']}"


def get_peram(k):
    if k in peram_mp:
        return peram_mp[k]
    return k

def calculate_recall_at(ground_truth, I, k1, k2):
    return ps.k_recall_at(ground_truth, I, k1, k2)

def read_vecs(filePath):
    return ps.read_vecs(dataset_path + filePath)

def read_vecs_v1(filePath):
    return ps.read_vecs(filePath)

def get_data_generic(dataset):
    # data we will search through
    xb = read_vecs(dataset+'/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs(dataset+'/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs(dataset+'/groundtruth.ivecs')
    return xb,xq,gt

def refined_recall(I_list, gt):
    total_hits = 0
    nq = min(len(gt),100)
    total_ids=0
    k_gt=gt.shape[1]
    for q in range(nq):
        gt_set = set(gt[q])
        combined = set()
        for I in I_list:
            combined.update(I[q])
        total_ids=total_ids+len(combined)
        hits = len(gt_set & combined)
        total_hits += hits
    return total_hits / (nq * k_gt), total_ids/(nq * k_gt)

os.chdir('/data/[username]/similarity-search/models/PM-LSH/pmlsh/logs')
files=os.listdir()
vecs=os.listdir('/data/[username]/similarity-search/models/PM-LSH/pmlsh/build/')
dataset_infos=['deep_hundred', 'glove_hundred', 'sun_hundred', 'audio_hundred', 'millionSong_hundred', 'nuswide_hundred', 'MNIST_hundred', 'notre_hundred', 'imageNet_hundred','sift_twenty']
# dataset_infos=['audio']
dct_pmlsh=dict()
pmlsh_logids=set()
for dataset in dataset_infos:
    dct_pmlsh[dataset]=[]
    for file in files:
        if dataset in file and '-' in file and '.txt' in file:
            with open(file, "r") as f:
                text = f.read()

                # Define regex patterns
                patterns = {
                    "hashing_time": r"FINISH HASHING WITH TIME:\s*([\d.]+)\s*s",
                    "build_time": r"FINISH BUILDING WITH TIME:\s*([\d.]+)\s*s",
                    "avg_query_time": r"AVG QUERY TIME:\s*([\d.]+)ms",
                    "total_query_time": r"TOTAL QUERY TIME:\s*([\d.]+)ms",
                    "recall": r"AVG RECALL:\s*([\d.]+)",
                    "avg_map": r"AVG MAP:\s*([\d.]+)",
                    "avg_ratio": r"AVG RATIO:\s*([\d.]+)",
                    "avg_cost": r"AVG COST:\s*([\d.]+)",
                    "avg_rounds": r"AVG ROUNDS:\s*([\d.]+)"
                }

                # Extract values
                results = {}
                for key, pattern in patterns.items():
                    match = re.search(pattern, text)
                    if match:
                        results[key] = float(match.group(1))
                results['construction_time']=results['hashing_time']+results['build_time']
                results['qps']=1/(results['avg_query_time']/1000)
                if file.replace('.txt','.ivecs') in vecs:
                    I=ps.read_vecs('/data/[username]/similarity-search/models/PM-LSH/pmlsh/build/' + file.replace('.txt','.ivecs'))
                    # I = I + 1
                    results['I']=I
                    pmlsh_logids.add(file.replace(dataset+'-','').replace('.txt',''))
                    results['logId']=file.replace(dataset+'-','').replace('.txt','')
                    dct_pmlsh[dataset].append(results)
print(dct_pmlsh)

vecs=os.listdir('/data/[username]/similarity-search/models/DB-LSH2/DB-LSH/dbLSH/build')

os.chdir('/data/[username]/similarity-search/models/DB-LSH2/DB-LSH/dbLSH/scripts')
dct_dblsh=dict()

with open('merged3.txt', "r") as f:
    text = f.read()
    logs=text.split('Using DB-LSH for ')[1:]
    for log in logs:
        text=log
        patterns = {
            "computing_time": r"COMPUTING TIME:\s*([\d.]+)s",
            "num_tables": r"THERE ARE\s+(\d+)\s+[\d\-A-Za-z]+\s+HASH TABLES",
            "tree_times": r"R\*-Tree has been built\. Elapsed time:\s*([\d.]+)s",
            "build_time": r"BUILDING TIME:\s*([\d.]+)s",
            "model_id": r"model_id=\s*([^\s]+)",
            "k": r"k=\s*(\d+)",
            "avg_query_time": r"AVG QUERY TIME:\s*([\d.]+)ms",
            "total_query_time": r"TOTAL QUERY TIME:\s*([\d.]+)ms",
            "recall": r"AVG RECALL:\s*([\d.]+)",
            "avg_map": r"AVG MAP:\s*([\d.]+)",
            "avg_ratio": r"AVG RATIO:\s*([\d.]+)",
            "avg_cost": r"AVG COST:\s*([\d.]+)",
            "avg_rounds": r"AVG ROUNDS:\s*([\d.]+)"
        }
        dataset=log.split(' ')[0]
        results = {}
        for key, pattern in patterns.items():
            if key == "tree_times":
                results[key] = [float(x) for x in re.findall(pattern, text)]
            else:
                match = re.search(pattern, text)
                if match:
                    results[key] = float(match.group(1)) if key not in ("model_id",) else match.group(1)
        
        results['construction_time']=results['computing_time']+results['build_time']
        results['qps']=1/(results['avg_query_time']/1000)
        I=ps.read_vecs('/data/[username]/similarity-search/models/DB-LSH2/DB-LSH/dbLSH/build/' + results['model_id'])
        results['I']=I
        if not dataset in dct_dblsh:
            dct_dblsh[dataset]=[]
        results['logId']=len(dct_dblsh[dataset])
        dct_dblsh[dataset].append(results)
print(dct_dblsh.keys(), dct_pmlsh.keys())
all_results=[]

for dataset in dataset_infos:
    xb,xq,gt=get_data_generic(dataset)
    pmlsh_results=dct_pmlsh[dataset]
    for result in pmlsh_results:
        rc=refined_recall([result['I']],gt)[0]
        all_results.append(['PM-LSH',dataset,rc,result['qps'],result['construction_time'],result['qps'],result['construction_time'],get_peram(result['logId']),''])
    dblsh_results=dct_dblsh[dataset]
    for result in dblsh_results:
        rc=refined_recall([result['I']],gt)[0]
        all_results.append(['DB-LSH',dataset,rc,result['qps'],result['construction_time'],result['qps'],result['construction_time'],'',get_peram(result['logId'])])
    for pmlsh_result in pmlsh_results:
        for dblsh_result in dblsh_results:
            rc=refined_recall([pmlsh_result['I'],dblsh_result['I']],gt)[0]
            all_results.append(['DB-LSH+PM-LSH',dataset,rc,min(dblsh_result['qps'],pmlsh_result['qps']),max(dblsh_result['construction_time'],pmlsh_result['construction_time']),1/((dblsh_result['avg_query_time']+pmlsh_result['avg_query_time'])/1000),(dblsh_result['construction_time']+pmlsh_result['construction_time']),get_peram(pmlsh_result['logId']),get_peram(dblsh_result['logId'])])
        
        
os.chdir('/data/[username]/similarity-search/models/polyvector_analysis_v1')
header = ['Method', 'Dataset','Refined Recall', 'QPS (parallel)', 'Construction Time (parallel)', 'QPS (sequential)', 'Construction Time (sequential)', 'PM-LSH peram', 'DB-LSH peram']

with open('hash_results_v1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_results)
    
print(pmlsh_logids)