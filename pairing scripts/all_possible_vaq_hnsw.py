import pickle
import os
import pandas as pd
import pysvs as ps
from collections import defaultdict
from multiprocessing import Process
import numpy as np

# os.chdir('/data/[username]/similarity-search/models/results_pickles_v7')
dataset_path = '/data/[username]/similarity-search/dataset/'
pickle_path='/data/[username]/similarity-search/models/results_pickles_v7/'


# Load the CSV
file_path = "Polyvector results - standalone graph based model's result memory.csv"
df = pd.read_csv(file_path)
file_path1 = "all_possible_standalone_vaq.csv"
df1 = pd.read_csv(file_path1)
def calculate_recall_at(ground_truth, I, k1, k2):
    return ps.k_recall_at(ground_truth, I, k1, k2)

def read_vecs(filePath):
    return ps.read_vecs(dataset_path + filePath)

def getStoredData(fileName):
    with open(pickle_path+fileName, 'rb') as f:
        I = pickle.load(f)
        return I

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

def refined_recall(gt, I_list, k_gt):
    total_hits = 0
    nq = len(gt)
    total_ids=0
    for q in range(nq):
        gt_set = set(gt[q])
        combined = set()
        for I in I_list:
            combined.update(I[q])
        total_ids=total_ids+len(combined)
        hits = len(gt_set & combined)
        total_hits += hits
    return total_hits / (nq * k_gt), total_ids/(nq * k_gt)

def compute_intersection_fill_recall(gt, I_list, k_gt):
    total_hits = 0
    nq = len(gt)
    n_models = len(I_list)

    for q in range(nq):
        gt_set = set(gt[q])
        candidate_lists = [set(I[q]) for I in I_list]

        # Intersection across all models
        intersection = set.intersection(*candidate_lists)

        selected = list(intersection)

        if len(selected) < k_gt:
            # Fill using rank sum among remaining candidates
            rank_sum = defaultdict(int)
            for I in I_list:
                for rank, neighbor in enumerate(I[q]):
                    rank_sum[neighbor] += rank + 1

            # Exclude already in intersection
            remaining = [n for n in rank_sum if n not in intersection]
            remaining_sorted = sorted(remaining, key=lambda x: rank_sum[x])

            selected.extend(remaining_sorted[:k_gt - len(selected)])

        selected = set(selected[:k_gt])
        hits = len(gt_set & selected)
        total_hits += hits

    return total_hits / (nq * k_gt)


def get_merged_approximation_ratio(I1, I2, gt, k, xb, xq):

    nq = xq.shape[0]
    ratios = []
    eps = 1e-12

    for qi in range(nq):
        # Combine candidate IDs from both algorithms
        merged = np.unique(np.concatenate((I1[qi], I2[qi])))

        # Compute distances between query and all merged candidates
        dists = np.linalg.norm(xb[merged] - xq[qi], axis=1)

        # Sort by distance and take top-k
        topk_indices = merged[np.argsort(dists)[:k]]

        dist_pred = np.linalg.norm(xb[topk_indices] - xq[qi], axis=1)
        dist_gt   = np.linalg.norm(xb[gt[qi]] - xq[qi], axis=1)

        ratios_q = (dist_pred[:k]+eps) / (dist_gt[:k] + eps)
        ratios.append(ratios_q)

    return np.mean(ratios)

def compute_approximation_ratio(I, gt, k, xb, xq):
    I = np.asarray(I)[:, :k]
    gt = np.asarray(gt)[:, :k]

    n, d = xq.shape
    eps = 1e-12
    ratios = []

    for qi in range(n):
        q = xq[qi]
        # compute distances for approximate and gt
        d_approx = np.linalg.norm(xb[I[qi]] - q, axis=1)
        d_gt = np.linalg.norm(xb[gt[qi]] - q, axis=1)

        ratio = (d_approx + eps) / (d_gt + eps)
        ratios.append(np.mean(ratio))

    return float(np.mean(ratios))


def compute_combined_recall(gt, I_list, k_gt,xb,xq):
    hnsw_recall=refined_recall(gt,[I_list[0]],k_gt)
    vaq_recall=refined_recall(gt,[I_list[1]],k_gt)
    recall1, multiplier = refined_recall(gt,I_list,k_gt)
    recall2= compute_intersection_fill_recall(gt,I_list,k_gt)
    return hnsw_recall, vaq_recall, recall1, multiplier, recall2, compute_approximation_ratio(I_list[0],gt,k_gt,xb,xq), compute_approximation_ratio(I_list[1],gt,k_gt,xb,xq), get_merged_approximation_ratio(I_list[0],I_list[1],gt,k_gt,xb,xq)

def get_all_possible_recall_qps(dataset,refinement):
    filtered_df1=df1[(df1["dataset"] == dataset) & (df1["refinementx"] == refinement)]
    return filtered_df1['fileName'].tolist(),filtered_df1['id'].tolist(),filtered_df1['qps'].tolist(),filtered_df1['construction_time'].tolist(),filtered_df1['memory'].tolist()

def gen_all_possible_comb_vaq_hnsw_comb(dataset,ref):
    all_possible_results=[]
    xb,xq,gt=get_data_generic(dataset)
    vaq_file_list,ids,qpss,indexing_times,memories=get_all_possible_recall_qps(dataset,ref)
    data = df[(df["dataset"] == dataset) & (df["model"] == 'HNSW')]
    filtered=df[(df['dataset'] == dataset) & (df['model'] == 'HNSW') & (df['flag1']==1)]
    file_names_hnsw,training_times_hnsw,memories_hnsw = filtered["fileName"].tolist(),filtered["construction time"].tolist(),filtered["memory"].tolist()
    for i in range(len(vaq_file_list)):
        k=20
        qn=200
        if dataset=='sift':
            k=100
            qn=10000
        path1='/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-[id]/results/'.replace('[id]',str(ids[i]))
        path2='/data/[username]/similarity-search/models/vaq-testing/analyse-mult-vaq-[id]/results/'.replace('[id]',str(ids[i]))
        
        vaq_file_name=vaq_file_list[i]
        if str(ids[i])=='40':
            vaq_file_name=path2+vaq_file_name.replace('vaq-log-','').replace('.txt','-'+str(k*ref)+'.ivecs')
        else:
            vaq_file_name=path1+vaq_file_name.replace('vaq-log-','').replace('.txt','-'+str(k*ref)+'.ivecs')
        qps_vaq=qpss[i]
        print(vaq_file_name)
        
        
        I_NN=ps.read_vecs(vaq_file_name)
        for j in range(len(file_names_hnsw)):
            file1=file_names_hnsw[j]
            hnsw_file_path=pickle_path+file1
            I1=getStoredData(file1)
            construction_time_hnsw=training_times_hnsw[j]
            memory_hnsw=memories_hnsw[j]
            qps_hnsw=qn/I1['search-time']
            hnsw_recall, vaq_recall, combined_recall,refine_multiplier,recall_intersection,ratio1,ratio2,ratio=compute_combined_recall(gt,[I1['I'],I_NN],gt.shape[1],xb,xq)
            hnsw_recall=hnsw_recall[0]
            vaq_recall=vaq_recall[0]
            all_possible_results.append([dataset,hnsw_file_path,vaq_file_name,min(qps_vaq,qps_hnsw),qn/(I1['search-time']+(qn/qps_vaq)),max(construction_time_hnsw,indexing_times[i]),construction_time_hnsw+indexing_times[i],memory_hnsw+memories[i],combined_recall,recall_intersection,hnsw_recall,vaq_recall,ratio1,ratio2,ratio])
    # print(all_possible_results)
    return all_possible_results

def compute(dataset,dummy):
    all_results=gen_all_possible_comb_vaq_hnsw_comb(dataset,10)
    y=['dataset','hnsw_file_path','vaq_file_path','parallal qps','sequential qps','parallel indexing time','sequential indexing time','memory','refined recall','majority voting recall','hnsw_recall','vaq_recall','hnsw ratio','vaq_ratio','merged_ratio']
    df_out = pd.DataFrame(all_results, columns=y)
    os.chdir('/data/[username]/similarity-search/models/polyvector_analysis_v1')
    # save as CSV
    output_file = 'all_possible_vaq_hnsw_combo_'+dataset+'.csv'
    df_out.to_csv(output_file, index=False) 
            
            
datasets = ['deep', 'glove', 'sun', 'audio', 'millionSong', 'nuswide', 'notre', 'sift', 'imageNet']
# datasets = ['audio']


if __name__ == "__main__":
    pss=[]
    for dataset in datasets:
        p = Process(target=compute, args=(dataset,'test'))
        pss.append(p)
    for _p in pss:
        _p.start()
    for _p in pss:
        _p.join()    
    
