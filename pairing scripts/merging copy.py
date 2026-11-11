import pickle
import os
import pandas as pd
import utils
import pysvs as ps
from collections import defaultdict
os.chdir('/data/[username]/similarity-search/models/results_pickles_v7')
dataset_path = '/data/[username]/similarity-search/dataset/'
pickle_path='/data/[username]/similarity-search/models/results_pickles_v7/'

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

def compute_combined_recall(gt, I_list, k_gt):
    recall1, multiplier = refined_recall(gt,I_list,k_gt)
    recall2= compute_intersection_fill_recall(gt,I_list,k_gt)
    return recall1, multiplier, recall2



dataset_infoss=['deep', 'glove', 'sun', 'audio', 'millionSong', 'nuswide', 'MNIST', 'notre', 'sift', 'imageNet']

results=[]
for dataset in dataset_infoss:
    qc=200
    if dataset=='sift':
        qc=10000
    xb,xq,gt=get_data_generic(dataset)
    I1=utils.getStoredData('NSG-'+dataset+'-8-32-.pkl')
    I2=utils.getStoredData('FLATNAV-'+dataset+'-32-16-32.pkl')
    construction_time_2=I2['construction_time']
    construction_time_1=I1['training-time']
    combined_recall,refine_multiplier,recall_intersection=compute_combined_recall(gt,[I1['I'],I2['I']],gt.shape[1])
    recall1r=I1['recallx'+str(2)]
    recall1=I1['recall@']
    search_time1=I1['search-time']
    # print(I1)
    search_time1r=I1['search-timex2']
    recall2r=I2['recallx'+str(2)]
    recall2=I2['recall']
    search_time2=I2['search_time']
    search_time2r=I2['search_timex2']
    result=[]
    if recall1>recall2:
        result.extend([search_time1,recall1])
    else:
        result.extend([search_time2,recall2])
    if recall1r>recall2r:
        result.extend([search_time1r,recall1r,construction_time_1])
    else:
        result.extend([search_time2r,recall2r,construction_time_2])
    result.extend([combined_recall,max(construction_time_1,construction_time_2),construction_time_1+construction_time_2,qc/search_time1,qc/search_time2,qc/(search_time1+search_time2),qc/max(search_time1,search_time2)])
    # print(result)
    results.append(result)
print('settings1=',results)

results=[]
for dataset in dataset_infoss:
    qc=200
    if dataset=='sift':
        qc=10000
    xb,xq,gt=get_data_generic(dataset)
    I1=utils.getStoredData('NSG-'+dataset+'-16-32-.pkl')
    I2=utils.getStoredData('FLATNAV-'+dataset+'-32-16-32.pkl')
    construction_time_2=I2['construction_time']
    construction_time_1=I1['training-time']
    combined_recall,refine_multiplier,recall_intersection=compute_combined_recall(gt,[I1['I'],I2['I']],gt.shape[1])
    recall1r=I1['recallx'+str(2)]
    recall1=I1['recall@']
    search_time1=I1['search-time']
    # print(I1)
    search_time1r=I1['search-timex2']
    recall2r=I2['recallx'+str(2)]
    recall2=I2['recall']
    search_time2=I2['search_time']
    search_time2r=I2['search_timex2']
    result=[]
    if recall1>recall2:
        result.extend([search_time1,recall1])
    else:
        result.extend([search_time2,recall2])
    if recall1r>recall2r:
        result.extend([search_time1r,recall1r,construction_time_1])
    else:
        result.extend([search_time2r,recall2r,construction_time_2])
    result.extend([combined_recall,max(construction_time_1,construction_time_2),construction_time_1+construction_time_2,qc/search_time1,qc/search_time2,qc/(search_time1+search_time2),qc/max(search_time1,search_time2)])
    # print(result)
    results.append(result)
print('settings2=',results)