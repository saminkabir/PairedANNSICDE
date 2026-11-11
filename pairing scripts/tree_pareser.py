import pickle
import os
import pandas as pd
import pysvs as ps
from collections import defaultdict
from multiprocessing import Process
import numpy as np

dataset_path = '/data/[username]/similarity-search/dataset/'
pickle_path='/data/[username]/similarity-search/models/results_pickles_v8/'

def read_vecs(filePath):
    return ps.read_vecs(dataset_path + filePath)

def getStoredData(fileName):
    with open(pickle_path+fileName, 'rb') as f:
        I = pickle.load(f)
        return I
    
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
            # print(I.shape)
            combined.update(I[q])
        total_ids=total_ids+len(combined)
        hits = len(gt_set & combined)
        total_hits += hits
    return total_hits / (nq * k_gt), total_ids/(nq * k_gt)

os.chdir('/data/[username]/similarity-search/models/PM-LSH/pmlsh/logs')
    
dataset_infos=['deep', 'glove', 'sun', 'audio', 'millionSong', 'nuswide', 'MNIST', 'notre', 'sift', 'imageNet']

cols=['model','dataset','construction_time (parallel)','construction_time (sequential)','search_time','search_timex_refined (parallel)','search_timex_refined (sequential)','recall','refined_recall','scann_peram','annoy_peram']
results=[]
for dataset in dataset_infos:
    for th in [(25),(50),(100),(200),(400)]:
        tmp='annoy-[dataset]-[th].pkl'
        annoy_file_name=tmp.replace('[th]',str(th))
        annoy_file_name=annoy_file_name.replace('[dataset]',str(dataset))
        pkl_annoy=getStoredData(annoy_file_name)
        rf_recall=pkl_annoy['recallx2']
        recall=pkl_annoy['recall']
        I_annoy=pkl_annoy['I']
        construction_annoy=pkl_annoy['construction-time']
        search_annoy=pkl_annoy['search-time']
        search_x2_annoy=pkl_annoy['search-timex2']
        row=['annoy',dataset,construction_annoy,construction_annoy,search_annoy,search_x2_annoy,search_x2_annoy,recall,rf_recall,'',pkl_annoy['tree']]
        results.append(row)
        
for dataset in dataset_infos:
    for t in [1000,100,10,200,25,400,50,600,75]:
        tmp='Scann-[dataset]-2000-[t]-250000-2.pkl'
        scann_file_name=tmp.replace('[t]',str(t))
        scann_file_name=scann_file_name.replace('[dataset]',str(dataset))
        pkl_scann=getStoredData(scann_file_name)
        peram=str(pkl_scann['num_leaves_'])+'-'+str(pkl_scann['num_leaves_to_search_'])+'-'+str(pkl_scann['training_sample_size_'])+'-'+str(pkl_scann['num_neighbors_'])+'-'+str(pkl_scann['anisotropic_quantization_threshold_'])
        # print(pkl_scann.keys())
        rf_recall=pkl_scann['recallx2']
        recall=pkl_scann['recall']
        I_scann=pkl_scann['I_1']
        construction_scann=pkl_scann['construction_time']
        search_scann=pkl_scann['search_time']
        search_x2_scann=pkl_scann['search_timex2']
        row=['scann',dataset,construction_scann,construction_scann,search_scann,search_x2_scann,search_x2_scann,recall,rf_recall,peram,'']
        results.append(row)
        
for dataset in dataset_infos:
    for th in [(25),(50),(100),(200),(400)]:
        tmp='annoy-[dataset]-[th].pkl'
        annoy_file_name=tmp.replace('[th]',str(th))
        annoy_file_name=annoy_file_name.replace('[dataset]',str(dataset))
        pkl_annoy=getStoredData(annoy_file_name)
        rf_recall=pkl_annoy['recallx2']
        recall=pkl_annoy['recall']
        I_annoy=pkl_annoy['I']
        construction_annoy=pkl_annoy['construction-time']
        search_annoy=pkl_annoy['search-time']
        search_x2_annoy=pkl_annoy['search-timex2']
        row=['annoy',dataset,construction_annoy,construction_annoy,search_annoy,search_x2_annoy,search_x2_annoy,recall,rf_recall]
        # results.append(row)
        for t in [1000,100,10,200,25,400,50,600,75]:
            tmp='Scann-[dataset]-2000-[t]-250000-2.pkl'
            scann_file_name=tmp.replace('[t]',str(t))
            scann_file_name=scann_file_name.replace('[dataset]',str(dataset))
            pkl_scann=getStoredData(scann_file_name)
            peram=str(pkl_scann['num_leaves_'])+'-'+str(pkl_scann['num_leaves_to_search_'])+'-'+str(pkl_scann['training_sample_size_'])+'-'+str(pkl_scann['num_neighbors_'])+'-'+str(pkl_scann['anisotropic_quantization_threshold_'])
            # print(pkl_scann.keys())
            rf_recall=pkl_scann['recallx2']
            recall=pkl_scann['recall']
            I_scann=pkl_scann['I_1']
            construction_scann=pkl_scann['construction_time']
            search_scann=pkl_scann['search_time']
            search_x2_scann=pkl_scann['search_timex2']
            row=['scann',dataset,construction_scann,construction_scann,search_scann,search_x2_scann,search_x2_scann,recall,rf_recall]
            # results.append(row)
            xb,xq,gt=get_data_generic(dataset)
            rf_com_recall=refined_recall([I_annoy,I_scann],gt)[0]
            row=['scann-annoy',dataset,max(construction_annoy,construction_scann),construction_annoy+construction_scann,'-',max(search_x2_annoy,search_x2_scann),search_x2_annoy+search_x2_scann,'-',rf_com_recall,peram,pkl_annoy['tree']]
            results.append(row)
            
df_out = pd.DataFrame(results, columns=cols)
df_out.to_csv('tree_results.csv', index=False) 
        
        
    
    


