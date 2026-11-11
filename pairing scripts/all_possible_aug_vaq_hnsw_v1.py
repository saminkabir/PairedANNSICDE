import pickle
import os
import pandas as pd
import pysvs as ps
from collections import defaultdict
from multiprocessing import Process

# os.chdir('/data/[username]/similarity-search/models/results_pickles_v7')
dataset_path = '/data/[username]/similarity-search/dataset/'
pickle_path_1='/data/[username]/similarity-search/models/results_pickles_v7/'
pickle_path_2='/data/[username]/similarity-search/models/results_pickles_v8/'


# Load the CSV
file_path = "all_augmentation_standalone_graph_methods.csv"
df = pd.read_csv(file_path)
file_path1 = "all_possible_standalone_vaq_v1.csv"
df1 = pd.read_csv(file_path1)
def calculate_recall_at(ground_truth, I, k1, k2):
    return ps.k_recall_at(ground_truth, I, k1, k2)

def read_vecs(filePath):
    return ps.read_vecs(dataset_path + filePath)

def getStoredData(fileName):
    pickle_path=pickle_path_1+''
    if ('hundred' in fileName or 'twenty' in fileName) and (not 'FLATNAV' in fileName):
        pickle_path=pickle_path_2+''
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

def compute_combined_recall(gt, I_list, k_gt):
    recall1, multiplier = refined_recall(gt,I_list,k_gt)
    recall2= compute_intersection_fill_recall(gt,I_list,k_gt)
    return recall1, multiplier, recall2

def get_all_possible_recall_qps(dataset,refinement):
    filtered_df1=df1[(df1["dataset"] == dataset) & (df1["refinementx"] == refinement)]
    return filtered_df1['fileName'].tolist(),filtered_df1['id'].tolist(),filtered_df1['qps'].tolist(),filtered_df1['construction_time'].tolist()

def gen_all_possible_comb_vaq_hnsw_comb(dataset,ref):
    all_possible_results=[]
    xb,xq,gt=get_data_generic(dataset)
    vaq_file_list,ids,qpss,indexing_times=get_all_possible_recall_qps(dataset,ref)
    data = df[(df["dataset"] == dataset) & (df["model"] == 'HNSW')]
    filtered=df[(df['dataset'] == dataset) & (df['model'] == 'HNSW')]
    file_names_hnsw,training_times_hnsw = filtered["fileName"].tolist(),filtered["construction-time"].tolist()
    for i in range(len(vaq_file_list)):
        k=20
        qn=200
        if dataset=='sift':
            k=100
            qn=10000
        path1='/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-[id]/results/'.replace('[id]',str(ids[i]))
        path2='/data/[username]/similarity-search/models/vaq-testing/analyse-mult-vaq-[id]/results/'.replace('[id]',str(ids[i]))
        
        vaq_file_name=vaq_file_list[i]
        raw_vaq_file_name=vaq_file_name+''
        if str(ids[i])=='40':
            vaq_file_name=path2+vaq_file_name.replace('vaq-log-','').replace('.txt','-'+str(k*ref)+'.ivecs')
        else:
            vaq_file_name=path1+vaq_file_name.replace('vaq-log-','').replace('.txt','-'+str(k*ref)+'.ivecs')
        qps_vaq=qpss[i]
        print(vaq_file_name)
        
        
        I_NN=ps.read_vecs(vaq_file_name)
        for j in range(len(file_names_hnsw)):
            file1=file_names_hnsw[j]
            
            pickle_path=pickle_path_1+''
            if ('hundred' in file1 or 'twenty' in file1) and (not 'FLATNAV' in file1):
                pickle_path=pickle_path_2+''
            hnsw_file_path=pickle_path+file1
            I1=getStoredData(file1)
            construction_time_hnsw=training_times_hnsw[j]
            qps_hnsw=qn/I1['search-time']
            combined_recall,refine_multiplier,recall_intersection=compute_combined_recall(gt,[I1['I'],I_NN],gt.shape[1])
            all_possible_results.append([dataset,hnsw_file_path,raw_vaq_file_name,vaq_file_name,min(qps_vaq,qps_hnsw),qn/(I1['search-time']+(qn/qps_vaq)),max(construction_time_hnsw,indexing_times[i]),construction_time_hnsw+indexing_times[i],combined_recall,recall_intersection])
    # print(all_possible_results)
    return all_possible_results

def compute(dataset,dummy):
    all_results=gen_all_possible_comb_vaq_hnsw_comb(dataset,10)
    y=['dataset','hnsw_file_path','raw_vaq_file_path','vaq_file_path','parallal qps','sequential qps','parallel indexing time','sequential indexing time','refined recall','majority voting recall']
    df_out = pd.DataFrame(all_results, columns=y)
    os.chdir('/data/[username]/similarity-search/models/polyvector_analysis_v1')
    # save as CSV
    output_file = 'all_possible_vaq_hnsw_combo_v1'+dataset+'.csv'
    df_out.to_csv(output_file, index=False) 
            
            
datasets = ['deep', 'glove', 'sun', 'audio', 'millionSong', 'nuswide', 'MNIST', 'notre', 'sift', 'imageNet','deep_hundred', 'glove_hundred', 'sun_hundred', 'audio_hundred', 'millionSong_hundred', 'nuswide_hundred', 'MNIST_hundred', 'notre_hundred', 'sift_twenty', 'imageNet_hundred']


if __name__ == "__main__":
    pss=[]
    for dataset in datasets:
        p = Process(target=compute, args=(dataset,'test'))
        pss.append(p)
    for _p in pss:
        _p.start()
    for _p in pss:
        _p.join()    
    
