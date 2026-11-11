import pickle
import os
import pandas as pd
import pysvs as ps
from multiprocessing import Process
from collections import defaultdict
# os.chdir('/data/[username]/similarity-search/models/results_pickles_v7')
dataset_path = '/data/[username]/similarity-search/dataset/'
pickle_path_1='/data/[username]/similarity-search/models/results_pickles_v7/'
pickle_path_2='/data/[username]/similarity-search/models/results_pickles_v8/'

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
    # print(dataset)
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

        # Step 1: intersection of all models
        intersection = set.intersection(*candidate_lists)
        selected = set(intersection)

        if len(selected) < k_gt:
            # Step 2: majority voting for remaining candidates
            votes = defaultdict(int)
            best_rank = {}

            for I in I_list:
                for rank, neighbor in enumerate(I[q]):
                    if neighbor in selected:
                        continue  # skip already selected
                    votes[neighbor] += 1
                    if neighbor not in best_rank:
                        best_rank[neighbor] = rank
                    else:
                        best_rank[neighbor] = min(best_rank[neighbor], rank)

            # Sort by votes (desc), then rank (asc)
            remaining_sorted = sorted(
                votes.keys(),
                key=lambda x: (-votes[x], best_rank[x])
            )

            # Fill until we reach k_gt
            for n in remaining_sorted:
                if len(selected) >= k_gt:
                    break
                selected.add(n)

        # Trim in case we overfilled
        selected = list(selected)[:k_gt]

        hits = len(gt_set & set(selected))
        total_hits += hits

    return total_hits / (nq * k_gt)

def compute_combined_recall(gt, I_list, k_gt):
    recall1, multiplier = refined_recall(gt,I_list,k_gt)
    recall2= compute_intersection_fill_recall(gt,I_list,k_gt)
    return recall1, multiplier, recall2


datasets = ['deep', 'glove', 'sun', 'audio', 'millionSong', 'nuswide', 'MNIST', 'notre', 'sift', 'imageNet','deep_hundred', 'glove_hundred', 'sun_hundred', 'audio_hundred', 'millionSong_hundred', 'nuswide_hundred', 'MNIST_hundred', 'notre_hundred', 'sift_twenty', 'imageNet_hundred']

# datasets = ['deep']

def gen_aug_2(_dataset,dummy):
    file_path = "all_augmentation_standalone_graph_methods.csv"
    df = pd.read_csv(file_path)
    results=[]
    y=['dataset','model1','file1','model1r','model1qps','model1r2r','model1qps2r','model1const','model2','file2','model2r','model2qps','model2r2r','model2qps2r','model2const','model3','file3','model3r','model3qps','model3r2r','model3qps2r','model3const','aug_recall_refine','parallel qps','sequential qps','augmented_const_parallel','augmented_const_sequential','aug_recall_intersection_majority_voting']
    for dataset in [_dataset]:
        
        # print(dataset)
        model='FLATNAV'
        filtered = df[(df['dataset'] == dataset) & (df['model'] == model)]
        file_names_flatnav = filtered["fileName"].tolist()
        model='NSG'
        filtered = df[(df['dataset'] == dataset) & (df['model'] == model)]
        file_names_nsg = filtered["fileName"].tolist()
        model='HNSW'
        filtered = df[(df['dataset'] == dataset) & (df['model'] == model)]
        file_names_hnsw = filtered["fileName"].tolist()
        
        
        cnt=0
        for file1 in file_names_nsg:
            for file2 in file_names_flatnav:
                cnt=cnt+1
                # print(cnt)
                qc=200
                if dataset=='sift':
                    qc=10000
                xb,xq,gt=get_data_generic(dataset)
                I1=getStoredData(file1)
                I2=getStoredData(file2)
                construction_time_2=I2['construction_time']
                construction_time_1=I1['training-time']
                combined_recall,refine_multiplier,recall_intersection=compute_combined_recall(gt,[I1['I'],I2['I']],gt.shape[1])
                recall1r=I1['recallx'+str(2)]
                recall1=I1['recall@']
                search_time1=I1['search-time']
                search_time1r=I1['search-timex2']
                recall2r=I2['recallx'+str(2)]
                recall2=I2['recall']
                search_time2=I2['search_time']
                search_time2r=I2['search_timex2']
                x=[dataset,'NSG',file1,recall1,qc/search_time1,recall1r,qc/search_time1r,construction_time_1,'FLATNAV',file2,recall2,qc/search_time2,recall2r,qc/search_time2r,construction_time_2,'-','-','-','-','-','-','-',combined_recall,qc/max(search_time1,search_time2),qc/(search_time1+search_time2),max(construction_time_1,construction_time_2),construction_time_1+construction_time_2,recall_intersection]
                results.append(x)
                
        
        
        for file1 in file_names_flatnav:
            for file2 in file_names_hnsw:
                # cnt=cnt+1
                # print(cnt)
                qc=200
                if dataset=='sift':
                    qc=10000
                xb,xq,gt=get_data_generic(dataset)
                I1=getStoredData(file1)
                I2=getStoredData(file2)
                construction_time_2=I2['training-time']
                construction_time_1=I1['construction_time']
                combined_recall,refine_multiplier,recall_intersection=compute_combined_recall(gt,[I1['I'],I2['I']],gt.shape[1])
                recall1r=I1['recallx'+str(2)]
                recall1=I1['recall']
                search_time1=I1['search_time']
                search_time1r=I1['search_timex2']
                recall2r=I2['recallx'+str(2)]
                recall2=I2['recall@']
                search_time2=I2['search-time']
                search_time2r=I2['search_timex2']
                x=[dataset,'FLATNAV',file1,recall1,qc/search_time1,recall1r,qc/search_time1r,construction_time_1,'HNSW',file2,recall2,qc/search_time2,recall2r,qc/search_time2r,construction_time_2,'-','-','-','-','-','-','-',combined_recall,qc/max(search_time1,search_time2),qc/(search_time1+search_time2),max(construction_time_1,construction_time_2),construction_time_1+construction_time_2,recall_intersection]
                results.append(x)
        
        for file1 in file_names_nsg:
            for file2 in file_names_hnsw:
                cnt=cnt+1
                # print(cnt)
                qc=200
                if dataset=='sift':
                    qc=10000
                xb,xq,gt=get_data_generic(dataset)
                I1=getStoredData(file1)
                I2=getStoredData(file2)
                construction_time_2=I2['training-time']
                construction_time_1=I1['training-time']
                combined_recall,refine_multiplier,recall_intersection=compute_combined_recall(gt,[I1['I'],I2['I']],gt.shape[1])
                recall1r=I1['recallx'+str(2)]
                recall1=I1['recall@']
                search_time1=I1['search-time']
                search_time1r=I1['search-timex2']
                recall2r=I2['recallx'+str(2)]
                recall2=I2['recall@']
                search_time2=I2['search-time']
                search_time2r=I2['search_timex2']
                x=[dataset,'NSG',file1,recall1,qc/search_time1,recall1r,qc/search_time1r,construction_time_1,'HNSW',file2,recall2,qc/search_time2,recall2r,qc/search_time2r,construction_time_2,'-','-','-','-','-','-','-',combined_recall,qc/max(search_time1,search_time2),qc/(search_time1+search_time2),max(construction_time_1,construction_time_2),construction_time_1+construction_time_2,recall_intersection]
                results.append(x)
                
                
    df_out = pd.DataFrame(results, columns=y)

    # save as CSV
    output_file = "augmented_model_results_2_com_"+_dataset+".csv"
    df_out.to_csv(output_file, index=False)   
    
def gen_aug_3(_dataset,dummy):
    file_path = "filtered_filtered_model_dataset_results.csv"
    df = pd.read_csv(file_path)
    results=[]
    y=['dataset','model1','file1','model1r','model1qps','model1r2r','model1qps2r','model1const','model2','file2','model2r','model2qps','model2r2r','model2qps2r','model2const','model3','file3','model3r','model3qps','model3r2r','model3qps2r','model3const','aug_recall_refine','parallel qps','sequential qps','augmented_const_parallel','augmented_const_sequential','aug_recall_intersection_majority_voting']
    for dataset in [_dataset]:
        
        # print(dataset)
        model='FLATNAV'
        filtered = df[(df['dataset'] == dataset) & (df['model'] == model)]
        file_names_flatnav = filtered["fileName"].tolist()
        model='NSG'
        filtered = df[(df['dataset'] == dataset) & (df['model'] == model)]
        file_names_nsg = filtered["fileName"].tolist()
        model='HNSW'
        filtered = df[(df['dataset'] == dataset) & (df['model'] == model)]
        file_names_hnsw = filtered["fileName"].tolist()
        
        
        cnt=0
        for file1 in file_names_nsg:
            for file2 in file_names_flatnav:
                for file3 in file_names_hnsw:   
                    cnt=cnt+1
                    # print(cnt)
                    qc=200
                    if dataset=='sift':
                        qc=10000
                    xb,xq,gt=get_data_generic(dataset)
                    I1=getStoredData(file1)
                    I2=getStoredData(file2)
                    I3=getStoredData(file3)
                    construction_time_2=I2['construction_time']
                    construction_time_1=I1['training-time']
                    construction_time_3=I3['training-time']
                    combined_recall,refine_multiplier,recall_intersection=compute_combined_recall(gt,[I1['I'],I2['I'],I3['I']],gt.shape[1])
                    recall1r=I1['recallx'+str(3)]
                    recall1=I1['recall@']
                    search_time1=I1['search-time']
                    search_time1r=I1['search-timex3']
                    recall2r=I2['recallx'+str(3)]
                    recall2=I2['recall']
                    search_time2=I2['search_time']
                    search_time2r=I2['search_timex3']
                    recall3r=I3['recallx'+str(3)]
                    recall3=I3['recall@']
                    search_time3=I3['search-time']
                    search_time3r=I3['search_timex3']
                    x=[dataset,'NSG',file1,recall1,qc/search_time1,recall1r,qc/search_time1r,construction_time_1,'FLATNAV',file2,recall2,qc/search_time2,recall2r,qc/search_time2r,construction_time_2,'HNSW',file3,recall3,qc/search_time3,recall3r,qc/search_time3r,construction_time_3,combined_recall,qc/max(search_time1,search_time2,search_time3),qc/(search_time1+search_time2+search_time3),max(construction_time_1,construction_time_2,construction_time_3),construction_time_1+construction_time_2+construction_time_3,recall_intersection]
                    results.append(x)
                
                
    df_out = pd.DataFrame(results, columns=y)

    # save as CSV
    output_file = "augmented_graph_model_results_3_com_"+_dataset+".csv"
    df_out.to_csv(output_file, index=False) 
    
if __name__ == "__main__":
    pss=[]
    for dataset in datasets:
        p = Process(target=gen_aug_2, args=(dataset,'test'))
        pss.append(p)
        p = Process(target=gen_aug_3, args=(dataset,'test'))
        pss.append(p)
    for _p in pss:
        _p.start()
    for _p in pss:
        _p.join()
        
