import pickle
import os
import pandas as pd
import pysvs as ps
from collections import defaultdict
from multiprocessing import Process

# os.chdir('/data/[username]/similarity-search/models/results_pickles_v7')
dataset_path = '/data/[username]/similarity-search/dataset/'
pickle_path='/data/[username]/similarity-search/models/results_pickles_v7/'

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

def compute_combined_recall(gt, I_list, k_gt):
    recall1, multiplier = refined_recall(gt,I_list,k_gt)
    recall2= compute_intersection_fill_recall(gt,I_list,k_gt)
    return recall1, multiplier, recall2




dataset_infoss=['deep', 'glove', 'sun', 'audio', 'millionSong', 'nuswide', 'notre', 'sift', 'imageNet']
def gen_aug_2(_dataset,dummy):
    y=['dataset','model1','file1','model1r','model1qps','model1r2r','model1qps2r','model1const','model2','file2','model2r','model2qps','model2r2r','model2qps2r','model2const','model3','file3','model3r','model3qps','model3r2r','model3qps2r','model3const','aug_recall_refine','parallel qps','sequential qps','augmented_const_parallel','augmented_const_sequential','aug_recall_intersection_majority_voting']
    dataset=_dataset
    file_path = "filtered_filtered_model_dataset_results.csv"
    df = pd.read_csv(file_path)
    results=[]
    model='FLATNAV'
    filtered = df[(df['dataset'] == dataset) & (df['model'] == model) & (df['flag1']==1)]
    file_names_flatnav = filtered["fileName"].tolist()
    model='NSG'
    filtered = df[(df['dataset'] == dataset) & (df['model'] == model) & (df['flag1']==1)]
    file_names_nsg = filtered["fileName"].tolist()
    model='HNSW'
    filtered = df[(df['dataset'] == dataset) & (df['model'] == model) & (df['flag1']==1)]
    file_names_hnsw = filtered["fileName"].tolist()
    file_path = "standalone_vaq.csv"
    df1 = pd.read_csv(file_path)
    df1 = df1[(df1['dataset'] == dataset)]
    for file1 in file_names_nsg:
        for index, row in df1.iterrows():
            qc=200
            k=20
            if dataset=='sift':
                qc=10000
                k=100
            vaq_file_name=row['fileName']
            vaq_file_name=vaq_file_name.replace('vaq-log-','').replace('.txt','-'+str(k)+'.ivecs')
            # /data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-34/results/
            I_NN=ps.read_vecs('/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-34/results/'+vaq_file_name)
            xb,xq,gt=get_data_generic(dataset)
            print(file1)
            I1=getStoredData(file1)
            construction_time_1=I1['training-time']
            recall1r=I1['recallx'+str(2)]
            recall1=I1['recall@']
            search_time1=I1['search-time']
            search_time1r=I1['search-timex2']
            combined_recall,refine_multiplier,recall_intersection=compute_combined_recall(gt,[I1['I'],I_NN],gt.shape[1])
            recall2r=row['recallx'+str(2)]
            recall2=row['recallx'+str(1)]
            search_time2=row['searchx1']
            search_time2r=row['searchx2']
            construction_time_2=row['construction-time']
            x=[dataset,'NSG',file1,recall1,qc/search_time1,recall1r,qc/search_time1r,construction_time_1,'VAQ',vaq_file_name,recall2,qc/search_time2,recall2r,qc/search_time2r,construction_time_2,'-','-','-','-','-','-','-',combined_recall,qc/max(search_time1,search_time2),qc/(search_time1+search_time2),max(construction_time_1,construction_time_2),construction_time_1+construction_time_2,recall_intersection]
            results.append(x)
    for file1 in file_names_hnsw:
        for index, row in df1.iterrows():
            qc=200
            k=20
            if dataset=='sift':
                qc=10000
                k=100
            vaq_file_name=row['fileName']
            vaq_file_name=vaq_file_name.replace('vaq-log-','').replace('.txt','-'+str(k)+'.ivecs')
            # /data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-34/results/
            I_NN=ps.read_vecs('/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-34/results/'+vaq_file_name)
            xb,xq,gt=get_data_generic(dataset)
            print(file1)
            I1=getStoredData(file1)
            construction_time_1=I1['training-time']
            recall1r=I1['recallx'+str(2)]
            recall1=I1['recall@']
            search_time1=I1['search-time']
            search_time1r=I1['search_timex2']
            combined_recall,refine_multiplier,recall_intersection=compute_combined_recall(gt,[I1['I'],I_NN],gt.shape[1])
            recall2r=row['recallx'+str(2)]
            recall2=row['recallx'+str(1)]
            search_time2=row['searchx1']
            search_time2r=row['searchx2']
            construction_time_2=row['construction-time']
            x=[dataset,'HNSW',file1,recall1,qc/search_time1,recall1r,qc/search_time1r,construction_time_1,'VAQ',vaq_file_name,recall2,qc/search_time2,recall2r,qc/search_time2r,construction_time_2,'-','-','-','-','-','-','-',combined_recall,qc/max(search_time1,search_time2),qc/(search_time1+search_time2),max(construction_time_1,construction_time_2),construction_time_1+construction_time_2,recall_intersection]
            results.append(x)
    for file1 in file_names_flatnav:
        for index, row in df1.iterrows():
            qc=200
            k=20
            if dataset=='sift':
                qc=10000
                k=100
            vaq_file_name=row['fileName']
            vaq_file_name=vaq_file_name.replace('vaq-log-','').replace('.txt','-'+str(k)+'.ivecs')
            # /data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-34/results/
            I_NN=ps.read_vecs('/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-34/results/'+vaq_file_name)
            xb,xq,gt=get_data_generic(dataset)
            print(file1)
            I1=getStoredData(file1)
            construction_time_1=I1['construction_time']
            recall1r=I1['recallx'+str(2)]
            recall1=I1['recall']
            search_time1=I1['search_time']
            search_time1r=I1['search_timex2']
            combined_recall,refine_multiplier,recall_intersection=compute_combined_recall(gt,[I1['I'],I_NN],gt.shape[1])
            recall2r=row['recallx'+str(2)]
            recall2=row['recallx'+str(1)]
            search_time2=row['searchx1']
            search_time2r=row['searchx2']
            construction_time_2=row['construction-time']
            x=[dataset,'FLATNAV',file1,recall1,qc/search_time1,recall1r,qc/search_time1r,construction_time_1,'VAQ',vaq_file_name,recall2,qc/search_time2,recall2r,qc/search_time2r,construction_time_2,'-','-','-','-','-','-','-',combined_recall,qc/max(search_time1,search_time2),qc/(search_time1+search_time2),max(construction_time_1,construction_time_2),construction_time_1+construction_time_2,recall_intersection]
            results.append(x)
    output_file = "augmented_model_vaq_results_2_com_"+_dataset+".csv"
    df_out = pd.DataFrame(results, columns=y)
    df_out.to_csv(output_file, index=False)


if __name__ == "__main__":
    pss=[]
    for dataset in dataset_infoss:
        p = Process(target=gen_aug_2, args=(dataset,'test'))
        pss.append(p)
    for _p in pss:
        _p.start()
    for _p in pss:
        _p.join()
            
 
        
