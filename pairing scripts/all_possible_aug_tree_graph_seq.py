import argparse
import pickle
import os
import pandas as pd
import pysvs as ps
from multiprocessing import Process
from collections import defaultdict
# os.chdir('/data/[username]/similarity-search/models/results_pickles_v7')
dataset_path = '/data/kabir/similarity-search/dataset/'
pickle_path_1='/data/kabir/similarity-search/models/results_pickles_v7/'
pickle_path_2='/data/kabir/similarity-search/models/results_pickles_v8/'

v8_files=os.listdir(pickle_path_2)
v7_files=os.listdir(pickle_path_1)

parser = argparse.ArgumentParser(description="Run MRPT ANN search on a dataset")
parser.add_argument("--dataset", type=str, required=True,
                    help="Name of the dataset (e.g. 'imageNet', 'glove-100')")

dataset = parser.parse_args().dataset

file_path = "graph_results_1.csv"
df = pd.read_csv(file_path)
model='FLATNAV'
filtered = df[(df['model'] == model) & (df['dataset'] == dataset)]
file_names_flatnav = filtered["fileName"].tolist()
file_path = "graph_results_2.csv"
df = pd.read_csv(file_path)
filtered = df[(df['model'] == model) & (df['dataset'] == dataset)]
file_names_flatnav.extend(filtered["fileName"].tolist())
file_names_flatnav=list(set(file_names_flatnav))

def read_vecs_v1(filePath):
    return ps.read_vecs(filePath)



def read_vecs(filePath):
    return ps.read_vecs(dataset_path + filePath)

def get_data_generic(dataset):
    # print(dataset)
    # data we will search through
    xb = read_vecs(dataset+'/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs(dataset+'/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs(dataset+'/groundtruth.ivecs')
    return xb,xq,gt

xb,xq,gt=get_data_generic(dataset)

def calculate_recall_at(ground_truth, I, k1, k2):
    return ps.k_recall_at(ground_truth, I, k1, k2)


def getStoredData(fileName):
    global v8_files, v7_files, pickle_path_1, pickle_path_2
    pickle_path=None
    if fileName in v8_files:
        pickle_path=pickle_path_2
    elif fileName in v7_files:
        pickle_path=pickle_path_1
    if not pickle_path is None:
        with open(pickle_path+fileName, 'rb') as f:
            print(pickle_path+fileName)
            I = pickle.load(f)
            return I
    return None



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


datasets=[
    'audio',
    'millionSong',
    'yahoomusic_20_variants',
    'agnews-mxbai-1024-euclidean_20_variants',
    'arxiv-nomic-768-normalized_20_variants',
    'ccnews-nomic-768-normalized_20_variants',
    'glove',
    'gooaq-distilroberta-768-normalized_20_variants',
    'deep',
    'imageNet',
    'MNIST',
    'notre',
    'nuswide',
    'sift',
    'sun',
    'laion-clip-512-normalized_20_variants',
    'llama-128-ip_20_variants',
    'coco-nomic-768-normalized_20_variants',
    'yandex-200-cosine_20_variants',
    'yi-128-ip_20_variants'
]


def get_dataset_name_from_file(file_name):
    for dataset in datasets:
        if dataset in file_name:
            return dataset
    return None



standalone_columns=['dataset','model','params1','params2','recall','qps','construction_time']
standalone_results=[]

for t in [1000,100,10,200,25,400,50,600,75]:
    tmp='Scann-[dataset]-2000-[t]-250000-2.pkl'
    tmp='scann-[dataset]-2000-[t]-2.pkl'
    qc=200
    if dataset=='sift':
        qc=10000
    scann_file_name=tmp.replace('[t]',str(t))
    scann_file_name=scann_file_name.replace('[dataset]',str(dataset))
    pkl_scann=getStoredData(scann_file_name)
    if pkl_scann is None:
        tmp='Scann-[dataset]-2000-[t]-250000-2.pkl'
        scann_file_name=tmp.replace('[t]',str(t))
        scann_file_name=scann_file_name.replace('[dataset]',str(dataset))
        pkl_scann=getStoredData(scann_file_name)
    print(pkl_scann)
    peram=scann_file_name
    # peram=str(pkl_scann['num_leaves_'])+'-'+str(pkl_scann['num_leaves_to_search_'])+'-'+str(pkl_scann['training_sample_size_'])+'-'+str(pkl_scann['num_neighbors_'])+'-'+str(pkl_scann['anisotropic_quantization_threshold_'])
    print(pkl_scann.keys())
    rf_recall_scann=pkl_scann['recallx2']
    recall_scann=pkl_scann['recall']
    construction_scann=pkl_scann['construction_time']
    search_scann=pkl_scann['search_time']
    search_x2_scann=pkl_scann['search_timex2']
    standalone_results.append([dataset,'ScaNN',peram,'',rf_recall_scann,qc/search_x2_scann,construction_scann])


for flatnav_result in file_names_flatnav:
    dataset = get_dataset_name_from_file(flatnav_result)
    if not dataset is None:
        qc=200
        if dataset=='sift':
            qc=10000
        if '.pkl' not in flatnav_result:
            flatnav_result=flatnav_result+'.pkl'
        I1=getStoredData(flatnav_result)
        print(flatnav_result)
        print(I1)
        construction_time_1=I1['construction_time']
        recall1r=I1['recallx'+str(2)]
        recall1=I1['recall']
        search_time1=I1['search_time']
        search_time1r=I1['search_timex2']
        standalone_results.append([dataset,'FLATNAV',file_names_flatnav,'',recall1r,qc/search_time1r,construction_time_1])
        for t in [1000,100,10,200,25,400,50,600,75]:
            tmp='Scann-[dataset]-2000-[t]-250000-2.pkl'

            tmp='scann-[dataset]-2000-[t]-2.pkl'
            scann_file_name=tmp.replace('[t]',str(t))
            scann_file_name=scann_file_name.replace('[dataset]',str(dataset))
            pkl_scann=getStoredData(scann_file_name)
            if pkl_scann is None:
                tmp='Scann-[dataset]-2000-[t]-250000-2.pkl'
                scann_file_name=tmp.replace('[t]',str(t))
                scann_file_name=scann_file_name.replace('[dataset]',str(dataset))
                pkl_scann=getStoredData(scann_file_name)
            peram=str(pkl_scann['num_leaves_'])+'-'+str(pkl_scann['num_leaves_to_search_'])+'-'+str(pkl_scann['training_sample_size_'])+'-'+str(pkl_scann['num_neighbors_'])+'-'+str(pkl_scann['anisotropic_quantization_threshold_'])
            rf_recall_scann=pkl_scann['recallx2']
            recall_scann=pkl_scann['recall']
            I_scann=pkl_scann['I_1']
            construction_scann=pkl_scann['construction_time']
            search_scann=pkl_scann['search_time']
            search_x2_scann=pkl_scann['search_timex2']
            combined_recall,refine_multiplier,recall_intersection=compute_combined_recall(gt,[I1['I'],I_scann],gt.shape[1])
            standalone_results.append([dataset,'ScaNN-FLATNAV',peram,file_names_flatnav,combined_recall,qc/(search_scann+search_time1),construction_scann+construction_time_1])

folder_name='scann-flatnav/'

df_out = pd.DataFrame(standalone_results, columns=standalone_columns)
output_file = folder_name+"seq-augmented_model_results"+dataset+".csv"
df_out.to_csv(output_file, index=False)   