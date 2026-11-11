import pickle
pickle_path='/data/[username]/similarity-search/models/results_pickles_v8/'
import numpy as np
import pandas as pd
import pysvs as ps

dataset_path = '/data/[username]/similarity-search/dataset/'

file_path = "c_quantization (copy).csv"
df = pd.read_csv(file_path)

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

win=0
lose=0
all_results=[]
def merge(file1,file2,dataset,peram1,peram2,row1,row2):
    xb,xq,gt=get_data_generic(dataset)
    D1=None
    D2=None
    I1=None
    I2=None
    with open(pickle_path+file1, 'rb') as f:
        II1 = pickle.load(f)
        D1=II1['D']
        I1=II1['I']
    with open(pickle_path+file2, 'rb') as f:
        II2 = pickle.load(f)
        D2=II2['D']
        I2=II2['I']
        
    qn=xq.shape[0]
    k=gt.shape[1]
    tot=0
    results=[]
    for q in range(0,qn):
        set_opq=set(I1[q])
        set_pq=set(I2[q])
        common=set_opq & set_pq
        candidates=[]
        for i in range(k):
            if not I1[q][i] in common:
                candidates.append((D1[q][i],I1[q][i]))
            if not I2[q][i] in common:
                candidates.append((D2[q][i],I2[q][i]))
        candidates.sort()
        for i in range(len(candidates)):
            if len(common)<k:
                common.add(candidates[i][1])
            else:
                break
        results.append(list(common))
    
    results=np.array(results)
    # recall1=calculate_recall_at(I1,gt,k,k)
    # recall2=calculate_recall_at(I2,gt,k,k)
    recall1=II1['recall@']
    recall2=II2['recall@']
    recall=calculate_recall_at(results,gt,k,k)
    global win,lose
    if (max(recall1,recall2)>recall):
        lose=lose+1
    else:
        win=win+1
    global all_results
    all_results.append([dataset,'OPQ',peram1,qn/row1['searchx1'],recall1,'PQ',peram2,qn/row2['searchx1'],recall2,qn/max(row1['searchx1'],row2['searchx1']),recall])
        
y=['dataset','model1','model 1 param','model1 qps','model1 recall','model2','model 2 param','model2 qps','model2 recall','augmented model qps','augmented model recall']
def gen_all_comb(dataset,dummy):
    xb,xq,gt=get_data_generic(dataset)
    k=gt.shape[1]
    results=[]
    filtered=df[(df['dataset']==dataset)]
    filtered_OPQ=filtered[(filtered['model']=='OPQ')]
    filtered_PQ=filtered[(filtered['model']=='PQ')]
    print(len(filtered_OPQ['file'].tolist()))
    print(len(filtered_PQ['file'].tolist()))
    occ=0
    for idx1, row1 in filtered_OPQ.iterrows():
        file1=row1['file']
        file_Name_1='OPQ-'+dataset+'-'+file1+'-'+str(k)+'.pkl'
        for idx2, row2 in filtered_PQ.iterrows():
            file2=row2['file']
            occ=occ+1
            print(occ)
            file_Name_2='PQ-'+dataset+'-'+file2+'-'+str(k)+'.pkl'
            merge(file_Name_1,file_Name_2,dataset,file1,file2,row1,row2)
            
gen_all_comb('sift','')

# merge('OPQ-sift-32-8-100.pkl','PQ-sift-32-8-100.pkl','sift')
dataset_infoss=['deep', 'glove', 'sun', 'audio', 'millionSong', 'nuswide', 'notre', 'sift', 'imageNet','MNIST']

for dataset in dataset_infoss:
    gen_all_comb(dataset,'')

print(win,lose)

df_out = pd.DataFrame(all_results, columns=y)

# save as CSV
output_file = "all_possible_quant_opq_pq.csv"
df_out.to_csv(output_file, index=False) 

