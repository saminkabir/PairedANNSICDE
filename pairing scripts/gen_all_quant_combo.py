import pandas as pd
import pysvs as ps
pickle_path='/data/[username]/similarity-search/models/results_pickles_v7/'
from multiprocessing import Process

import pickle

file_path = "c_quantization (copy).csv"
df = pd.read_csv(file_path)

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

dataset_infoss=['deep', 'glove', 'sun', 'audio', 'millionSong', 'nuswide', 'notre', 'sift', 'imageNet','MNIST']
dataset_infoss=['MNIST']

# dataset_infoss=['deep']


def get_I(model,fileName,dataset):
    k=20
    if dataset=='sift':
        k=100
    if model=='VAQ':
        vaq_file_name=fileName
        vaq_file_name=vaq_file_name.replace('vaq-log-','').replace('.txt','-'+str(k)+'.ivecs')
        pth='/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-34/results/'
        if 'MNIST' in vaq_file_name:
            pth='/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-35/results/'
            
        I_NN=ps.read_vecs(pth+vaq_file_name)
        # print(vaq_file_name)
        return I_NN
    file_Name=None
    # print(model)
    if model=='OPQ':
        file_Name='OPQ-'+dataset+'-'+fileName+'-'+str(k)+'.pkl'
        # print(file_Name)
        with open(pickle_path+file_Name, 'rb') as f:
            I = pickle.load(f)
            return I['I']
    if model=='PQ':
        file_Name='PQ-'+dataset+'-'+fileName+'-'+str(k)+'.pkl'
        # print(file_Name)
        with open(pickle_path+file_Name, 'rb') as f:
            I = pickle.load(f)
            return I['I']
    if model=='IVFPQ':
        file_Name='IVFPW-'+dataset+'-'+fileName+'-'+str(k)+'.pkl'
        # print(file_Name)
        with open(pickle_path+file_Name, 'rb') as f:
            I = pickle.load(f)
            return I['I_pq']
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
    return total_hits / (nq * k_gt)

combs_count=0

def combine_recall(Infos,dataset):
    # print(Infos,dataset)
    global combs_count
    
    combs_count=combs_count+1
    # print(combs_count)
    Is=[]
    ln=len(Infos)
    ln=int(ln/2)
    for i in range(0,ln):
        Is.append(get_I(Infos[i*2],Infos[i*2+1],dataset))
    # return 0
    k_gt=Is[0].shape[1]
    xb,xq,gt=get_data_generic(dataset)
    return refined_recall(gt,Is,k_gt)
    

def gen_all_comb(dataset,dummy):
    results=[]
    filtered=df[(df['dataset']==dataset)]
    filtered_VAQ=filtered[(filtered['model']=='VAQ')]
    filtered_IVFPQ=filtered[(filtered['model']=='IVFPQ')]
    filtered_OPQ=filtered[(filtered['model']=='OPQ')]
    filtered_PQ=filtered[(filtered['model']=='PQ')]
    refine_k=2
    cnt=0
    for idx1, row1 in filtered_VAQ.iterrows():
        for idx2, row2 in filtered_IVFPQ.iterrows():
            cnt=cnt+1
            # print(cnt)
            recall=combine_recall(['VAQ',row1['file'],'IVFPQ',row2['file']],dataset)
            results.append([dataset,'VAQ',row1['file'],row1['searchx1'],row1['recallx1'],row1['searchx'+str(refine_k)],row1['recallx'+str(refine_k)],row1['construction-time'],row1['memory_usage'],'IVFPQ',row2['file'],row2['searchx1'],row2['recallx1'],row2['searchx'+str(refine_k)],row2['recallx'+str(refine_k)],row2['construction-time'],row2['memory_usage'],'-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-',recall])
    for idx1, row1 in filtered_VAQ.iterrows():
        for idx2, row2 in filtered_OPQ.iterrows():
            cnt=cnt+1
            # print(cnt)
            recall=combine_recall(['VAQ',row1['file'],'OPQ',row2['file']],dataset)
            results.append([dataset,'VAQ',row1['file'],row1['searchx1'],row1['recallx1'],row1['searchx'+str(refine_k)],row1['recallx'+str(refine_k)],row1['construction-time'],row1['memory_usage'],'OPQ',row2['file'],row2['searchx1'],row2['recallx1'],row2['searchx'+str(refine_k)],row2['recallx'+str(refine_k)],row2['construction-time'],row2['memory_usage'],'-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-',recall])
    for idx1, row1 in filtered_VAQ.iterrows():
        for idx2, row2 in filtered_PQ.iterrows():
            cnt=cnt+1
            # print(cnt)
            recall=combine_recall(['VAQ',row1['file'],'PQ',row2['file']],dataset)
            results.append([dataset,'VAQ',row1['file'],row1['searchx1'],row1['recallx1'],row1['searchx'+str(refine_k)],row1['recallx'+str(refine_k)],row1['construction-time'],row1['memory_usage'],'PQ',row2['file'],row2['searchx1'],row2['recallx1'],row2['searchx'+str(refine_k)],row2['recallx'+str(refine_k)],row2['construction-time'],row2['memory_usage'],'-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-',recall])
    
    
    for idx1, row1 in filtered_OPQ.iterrows():
        for idx2, row2 in filtered_PQ.iterrows():
            cnt=cnt+1
            # print(cnt)
            recall=combine_recall(['OPQ',row1['file'],'PQ',row2['file']],dataset)
            results.append([dataset,'OPQ',row1['file'],row1['searchx1'],row1['recallx1'],row1['searchx'+str(refine_k)],row1['recallx'+str(refine_k)],row1['construction-time'],row1['memory_usage'],'PQ',row2['file'],row2['searchx1'],row2['recallx1'],row2['searchx'+str(refine_k)],row2['recallx'+str(refine_k)],row2['construction-time'],row2['memory_usage'],'-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-',recall])
            
    
    
    
    for idx1, row1 in filtered_OPQ.iterrows():
        for idx2, row2 in filtered_IVFPQ.iterrows():
            cnt=cnt+1
            # print(cnt)
            recall=combine_recall(['OPQ',row1['file'],'IVFPQ',row2['file']],dataset)
            results.append([dataset,'OPQ',row1['file'],row1['searchx1'],row1['recallx1'],row1['searchx'+str(refine_k)],row1['recallx'+str(refine_k)],row1['construction-time'],row1['memory_usage'],'IVFPQ',row2['file'],row2['searchx1'],row2['recallx1'],row2['searchx'+str(refine_k)],row2['recallx'+str(refine_k)],row2['construction-time'],row2['memory_usage'],'-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-',recall])
            
    
    for idx1, row1 in filtered_PQ.iterrows():
        for idx2, row2 in filtered_IVFPQ.iterrows():
            cnt=cnt+1
            # print(cnt)
            recall=combine_recall(['PQ',row1['file'],'IVFPQ',row2['file']],dataset)
            results.append([dataset,'PQ',row1['file'],row1['searchx1'],row1['recallx1'],row1['searchx'+str(refine_k)],row1['recallx'+str(refine_k)],row1['construction-time'],row1['memory_usage'],'IVFPQ',row2['file'],row2['searchx1'],row2['recallx1'],row2['searchx'+str(refine_k)],row2['recallx'+str(refine_k)],row2['construction-time'],row2['memory_usage'],'-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-',recall])
            
    refine_k=3
    for idx1, row1 in filtered_VAQ.iterrows():
        for idx2, row2 in filtered_IVFPQ.iterrows():
            for idx3, row3 in filtered_OPQ.iterrows():
                cnt=cnt+1
                # print(cnt)
                recall=combine_recall(['VAQ',row1['file'],'IVFPQ',row2['file'],'OPQ',row3['file']],dataset)
                results.append([dataset,'VAQ',row1['file'],row1['searchx1'],row1['recallx1'],row1['searchx'+str(refine_k)],row1['recallx'+str(refine_k)],row1['construction-time'],row1['memory_usage'],'IVFPQ',row2['file'],row2['searchx1'],row2['recallx1'],row2['searchx'+str(refine_k)],row2['recallx'+str(refine_k)],row2['construction-time'],row2['memory_usage'],'OPQ',row3['file'],row3['searchx1'],row3['recallx1'],row3['searchx'+str(refine_k)],row3['recallx'+str(refine_k)],row3['construction-time'],row3['memory_usage'],'-','-','-','-','-','-','-','-',recall])
    
    for idx1, row1 in filtered_VAQ.iterrows():
        for idx2, row2 in filtered_IVFPQ.iterrows():
            for idx3, row3 in filtered_PQ.iterrows():
                cnt=cnt+1
                # print(cnt)
                recall=combine_recall(['VAQ',row1['file'],'IVFPQ',row2['file'],'PQ',row3['file']],dataset)
            
                results.append([dataset,'VAQ',row1['file'],row1['searchx1'],row1['recallx1'],row1['searchx'+str(refine_k)],row1['recallx'+str(refine_k)],row1['construction-time'],row1['memory_usage'],'IVFPQ',row2['file'],row2['searchx1'],row2['recallx1'],row2['searchx'+str(refine_k)],row2['recallx'+str(refine_k)],row2['construction-time'],row2['memory_usage'],'PQ',row3['file'],row3['searchx1'],row3['recallx1'],row3['searchx'+str(refine_k)],row3['recallx'+str(refine_k)],row3['construction-time'],row3['memory_usage'],'-','-','-','-','-','-','-','-',recall])
    
    for idx1, row1 in filtered_OPQ.iterrows():
        for idx2, row2 in filtered_IVFPQ.iterrows():
            for idx3, row3 in filtered_PQ.iterrows():
                cnt=cnt+1
                # print(cnt)
                recall=combine_recall(['OPQ',row1['file'],'IVFPQ',row2['file'],'PQ',row3['file']],dataset)
                results.append([dataset,'OPQ',row1['file'],row1['searchx1'],row1['recallx1'],row1['searchx'+str(refine_k)],row1['recallx'+str(refine_k)],row1['construction-time'],row1['memory_usage'],'IVFPQ',row2['file'],row2['searchx1'],row2['recallx1'],row2['searchx'+str(refine_k)],row2['recallx'+str(refine_k)],row2['construction-time'],row2['memory_usage'],'PQ',row3['file'],row3['searchx1'],row3['recallx1'],row3['searchx'+str(refine_k)],row3['recallx'+str(refine_k)],row3['construction-time'],row3['memory_usage'],'-','-','-','-','-','-','-','-',recall])
    
    k=4
    for idx1, row1 in filtered_VAQ.iterrows():
        for idx2, row2 in filtered_IVFPQ.iterrows():
            for idx3, row3 in filtered_OPQ.iterrows():
                for idx4, row4 in filtered_PQ.iterrows():
                    cnt=cnt+1
                    # print(cnt)
                    recall=combine_recall(['VAQ',row1['file'],'IVFPQ',row2['file'],'OPQ',row3['file'],'PQ',row4['file']],dataset)
                    results.append([dataset,'VAQ',row1['file'],row1['searchx1'],row1['recallx1'],row1['searchx'+str(refine_k)],row1['recallx'+str(refine_k)],row1['construction-time'],row1['memory_usage'],'IVFPQ',row2['file'],row2['searchx1'],row2['recallx1'],row2['searchx'+str(refine_k)],row2['recallx'+str(refine_k)],row2['construction-time'],row2['memory_usage'],'OPQ',row3['file'],row3['searchx1'],row3['recallx1'],row3['searchx'+str(refine_k)],row3['recallx'+str(refine_k)],row3['construction-time'],row3['memory_usage'],'PQ',row4['file'],row4['searchx1'],row4['recallx1'],row4['searchx'+str(refine_k)],row4['recallx'+str(refine_k)],row4['construction-time'],row4['memory_usage'],recall])
    # print(cnt)
    y=['dataset','model1','model1 parameter','model1 search-time without refinement','model1 recall without refinement','model1 search-time with refinement','model1 recall with refinement','model1 construction time','model1 memory usage','model2','model2 parameter','model2 search-time without refinement','model2 recall without refinement','model2 search-time with refinement','model2 recall with refinement','model2 construction time','model2 memory usage','model3','model3 parameter','model3 search-time without refinement','model3 recall without refinement','model3 search-time with refinement','model3 recall with refinement','model3 construction time','model3 memory usage','model4','model4 parameter','model4 search-time without refinement','model4 recall without refinement','model4 search-time with refinement','model4 recall with refinement','model4 construction time','model4 memory usage','augmented model recall']

    df_out = pd.DataFrame(results, columns=y)

    # save as CSV
    output_file = "all_possible_quant_optimization"+dataset+".csv"
    df_out.to_csv(output_file, index=False) 
    
    

if __name__ == "__main__":
    pss=[]
    for dataset in dataset_infoss:
        p = Process(target=gen_all_comb, args=(dataset,'test'))
        pss.append(p)
    for _p in pss:
        _p.start()
    for _p in pss:
        _p.join()
