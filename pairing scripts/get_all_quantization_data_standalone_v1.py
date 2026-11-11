import pickle
import os
import pandas as pd
import pysvs as ps
pickle_path='/data/[username]/similarity-search/models/results_pickles_v8/'
all_results=[]
cnt=0
def getStoredData(dataset,model,k):
    dicts=[]
    global cnt
    lss=[(8,8),(16,4),(16,8),(32,4),(32,8),(64,4)]
    for per in lss:
        r=[]
        r.append(model)
        r.append(dataset)
        r.append(str(per[0])+'-'+str(per[1]))
        for mult in range(1,6):
            isExist = os.path.exists(pickle_path+model+'-'+dataset+'-'+str(per[0])+'-'+str(per[1])+'-'+str(k*mult)+'.pkl') 
            if isExist:
                with open(pickle_path+model+'-'+dataset+'-'+str(per[0])+'-'+str(per[1])+'-'+str(k*mult)+'.pkl', 'rb') as f:
                    I = pickle.load(f)
                    if mult==1:
                        r.append(I['construction-time'])
                    r.append(I['recall@'])
                    r.append(I['search-time'])
            else:
                cnt=cnt+1
                if mult==1:
                    r.append('-')
                r.append('-')
                r.append('-')
                    
        dicts.append(r)
    return dicts
y=['model','dataset','parameter','construction-time','recallx1','searchx1','recallx2','searchx2','recallx3','searchx3','recallx4','searchx4','recallx5','searchx5']
datasets=['deep_hundred', 'glove_hundred', 'sun_hundred', 'audio_hundred', 'millionSong_hundred', 'nuswide_hundred', 'MNIST_hundred', 'notre_hundred', 'sift_twenty', 'imageNet_hundred']


for dataset in datasets:
    for model in ['PQ','OPQ','IVFPW']:
        k=100
        if dataset=='sift_twenty':
            k=20
        all_results.extend(getStoredData(dataset,model,k))
                
print(len(all_results))
print(cnt)
df_out = pd.DataFrame(all_results, columns=y)

# save as CSV
output_file = "standalone_quantization_v1.csv"
df_out.to_csv(output_file, index=False) 
