import os 

import pandas as pd

mp=dict()
            
os.chdir('/relative_path/Research-Implementation/models/VAQ22/data/faiss-library/logHNSW')

for file in os.listdir():
    with open(file, "r") as f:
        data=f.read()
        if 'Maximum resident set size (kbytes): ' in data:
            memory_size=int(data.split('Maximum resident set size (kbytes): ')[1].split('\t')[0])
            peram=file.replace('.txt','')
            perams=peram.split('-')
            mp['HNSW-'+perams[0]+'-'+perams[1]+'-'+perams[3]+'-'+perams[2]+'.pkl']=memory_size
            

os.chdir('/relative_path/Research-Implementation/models/VAQ22/data/faiss-library/flatnavogs')
for file in os.listdir():
    with open(file, "r") as f:
        data=f.read()
        if 'Maximum resident set size (kbytes): ' in data:
            memory_size=int(data.split('Maximum resident set size (kbytes): ')[1].split('\t')[0])
            peram=file.replace('.txt','')
            perams=peram.split('-')
            mp['FLATNAV-'+perams[0]+'-'+perams[1]+'-'+perams[2]+'-'+perams[3]+'.pkl']=memory_size
            
os.chdir('/relative_path/Research-Implementation/models/VAQ22/data/faiss-library/logNSG')
for file in os.listdir():
    with open(file, "r") as f:
        data=f.read()
        if 'Maximum resident set size (kbytes): ' in data:
            memory_size=int(data.split('Maximum resident set size (kbytes): ')[1].split('\t')[0])
            peram=file.replace('.txt','')
            perams=peram.split('-')
            mp['NSG-'+perams[0]+'-'+perams[1]]=memory_size
            
print(len(mp.keys()))
            
            
os.chdir('/home/saminyeaser/polyvector_analysis_v1')

df = pd.read_csv("Polyvector results - graph_all_combination_augmented_results.csv")


def get_memorty_result(model,fileName):
    # print(model,fileName)
    if model=='FLATNAV':
        return mp[fileName]
    if model=='HNSW':
        return mp[fileName]
    if model=='NSG':
        ky=fileName.split('-')
        kys=ky[0]+'-'+ky[1]+'-'+ky[2]
        return mp[kys]
    return '-'
def get_num(val):
    if val=='-':
        return 0
    return int(val)
    
# Option 1: Iterate using iterrows() (row as Series)
for index, row in df.iterrows():
    mem1=get_memorty_result(row['model1'],row['file1'])
    df.at[index, "model1 memory"] =mem1
    mem2=get_memorty_result(row['model2'],row['file2'])
    df.at[index, "model2 memory"] =mem2
    mem3=get_memorty_result(row['model3'],row['file3'])
    df.at[index, "model3 memory"] =mem3
    df.at[index, "augmented model memory"] =get_num(mem1)+get_num(mem2)+get_num(mem3)

output_file = "Polyvector results - graph_all_combination_augmented_results_memory.csv"
df.to_csv(output_file, index=False) 


# def get_memorty_result(model,fileName):
#     # print(model,fileName)
#     if model=='FLATNAV':
#         return mp[fileName]
#     if model=='HNSW':
#         return mp[fileName]
#     if model=='NSG':
#         ky=fileName.split('-')
#         kys=ky[0]+'-'+ky[1]+'-'+ky[2]
#         return mp[kys]
#     return '-'
# Option 1: Iterate using iterrows() (row as Series)


df = pd.read_csv("Polyvector results - standalone graph based model's result.csv")

for index, row in df.iterrows():
    mem1=get_memorty_result(row['model'],row['fileName'])
    df.at[index, "memory"] =mem1
    
    
output_file = "Polyvector results - standalone graph based model's result memory.csv"
df.to_csv(output_file, index=False) 


