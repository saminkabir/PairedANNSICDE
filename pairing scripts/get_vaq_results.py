import os
import pandas as pd

from multiprocessing import Process


path='/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-35'
os.chdir(path)

mem_mp={}

for fileName in os.listdir():
    if '-bulk-testing-27,' in fileName:
        nbits=int(fileName.split('VAQ')[1].split('m')[0])
        nsubs=int(fileName.split('VAQ')[1].split('m')[1].split('min')[0])
        dataset=fileName.split('-bulk-testing-27')[0]
        nbits=int(nbits/nsubs)
        with open(fileName, "r") as file:
            data=file.read()
            mem=int(data.split('Maximum resident set size (kbytes): ')[1].split('\n')[0])
            mem_mp[(dataset,nsubs,nbits)]=mem
            
print(mem_mp)



path='/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-35/finalTests2/'
files=os.listdir(path)
datasets=set()
best_recalls=dict()
for fileName in files:
    search_times=[]
    recalls=[]
    construction_time=0
    dataset=None
    m=None
    nbits=None
    m=None
    visit_cluster=None
    mem_usage=None
    with open(path+fileName, "r") as file:
        lines = file.readlines()
        for line in lines:
            if 'recall@R (Probably correct): ' in line:
                recall=float(line.split('recall@R (Probably correct): ')[1].split(' ')[0])
                recalls.append(recall)
            if '== Querying time: ' in line:
                search_time=float(line.split('== Querying time: ')[1].split(' ')[0])
                search_times.append(search_time)
            if '== Training time: ' in line:
                construction_time=construction_time+float(line.split('== Training time: ')[1].split(' ')[0])
            if 'method = VAQ' in line:
                nbits=int(line.split('method = VAQ')[1].split('m')[0])
                m=int(line.split('method = VAQ')[1].split('m')[1].split('min')[0])
            if 'groundtruth = ../../../dataset/' in line:
                dataset=line.split('groundtruth = ../../../dataset/')[1].split('/groundtruth.ivecs')[0]
            if 'visit-cluster = ' in line:
                visit_cluster=float(line.split('visit-cluster = ')[1].replace('\n',''))
            elif "Maximum resident set size" in line:
                mem_usage = int(line.split(":")[1].strip())
    nbits=int(nbits/m)
    if len(recalls)>=5 and visit_cluster==0.125:
        mm_usage=0
        if (dataset,m,nbits) in mem_mp:
            mm_usage=mem_mp[(dataset,m,nbits)]
            print(mem_mp[(dataset,m,nbits)])
        
        data=[dataset,str(m)+'-'+str(nbits),construction_time,recalls,search_times]
        ky=dataset,str(m)+'-'+str(nbits)+'-'+str(visit_cluster)
        if not ky in best_recalls:
            best_recalls[ky]=['VAQ',dataset,ky[1],construction_time,recalls[0],search_times[0],recalls[1],search_times[1],recalls[2],search_times[2],recalls[3],search_times[3],recalls[4],search_times[4],mm_usage,fileName]
        elif best_recalls[ky][4]<recalls[0]:
            best_recalls[ky]=['VAQ',dataset,ky[1],construction_time,recalls[0],search_times[0],recalls[1],search_times[1],recalls[2],search_times[2],recalls[3],search_times[3],recalls[4],search_times[4],mm_usage,fileName]
# print(best_recalls)
results=[]
for k in best_recalls.keys():
    results.append(best_recalls[k])
# print(results)
y=['model','dataset','parameter','construction-time','recallx1','searchx1','recallx2','searchx2','recallx3','searchx3','recallx4','searchx4','recallx5','searchx5','memory_usage','fileName']

df_out = pd.DataFrame(results, columns=y)
os.chdir('/data/[username]/similarity-search/models/polyvector_analysis_v1')
# save as CSV
output_file = "standalone_vaq.csv"
df_out.to_csv(output_file, index=False) 
