import os
import pandas as pd

mem_mp={}

path='/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-[id]/'

for id in ['35','36','37','38','39']:
    current_path=path.replace('[id]',id)
    os.chdir(current_path)
    for fileName in os.listdir():
        if '-mem.txt' in fileName:
            with open(fileName, "r") as file:
                raw_file_name='vaq-log-'+fileName.replace('-mem.txt','.txt')
                data=file.read()
                mem=int(data.split('Maximum resident set size (kbytes): ')[1].split('\n')[0])
                mem_mp[raw_file_name]=mem

path='/data/[username]/similarity-search/models/vaq-testing/analyse-mult-vaq-[id]/'

for id in ['40']:
    current_path=path.replace('[id]',id)
    os.chdir(current_path)
    for fileName in os.listdir():
        if '-mem.txt' in fileName:
            with open(fileName, "r") as file:
                raw_file_name='vaq-log-'+fileName.replace('-mem.txt','.txt')
                data=file.read()
                mem=int(data.split('Maximum resident set size (kbytes): ')[1].split('\n')[0])
                mem_mp[raw_file_name]=mem
            
    
    
    


path='/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-[id]/finalTests2/'

recall_qps_list=[]
for id in ['35','36','37','38','39']:
    current_path=path.replace('[id]',id)
    os.chdir(current_path)
    for fileName in os.listdir():
        with open(fileName, "r") as file:
            recalls=[]
            qpss=[]
            construction_time=0
            dataset=None
            lines = file.readlines()
            bits=None
            if ',VAQ' in fileName:
                bits=int(fileName.split(',VAQ')[1].split('m')[0])
            for line in lines:
                if 'recall@R (Probably correct): ' in line:
                    recall=float(line.split('recall@R (Probably correct): ')[1].split(' ')[0])
                    recalls.append(recall)
                if '== Querying time: ' in line:
                    search_time=float(line.split('== Querying time: ')[1].split(' ')[0])
                    qpss.append(search_time)
                if 'groundtruth = ../../../dataset/' in line:
                    dataset=line.split('groundtruth = ../../../dataset/')[1].split('/groundtruth.ivecs')[0]
                if '== Training time: ' in line:
                    construction_time=construction_time+float(line.split('== Training time: ')[1].split(' ')[0])
                if '== Encoding time: ' in line:
                    construction_time=construction_time+float(line.split('== Encoding time: ')[1].split(' ')[0])
            if dataset!=None:
                for i in range(min(len(recalls),len(qpss))):
                    qn=200
                    if dataset=='sift':
                        qn=10000
                    recall_qps_list.append([dataset,recalls[i],qn/qpss[i],construction_time,mem_mp[fileName],i+1,bits,fileName,id])
    
path='/data/[username]/similarity-search/models/vaq-testing/analyse-mult-vaq-[id]/finalTests2/'

                    
for id in ['40']:
    current_path=path.replace('[id]',id)
    os.chdir(current_path)
    for fileName in os.listdir():
        with open(fileName, "r") as file:
            recalls=[]
            qpss=[]
            construction_time=0
            dataset=None
            lines = file.readlines()
            bits=None
            # print(fileName)
            if ',VAQ' in fileName:
                # print(fileName)
                bits=int(fileName.split(',VAQ')[1].split('m')[0])
                # print(bits)
            for line in lines:
                if 'recall@R (Probably correct): ' in line:
                    recall=float(line.split('recall@R (Probably correct): ')[1].split(' ')[0])
                    recalls.append(recall)
                if '== Querying time: ' in line:
                    search_time=float(line.split('== Querying time: ')[1].split(' ')[0])
                    qpss.append(search_time)
                if 'groundtruth = ../../../dataset/' in line:
                    dataset=line.split('groundtruth = ../../../dataset/')[1].split('/groundtruth.ivecs')[0]
                if '== Training time: ' in line:
                    construction_time=construction_time+float(line.split('== Training time: ')[1].split(' ')[0])
                if '== Encoding time: ' in line:
                    construction_time=construction_time+float(line.split('== Encoding time: ')[1].split(' ')[0])
            if dataset!=None:
                for i in range(min(len(recalls),len(qpss))):
                    qn=200
                    if dataset=='sift':
                        qn=10000
                    recall_qps_list.append([dataset,recalls[i],qn/qpss[i],construction_time,mem_mp[fileName],i+1,bits,fileName,id])
                    
        
y=['dataset','recall','qps','construction_time','memory','refinementx','bits','fileName','id']
df_out = pd.DataFrame(recall_qps_list, columns=y)
os.chdir('/data/[username]/similarity-search/models/polyvector_analysis_v1')
# save as CSV
output_file = "all_possible_standalone_vaq.csv"
df_out.to_csv(output_file, index=False) 
