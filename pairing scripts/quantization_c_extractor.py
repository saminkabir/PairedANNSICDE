import os
import pandas as pd

os.chdir('/home/saminyeaser/faiss-c-logs')
all_results={}
def parse_metrics(file_path):
    results = {}
    with open(file_path, "r") as f:
        if 'IVF' in file_path:
            results['model']='IVFPQ'
        elif 'OPQ' in file_path:
            results['model']='OPQ'
        else:
            results['model']='PQ'
        results['parameter']=file_path.split('PQ')[-1].replace('.txt','').replace('x','-')
        results['dataset']=file_path.split('-')[0]
        for line in f:
            if "Recall@k" in line:
                results["Recall@k"] = float(line.split("=")[1].strip())
            elif "Construction time:" in line:
                # keep ms as integer
                results["Construction time (ms)"] = float(line.split(":")[1].strip().split()[0])/1000.0
            elif "Search time:" in line:
                results["Search time (ms)"] = float(line.split(":")[1].strip().split()[0])/1000.0
            elif "Maximum resident set size" in line:
                results["Memory usage"] = int(line.split(":")[1].strip())
    all_results[results['model']+'-'+results['dataset']+'-'+results['parameter']]=results
    print(results)

for file in os.listdir():
    parse_metrics(file)

os.chdir('/home/saminyeaser/polyvector_analysis_v1')


file_path = "standalone_quantization (copy).csv"
df = pd.read_csv(file_path)
print(len(all_results))    
cnt=0
for idx, row in df.iterrows():
    ky = row['model'] + '-' + row['dataset'] + '-' + row['parameter']
    if ky in all_results:
        df.loc[idx, 'memory_usage'] = all_results[ky]['Memory usage']
        df.loc[idx, 'construction-time'] = all_results[ky]['Construction time (ms)']
        for i in range(2,5):
            df.loc[idx, 'recallx'+str(i)]=(row['recallx'+str(i)]/row['recallx'+str(1)])*all_results[ky]["Recall@k"]
            df.loc[idx, 'searchx'+str(i)]=(row['searchx'+str(i)]/row['searchx'+str(1)])*all_results[ky]["Search time (ms)"]
        
        df.loc[idx, 'recallx'+str(1)]=all_results[ky]["Recall@k"]
        df.loc[idx, 'searchx'+str(1)]=all_results[ky]["Search time (ms)"]
            
    else:
        df.loc[idx, 'memory_usage'] = 0
        
print(cnt)
output_file = "c_quantization.csv"
df.to_csv(output_file, index=False)
