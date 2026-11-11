import pickle
import os
import pandas as pd
import pysvs as ps
pickle_path='/data/[username]/similarity-search/models/results_pickles_v7/'

os.chdir('/data/[username]/similarity-search/models/results_pickles_v7')
def getStoredData(fileName):
    with open(pickle_path+fileName, 'rb') as f:
        I = pickle.load(f)
        return I
def get_nearest_neighbors(model,refine_multiplier):
    pass
def get_performance_results(fileName,I):
    print(fileName)
    recalls=[]
    search_times=[]
    model=None
    dataset=None
    construction_time=None
    if ('HNSW' in fileName) or ('NSG' in fileName) or ('FLATNAV' in fileName):
        if 'HNSW-' in fileName:
            model='HNSW'
            recalls.append(I['recall@'])
            search_times.append(I['search-time'])
            dataset=I['dataset_name']
            construction_time=I['training-time']
        if 'NSG-' in fileName:
            model='NSG'
            recalls.append(I['recall@'])
            search_times.append(I['search-time'])
            dataset=I['dataset_name']
            construction_time=I['training-time']
        if 'FLATNAV-' in fileName:
            model='FLATNAV'
            recalls.append(I['recall'])
            search_times.append(I['search_time'])
            dataset=fileName.split('FLATNAV-')[1].split('-')[0]
            construction_time=I['construction_time']
        for i in range(2,6):
            recalls.append(I['recallx'+str(i)])
            if ('search-timex'+str(i) in I):
                search_times.append(I['search-timex'+str(i)])
            else:
                search_times.append(I['search_timex'+str(i)])
            
        return [model,dataset,fileName,recalls[0],search_times[0],recalls[1],search_times[1],recalls[2],search_times[2],recalls[3],search_times[3],recalls[4],search_times[4],construction_time]
    return None
def get_all_performance_results():
    results=[]
    for file in os.listdir():
        with open(file, 'rb') as f:
            I = pickle.load(f)
            result=get_performance_results(file,I)
            if result!=None:
                results.append(result)
    os.chdir('/data/[username]/similarity-search/models/polyvector_analysis_v1')
    df = pd.DataFrame(results, columns=['model', 'dataset', 'fileName', 'recallx1', 'search_time_x1', 'recallx2', 'search_time_x2', 'recallx3', 'search_time_x3', 'recallx4', 'search_time_x4', 'recallx5', 'search_time_x5','construction-time'])
    df.to_csv('all_augmentation_standalone_graph_methods.csv', index=False)

get_all_performance_results()
