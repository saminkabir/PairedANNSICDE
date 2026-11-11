import os
import pickle

memory_results={}
import pandas as pd

import pysvs as ps
import numpy as np
import math
from multiprocessing import Process, Queue

dataset_path = '/data/[username]/similarity-search/dataset/'

def read_vecs(filePath):
    return ps.read_vecs(dataset_path + filePath)

def get_data_generic(dataset):
    # data we will search through
    xb = read_vecs(dataset+'/base.fvecs')  # 1M samples
    # also get some query vectors to search with
    xq = read_vecs(dataset+'/query.fvecs')
    # take just one query (there are many in sift_learn.fvecs)
    gt = read_vecs(dataset+'/groundtruth.ivecs')
    return xb,xq,gt


datasets=['llama-128-ip_20_variants', 'coco-nomic-768-normalized_20_variants', 'yi-128-ip_20_variants', 'agnews-mxbai-1024-euclidean_20_variants', 'gooaq-distilroberta-768-normalized_20_variants', 'ccnews-nomic-768-normalized_20_variants', 'yandex-200-cosine_20_variants', 'arxiv-nomic-768-normalized_20_variants', 'laion-clip-512-normalized_20_variants','yahoomusic_20_variants']

polyvector_datasets=['llama-128-ip_20_variants', 'coco-nomic-768-normalized_20_variants', 'yi-128-ip_20_variants', 'agnews-mxbai-1024-euclidean_20_variants', 'gooaq-distilroberta-768-normalized_20_variants', 'ccnews-nomic-768-normalized_20_variants', 'yandex-200-cosine_20_variants', 'arxiv-nomic-768-normalized_20_variants', 'laion-clip-512-normalized_20_variants','yahoomusic_20_variants']




datasets=['notre_rc_1.5','notre_rc_1.25','sun_rc_2.0','sun_rc_1.5','sun_rc_1.25','deep_rc_1.25','deep_rc_1.5','deep_rc_2.0','sift_rc_1.25','sift_rc_1.5','sift_rc_2.0','audio_rc_1.25','audio_rc_1.5','audio_rc_2.0','millionSong_rc_1.25','millionSong_rc_1.5','millionSong_rc_2.0','yi-128-ip_20_variants_rc_1.25','yi-128-ip_20_variants_rc_1.5','yi-128-ip_20_variants_rc_2.0','llama-128-ip_20_variants_rc_1.25','llama-128-ip_20_variants_rc_1.5','llama-128-ip_20_variants_rc_2.0','laion-clip-512-normalized_20_variants_rc_1.25','laion-clip-512-normalized_20_variants_rc_1.5','laion-clip-512-normalized_20_variants_rc_2.0']

polyvector_datasets=['notre_rc_1.5','notre_rc_1.25','sun_rc_2.0','sun_rc_1.5','sun_rc_1.25','deep_rc_1.25','deep_rc_1.5','deep_rc_2.0','sift_rc_1.25','sift_rc_1.5','sift_rc_2.0','audio_rc_1.25','audio_rc_1.5','audio_rc_2.0','millionSong_rc_1.25','millionSong_rc_1.5','millionSong_rc_2.0','yi-128-ip_20_variants_rc_1.25','yi-128-ip_20_variants_rc_1.5','yi-128-ip_20_variants_rc_2.0','llama-128-ip_20_variants_rc_1.25','llama-128-ip_20_variants_rc_1.5','llama-128-ip_20_variants_rc_2.0','laion-clip-512-normalized_20_variants_rc_1.25','laion-clip-512-normalized_20_variants_rc_1.5','laion-clip-512-normalized_20_variants_rc_2.0']

datasets=['yi-128-ip_20_variants_50k', 'yi-128-ip_20_variants_500k', 'yi-128-ip_20_variants_10k', 'yi-128-ip_20_variants_100k', 'yi-128-ip_20_variants', 'yandex-200-cosine_20_variants_50k', 'yandex-200-cosine_20_variants_500k', 'yandex-200-cosine_20_variants_10k', 'yandex-200-cosine_20_variants_100k', 'yandex-200-cosine_20_variants', 'uqv_20_variants_50k', 'uqv_20_variants_500k', 'uqv_20_variants_10k', 'uqv_20_variants_100k', 'uqv_20_variants', 'sift_50k', 'sift_500k', 'sift_10k', 'sift_100k', 'sift', 'millionSong_50k', 'millionSong_500k', 'millionSong_10k', 'millionSong_100k', 'millionSong', 'gooaq-distilroberta-768-normalized_20_variants_50k', 'gooaq-distilroberta-768-normalized_20_variants_500k', 'gooaq-distilroberta-768-normalized_20_variants_10k', 'gooaq-distilroberta-768-normalized_20_variants_100k', 'gooaq-distilroberta-768-normalized_20_variants', 'deep_50k', 'deep_500k', 'deep_10k', 'deep_100k', 'deep', 'arxiv-nomic-768-normalized_20_variants_50k', 'arxiv-nomic-768-normalized_20_variants_500k', 'arxiv-nomic-768-normalized_20_variants_10k', 'arxiv-nomic-768-normalized_20_variants_100k', 'arxiv-nomic-768-normalized_20_variants']



polyvector_datasets=['notre_rc_1.25', 'notre_rc_1.5', 'notre_rc_2.0', 'sun_rc_1.25', 'sun_rc_1.5', 'sun_rc_2.0', 'deep_rc_1.25', 'deep_rc_1.5', 'deep_rc_2.0', 'sift_rc_1.25', 'sift_rc_1.5', 'sift_rc_2.0', 'audio_rc_1.25', 'audio_rc_1.5', 'audio_rc_2.0', 'millionSong_rc_1.25', 'millionSong_rc_1.5', 'millionSong_rc_2.0', 'yi-128-ip_20_variants_rc_1.25', 'yi-128-ip_20_variants_rc_1.5', 'yi-128-ip_20_variants_rc_2.0', 'llama-128-ip_20_variants_rc_1.25', 'llama-128-ip_20_variants_rc_1.5', 'llama-128-ip_20_variants_rc_2.0', 'laion-clip-512-normalized_20_variants_rc_1.25', 'laion-clip-512-normalized_20_variants_rc_1.5', 'laion-clip-512-normalized_20_variants_rc_2.0', 'yandex-200-cosine_20_variants_rc_1.25', 'yandex-200-cosine_20_variants_rc_1.5', 'yandex-200-cosine_20_variants_rc_2.0', 'coco-nomic-768-normalized_20_variants_rc_1.25', 'coco-nomic-768-normalized_20_variants_rc_1.5', 'coco-nomic-768-normalized_20_variants_rc_2.0', 'ccnews-nomic-768-normalized_20_variants_rc_1.25', 'ccnews-nomic-768-normalized_20_variants_rc_1.5', 'ccnews-nomic-768-normalized_20_variants_rc_2.0', 'yahoomusic_20_variants_rc_1.25', 'yahoomusic_20_variants_rc_1.5', 'yahoomusic_20_variants_rc_2.0', 'imageNet_rc_1.25', 'imageNet_rc_1.5', 'imageNet_rc_2.0', 'glove_rc_1.25', 'glove_rc_1.5', 'glove_rc_2.0', 'MNIST_rc_1.25', 'MNIST_rc_1.5', 'MNIST_rc_2.0', 'nuswide_rc_1.25', 'nuswide_rc_1.5', 'nuswide_rc_2.0']


datasets=['notre_rc_1.25', 'notre_rc_1.5', 'notre_rc_2.0', 'sun_rc_1.25', 'sun_rc_1.5', 'sun_rc_2.0', 'deep_rc_1.25', 'deep_rc_1.5', 'deep_rc_2.0', 'sift_rc_1.25', 'sift_rc_1.5', 'sift_rc_2.0', 'audio_rc_1.25', 'audio_rc_1.5', 'audio_rc_2.0', 'millionSong_rc_1.25', 'millionSong_rc_1.5', 'millionSong_rc_2.0', 'yi-128-ip_20_variants_rc_1.25', 'yi-128-ip_20_variants_rc_1.5', 'yi-128-ip_20_variants_rc_2.0', 'llama-128-ip_20_variants_rc_1.25', 'llama-128-ip_20_variants_rc_1.5', 'llama-128-ip_20_variants_rc_2.0', 'laion-clip-512-normalized_20_variants_rc_1.25', 'laion-clip-512-normalized_20_variants_rc_1.5', 'laion-clip-512-normalized_20_variants_rc_2.0', 'yandex-200-cosine_20_variants_rc_1.25', 'yandex-200-cosine_20_variants_rc_1.5', 'yandex-200-cosine_20_variants_rc_2.0', 'coco-nomic-768-normalized_20_variants_rc_1.25', 'coco-nomic-768-normalized_20_variants_rc_1.5', 'coco-nomic-768-normalized_20_variants_rc_2.0', 'ccnews-nomic-768-normalized_20_variants_rc_1.25', 'ccnews-nomic-768-normalized_20_variants_rc_1.5', 'ccnews-nomic-768-normalized_20_variants_rc_2.0', 'yahoomusic_20_variants_rc_1.25', 'yahoomusic_20_variants_rc_1.5', 'yahoomusic_20_variants_rc_2.0', 'imageNet_rc_1.25', 'imageNet_rc_1.5', 'imageNet_rc_2.0', 'glove_rc_1.25', 'glove_rc_1.5', 'glove_rc_2.0', 'MNIST_rc_1.25', 'MNIST_rc_1.5', 'MNIST_rc_2.0', 'nuswide_rc_1.25', 'nuswide_rc_1.5', 'nuswide_rc_2.0']



polyvector_datasets=['notre_rc_1.25', 'notre_rc_1.5', 'notre_rc_2.0', 'sun_rc_1.25', 'sun_rc_1.5', 'sun_rc_2.0', 'deep_rc_1.25', 'deep_rc_1.5', 'deep_rc_2.0', 'sift_rc_1.25', 'sift_rc_1.5', 'sift_rc_2.0', 'audio_rc_1.25', 'audio_rc_1.5', 'audio_rc_2.0', 'millionSong_rc_1.25', 'millionSong_rc_1.5', 'millionSong_rc_2.0', 'yi-128-ip_20_variants_rc_1.25', 'yi-128-ip_20_variants_rc_1.5', 'yi-128-ip_20_variants_rc_2.0', 'llama-128-ip_20_variants_rc_1.25', 'llama-128-ip_20_variants_rc_1.5', 'llama-128-ip_20_variants_rc_2.0', 'laion-clip-512-normalized_20_variants_rc_1.25', 'laion-clip-512-normalized_20_variants_rc_1.5', 'laion-clip-512-normalized_20_variants_rc_2.0', 'yandex-200-cosine_20_variants_rc_1.25', 'yandex-200-cosine_20_variants_rc_1.5', 'yandex-200-cosine_20_variants_rc_2.0', 'coco-nomic-768-normalized_20_variants_rc_1.25', 'coco-nomic-768-normalized_20_variants_rc_1.5', 'coco-nomic-768-normalized_20_variants_rc_2.0', 'ccnews-nomic-768-normalized_20_variants_rc_1.25', 'ccnews-nomic-768-normalized_20_variants_rc_1.5', 'ccnews-nomic-768-normalized_20_variants_rc_2.0', 'yahoomusic_20_variants_rc_1.25', 'yahoomusic_20_variants_rc_1.5', 'yahoomusic_20_variants_rc_2.0', 'imageNet_rc_1.25', 'imageNet_rc_1.5', 'imageNet_rc_2.0', 'glove_rc_1.25', 'glove_rc_1.5', 'glove_rc_2.0', 'MNIST_rc_1.25', 'MNIST_rc_1.5', 'MNIST_rc_2.0', 'nuswide_rc_1.25', 'nuswide_rc_1.5', 'nuswide_rc_2.0']





dataset_info={}

dataset_infos=set()


for dataset in polyvector_datasets:
    xb,xq,gt=get_data_generic(dataset)
    dataset_info[dataset]=(xb.shape[0],xq.shape[0],xb.shape[1],gt.shape[1])
    
for dataset in datasets:
    xb,xq,gt=get_data_generic(dataset)
    dataset_infos.add((dataset,xb.shape[0],xq.shape[0],xq.shape[0],xb.shape[1],gt.shape[1]))
    
print(dataset_infos)

def get_qn(dataset):
    if not dataset in datasets:
        return 200
    return dataset_info[dataset][1]
def get_k(dataset):
    if not dataset in datasets:
        return 100
    return dataset_info[dataset][3]

def refined_recall(dataset, I_list):
    xb,xq,gt=get_data_generic(dataset)
    k_gt=gt.shape[1]
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

def get_dataset(fileName):
    print(fileName)
    fileName=fileName.replace('FLATNAV-','')
    fileNames=fileName.split('-')
    dd=''
    for i in range(len(fileNames)-3):
        if not (i==0):
            dd=dd+'-'
        dd=dd+fileNames[i]
    print(dd)
    for dataset in datasets:
        if dataset == dd:
            return dataset
    return None




def get_memory_info_PQ_OPQ_IVFPQ():
    global memory_results
    os.chdir('/data/[username]/similarity-search/models/VAQ-SINGLE/data/faiss-library/logIVFPQ')
    for file in os.listdir():
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            if "'dataset_name': '" in content and "Maximum resident set size (kbytes): " in content:
                dataset_name=content.split("'dataset_name': '")[1].split("'")[0]
                # if dataset_name in datasets:
                memory=int(content.split("Maximum resident set size (kbytes): ")[1].split('\n')[0])
                model=content.split("'model_name': '")[1].split("'")[0]
                m=int(content.split("'m': ")[1].split(',')[0])
                nbits=int(content.split("'nbits': ")[1].split(',')[0])
                multiplier=1
                if model=='PQ':
                    multiplier=1053692/2572864
                if model=='OPQ':
                    multiplier=2060600/2613884
                if model=='IVFPQ':
                    multiplier=2303012/2568492
                    
                memory_results[model+'-'+dataset_name+'-'+str(m)+'-'+str(nbits)]=int(memory*multiplier)
                
def get_memory_info_generic(model,loc):
    global memory_results
    os.chdir(loc)
    for file in os.listdir():
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            if "Maximum resident set size (kbytes): " in content:
                memory=int(content.split("Maximum resident set size (kbytes): ")[1].split('\n')[0])
                ks=model+'-'+file.split('.txt')[0]
                if model=='scann' and (not ks.endswith('-0.2')):
                    ks=ks+'-0.2'
                memory_results[ks]=memory
                

results=[]
vaq_standalone_results=[]
                    
def get_hnsw_results():
    global results
    os.chdir('/data/[username]/similarity-search/models/results_pickles_v8')
    for file in os.listdir():
        if 'HNSW' in file and '.pkl' in file:
            with open(file, "rb") as f:
                info = pickle.load(f)
                # print(info)
                res=dict()
                recallx1=info['recall@']
                recallx2=info['recallx2']
                dataset=info['dataset_name']
                I=info['I']
                search_timex1=info['search-time']
                search_timex2=info['search_timex2']
                construction_time=info['training-time']
                ks='HNSW-'+dataset+'-'+str(info['m'])+'-'+str(info['ef_search'])+'-'+str(info['ef_construction'])
                if dataset in datasets:
                    # print(ks)
                    if ks in memory_results:
                        ds={
                            'model': 'HNSW',
                            'dataset': dataset,
                            'recallx1': recallx1,
                            'recallx2': recallx2,
                            'I': I,
                            'search_timex1': search_timex1,
                            'search_timex2': search_timex2,
                            'construction_time': construction_time,
                            'memory':memory_results[ks],
                            'ks': ks
                        }
                        results.append(ds)
    return results




                    
def get_nsg_results():
    global results
    os.chdir('/data/[username]/similarity-search/models/results_pickles_v8')
    for file in os.listdir():
        if 'NSG' in file and '.pkl' in file:
            with open(file, "rb") as f:
                info = pickle.load(f)
                # print(info)
                res=dict()
                recallx1=info['recall@']
                recallx2=info['recallx2']
                dataset=info['dataset_name']
                I=info['I']
                search_timex1=info['search-time']
                search_timex2=info['search-timex2']
                construction_time=info['training-time']
                ks='NSG-'+dataset+'-'+str(info['m'])
                if dataset in datasets:
                    # print(ks)
                    if ks in memory_results:
                        ds={
                            'model': 'NSG',
                            'dataset': dataset,
                            'recallx1': recallx1,
                            'recallx2': recallx2,
                            'I': I,
                            'search_timex1': search_timex1,
                            'search_timex2': search_timex2,
                            'construction_time': construction_time,
                            'memory':memory_results[ks],
                            'ks': 'NSG-'+dataset+'-'+str(info['m'])+'-'+str(info['search_L'])
                        }
                        results.append(ds)
    return results




                    
def get_flatnav_results():
    global results
    os.chdir('/data/[username]/similarity-search/models/results_pickles_v7')
    for file in os.listdir():
        if 'FLATNAV' in file and '.pkl' in file:
            with open(file, "rb") as f:
                info = pickle.load(f)
                # print(info)
                res=dict()
                recallx1=info['recall']
                recallx2=info['recallx2']
                dataset=get_dataset(file)
                if not dataset == None:
                    I=info['I']
                    search_timex1=info['search_time']
                    search_timex2=info['search_timex2']
                    construction_time=info['construction_time']
                    ks='FLATNAV-'+dataset+'-'+str(info['max_edges_per_node'])+'-'+str(info['ef_construction'])+'-'+str(info['ef_search'])
                    # print(ks)
                    if not dataset == None:
                        # print(ks)
                        if ks in memory_results:
                            ds={
                                'model': 'FLATNAV',
                                'dataset': dataset,
                                'recallx1': recallx1,
                                'recallx2': recallx2,
                                'I': I,
                                'search_timex1': search_timex1,
                                'search_timex2': search_timex2,
                                'construction_time': construction_time,
                                'memory':memory_results[ks],
                                'ks': ks
                            }
                            results.append(ds)
    return results


def get_memory_info_VAQ(location):
    global memory_results
    os.chdir(location)
    for file in os.listdir():
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            if "Maximum resident set size (kbytes): " in content:
                memory=int(content.split("Maximum resident set size (kbytes): ")[1].split('\n')[0])
                ks='vaq-log-'+file.replace('-mem.txt','')
                memory_results[ks]=int(memory)
    os.chdir('/data/[username]/similarity-search/models/polyvector_analysis_v1')



def extract_vaq_log(location,id):
    os.chdir(location)
    global memory_results
    global results
    datas=[]
    for fileName in os.listdir():
        recalls=[]
        search_time=[]
        training_time=None
        construction_time=None
        dataset=None
        with open(fileName, "r", encoding="utf-8") as f:
            for line in f:
                if '== Training time: ' in line:
                    training_time=float(line.split('== Training time: ')[1].split(' s')[0])
                if '== Encoding time: ' in line:
                    construction_time=float(line.split('== Encoding time: ')[1].split(' s')[0])
                if 'Querying time: ' in line:
                    search_time.append(float(line.split('Querying time: ')[1].split(' s')[0]))
                if 'recall@R (Probably correct): ' in line:
                    recalls.append(float(line.split('recall@R (Probably correct): ')[1].split(' ')[0]))
                if 'dataset = ../../../dataset/' in line:
                    dataset=line.split('dataset = ../../../dataset/')[1].split('/')[0]
            if dataset!=None and dataset in polyvector_datasets:
                dataset=get_dataset(dataset)
                ks=fileName.replace('.txt','')
                if len(recalls)>=2 and len(search_time)>=2:
                    ivec_file_location='../results/'+ks.replace('vaq-log-','')+'-'+str(get_k(dataset))+'.ivecs'
                    I=ps.read_vecs(ivec_file_location)
                    with open('../'+ks.replace('vaq-log-','')+'-mem.txt', "r", encoding="utf-8") as f:
                        content = f.read()
                        if "Maximum resident set size (kbytes): " in content:
                            memory=int(content.split("Maximum resident set size (kbytes): ")[1].split('\n')[0])
                            ds={
                                'model': 'VAQ',
                                'dataset': dataset,
                                'recallx1': recalls[0],
                                'recallx2': recalls[1],
                                'recalls': recalls,
                                'search_times':search_time,
                                'I': I,
                                'search_timex1': search_time[0],
                                'search_timex2': search_time[1],
                                'construction_time': training_time+construction_time,
                                'memory':memory,
                                'ks': ks,
                                'id': id,
                                'bits': int(ks.split('VAQ')[1].split('m')[0])
                            }
                            results.append(ds)
    os.chdir('/data/[username]/similarity-search/models/polyvector_analysis_v1')



            

            
            
                    
def get_annoy_results():
    global results
    os.chdir('/data/[username]/similarity-search/models/results_pickles_v8')
    for file in os.listdir():
        if 'annoy' in file and '.pkl' in file:
            with open(file, "rb") as f:
                info = pickle.load(f)
                # print(info)
                res=dict()
                recallx1=info['recall']
                recallx2=info['recallx2']
                dataset=info['dataset_name']
                if not dataset == None:
                    I=info['I']
                    search_timex1=info['search-time']
                    search_timex2=info['search-timex2']
                    construction_time=info['construction-time']
                    ks='annoy-'+dataset+'-'+str(info['tree'])+'-'+'8'
                    # print(ks)
                    if dataset in polyvector_datasets:
                        # print(ks)
                        if ks in memory_results:
                            ds={
                                'model': 'annoy',
                                'dataset': dataset,
                                'recallx1': recallx1,
                                'recallx2': recallx2,
                                'I': I,
                                'search_timex1': search_timex1,
                                'search_timex2': search_timex2,
                                'construction_time': construction_time,
                                'memory':memory_results[ks],
                                'ks': ks
                            }
                            results.append(ds)
    return results



def get_scann_results():
    global results
    os.chdir('/data/[username]/similarity-search/models/results_pickles_v8')
    for file in os.listdir():
        if 'scann' in file and '.pkl' in file:
            with open(file, "rb") as f:
                info = pickle.load(f)
                # print(info)
                res=dict()
                recallx1=info['recall']
                recallx2=info['recallx2']
                dataset=info['dataset_name']
                I=info['I_1']
                search_timex1=info['search_time']
                search_timex2=info['search_timex2']
                construction_time=info['construction_time']
                ks='scann-'+dataset+'-'+str(info['num_leaves_'])+'-'+str(info['num_leaves_to_search_'])+'-'+str(info['num_neighbors_'])+'-'+str(info['anisotropic_quantization_threshold_'])
                if dataset in polyvector_datasets:
                    # print(ks)
                    if ks in memory_results:
                        ds={
                            'model': 'scann',
                            'dataset': dataset,
                            'recallx1': recallx1,
                            'recallx2': recallx2,
                            'I': I,
                            'search_timex1': search_timex1,
                            'search_timex2': search_timex2,
                            'construction_time': construction_time,
                            'memory':memory_results[ks],
                            'ks': ks
                        }
                        print(ks)
                        results.append(ds)
    return results

def get_pq_results():
    global results
    os.chdir('/data/[username]/similarity-search/models/results_pickles_v8')
    for file in os.listdir():
        if 'PQ' in file and ('-8.pkl' in file or '-4.pkl' in file):
            with open(file, "rb") as f:
                info = pickle.load(f)
                # print(info)
                res=dict()
                dataset=info['dataset_name']
                if dataset in polyvector_datasets:
                    _k=get_k(dataset)
                    recallx1=info['recall@x'+str(_k)]
                    recallx2=info['recall@x'+str(2*_k)]
                    print(file)
                    I=None
                    if 'I' in info.keys():
                        I=info['I']
                    else:
                        I=info['I_pq']
                    search_timex1=info['search-timex'+str(_k)]
                    search_timex2=info['search-timex'+str(_k*2)]
                    construction_time=info['training-time']
                    ks=info['model_name']+'-'+dataset+'-'+str(info['m'])+'-'+str(info['nbits'])
                    if dataset in datasets:
                        # print(ks)
                        if ks in memory_results:
                            ds={
                                'model': info['model_name'],
                                'dataset': dataset,
                                'recallx1': recallx1,
                                'recallx2': recallx2,
                                'I': I,
                                'search_timex1': search_timex1,
                                'search_timex2': search_timex2,
                                'construction_time': construction_time,
                                'memory':memory_results[ks],
                                'ks': ks
                            }
                            results.append(ds)
                            print(ks)
    return results

get_memory_info_PQ_OPQ_IVFPQ()
get_memory_info_generic('scann','/data/[username]/similarity-search/models/VAQ-SINGLE/data/faiss-library/logScann')
get_memory_info_generic('annoy','/data/[username]/similarity-search/models/VAQ-SINGLE/data/faiss-library/logANNOY')
get_memory_info_generic('HNSW','/data/[username]/similarity-search/models/VAQ-SINGLE/data/faiss-library/logHNSW')
get_memory_info_generic('NSG','/data/[username]/similarity-search/models/VAQ-SINGLE/data/faiss-library/logNSG')
get_memory_info_generic('FLATNAV','/data/[username]/similarity-search/models/VAQ-SINGLE/data/faiss-library/flatnavogs')

                

get_hnsw_results()
get_nsg_results()
get_flatnav_results()
get_annoy_results()
get_scann_results()
get_pq_results()
# extract_vaq_log('/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-42/finalTests2',42)
# extract_vaq_log('/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-41/finalTests2',41)
extract_vaq_log('/data/[username]/similarity-search/models/vaq-testing/analyse-multi-vaq-46/finalTests2',46)
 
vaq_standalone=[]
for result in results:
    if result['model']=='VAQ':
        model='VAQ'
        dataset=result['dataset']
        construction_time=result['construction_time']
        memory=result['memory']
        qs=get_qn(dataset)
        fileName=result['ks']+'.txt'
        for i in range(len(result['recalls'])):
            data=[dataset,1,result['recalls'][i],qs/result['search_times'][i],construction_time,memory,i+1,result['bits'],fileName,result['id']]
            vaq_standalone.append(data)

y=['dataset','multiplier','recall','qps','construction_time','memory','refinementx','bits','fileName','id']
df_out = pd.DataFrame(vaq_standalone, columns=y)
os.chdir('/data/[username]/similarity-search/models/polyvector_analysis_v1')
# save as CSV
output_file = "all_possible_standalone_vaq_new_datasets_k_20.csv"
# df_out.to_csv(output_file, index=False) 

models=['NSG','HNSW','FLATNAV']
tree_models=['scann','annoy']

columns_standalone_graph=['model','qn','dataset','fileName','recall without refinemet','search_time without refinemnt','qps','recall with 2k refinement','search_time with 2k refienement','qps with 2k refienement','construction time','memory']


columns_augmented_graph=['dataset','model1','file1',"model1's recall","model1's qps","model1's recall with refinement","model1's qps with refinement","model1's construction time","model2","file2","model2's recall","model2's qps","model2's recall with refinement","model2's qps with refinement","model2's construction time","augmented model's recall with refinement","oracle's recall with refinement","augmented model's qps with refinement in parallal settings","augmented model's qps with refinement in sequential settings","augmented model's construction time in parallal settings","augmented model's construction time in sequential settings","model1 memory","model2 memory","augmented model memory"]



def get_graph_results(dataset,test):
    results_augmented_graph=[]
    results_standalone_graph=[]
    for i in range(0,len(models)):
        model1_results=[]
        for result in results:
            if result['model']==models[i] and result['dataset']==dataset:
                model1_results.append(result)
        for result in model1_results:
            data=[result['model'],get_qn(result['dataset']),result['dataset'],result['ks'],result['recallx1'],result['search_timex1'],get_qn(result['dataset'])/result['search_timex1'],result['recallx2'],result['search_timex2'],get_qn(result['dataset'])/result['search_timex2'],result['construction_time'],result['memory']]
            results_standalone_graph.append(data)
        for j in range(i+1,len(models)):
            model2_results=[]
            for result in results:
                if result['model']==models[j] and result['dataset']==dataset:
                    model2_results.append(result)
            for result1 in model1_results:
                for result2 in model2_results:
                    if result1['dataset']==result2['dataset']:
                        augmented_model_recall=refined_recall(result1['dataset'],[result1['I'],result2['I']])
                        data=[result1['dataset'],result1['model'],result1['ks'],result1['recallx1'],get_qn(result1['dataset'])/result1['search_timex1'],result1['recallx2'],get_qn(result1['dataset'])/result1['search_timex2'],result1['construction_time'],result2['model'],result2['ks'],result2['recallx1'],get_qn(result2['dataset'])/result2['search_timex1'],result2['recallx2'],get_qn(result2['dataset'])/result2['search_timex2'],result2['construction_time'],augmented_model_recall,max(result2['recallx2'],result1['recallx2']),get_qn(result2['dataset'])/max(result1['search_timex1'],result2['search_timex1']),get_qn(result2['dataset'])/(result1['search_timex1']+result2['search_timex1']),max(result1['construction_time'],result2['construction_time']),(result1['construction_time']+result2['construction_time']),result1['memory'],result2['memory'],result1['memory']+result2['memory']]
                        results_augmented_graph.append(data)
    df_standalone_graph = pd.DataFrame(results_standalone_graph, columns=columns_standalone_graph)
    df_augmented_graph = pd.DataFrame(results_augmented_graph, columns=columns_augmented_graph)
    os.chdir('/data/[username]/similarity-search/models/polyvector_analysis_v1')
    df_standalone_graph.to_csv('graph_new_datasets_standalone_results_rc_v2/new_datasets_standalone_graph_methods('+dataset+').csv', index=False) 
    df_augmented_graph.to_csv('graph_new_datasets_augmented_results_rc_v2/new_datasets_augmenred_graph_methods('+dataset+').csv', index=False) 
    
tree_models=['scann','annoy']
columns_augmented_tree=["model","dataset","construction_time (parallel)","construction_time (sequential)","search_time","search_timex_refined (parallel)","search_timex_refined (sequential)","recall","refined_recall","memory (sequential)","scann_peram","annoy_peram"]


def get_tree_results(dataset,test):
    results_augmented_graph=[]
    for i in range(0,len(tree_models)):
        model1_results=[]
        for result in results:
            if result['model']==tree_models[i] and result['dataset']==dataset:
                model1_results.append(result)
        for result in model1_results:
            data=[result['model'],result['dataset'],result['construction_time'],result['construction_time'],result['search_timex1'],result['search_timex2'],result['search_timex2'],result['recallx1'],result['recallx2'],result['memory']]
            if result['model']=='scann':
                data.append(result['ks'])
                data.append('')
            else:
                data.append('')
                data.append(result['ks'])
            results_augmented_graph.append(data)
        for j in range(i+1,len(tree_models)):
            model2_results=[]
            for result in results:
                if result['model']==tree_models[j] and result['dataset']==dataset:
                    model2_results.append(result)
            for result1 in model1_results:
                for result2 in model2_results:
                    if result1['dataset']==result2['dataset']:
                        augmented_model_recall=refined_recall(result1['dataset'],[result1['I'],result2['I']])
                        data=['scann-annoy',result1['dataset'],max(result1['construction_time'],result2['construction_time']),result1['construction_time']+result2['construction_time'],'-',max(result1['search_timex1'],result2['search_timex1']),result1['search_timex1']+result2['search_timex1'],'-',augmented_model_recall,result1['memory']+result2['memory'],result1['ks'],result2['ks']]
                        results_augmented_graph.append(data)
    df_augmented_graph = pd.DataFrame(results_augmented_graph, columns=columns_augmented_tree)
    os.chdir('/data/[username]/similarity-search/models/polyvector_analysis_v1')
    df_augmented_graph.to_csv('tree_new_datasets_results_k_20/new_datasets_augmenred_tree_methods('+dataset+').csv', index=False) 

quantization_models=['VAQ','PQ','OPQ','IVFPQ']

columns_standalone_quant=['model','qs','dataset','parameter','construction-time','recallx1','searchx1','qpsx1','recallx2','searchx2','qpsx2','memory_usage','file']





columns_augmented_quant=["dataset","qs","model1","model1 parameter","model1 search-time without refinement","model1 recall without refinement","model1 search-time with refinement","model1 recall with refinement","model1 construction time","model1 memory usage","model2","model2 parameter","model2 search-time without refinement","model2 recall without refinement","model2 search-time with refinement","model2 recall with refinement","model2 construction time","model2 memory usage","augmented model recall","augmented qps","augmented construction time","augmented memory"]



def get_quant_results(dataset,test):
    results_augmented_graph=[]
    results_standalone_graph=[]
    for i in range(0,len(quantization_models)):
        model1_results=[]
        for result in results:
            if result['model']==quantization_models[i] and result['dataset']==dataset:
                model1_results.append(result)
        for result in model1_results:
            data=[result['model'],get_qn(result['dataset']),result['dataset'],result['ks'],result['construction_time'],result['recallx1'],result['search_timex1'],get_qn(result['dataset'])/result['search_timex1'],result['recallx2'],result['search_timex2'],get_qn(result['dataset'])/result['search_timex2'],result['memory'],result['ks']]
            results_standalone_graph.append(data)
        for j in range(i+1,len(quantization_models)):
            model2_results=[]
            for result in results:
                if result['model']==quantization_models[j] and result['dataset']==dataset:
                    model2_results.append(result)
            for result1 in model1_results:
                for result2 in model2_results:
                    if result1['dataset']==result2['dataset']:
                        augmented_model_recall=refined_recall(result1['dataset'],[result1['I'],result2['I']])

                        data=[result1['dataset'],get_qn(result2['dataset']),result1['model'],result1['ks'],result1['search_timex1'],result1['recallx1'],result1['search_timex2'],result1['recallx2'],result1['construction_time'],result1['memory'],result2['model'],result2['ks'],result2['search_timex1'],result2['recallx1'],result2['search_timex2'],result2['recallx2'],result2['construction_time'],result2['memory'],augmented_model_recall,get_qn(result2['dataset'])/max(result1['search_timex1'],result2['search_timex1']),max(result1['construction_time'],result2['construction_time']),result1['memory']+result2['memory']]
                        results_augmented_graph.append(data)
                        
    df_standalone_graph = pd.DataFrame(results_standalone_graph, columns=columns_standalone_quant)
    df_augmented_graph = pd.DataFrame(results_augmented_graph, columns=columns_augmented_quant)
    os.chdir('/data/[username]/similarity-search/models/polyvector_analysis_v1')
    df_standalone_graph.to_csv('quant_new_datasets_standalone_results_k_20/new_datasets_standalone_quant_methods('+dataset+').csv', index=False) 
    df_augmented_graph.to_csv('quant_new_datasets_augmented_results_k_20/new_datasets_augmenred_quant_methods('+dataset+').csv', index=False) 

if __name__ == "__main__":
    pss=[]
    for dataset in datasets:
        p = Process(target=get_graph_results, args=(dataset,'test'))
        pss.append(p)
    # for dataset in polyvector_datasets:
    #     p = Process(target=get_tree_results, args=(dataset,'test'))
    #     pss.append(p)
    # for dataset in datasets:
    #     p = Process(target=get_quant_results, args=(dataset,'test'))
    #     pss.append(p)
    for _p in pss:
        _p.start()
    for _p in pss:
        _p.join()

    
                    
    
