tmp='python all_possible_aug_tree_graph.py --dataset [dataset]'

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

for dataset in datasets:
    command = tmp.replace("[dataset]", dataset)
    print(command,end=" & ")
print('echo 1')
