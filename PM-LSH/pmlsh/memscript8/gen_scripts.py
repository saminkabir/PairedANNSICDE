infos=[('ISC_EHB_DepthPhases', 1000000, 300, 300, 256, 100), ('Iquique', 578853, 300, 300, 256, 100), ('MNIST', 69000, 200, 200, 784, 20), ('Meier2019JGR', 1000000, 300, 300, 256, 100), ('Music', 1000000, 100, 100, 100, 100), ('NEIC', 1000000, 300, 300, 256, 100), ('OBS', 1000000, 300, 300, 256, 100), ('OBST2024', 1000000, 300, 300, 256, 100), ('PNW', 1000000, 300, 300, 256, 100), ('Yelp', 77079, 100, 100, 50, 100), ('agnews-mxbai-1024-euclidean', 769382, 100, 100, 1024, 100), ('arxiv-nomic-768-normalized', 1000000, 100, 100, 768, 100), ('astro1m', 1000000, 100, 100, 256, 100), ('audio', 53387, 200, 200, 192, 20), ('bigann', 1000000, 100, 100, 128, 100), ('ccnews-nomic-768-normalized', 495328, 100, 100, 768, 100), ('celeba-resnet-2048-cosine', 201599, 100, 100, 2048, 100), ('cifar', 50000, 200, 200, 512, 20), ('coco-nomic-768-normalized', 282360, 100, 100, 768, 100), ('codesearchnet-jina-768-cosine', 1000000, 100, 100, 768, 100), ('crawl', 1989995, 10000, 10000, 300, 100), ('deep', 1000000, 200, 200, 256, 20), ('enron', 94987, 200, 200, 1369, 20), ('ethz', 36643, 100, 100, 256, 100), ('geofon', 275174, 100, 100, 128, 100), ('gist', 1000000, 1000, 1000, 960, 100), ('glove', 1192514, 200, 200, 100, 20), ('gooaq-distilroberta-768-normalized', 1000000, 100, 100, 768, 100), ('imageNet', 2340373, 200, 200, 150, 20), ('instancegm', 1000000, 100, 100, 128, 100), ('laion-clip-512-normalized', 1000000, 100, 100, 512, 100), ('landmark-nomic-768-normalized', 760757, 100, 100, 768, 100), ('lastfm', 292385, 100, 100, 65, 100), ('lendb', 1000000, 100, 100, 256, 100), ('llama-128-ip', 256921, 100, 100, 128, 100), ('millionSong', 992272, 200, 200, 420, 20), ('movielens', 10677, 1000, 1000, 150, 100), ('netflix', 17770, 1000, 1000, 300, 100), ('notre', 332668, 200, 200, 128, 20), ('nuswide', 268643, 200, 200, 500, 20), ('nytimes', 290000, 100, 100, 256, 100), ('random', 1000000, 200, 200, 100, 20), ('sald1m', 1000000, 100, 100, 128, 100), ('seismic1m', 1000000, 100, 100, 256, 100), ('space1V', 1000000, 100, 100, 100, 100), ('stead', 1000000, 100, 100, 256, 100), ('sun', 79106, 200, 200, 512, 20), ('text-to-image', 1000000, 100, 100, 200, 100), ('tiny5m', 5000000, 1000, 1000, 384, 100), ('trevi', 99900, 200, 200, 4096, 20), ('txed', 519589, 100, 100, 256, 100), ('ukbench', 1097907, 200, 200, 128, 20), ('uqv', 1000000, 10000, 10000, 256, 100), ('vcseis', 160178, 100, 100, 256, 100), ('word2vec', 1000000, 1000, 1000, 300, 100), ('yahoo-minilm-384-normalized', 677305, 100, 100, 384, 100), ('yahoomusic', 136736, 100, 100, 300, 100), ('yandex-200-cosine', 1000000, 100, 100, 200, 100)]



import sys
final_run=''
content=open("big-script-template-1.sh", "r").read()
for info in infos:
    ds=info[0]
    n=str(info[1])
    d=str(info[4])
    k=str(info[5])
    log_file_name = ds+".sh"
    final_run=final_run+'bash '+log_file_name+' & '
    sys.stdout = open(log_file_name, "w")
    new_content=content+''
    new_content=new_content.replace('[N]',n)
    new_content=new_content.replace('[dim]',d)
    new_content=new_content.replace('[K]',k)
    new_content=new_content.replace('[dataset]',ds)
    print(new_content)
final_run=final_run+'echo 1'
sys.stdout = open('run_all.sh', "w")          
print(final_run)           
    
    
