infos=[('MNIST_hundred', 69000, 200, 200, 784, 100), ('audio_hundred', 53387, 200, 200, 192, 100), ('deep_hundred', 1000000, 200, 200, 256, 100), ('glove_hundred', 1192514, 200, 200, 100, 100), ('imageNet_hundred', 2340373, 200, 200, 150, 100), ('millionSong_hundred', 992272, 200, 200, 420, 100), ('notre_hundred', 332668, 200, 200, 128, 100), ('nuswide_hundred', 268643, 200, 200, 500, 100), ('sun_hundred', 79106, 200, 200, 512, 100), ('sift_twenty', 1000000, 10000, 10000, 128, 20)]



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
    
    
