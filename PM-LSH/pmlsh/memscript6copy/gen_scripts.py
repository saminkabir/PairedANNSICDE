infos=[('deep', 1000000, 200, 200, 256, 20),('glove', 1192514, 200, 200, 100, 20),('sun', 79106, 200, 200, 512, 20),('audio', 53387, 200, 200, 192, 20),('millionSong', 992272, 200, 200, 420, 20),('nuswide', 268643, 200, 200, 500, 20),('MNIST', 69000, 200, 200, 784, 20),('notre', 332668, 200, 200, 128, 20),('sift', 1000000, 10000, 10000, 128, 100),('imageNet', 2340373, 200, 200, 150, 20)]



import sys
final_run=''
content=open("big-script-template-1.sh", "r").read()
for info in infos:
    ds=info[0]
    n=str(info[1])
    d=str(info[4])
    qn=str(info[5])
    log_file_name = ds+".sh"
    final_run=final_run+'bash '+log_file_name+' & '
    sys.stdout = open(log_file_name, "w")
    new_content=content+''
    new_content=new_content.replace('[N]',n)
    new_content=new_content.replace('[dim]',d)
    new_content=new_content.replace('[K]','100')
    new_content=new_content.replace('[dataset]',ds)
    print(new_content)
final_run=final_run+'echo 1'
sys.stdout = open('run_all.sh', "w")          
print(final_run)           
    
    
