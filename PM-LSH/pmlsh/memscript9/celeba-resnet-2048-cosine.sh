cd ../build

K=100
dim=2048
dataset=celeba-resnet-2048-cosine
N=201599


lowDim=5
c=1.2
T=0.2
R_min=1.0
logId=log31

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt

lowDim=5
c=1.15
T=0.2
R_min=1.0
logId=log32

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt

lowDim=5
c=1.1
T=0.2
R_min=1.0
logId=log33

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt



