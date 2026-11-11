cd ../build
lowDim=15
c=1.75
T=0.2
R_min=1.0
K=20
logId=log2

dataset=sift
N=1000000
dim=128
K=100

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


dataset=uqv
N=1000000
dim=256
K=100


/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


dataset=ukbench
N=1097907
dim=128
K=20


/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt



dataset=deep
N=1000000
dim=256
K=20


/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


dataset=imageNet
N=2340373
dim=150
K=20


/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


