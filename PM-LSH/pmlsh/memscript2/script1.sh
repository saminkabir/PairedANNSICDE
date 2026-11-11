cd ../build
lowDim=15
c=1.5
T=0.2
R_min=1.0
K=20
logId=log1

dim=256
K=100

dataset=ethz
N=36643

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


dataset=vcseis
N=160178


/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


dataset=txed
N=519589


/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt



dataset=lendb
N=1000000


/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


dataset=stead
N=1000000


/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt

dim=128



dataset=geofon
N=275174


/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


dataset=instancegm
N=1000000


/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


