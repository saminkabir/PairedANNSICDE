cd ../build

K=100
dim=256
dataset=OBST2024
N=1000000

lowDim=15
c=1.5
T=0.2
R_min=1.0
logId=log1

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt

lowDim=15
c=1.75
T=0.2
R_min=1.0
logId=log2

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt

lowDim=15
c=1.25
T=0.2
R_min=1.0
logId=log3

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


lowDim=15
c=1.5
T=0.2
R_min=20.0
logId=log4

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


lowDim=15
c=1.5
T=0.2
R_min=300
logId=log5

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


lowDim=15
c=1.5
T=0.1
R_min=1.0
logId=log6

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


lowDim=15
c=1.5
T=0.4
R_min=1.0
logId=log7

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt



