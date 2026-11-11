cd ../build

K=100
dim=256
dataset=PNW
N=1000000

lowDim=5
c=1.5
T=0.2
R_min=1.0
logId=log15

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt

lowDim=5
c=1.75
T=0.2
R_min=1.0
logId=log16

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt

lowDim=5
c=1.25
T=0.2
R_min=1.0
logId=log17

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


lowDim=5
c=1.5
T=0.2
R_min=20.0
logId=log18

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


lowDim=5
c=1.5
T=0.2
R_min=300
logId=log19

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


lowDim=5
c=1.5
T=0.1
R_min=1.0
logId=log20

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


lowDim=5
c=1.5
T=0.4
R_min=1.0
logId=log21

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


