cd ../build

K=100
dim=512
dataset=sun
N=79106

lowDim=10
c=1.5
T=0.2
R_min=1.0
logId=log8

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt

lowDim=10
c=1.75
T=0.2
R_min=1.0
logId=log9

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt

lowDim=10
c=1.25
T=0.2
R_min=1.0
logId=log10

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


lowDim=10
c=1.5
T=0.2
R_min=20.0
logId=log11

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


lowDim=10
c=1.5
T=0.2
R_min=300
logId=log12

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


lowDim=10
c=1.5
T=0.1
R_min=1.0
logId=log13

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


lowDim=10
c=1.5
T=0.4
R_min=1.0
logId=log14

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt


