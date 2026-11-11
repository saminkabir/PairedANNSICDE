cd ../build

K=[K]
dim=[dim]
dataset=[dataset]
N=[N]


lowDim=10
c=1.5
T=0.2
R_min=1.0
logId=log8

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt

lowDim=10
c=1.75
T=0.2
R_min=1.0
logId=log9

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt

lowDim=10
c=1.25
T=0.2
R_min=1.0
logId=log10

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt


lowDim=10
c=1.5
T=0.2
R_min=20.0
logId=log11

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt


lowDim=10
c=1.5
T=0.2
R_min=300
logId=log12

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt


lowDim=10
c=1.5
T=0.1
R_min=1.0
logId=log13

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt


lowDim=10
c=1.5
T=0.4
R_min=1.0
logId=log14

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt


lowDim=5
c=1.5
T=0.2
R_min=1.0
logId=log15

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt

lowDim=5
c=1.75
T=0.2
R_min=1.0
logId=log16

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt

lowDim=5
c=1.25
T=0.2
R_min=1.0
logId=log17

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt


lowDim=5
c=1.5
T=0.2
R_min=20.0
logId=log18

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt


lowDim=5
c=1.5
T=0.2
R_min=300
logId=log19

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt


lowDim=5
c=1.5
T=0.1
R_min=1.0
logId=log20

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt


lowDim=5
c=1.5
T=0.4
R_min=1.0
logId=log21

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt


lowDim=5
c=1.2
T=0.2
R_min=1.0
logId=log31

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt

lowDim=5
c=1.15
T=0.2
R_min=1.0
logId=log32

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt

lowDim=5
c=1.1
T=0.2
R_min=1.0
logId=log30

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt


lowDim=4
c=1.25
T=0.2
R_min=1.0
logId=log33

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt

lowDim=3
c=1.25
T=0.2
R_min=1.0
logId=log34

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt

lowDim=2
c=1.25
T=0.2
R_min=1.0
logId=log35

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K ${dataset}-${logId}.ivecs &> ../logs/${dataset}-${logId}.txt



