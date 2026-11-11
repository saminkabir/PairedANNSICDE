cd ../build



dataset=imageNet
N=2340373
dim=150

lowDim=15
c=1.25
T=0.4
R_min=1.0
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

lowDim=20
c=1.25
T=0.4
R_min=1.0
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

lowDim=30
c=1.25
T=0.4
R_min=1.0
K=20

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

lowDim=30
c=1.25
T=0.4
R_min=300
K=20

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K
