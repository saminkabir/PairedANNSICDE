cd ../build



dataset=nytimes
N=290000
dim=256
K=100

lowDim=15
c=1.75
T=0.2
R_min=1.0
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

lowDim=20
c=1.75
T=0.4
R_min=1.0
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

lowDim=30
c=1.75
T=0.2
R_min=1.0
K=20
./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

lowDim=30
c=1.75
T=0.4
R_min=1.0
K=20

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K
