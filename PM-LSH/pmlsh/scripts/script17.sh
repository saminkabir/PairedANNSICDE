cd ../build
c=1.5
T=0.4
R_min=1.0





dataset=random
N=1000000
dim=1000000
K=20

lowDim=30

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K



T=0.5

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

dataset=seismic1m
N=1000000
dim=256
K=100

lowDim=30

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


