cd ../build
lowDim=15
c=1.5
T=0.1
R_min=1.0

dataset=crawl
N=1989995
dim=300
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

dataset=tiny5m
N=5000000
dim=384
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


T=0.075

dataset=crawl
N=1989995
dim=300
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

dataset=tiny5m
N=5000000
dim=384
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K



T=0.05

dataset=crawl
N=1989995
dim=300
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

dataset=tiny5m
N=5000000
dim=384
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

