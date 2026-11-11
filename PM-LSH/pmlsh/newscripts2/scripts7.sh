cd ../build
lowDim=10
c=1.5
T=0.08
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



lowDim=15
c=2
T=0.08
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




lowDim=10
c=1.75
T=0.08
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

