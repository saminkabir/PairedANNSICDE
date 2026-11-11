cd ../build
lowDim=15
c=1.5
T=0.2
R_min=300
K=20




dataset=crawl
N=1989995
dim=300
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K



