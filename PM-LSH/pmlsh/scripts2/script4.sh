cd ../build
lowDim=15
c=1.5
T=0.2
R_min=20.0
K=20

dataset=glove
N=1192514
dim=100
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=lastfm
N=292385
dim=65
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=movielens
N=10677
dim=150
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=netflix
N=17770
dim=300
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=nytimes
N=290000
dim=256
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=crawl
N=1989995
dim=300
K=100


# ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K



