cd ../build
lowDim=15
c=1.25
T=0.2
R_min=1.0
K=20

dataset=sun
N=79106
dim=512
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

dataset=trevi
N=99900
dim=4096
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

dataset=tiny5m
N=5000000
dim=384
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

dataset=word2vec
N=1000000
dim=300
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

dataset=yahoomusic
N=136736
dim=300
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


