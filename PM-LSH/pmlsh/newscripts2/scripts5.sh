cd ../build
lowDim=15
c=2
T=0.08
R_min=1.0


dataset=nuswide
N=268643
dim=500
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=nytimes
N=290000
dim=256
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=random
N=1000000
dim=100
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=sald1m
N=1000000
dim=128
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=seismic1m
N=1000000
dim=256
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=sift
N=1000000
dim=128
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=space1V
N=1000000
dim=100
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=sun
N=79106
dim=512
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=text-to-image
N=1000000
dim=200
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K



dataset=trevi
N=99900
dim=4096
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=uqv
N=1000000
dim=256
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=ukbench
N=1097907
dim=128
K=20


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





