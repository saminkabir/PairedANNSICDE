cd ../build
lowDim=10
c=1.5
T=0.08
R_min=1.0

dataset=astro1m
N=1000000
dim=256
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=audio
N=53387
dim=192
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=bigann
N=1000000
dim=128
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=cifar
N=50000
dim=512
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K





dataset=deep
N=1000000
dim=256
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=enron
N=94987
dim=1369
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=gist
N=1000000
dim=960
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=glove
N=1192514
dim=100
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=imageNet
N=2340373
dim=150
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=lastfm
N=292385
dim=65
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=millionSong
N=992272
dim=420
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=movielens
N=10677
dim=150
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=MNIST
N=69000
dim=784
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=netflix
N=17770
dim=300
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=notre
N=332668
dim=128
K=20


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K