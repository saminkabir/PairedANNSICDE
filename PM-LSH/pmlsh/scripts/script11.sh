cd ../build
lowDim=15
c=1.5
T=0.2
R_min=20.0


dataset=uqv
N=1000000
dim=256
K=100


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=random
N=1000000
dim=1000000
K=20

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K

dataset=astro1m
N=1000000
dim=256
K=100

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=seismic1m
N=1000000
dim=256
K=100

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=sald1m
N=1000000
dim=128
K=100

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=bigann
N=1000000
dim=128
K=100

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=space1V
N=1000000
dim=100
K=100

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=text-to-image
N=1000000
dim=200
K=100

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K



dataset=gist
N=1000000
dim=960
K=100

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K



dataset=text-to-image
N=1000000
dim=128
K=100

./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K
