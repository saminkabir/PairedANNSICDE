cd ../build
lowDim=15
c=1.5
T=0.4
R_min=1.0
K=20


dataset=audio
N=53387
dim=192


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=imageNet
N=2340373
dim=150


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=notre
N=332668
dim=128


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=ukbench
N=1097907
dim=128


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=cifar
N=50000
dim=512


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=enron
N=94987
dim=1369


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=millionSong
N=992272
dim=420


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=MNIST
N=69000
dim=784


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=nuswide
N=268643
dim=500


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=sun
N=79106
dim=512


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=deep
N=1000000
dim=256


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K


dataset=trevi
N=99900
dim=4096


./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K