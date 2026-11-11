cd ../build
lowDim=15
c=1.5
T=0.1
R_min=1.0
K=20
logId=log6


K=100

dim=100
dataset=Music
N=1000000

/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt

dim=50
dataset=Yelp
N=77079


/usr/bin/time -v ./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K &> ../logs/${dataset}-${logId}.txt
