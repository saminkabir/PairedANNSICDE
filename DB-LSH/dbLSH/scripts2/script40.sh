cd ..



datasetName=crawl
k=100




c=1.75
k=20
L=5
K=10
beta=0.1
R_min=6.50

./dblsh $datasetName $c $k $L $K $beta $R_min

c=1.25
k=20
L=5
K=10
beta=0.1
R_min=6.50

./dblsh $datasetName $c $k $L $K $beta $R_min

c=1.5
k=20
L=5
K=10
beta=0.1
R_min=300

./dblsh $datasetName $c $k $L $K $beta $R_min

c=1.5
k=20
L=5
K=10
beta=0.1
R_min=600

./dblsh $datasetName $c $k $L $K $beta $R_min

c=1.5
k=20
L=10
K=10
beta=0.1
R_min=6.50

./dblsh $datasetName $c $k $L $K $beta $R_min

c=1.5
k=20
L=15
K=10
beta=0.1
R_min=6.50

./dblsh $datasetName $c $k $L $K $beta $R_min


c=1.5
k=20
L=5
K=12
beta=0.1
R_min=6.50


./dblsh $datasetName $c $k $L $K $beta $R_min

c=1.5
k=20
L=5
K=14
beta=0.1
R_min=6.50

./dblsh $datasetName $c $k $L $K $beta $R_min