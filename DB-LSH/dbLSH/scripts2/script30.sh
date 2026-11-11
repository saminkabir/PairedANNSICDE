cd ..

c=1.5
k=20
L=5
K=14
beta=0.1
R_min=6.50


datasetName=astro1m
k=100

beta=0.25
./dblsh $datasetName $c $k $L $K $beta $R_min
beta=0.35
./dblsh $datasetName $c $k $L $K $beta $R_min
beta=0.5
./dblsh $datasetName $c $k $L $K $beta $R_min
beta=0.55
./dblsh $datasetName $c $k $L $K $beta $R_min
beta=0.65
./dblsh $datasetName $c $k $L $K $beta $R_min

