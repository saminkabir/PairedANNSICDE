cd ..

c=1.5
k=20
L=20
K=40
beta=0.1
R_min=6.50



datasetName=uqv
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min

beta=0.2

./dblsh $datasetName $c $k $L $K $beta $R_min

beta=0.3

./dblsh $datasetName $c $k $L $K $beta $R_min

beta=0.4

./dblsh $datasetName $c $k $L $K $beta $R_min

beta=0.5

./dblsh $datasetName $c $k $L $K $beta $R_min

beta=0.6

./dblsh $datasetName $c $k $L $K $beta $R_min

beta=0.7

./dblsh $datasetName $c $k $L $K $beta $R_min

beta=0.8

./dblsh $datasetName $c $k $L $K $beta $R_min

beta=0.9

./dblsh $datasetName $c $k $L $K $beta $R_min

