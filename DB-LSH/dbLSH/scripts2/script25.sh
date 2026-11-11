cd ..

c=1.5
k=20
L=15
K=10
beta=0.1
R_min=6.50




datasetName=glove
k=20

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=lastfm
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=movielens
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=netflix
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=nytimes
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min



datasetName=word2vec
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=yahoomusic
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min




