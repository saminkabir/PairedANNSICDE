cd ..

c=1.5
k=20
L=5
K=10
beta=0.1
R_min=300

datasetName=uqv
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=random
k=20

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=astro1m
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=seismic1m
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=sald1m
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=bigann
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=space1V
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=text-to-image
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=gist
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=sift
k=100

./dblsh $datasetName $c $k $L $K $beta $R_min


