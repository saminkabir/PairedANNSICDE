cd ..
datasetName=audio
c=1.5
k=20
L=5
K=10
beta=0.1
R_min=300

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=imageNet

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=notre

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=ukbench

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=cifar

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=enron

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=millionSong

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=MNIST

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=nuswide

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=sun

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=deep

./dblsh $datasetName $c $k $L $K $beta $R_min

datasetName=trevi

./dblsh $datasetName $c $k $L $K $beta $R_min


