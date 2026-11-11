cd ..
c=1.5
k=20
L=5
K=10
beta=0.8
R_min=6.50

declare -a arr=("cifar" "deep" "MNIST" "imageNet" "glove" "random")

for datasetName in "${arr[@]}"
do
    beta=0.85

    ./dblsh $datasetName $c $k $L $K $beta $R_min

    beta=0.7


    ./dblsh $datasetName $c $k $L $K $beta $R_min


    beta=0.6

    ./dblsh $datasetName $c $k $L $K $beta $R_min


    beta=0.45

    ./dblsh $datasetName $c $k $L $K $beta $R_min

done


k=20

declare -a arr=("movielens" "yahoomusic" "netflix")

for datasetName in "${arr[@]}"
do
    beta=0.85
    
    ./dblsh $datasetName $c $k $L $K $beta $R_min

    beta=0.7


    ./dblsh $datasetName $c $k $L $K $beta $R_min


    beta=0.6

    ./dblsh $datasetName $c $k $L $K $beta $R_min


    beta=0.45

    ./dblsh $datasetName $c $k $L $K $beta $R_min

done
