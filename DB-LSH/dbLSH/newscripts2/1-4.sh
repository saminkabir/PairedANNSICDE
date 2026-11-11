k=100

cd ..
declare -a arr=("ethz" "vcseis" "txed")

for datasetName in "${arr[@]}"
do
    c=1.5
    L=5
    K=10
    beta=0.1
    R_min=0.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-1

done
