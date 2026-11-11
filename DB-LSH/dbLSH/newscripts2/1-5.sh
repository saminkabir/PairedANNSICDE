k=100

cd ..
declare -a arr=("Music" "Yelp")

for datasetName in "${arr[@]}"
do
    c=1.5
    L=5
    K=10
    beta=0.1
    R_min=0.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-1

    c=1.75
    L=5
    K=10
    beta=0.1
    R_min=0.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-2

    c=1.25
    L=5
    K=10
    beta=0.1
    R_min=0.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-3

    c=1.5
    L=10
    K=10
    beta=0.1
    R_min=0.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-4

    c=1.5
    L=15
    K=10
    beta=0.1
    R_min=0.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-5

    c=1.5
    L=5
    K=12
    beta=0.1
    R_min=0.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-6

    c=1.5
    L=5
    K=14
    beta=0.1
    R_min=0.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-7

    c=1.5
    L=5
    K=10
    beta=0.1
    R_min=6.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-8

    c=1.75
    L=5
    K=10
    beta=0.1
    R_min=6.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-9

    c=1.25
    L=5
    K=10
    beta=0.1
    R_min=6.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-10

    c=1.5
    L=5
    K=10
    beta=0.1
    R_min=300

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-11

    c=1.5
    L=5
    K=10
    beta=0.1
    R_min=600

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-12

    c=1.5
    L=10
    K=10
    beta=0.1
    R_min=6.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-13

    c=1.5
    L=15
    K=10
    beta=0.1
    R_min=6.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-14

    c=1.5
    L=5
    K=12
    beta=0.1
    R_min=6.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-15

    c=1.5
    L=5
    K=14
    beta=0.1
    R_min=6.50

    ./dblsh $datasetName $c $k $L $K $beta $R_min # > ${datasetName}-16

done
