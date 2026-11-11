with open('stat.py', 'r') as file:
    Lines = file.readlines()
    for ln in Lines:
        spl=ln.split(' (')
        # print(spl)
        dataset=spl[0]
        N=spl[1].split(', ')[0]
        dim=spl[1].split(', ')[1].split(')')[0]
        K=spl[3].split(', ')[1].split(')')[0]
        # print(dataset, N, dim, K)
        print('dataset='+dataset)
        print('N='+N)
        print('dim='+dim)
        print('K='+K)
        print('\n')
        print('./pmlsh $dataset $N $dim $lowDim $c $T $R_min $K')
        print('\n')