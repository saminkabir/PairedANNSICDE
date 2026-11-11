def getTotalQuery(s):
    # print(s)
    return s.split('TOTAL QUERY TIME:  ')[1].split('ms')[0]
def getRecall(s):
    # print(s)
    return s.split('AVG RECALL:        ')[1].split('\n')[0]
def getMap(s):
    # print(s)
    return s.split('AVG MAP:           ')[1].split('\n')[0]
def getConstructionTime(s):
    # print(s)
    return s.split('FINISH BUILDING WITH TIME: ')[1].split(' s')[0]
def getDataset(s):
    # print(s)
    return s.split('Using PM-LSH for ')[1].split(' ...')[0]
def getK(s):
    # print(s)
    return s.split('k=        ')[1].split('\n')[0]

results=[]
def extractFile(fileName):
    with open(fileName, 'r') as file:
        st = file.read()
        # print(st)
        splitted = st.split('AVG ROUNDS:')
        # print(len(splitted))
        for i in range(0,len(splitted)-1):
            s=splitted[i]
            ds=getDataset(s)
            ct=getConstructionTime(s)
            qt=getTotalQuery(s)
            rc=getRecall(s)
            mp=getMap(s)
            results.append(ds+','+'PM-LSH'+','+getK(s)+','+ct+','+qt+','+rc+','+mp)
    
        
for i in range(1,8):
    extractFile('log'+str(i)+'.txt')
    
results.sort()
for r in sorted(results):
    print(r)