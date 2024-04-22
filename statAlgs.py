import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def prcomp(x):
    '''x is a pandas df'''

    #step 1 is to center data to 0
    x = x - x.mean()


    #step 2 is to find covarience matrix
    n = x.shape[0]
    cov = np.matmul(x.T, x) / n

    #print(cov)
    #cov = np.cov(x)


    #step 3 is to find eigenvalues and eigenvectors of cov matrix
    eigen = np.linalg.eig(cov)
    eigVal = eigen[0].tolist()
    eigVect = eigen[1].tolist()
    print(eigVal)
    for i in range(0,len(eigVal)):
        eigVal[i] = eigVal[i].real
        for j in range(0,len(eigVect[i])):
            eigVect[i][j] = eigVect[i][j].real 
    
    #step 4 - start from the greatest eigenvalue and work down to find first to last principle components
    #note that np does not automatically sort by eigenvalue so a little sorting by size is needed
    maxEigVal = -1
    prc = []
    prcEigVal = []
    for j in range(0, len(eigVal)):
        for i in range(0,len(eigVal)):
            if eigVal[i] > maxEigVal:
                maxEigVal = eigVal[i]
                maxEigValIndex = i
        prc.append(eigVect[maxEigValIndex])
        prcEigVal.append(eigVal[maxEigValIndex])
        eigVal.pop(maxEigValIndex)
        eigVect.pop(maxEigValIndex)
        maxEigVal = -1
    

    #prc contains the principle components, prcEigVal contain how much of the varience each component contains
    #step 5 is to calculate varience of each component using eigenvalues
    totalVar = sum(prcEigVal)
    cumVar = []
    var = []
    for i in range(0, len(prc)):
        cumVar.append(sum(prcEigVal[0:i+1]) / totalVar)
        var.append(prcEigVal[i] / totalVar)

    rows = []
    #step 6 is to print info out nicely
    header = []
    for i in range(1, len(x.columns)+1):
        header.append('PC'+str(i))
    for i in range(0, len(x.columns)):
        rows.append([])
        for j in range(0, len(prc[i])):
            rows[i].append(prc[j][i])
    rows.append(cumVar)
    rows.append(var)

    rowNames = []
    for variable in x.columns:
        rowNames.append(variable)
    rowNames.extend(['Cumulative Varience', 'Varience'])

    table = pd.DataFrame(rows, columns=header, index=rowNames)
    print(table)

    '''
    pca = PCA(n_components=5)
    pca.fit(x)
    print(pca.components_)
    print(pca.explained_variance_ratio_)
    '''




np.random.seed(1234)
x = pd.DataFrame(np.random.randint(0,100,size = (100, 5)), columns=['x1', 'x2', 'x3', 'x4', 'x5'])

prcomp(x)