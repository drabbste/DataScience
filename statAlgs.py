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

def OLS(xDf, eq):
    '''
    Take in df, use eq to make a suitable x matrix to run OLS
    Col names in df should be used in the eq string
    EX: 'y~x1+x2+x2^2+x^3'
    '''

    #step 1 is to get x and y matrix
    #the x matrix needs to account for exponents and other terms from the df
    yName = eq.split('~')[0]
    terms = eq.split('~')[1].split('+')

    y = xDf[yName]
    x = []


    for i in range(0, xDf.shape[0]):
        xRow = [1]                      #need first columm to be 1 because of y-int (b_0) term
        for term in terms:
            termSplit = term.split('^')
            var = termSplit[0]
            if len(termSplit) == 1:
                exp = 1
            else:
                exp = int(termSplit[1])
            xRow.append(xDf[var][i] ** exp)
        x.append(xRow)
    x = np.array(x)

    #find beta coefficients
    b = np.linalg.inv(x.T @ x) @ x.T @ y
    #print("!!!")
    #print(y)
    print(b)
    #print(x.shape)
    #calculate r^2 - first find residuals and ybar
    yhat = x @ b
    squareResid = np.square(y-yhat)
    #print(y)
    #print(yhat)
    #print(y)
    ybar = np.mean(y)
    sse = np.sum(squareResid) / (x.shape[0] - len(terms))   #note the -1 goes away because of b0 isn't contained in the terms list
    sst = np.sum(np.square(ybar-y)) / (x.shape[0] - 1)
    #print(sse, sst)
    r2 = 1 - (sse/sst)
    print(r2)

    #significance test for each coeff
    #TODO




np.random.seed(1234)

#pr comp test code
'''
x = pd.DataFrame(np.random.randint(0,100,size = (100, 5)), columns=['x1', 'x2', 'x3', 'x4', 'x5'])
prcomp(x)
'''

#ols test code
x = pd.DataFrame(np.random.randint(0,100,size = (100, 5)), columns=['x1', 'x2', 'x3', 'x4', 'y'])
np.random.seed(12)
x_var = pd.DataFrame(np.random.uniform(low=-10, high=10, size=(100,4)), columns=['x1', 'x2', 'x3', 'x4'])
y_var = pd.DataFrame(x_var.iloc[:,0] + x_var.iloc[:,1] + x_var.iloc[:,2] + x_var.iloc[:,3] + 9 + np.random.normal(scale=10,size=100), columns=['y'])
x = pd.concat([y_var, x_var], axis=1)
#print(x)
OLS(x,'y~x1+x2+x3+x4')
#matrix = [[1,1,1], [2,2,2], [6,2,3]]
#x = pd.DataFrame(matrix, columns = ['y', 'x1', 'x2'])
#OLS(x,'y~x1+x2')

