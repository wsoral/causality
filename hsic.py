import numpy as np 
import numpy.matlib as npml 
from scipy.stats import gamma 


def rbf_dot(patterns1, patterns2, deg):
    size1 = patterns1.shape
    size2 = patterns2.shape
    
    G = np.sum(np.multiply(patterns1, patterns1))
    H = np.sum(np.multiply(patterns2, patterns2))
    
    Q = npml.repmat(G, 1, size2[0])
    R = npml.repmat(H.T, size1[0], 1)
    
    H = Q + R - 2*np.outer(patterns1,patterns2.T)   
    
    H = np.exp(-H/2./float(deg**2))
    
    return H


def hsicTestGamma(X,Y,alpha=0.05,params =[-1.,-1.]):
    m = float(X.shape[0])
    
    if params[0] == -1:
        size1=X.shape[0]
        if size1>100:
            Xmed = X[:99]
            size1 = 100
        else:
            Xmed = X
        G = np.sum(Xmed * Xmed)
        Q = npml.repmat(G, 1, size1)
        R = npml.repmat(G.T, size1, 1)
        dists = Q + R - 2*np.outer(Xmed,Xmed.T)
        dists = dists - npml.tril(dists)
        dists = np.reshape(dists, size1**2,1)
        params[0] = np.sqrt(0.5*np.median(dists[dists>0])) # not working?
    
    if params[1] == -1:
        size1=Y.shape[0]
        if size1>100:
            Ymed = Y[:99]
            size1 = 100
        else:
            Ymed = Y
        G = np.sum(Ymed * Ymed)
        Q = npml.repmat(G, 1, size1)
        R = npml.repmat(G.T, size1, 1)
        dists = Q + R - 2*np.outer(Ymed,Ymed.T)
        dists = dists - npml.tril(dists)
        dists = np.reshape(dists, size1**2,1)
        params[1] = np.sqrt(0.5*np.median(dists[dists>0])) # not working?
    
    K = rbf_dot(X,X,params[0])
    L = rbf_dot(Y,Y,params[1])
    
    bone = npml.ones((m,1))
    H = npml.eye(m)-1/m*npml.ones((m,m))

    Kc = H.dot(K).dot(H)
    Lc = H.dot(L).dot(H)

    anmHSIC = ((m - 1)**-2) * np.sum(npml.diag(K.dot(H).dot(L).dot(H)))

    testStat = 1/m * np.sum(np.multiply(Kc.T, Lc))
    varHSIC = np.square(1/6. * np.multiply(Kc,Lc))
    varHSIC = 1/m/(m-1)* (np.sum(varHSIC) - np.sum(npml.diag(varHSIC)))
    varHSIC = 72*(m-4)*(m-5)/m/(m-1)/(m-2)/(m-3) * varHSIC
    K = K-npml.diag(npml.diag(K))
    L = L-npml.diag(npml.diag(L))
    muX = 1/m/(m-1)*bone.T.dot(K.dot(bone))
    muY = 1/m/(m-1)*bone.T.dot(L.dot(bone))
    
    mHSIC = 1/m * (1 + muX * muY - muX - muY)
    al = np.square(mHSIC) / varHSIC   
    bet = varHSIC*m / mHSIC
    treshold = gamma(al, bet).ppf(1-alpha)
#    return {"testStat" : testStat, "treshold":treshold,
 #           "mHSIC": mHSIC, "varHSIC" : varHSIC }
    return anmHSIC
