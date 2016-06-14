import numpy as np 
import numpy.matlib as npml 


def rbf_dot(patterns1, patterns2, deg):
    size1 = patterns1.shape
    size2 = patterns2.shape
    
    G = np.sum(patterns1 * patterns1)
    H = np.sum(patterns2 * patterns2)
    
    Q = npml.repmat(G, 1, size2[0])
    R = npml.repmat(H.T, size1[0], 1)
    
    H = Q + R - 2*np.outer(patterns1,patterns2.T)   
    
    H = np.exp(-H/2/deg**2)
    
    return(H)


def hsicTestGamma(X,Y,alpha=0.05,params =[-1.,-1.]):
    m = X.shape[0]
    
    if params[0] == -1:
        size1=X.shape[0]
        if size1>100:
            Xmed = X[0:99]
            size1 = 100
        else:
            Xmed = X
        G = np.sum(Xmed * Xmed), 2)
        Q = npml.repmat(G, 1, size1)
        R = npml.repmat(G.T, size1, 1)
        dists = Q + R - 2*np.outer(Xmed,Xmed.T)
        dists = dists - npmp.tril(dists)
        dists = np.reshape(dists, size1**2,1)
        params[0] = sqrt(0.5*median(dists[dists.>0])) # not working?
    end
    
    if params[2] == -1
        size1=size(Y,1)
        if size1>100
            Ymed = Y[1:100,:]
            size1 = 100
        else
            Ymed = Y
        end
        G = sum((Ymed .* Ymed), 2)
        Q = repmat(G, 1, size1)
        R = repmat(G', size1, 1)
        dists = Q + R - 2*Ymed*Ymed'
        dists = dists - tril(dists)
        dists = reshape(dists, size1^2,1)
        params[2] = sqrt(0.5*median(dists[dists.>0])) # not working?
    end
    
    K = rbf_dot(X,X,params[1])
    L = rbf_dot(Y,Y,params[2])
    
    bone = ones(m,1)
    H = eye(m)-1/m*ones(m,m)
    
    Kc = H*K*H
    Lc = H*L*H
    
    testStat = 1/m * sum(sum(Kc' .* Lc))
    
    varHSIC = (1/6 * Kc .* Lc).^2
    
    varHSIC = 1/m/(m-1)* (sum(sum(varHSIC)) - sum(diag(varHSIC)))
    
    varHSIC = 72*(m-4)*(m-5)/m/(m-1)/(m-2)/(m-3) * varHSIC

    K = K-diagm(diag(K))
    L = L-diagm(diag(L))

    muX = 1/m/(m-1)*bone'*(K*bone)
    muY = 1/m/(m-1)*bone'*(L*bone)
    
    mHSIC = 1/m * (1 + muX*muY - muX - muY)
    
    al = mHSIC^2 / varHSIC   
    bet = varHSIC*m / mHSIC[1,1]
    treshold = quantile(Gamma(al[1,1], bet), 1-alpha)
    Dict("testStat" => testStat, "treshold" => treshold,
    "mHSIC" => mHSIC[1], "varHSIC" => varHSIC)
end
