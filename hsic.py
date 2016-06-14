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
