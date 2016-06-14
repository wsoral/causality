import numpy as np
import hsic

x = np.array([1,2,3,4,5,6])
y = np.array([2,2,3,2,4,2])
out = hsic.hsicTestGamma(x,y)
print(out)
