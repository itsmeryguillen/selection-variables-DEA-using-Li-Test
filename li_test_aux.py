import numpy as np
from scipy import stats

"""
    Silverman (1986) rule of thumb
"""
def Silv_h_Robust(x):
    n = x.shape[0]
    s = np.std(x)
    r = stats.iqr(x)
    h = 1.06*min([x*n**(-1/5) for x in [s, r/1.349]])
    return h


"""
    This function computes values of gaussian kernel: for density of x evaluated at y
"""
def Nkernel_Li(x,y,h):
    nx = len(x)
    ny = len(y)
    dx = np.kron(np.array([1]*ny),np.array(x))
    dx = np.reshape(dx, (ny,nx)).transpose()
    dy = np.kron(np.array([1]*nx),np.array(y))
    dy = np.reshape(dy, (ny,nx)).transpose()
    a = (dx - dy.transpose()) / h
    k = (1/np.sqrt(2*np.pi))*np.exp(-0.5*(a**2))
    return k


"""
    This program computes the Fan and Ullah 1999 test (same as Li-1996 test!) adapted from Simar and Zelenyuk (2003) 
"""
def li_1996_test(x,y,hx,hy):
    nx = len(x)
    ny = len(y)
    h = min(hx,hy)
    if h < 0.000001:
        h = 0.000001
    hx, hy = h, h

    Kxx = Nkernel_Li(x,x,hx);       
    Kyy = Nkernel_Li(y,y,hy)
    Kxy = Nkernel_Li(x,y,hx);       
    Kyx = Nkernel_Li(y,x,hy)

    lambdaa = nx/ny
    K1 = sum(sum(Kxx))
    K2 = sum(sum(Kyy))
    K3 = sum(sum(Kxy))
    K4 = sum(sum(Kyx))

    shat2 = (2/h)*( K1/(nx**2) + K2*(lambdaa**2)/(ny**2) + K3*lambdaa/(nx*ny) + K4*lambdaa/(nx*ny)) * (1/(2*np.sqrt(np.pi)))
    shat = np.sqrt(shat2)

    insid = K1/(nx*(nx-1)) + K2/(ny*(ny-1)) - K3/(nx*(ny-1)) - K4/(ny*(nx-1))

    Inxny2 = (insid - sum(Kxx.diagonal())/(nx*(nx-1)) - sum(Kyy.diagonal())/(ny*(ny-1)) + sum(Kxy.diagonal())/(nx*(ny-1)) + sum(Kyx.diagonal())/(ny*(nx-1)) ) / h 
    
    Lit =  nx*(h**0.5)*Inxny2 /(shat2**0.5)
    return [Lit, shat]


"""
    Bootstrap for Li (1996) test (proposed by Simar and Zelenyuk, 2006)
"""
def Li_Test_Naive_Boot(B, x, y):

    # Ensure we are subsampling from the the largest sample!
    nx = len(x)
    ny = len(y)
    if nx < ny:
        x = y
    a = []

    #  start the bootstrap loop
    for iboot in range(B): 

        #  Naive Bootstrap for x and y drawn from x (without reflection)
        
        # FOR GROUP 1
        xb = np.random.choice(x, nx, replace = True)
        hx = Silv_h_Robust(xb) # This bandwidth-rule shall be the same as the one sed in the main function that called this bootstrap

        # FOR GROUP 2
        yb = np.random.choice(x, ny, replace = True) 
        hy = Silv_h_Robust(yb) #  This bandwidth-rule shall be the same as the one used in the main function that called this bootstrap

        out = li_1996_test(xb, yb, hx, hy) # this function uses minimal of the two bandwidths

        a.append(out[0])

    return a