import pandas as pd
import numpy as np
from li_test_aux import *

"""
    Calculate the adapated Li test (Simar and Zelenyuk, 2006) given two effiency scores computed under the output-oriented radial DEA model
 
    Args:
        data (dataframe): dataframe containg the effienciy scores to be compared
        col1 (str): name of the column containing the first vector of scores
        col2 (str): name of the column containing the second vector of scores
        nX (int): number of inputs of the DEA model
        nY (int): number of outputs of the DEA model
        seed (int): seed 
 
    Returns:
        float: adapted Li test p-value
"""


def li_test(data, col1, col2, nX, nY, seed = 1111):
    
    np.random.seed(seed)

    B = 1000            # Boostrapping iterations
    n = data.shape[0]   # Number of DMUs

    # Select columns of scores
    eff1 = data.loc[:,col1].copy()
    eff2 = data.loc[:,col2].copy()

    # Smooth the DEA estimates that equal 1, to be used in the Li-test
    # ALTERNATIVELY: Eliminate the points at the BOUNDARY ('1') in estimation of both bandwidth and density
    eff_eps1 = np.array(eff1.copy()).round(4)
    if len(eff_eps1[eff_eps1 > 1]) != 0:
        quantile1 = np.quantile(eff_eps1[eff_eps1 > 1], 0.05, method = 'inverted_cdf')-1
    else:
        quantile1 = np.inf
    eps1 = round(min(quantile1, n**(-2/(nX+nY+1))),4)
    for i in range(n):
        if eff_eps1[i] == 1:
            eff_eps1[i] += np.random.uniform(0,eps1)

    eff_eps2 = np.array(eff2.copy()).round(4)
    if len(eff_eps2[eff_eps2 > 1]) != 0:
        quantile2 = np.quantile(eff_eps2[eff_eps2 > 1], 0.05, method = 'inverted_cdf')-1
    else:
        quantile2 = np.inf
    eps2 = round(min(quantile2, n**(-2/(nX+nY+1))),4)
    for i in range(n):
        if eff_eps2[i] == 1:
            eff_eps2[i] += np.random.uniform(0,eps2)

            
    # compute the bandwidths using the Silverman (1986) adaptive rule of thumb 
    # ALTERNATIVELY: another rule can be chosen , e.g., Sheather and Jones (1991) method, but then this same rule has to be used in each bootstrap iteration
    h1 = Silv_h_Robust(eff_eps1)
    h2 = Silv_h_Robust(eff_eps2)

    # computation of empirical p-values for true eff without reflection, via Naive bootstrap
    Li_test = li_1996_test(eff_eps1, eff_eps2, h1, h2)
    Tb = Li_Test_Naive_Boot(B, eff_eps1, eff_eps2)

    partial_pvalb = [1 if abs(Tb[i]) > abs(Li_test[0]) else 0 for i in range(B)]
    pvalb = sum(partial_pvalb) / B

    return pvalb