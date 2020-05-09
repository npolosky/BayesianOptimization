import numpy as np
from typing import Callable

def CMAES(initialMean:np.ndarray, initialSigma:float, numIterations:int, f:Callable, params, minimize:bool):
    """
        This is an implementation of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
        It is a python implementation translated from the C++ implementation of CMA-ES provided by
        Phil Thomas, which is openly available at https://aisafety.cs.umass.edu/tutorial4cpp.html
        Phil's implementation is located in the HelperFunctions.hpp file which can be directly 
        downloaded from the aforementioned link.

    :param initialMean: 
    :param initialSigma: 
    :param numIterations:
    :param f:
    :param params:
    :param minimize:    
    """

    # Initialize all parameters of the algorithm
    N = initialMean.size
    lam = int(4 + np.floor(3.0 * np.log(N)))
    #hsig = 0
    sigma = initialSigma
    mu = lam / 2.0
    eigeneval = 0
    chiN = (N ** 0.5) * (1.0 - 1.0 / (4.0 * N) + 1.0 / (21.0 * N * N))
    xmean = initialMean
    weights = np.arange(1,mu+1,1)
    weights = np.log(mu + 1.0 / 2.0) - np.log(weights)
    mu = np.floor(mu)
    weights = weights / np.sum(weights)
    mueff = np.sum(weights) * np.sum(weights) / np.dot(weights, weights)
    cc = (4.0 + mueff / N) / (N + 4.0 + 2.0 * mueff / N)
    cs = (mueff + 2.0) / (N + mueff + 5.0)
    c1 = 2.0 / ((N + 1.3) * (N + 1.3) + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((N + 2.0) * (N + 2.0) + mueff))
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (N + 1.0)) - 1.0) + cs
    pc = np.zeros(N)
    ps = np.zeros(N)
    D = np.ones(N)
    DSquared = np.power(D, 2.)
    DInv = 1.0 / D
    xold = np.zeros(xmean.shape)
    # oneOverD
    B = np.eye(N, N)
    C = B.dot(np.diag(DSquared).dot(B.transpose()))
    invsqrtC = B.dot(np.diag(DInv).dot(B.transpose()))
    arx = np.zeros((N, lam)) 
    repmat = np.zeros((xmean.size , int(mu + .1)))
    # artmp
    arxSubMatrix = np.zeros((N, int(mu + .1)))
    arfitness = np.zeros(lam).astype(np.float)
    arindex = np.zeros(lam).astype(np.int)

    # Iterate over num iterations
    for counteval in range(0, numIterations):
        # Sample the population
        for k in range(0, lam):
            randomVector = D * np.random.normal(loc=0.0, scale=1.0, size=D.shape)
            arx[:,k] = xmean + sigma * np.dot(B, randomVector)
        # Eval the population
        for i in range(0, lam):
            if params:
                arfitness[i] = ((int(minimize)*2.)-1.0) * f(arx[:,i], params)
            else:
                arfitness[i] = ((int(minimize)*2.)-1.0) * f(arx[:,i])

        counteval += lam
        xold = xmean
        # print("arfitness: {}".format(arfitness))
        # arindex in a list of tuples, holding (index, fitness), sorted on fitness in ascending order
        arindex = sorted([(i,j) for (i,j) in enumerate(arfitness)], key=lambda x: x[1])
        for col in range(0, int(mu)):
            arxSubMatrix[:,col] = arx[:,arindex[col][0]]
        xmean = np.dot(arxSubMatrix, weights)
        ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * invsqrtC.dot((xmean - xold) / sigma)
        hsig = 0.0
        if (np.linalg.norm(ps) / np.sqrt(1.0 - np.power(1.0 - cs, 2.0 * counteval / lam)) / chiN) < (1.4 + 2.0 / (N + 1.0)):
            hsig = 1.0
        pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (xmean - xold) / sigma
        for i in range(0, repmat.shape[1]):
            repmat[:,i] = xold
        artmp = (1.0 / sigma) * (arxSubMatrix - repmat)
        C = (1 - c1 - cmu) * C + c1 * (pc.dot(pc.T) + (1.0 - hsig) * cc * (2.0 - cc) * C) + cmu * artmp.dot(np.diag(weights).dot(artmp.T))
        if(np.any(np.isnan(C))):
            print("gosh darnit: {}".format(xmean))
        sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1.0))
        if ((1.0)*counteval - eigeneval > 1.0*lam / (c1 + cmu) / 1.0*N / 10.0):
            eigeneval = counteval
            for r in range(0, C.shape[0]):
                for c in range(r + 1, C.shape[1]):
                    C[r, c] = C[c, r]
            D,B = np.linalg.eig(C)
            D = np.sqrt(np.real(D))
            B = np.real(B)
            for i in range(0, B.shape[1]):
                B[:,i] = B[:,i]/np.linalg.norm(B[:,i])
            invsqrtC = B.dot(np.diag(1.0/D).dot(B.T))
    return arx[:,0]


if __name__ == '__main__':
    np.random.seed(0)
    initX = np.random.rand(10)*100.
    print("init: {}\ninit norm: {}".format(initX,np.linalg.norm(initX)))
    sol = CMAES(initX, 1.0, 1000, np.linalg.norm, None, True)
    print("solution: {}\nnorm val: {}".format(sol,np.linalg.norm(sol)))