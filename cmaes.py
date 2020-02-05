import numpy as np

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
    lam = 4 + np.floor(3.0 * np.log(N))
    #hsig = 0
    sigma = initialSigma
    mu = lam / 2.0
    eigeneval = 0
    chiN = (N ** 0.5) * (1.0 - 1.0 / (4.0 * N) + 1.0 / (21.0 * N * N))
    xmean = initialMean
    weights = np.arange(1,mu,1)
    weights = np.log(mu + 1.0 / 2.0) - np.log(weights)
    mu = np.floor(mu)
    weights = weights / np.sum(weights)
    mueff = np.sum(weights) * np.sum(weights) / np.dot(weights, weights)
    cc = (4.0 + mueff / N) / (N + 4.0 + 2.0 * mueff / N)
    cs = (mueff + 2.0) / (N + mueff + 5.0)
    c1 = 2.0 / ((N + 1.3) * (N + 1.3) + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((N + 2.0) * (N + 2.0) + mueff))
    damps = 1.0 + 2.0 * max(0.0, sqrt((mueff - 1.0) / (N + 1.0)) - 1.0) + cs
    pc = np.zeros(N)
    ps = np.zeros(N)
    D = np.ones(N)
    DSquared = np.power(D, 2.)
    DInv = 1.0 / D
    xold = np.zeros(xmean.shape)
    # oneOverD
    B = np.eye(N, N)
    C = B * np.diag(DSquared) * B.transpose()
    invsqrtC = B * np.diag(DInv) * B.transpose()
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
			arx[k,:] = xmean + sigma * np.dot(B, randomVector)
		# Eval the population
		for i in range(0, lam):
			arfitness[i] = ((int(minimize)*2.)-1.0) * f(arr[:,i], params)

		counteval += lam
		xold = xmean
		# arindex in a list of tuples, holding (index, fitness), sorted on fitness in ascending order
		arindex = sorted([(i,j) for (i,j) in enumerate(arfitness)], key=lambda x: x[1])
		for row in range(0, int(mu)):
			arxSubMatrix[row,:] = arx[arindex[row][0]]
		xmean = np.dot(arxSubMatrix.T, weights)
