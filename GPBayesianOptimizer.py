import numpy as np
import time
from typing import Callable

from cmaes import CMAES

def MorletWavelet(vec:np.ndarray)->float:
    """
        Returns the value of a Morlet Wavelet evaluated at vec=(x,y). Specifically,
        the function below is a Morlet Wavelet in the first dimension and a bump
        function in the second dimension.

    :param vec: the input value at which to evaluate the Morlet Wavelet Function
    """
    x = vec[0]
    y = vec[1]
    return ((1./np.sqrt(0.5*np.pi))*np.exp(-x**2./4.0) * 2.0*np.sin(2.*np.pi*0.3*x)) * (1./np.sqrt(0.1*np.pi))*np.exp(-y**2./0.1)

def bayesOpt(inputs:np.ndarray, outputs:np.ndarray, func:Callable, theta_d:np.ndarray, num_iters:int,
             maximize:bool=True, chol:bool =False):
    """
        Function which performs bayesian optimization using Gaussian Processes and
        the CMA-ES optimization algorithm.

    :param inputs: a NumPy array containing the previously explored values of the objective function.
    :param outputs: the values of the objective function evaluated at each input value.
    :param func: the objective function.
    :param theta_d: hyperparameter for the Gaussian Process.
    :param num_iters: number of GP iteration to run.
    :param maximize: indicate whether the objective function should be maximized (True) or minimized (False).
    :param chol: indicate whether to use Cholesky decomposition for GP evaluation (True) or to use
                 matrix inversion (False).
    """
    if len(inputs.shape) < 2: inputs = np.expand_dims(inputs, -1)
    if maximize: UCB = -np.inf 
    else: UCB = np.inf
    
    x_t = np.zeros(inputs.shape[1])
    t = 0
    bestx_t = x_t
    bestUCB = UCB
    #GP iterations
    while t<num_iters:
        #this function computes the upper confidence value of the GP at x_t
        def kernelFunc(x_t:np.ndarray, maximize:bool, chol:bool):
            theta_0 = 0.1
            kappa = 0.1

            kernel = np.hstack([inputs.reshape(inputs.shape[0],1,inputs.shape[1])] * inputs.shape[0])
            kernelT = np.transpose(kernel, (1,0,2))
            
            #RBF kernel for covariance matrix
            K = theta_0 * np.exp(-0.5 * np.sum(np.power(kernel - kernelT, 2.) / np.power(theta_d, 2.), axis=-1))
            # K = theta_0 * np.fromfunction(lambda i, j: np.exp( -0.5 * np.sum(np.power(inputs[i.astype(np.int),:] - inputs[j.astype(np.int),:], 2.) / np.power(theta_d, 2.), axis=-1)),
            #                         (25,25), dtype=inputs.dtype)

            K_t = theta_0 * np.exp(-0.5 * np.sum(np.power(inputs - x_t, 2.) / np.power(theta_d, 2.), axis=-1))
            K_tt = theta_0 * (np.exp(-0.5 * np.sum(np.power(x_t - x_t, 2.) / np.power(theta_d, 2.), axis=-1)) + np.random.normal(0,1e-5))
            
            if chol:
                L = np.linalg.cholesky(K + 1e-3*np.eye(inputs.shape[0]))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L,outputs))
                mean = K_t.dot(alpha)
                v = np.linalg.solve(L,K_t)
                var = K_tt - v.dot(v)
            else:
                KI_inv = np.linalg.inv(K + 1e-3*np.eye(inputs.shape[0]))
                mean = np.dot(K_t.transpose(), np.dot(KI_inv, outputs))
                var = K_tt - np.dot(K_t.transpose(), np.dot(KI_inv, K_t))

            assert var > 0, "Var < 0, please reduce K_tt noise"

            if maximize: UCB = mean + kappa*np.sqrt(var)
            else: UCB = mean - kappa*np.sqrt(var)
            return UCB

        #run CMA-ES to find maximum of the GP
        if maximize: xmean = inputs[np.argmax(outputs)]
        else: xmean = inputs[np.argmin(outputs)]

        bestX = CMAES(xmean, 0.1, 100, 
                      lambda x: kernelFunc(x, maximize, chol), None, not maximize)

        inputs = np.vstack((inputs, bestX))
        outputs = np.hstack((outputs, func(bestX)))
        if maximize:
            bestx_t = inputs[np.argmax(outputs)]
            print("Current best x_t: {}, best val: {}".format(bestx_t, np.max(outputs)))
        else:
            bestx_t = inputs[np.argmin(outputs)]
            print("Current best x_t: {}, best val: {}".format(bestx_t, np.min(outputs)))
        t+=1
    return inputs, outputs


def main():
    inputsx = np.arange(-5.0, 5.0, 1.8)
    inputsy = np.arange(-5.0, 5.0, 1.8)
    
    xx,yy = np.meshgrid(inputsx, inputsy)
    outputs = np.array([MorletWavelet(np.array([x, y])) for (x,y) in zip(xx,yy)])

    inputs = np.vstack((np.ravel(xx), np.ravel(yy))).transpose()
    length = np.array([0.1, 2.])
    num_iters = 10
    maximize = True
    xs, ys = bayesOpt(inputs, np.ravel(outputs), MorletWavelet, length, num_iters, maximize=maximize)

    if maximize:
        print("max of inputs: {}".format(np.max(outputs)))
        print("max of Bopt: {}".format(np.max(ys)))
        print("Improvement: {}".format(abs(np.max(outputs) - np.max(ys))))
    else:
        print("min of inputs: {}".format(np.min(outputs)))
        print("min of Bopt: {}".format(np.min(ys)))
        print("Improvement: {}".format(abs(np.min(outputs) - np.min(ys))))

if __name__ == '__main__':
    main()