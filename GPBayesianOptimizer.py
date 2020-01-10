import cma
import numpy as np
from typing import Callable
from matplotlib import pyplot as plt

def inputFunction(x:float)->float:
    return 10. + (-1. * x**(4.) * 2.0*np.sin(2.*np.pi*4.*x))

def MorletWavelet(x:float)->float:
    """
        Returns the value of a Morlet Wavelet evaluated at x

    :param x: the input value at which to evaluate the Morlet Wavelet Function
    """
    return (1./np.sqrt(2.*np.pi*4.0))*np.exp(-x**2./4.0) * 2.0*np.sin(2.*np.pi*1.*x)

def bayesOpt(inputs:np.ndarray, outputs:np.ndarray, func:Callable, theta_d:np.ndarray, maximize:bool=True):
    """
        Function which performs bayesian optimization using Gaussian Processes and
        the CMA-ES optimization algorithm.

    :param inputs: a NumPy array containing the previously explored values of the objective function.
    :param outputs: the values of the objective function evaluated at each input value.
    :param func: the objective function.
    :param theta_d: hyperparameter for the Gaussian Process.
    :param maximize: indicate whether the objective function should be maximized (True) or minimized (False).
    """
    if len(inputs.shape) < 2: inputs = np.expand_dims(inputs, -1)
    kernel = np.hstack([inputs.reshape(inputs.shape[0],1,inputs.shape[1])] * inputs.shape[0])
    kernelT = np.transpose(kernel, (1,0,2))
    theta_0 = 0.5
    
    #RBF kernel for covariance matrix
    K = theta_0 * np.exp(-0.5 * np.sum(np.power(kernel - kernelT, 2.) / np.power(theta_d, 2.), axis=-1))
    if maximize: UCB = -np.inf 
    else: UCB = np.inf
    maxRet = np.max(outputs)
    x_t = np.zeros(inputs.shape[1])
    t = 0
    bestx_t = x_t
    bestUCB = UCB
    
    #GP iterations
    while t<10:
        #this function computes the upper confidence value of the GP at x_t
        def kernelFunc(x_t:np.ndarray):
            K_t = theta_0 * np.exp(-0.5 * np.sum(np.power(inputs - x_t, 2.) / np.power(theta_d, 2.), axis=-1))
            K_tt = theta_0 * (np.exp(-0.5 * np.sum(np.power(x_t - x_t, 2.) / np.power(theta_d, 2.), axis=-1)) + np.random.normal(0,1))
            KI_inv = np.linalg.inv(K + 1e-3*np.eye(inputs.shape[0]))
            mean = np.dot(K_t.transpose(), np.dot(KI_inv, outputs))
            var = K_tt + 1 - np.dot(K_t.transpose(), np.dot(KI_inv, K_t))
            UCB = mean + 0.5*np.sqrt(var)
            return UCB

        #run CMA-ES to find maximum of the GP
        es = cma.CMAEvolutionStrategy(inputs[np.argmax(outputs)], 0.5)
        for _ in range(0, 10):
            new_xs = es.ask()
            new_outputs = [kernelFunc(x) for x in new_xs]
            if maximize: new_outputs = [-x for x in new_outputs]
            es.tell(new_xs, new_outputs)

        inputs = np.vstack((inputs, es.best.x))
        outputs = np.vstack((outputs, func(es.best.x)))
        t+=1
    return inputs, outputs


def main():
    inputs = np.arange(-5.0, 5.0, 2.2)
    outputs = np.array([MorletWavelet(x) for x in inputs])
    fig, ax = plt.subplots()
    ax.plot(inputs, outputs)
    ax.set(xlim=(-5.5, 5.5), ylim=(-0.5, 0.5))
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.set_title("Initial Samples")
    plt.show()

    xs, ys = bayesOpt(inputs, outputs, MorletWavelet, np.array([1.0]))
    ax.plot(xs, ys)
    ax.set(xlim=(-5.5, 5.5), ylim=(-0.5, 0.5))
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.set_title("Gaussian Processs")
    plt.show()    

    fig, ax = plt.subplots()
    xVals = np.arange(-5.0, 5.0, 0.01)
    yVals = np.array([MorletWavelet(x) for x in xVals])
    ax.plot(xVals, yVals)
    ax.set(xlim=(-5.5, 5.5), ylim=(-0.5, 0.5))
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.set_title("Ground Truth")
    plt.show()

if __name__ == '__main__':
    main()