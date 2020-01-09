import cma
import numpy as np
from typing import Callable
from matplotlib import pyplot as plt

def inputFunction(x:float)->float:
    return 10. + (-1. * x**(4.) * 2.0*np.sin(2.*np.pi*4.*x))


def bayesOpt(inputs:np.ndarray, outputs:np.ndarray, func:Callable, theta_d:np.ndarray, maximize:bool=True):
    x_init = inputs
    kernel = np.hstack([x_init.reshape(x_init.shape[0],1,x_init.shape[1])] * x_init.shape[0])
    kernelT = np.transpose(kernel, (1,0,2))
    theta_0 = 0.5
    
    #RBF kernel for covariance matrix
    K = theta_0 * np.exp(-0.5 * np.sum(np.power(kernel - kernelT, 2.) / np.power(theta_d, 2.), axis=-1))
    UCB = -np.inf
    maxRet = np.max(outputs)
    x_t = np.zeros(x_init.shape[1])
    t = 0
    bestx_t = x_t
    bestUCB = UCB
    
    #GP iterations
    while t<10:
        #this function computes the upper confidence value of the GP at x_t
        def kernelFunc(x_t:np.ndarray):
            K_t = theta_0 * np.exp(-0.5 * np.sum(np.power(x_init - x_t, 2.) / np.power(theta_d, 2.), axis=-1))
            K_tt = theta_0 * (np.exp(-0.5 * np.sum(np.power(x_t - x_t, 2.) / np.power(theta_d, 2.), axis=-1)) + np.random.normal(0,1))
            KI_inv = np.linalg.inv(K + 1e-3*np.eye(x_init.shape[0]))
            mean = np.dot(K_t.transpose(), np.dot(KI_inv, outputs))
            var = K_tt + 1 - np.dot(K_t.transpose(), np.dot(KI_inv, K_t))
            UCB = mean + 0.5*np.sqrt(var)
            return UCB

        #run CMA-ES to find maximum of the GP
        es = cma.CMAEvolutionStrategy(inputs[np.argmax(outputs)], 0.5)
        for _ in range(0, 10):
            new_xs = es.ask()
            new_outputs = [kernelFunc(x) for x in new_xs]
            es.tell(new_xs, new_outputs)

        inputs = np.vstack((inputs, es.best.x))
        outputs = np.vstack((outputs, func(es.best.x)))
        t+=1
    return outputs[-1]

def main():
    xVals = np.arange(-1., 1.1, 0.1)
    print(xVals)
    yVals = np.array([inputFunction(x) for x in xVals])

    plt.plot(xVals, yVals)
    plt.show()

if __name__ == '__main__':
    main()