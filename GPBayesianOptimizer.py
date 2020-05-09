import numpy as np
import time
from typing import Callable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from cmaes import CMAES

def inputFunction(x:float)->float:
    return 10. + (-1. * x**(4.) * 2.0*np.sin(2.*np.pi*4.*x))

def MorletWavelet(vec:np.ndarray)->float:
    """
        Returns the value of a Morlet Wavelet evaluated at x

    :param x: the input value at which to evaluate the Morlet Wavelet Function
    """
    x = vec[0]
    y = vec[1]
    return ((1./np.sqrt(2.*np.pi*4.0))*np.exp(-x**2./4.0) * 2.0*np.sin(2.*np.pi*0.25*x)) * (1./np.sqrt(2.*np.pi*4.0))*np.exp(-y**2./4.0)

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
    if maximize: UCB = -np.inf 
    else: UCB = np.inf
    # maxRet = np.max(outputs)
    
    x_t = np.zeros(inputs.shape[1])
    t = 0
    bestx_t = x_t
    bestUCB = UCB
    #GP iterations
    while t<10:
        #this function computes the upper confidence value of the GP at x_t
        def kernelFunc(x_t:np.ndarray):
            theta_0 = 0.5
            kappa = 0.1

            kernel = np.hstack([inputs.reshape(inputs.shape[0],1,inputs.shape[1])] * inputs.shape[0])
            # print(kernel.shape)
            kernelT = np.transpose(kernel, (1,0,2))
            # print(kernelT.shape)
            #RBF kernel for covariance matrix
            K = theta_0 * np.exp(-0.5 * np.sum(np.power(kernel - kernelT, 2.) / np.power(theta_d, 2.), axis=-1))
            fake_K = theta_0 * np.fromfunction(lambda i, j: np.exp( -0.5 * np.sum(np.power(inputs[i.astype(np.int),:] - inputs[j.astype(np.int),:], 2.) / np.power(theta_d, 2.), axis=-1)),
                                    (25,25), dtype=inputs.dtype)
            # print("fake_K: {}".format(fake_K))
            # print("K: {}".format(K))
            # print("equal?: {}".format(np.all(K==fake_K)))

            # print("x_t: {}".format(x_t))
            K_t = theta_0 * np.exp(-0.5 * np.sum(np.power(inputs - x_t, 2.) / np.power(theta_d, 2.), axis=-1))
            # print("K_t: {}".format(K_t))
            
            K_tt = theta_0 * (np.exp(-0.5 * np.sum(np.power(x_t - x_t, 2.) / np.power(theta_d, 2.), axis=-1)))# + np.random.normal(0,0.01))
            KI_inv = np.linalg.inv(K + 1e-3*np.eye(inputs.shape[0]))
            mean = np.dot(K_t.transpose(), np.dot(KI_inv, outputs))
            var = K_tt + 1e-3 - np.dot(K_t.transpose(), np.dot(KI_inv, K_t))
            if var < 0:
                print("K_tt: {}".format(K_tt))
                print("mean: {}".format(mean))
                print("var: {}".format(var))
                exit()
            UCB = mean + kappa*np.sqrt(var)
            return UCB

        #run CMA-ES to find maximum of the GP
        bestX = CMAES(inputs[np.argmax(outputs)], 0.1, 100, kernelFunc, None, False)
        # es = cma.CMAEvolutionStrategy(inputs[np.argmax(outputs)], 0.1)
        # for _ in range(0, 1000):
        #     new_xs = es.ask()
        #     # print(new_xs)
        #     new_outputs = [kernelFunc(x) for x in new_xs]
        #     # print(new_outputs)
        #     if maximize: new_outputs = [-x for x in new_outputs]
        #     es.tell(new_xs, new_outputs)

        inputs = np.vstack((inputs, bestX))
        outputs = np.hstack((outputs, func(bestX)))
        t+=1
    return inputs, outputs


def main():
    inputsx = np.arange(-5.0, 5.0, 2.2)
    inputsy = np.arange(-5.0, 5.0, 2.2)
    xx,yy = np.meshgrid(inputsx, inputsy)
    outputs = np.array([MorletWavelet(np.array([x, y])) for (x,y) in zip(xx,yy)])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.contour(inputsx, inputsy, outputs)
    # ax.set(xlim=(-5.5, 5.5), ylim=(-5.5, 5.5), zlim=(-0.5, 0.5))
    # ax.set_xlabel("Input")
    # ax.set_zlabel("Output")
    # ax.set_title("Initial Samples")
    # plt.show()

    #num samples x num parameters
    print(xx.shape)
    print(np.ravel(yy).shape)
    print(np.ravel(outputs).shape)
    inputs = np.vstack((np.ravel(xx), np.ravel(yy))).transpose()
    xs, ys = bayesOpt(inputs, np.ravel(outputs), MorletWavelet, np.array([10., 10.]))
    print(xs)
    print(ys)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.contour(xs[:0], xs[:1], ys)
    ax.set(xlim=(-5.5, 5.5), ylim=(-5.5, 5.5))
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.set_title("Gaussian Processs")
    plt.show()    

    xVals = np.arange(-5.0, 5.0, 0.01)
    yVals = np.arange(-5.0, 5.0, 0.01)
    xx, yy = np.meshgrid(xVals, yVals)
    outs = np.array([MorletWavelet(np.ravel(xx), np.ravel(yy))])
    outs = outs.reshape(xx.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.contour(xVals, yVals, outs)
    ax.set(xlim=(-5.5, 5.5), ylim=(-5.5, 5.5))
    ax.set_xlabel("Input")
    ax.set_zlabel("Output")
    ax.set_title("Ground Truth")
    plt.show()

if __name__ == '__main__':
    main()