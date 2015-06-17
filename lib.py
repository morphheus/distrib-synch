#!/usr/bin/env python
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
#import scipy.signal.fftconvolve as fftconvlve



def zadoff(u, N,oversampling=1,  q=0):
    """Returns a zadoff-chu sequence betwee with start and endpoints given by
    the list"""

    if N % 2 != 1:
        raise ValueError('N must be an odd integer for ZC sequences')

    x = np.linspace(0, N, N*oversampling+1, dtype='float64')
    return np.exp(-1j*pi*u*x*(x+1+q)/N)


def cplx_gaussian(shape, noise_variance):
    """Assume jointly gaussian noise. Real and Imaginary parts have
    noise_variance/2 actual variance"""
    """The magnitude of the noise will have a variance of 'noise_variance'"""
    x = np.random.normal(size=shape, scale=noise_variance) + 1j*np.random.normal(size=shape, scale=noise_variance)
    return x*np.sqrt(1/2)



def barycenter_correlation(f,g,power_weight=2, method='regular'):
    """Outputs the barycenter location of 'f' in 'g'. g is expected to be the
    longer array"""
    if len(g) < len(f):
        raise AttributeError("Expected 'g' to be longer than 'f'")
    
    if method == 'regular':
        cross_correlation = np.convolve(f,g,mode='full')
    elif method == 'fft':
        cross_correlation = fftconvolve(f,g,mode='full')
    else: raise ValueError("Unkwnown '" + method +"' method")

    cross_correlation = abs(cross_correlation)**power_weight
    
    lag = np.indices(cross_correlation.shape)[0]+1
    bary = np.sum(cross_correlation*lag)/np.sum(cross_correlation)

    return bary



f = np.array([0.5,1,0.5])
g = np.array([ 0,0,0,0,0,0,0,0.5,1,0.5,0, 0,0])


tmp = barycenter_correlation(f,g)
print(np.convolve(f,g))
print(tmp)

"""
TODO:
0. Complex conjugate on f! in Barycenter! (cross correlation requires complex conjugation)

1. Properly test barycenter_correlation

2. Implement clock class and scheduler

"""



"""
OPTIMIZATION LIST:


BARYCENTER_CORRELATION()
1. Use scipy.signal.fftconvolve instead of numpy.convolve

2. Use mode='valid' instead of full (to save on some computation). This will require adjusting the value of the output, as the cross-correlation matrix will not have the same size as g


"""
