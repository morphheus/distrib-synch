#!/usr/bin/env python
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt



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



"""
TODO:
1. Implement channel management. 

2. Implement clock class, with some sort of scheduler for events.

"""
