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


"""
TODO:
0. MAKE GIT

1. Implement channel management
-Channel creation with AWGN in it already. complex AWGN (real and
imaginary part are jointly gaussian)
-One "reception" channel per device.

2. Implement clock class, with some sort of scheduler for events.


"""

# HAI
