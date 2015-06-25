#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import collections
import bisect

from numpy import pi
from pprint import pprint
#import scipy.signal.fftconvolve as fftconvlve




########################
# SUBROUTINES
#######################


#---------
def zadoff(u, N,oversampling=1,  q=0):
    """Returns a zadoff-chu sequence betwee with start and endpoints given by
    the list"""


    x = np.linspace(0, N-1, N*oversampling, dtype='float64')
    return np.exp(-1j*pi*u*x*(x+1+2*q)/N)



#---------
def cplx_gaussian(shape, noise_variance):
    """Assume jointly gaussian noise. Real and Imaginary parts have
    noise_variance/2 actual variance"""
    """The magnitude of the noise will have a variance of 'noise_variance'"""
    # If variance is equal to zero, numpy returns an error
    if noise_variance:
        x = np.random.normal(size=shape, scale=noise_variance) + 1j*np.random.normal(size=shape, scale=noise_variance)
    else:
        x = np.array([0+0j]*shape[0]*shape[1]).reshape(shape[0],-1)
    return x



#---------
def barycenter_correlation(f,g, power_weight=2, method='regular'):
    """Outputs the barycenter location of 'f' in 'g'. g is expected to be the
    longer array"""
    if len(g) < len(f):
        raise AttributeError("Expected 'g' to be longer than 'f'")
    
    if method == 'regular':
        cross_correlation = np.convolve(f.conjugate(),g,mode='valid')
    elif method == 'fft':
        cross_correlation = fftconvolve(f.conjugate(),g,mode='valid')
    else: raise ValueError("Unkwnown '" + method +"' method")

    # NOTE: To take the cross-correlation, we would have to flip back the solution on itself.
    # This is not implemented yet.
    
    weight = np.absolute(cross_correlation)**power_weight
    weightsum = np.sum(weight)
    lag = np.indices(weight.shape)[0]+1

    # If empty cross_correlation, return -1
    if not weightsum:
        barycenter = 0
    else:
        barycenter = np.sum(weight*lag)/weightsum

    offset = (len(g) - len(cross_correlation))/2 + 1 # Correction for 'valid' comvolution

    return barycenter-offset, cross_correlation



def d_to_a(values, pulse, spacing):
    """outputs an array with modulated pulses"""
    plen = len(pulse)
    if plen % 2 == 0:
        offset = plen/2
    else:
        offset = plen/2
     
    output = np.zeros((len(values)-1)*(spacing)+plen) + 0j
    idx = 0;
    for val in values:
        output[idx:idx+plen] += val*pulse
        idx += spacing

    return output

def test_raised_cosine():
    return 1


########################
# THEORY AND STATIC GRAPHS
########################
def test_crosscorr():
    Nzc = 53
    N = Nzc*10

    f = zadoff(1,Nzc,q=0)
    pulse = np.array(list(f)*7)
    channel = cplx_gaussian([1,N],0)
    g = channel[0]
    
    
    pulse_idx = round(len(g)*0.1)
    g[pulse_idx:pulse_idx+len(pulse)] += pulse
    
    barycenter, crosscorr = barycenter_correlation(f,g,power_weight=2)
    print(barycenter)
    plt.plot(abs(crosscorr))
    #plt.plot(np.real(g))
    plt.show()






"""
TODO:

0 EMACS: implement shift-enter as return and remove one tab



3. Consider the interpolation that will be made by the physical device. Yes, it will generate
discrete samples, but in practice, it will oscillate from one sample point to the other.
Oversampling the generator does not function for varying reasons.

What gives, then? Spline interpolation? Sinc interpolation? 

10. Implement some sort of check for the "variance" of the barycenter. In the cross
correlation, maybe drop all entries 10% from the min?

11. Find papers/textbooks on the kind of SNR that would be observed

"""



"""
OPTIMIZATION LIST:

ZADOFF:
1. If the q, u, and oversampling are not used, just make a zadoff_lite(N) function to save on computation time


BARYCENTER_CORRELATION()
1. Use scipy.signal.fftconvolve instead of numpy.convolve

2. Use mode='valid' instead of full (to save on some computation). This will require adjusting the value of the output, as the cross-correlation matrix will not have the same size as g

3. As zadoff-chu sequences have constant amplitude, maybe you could only store the phase? instead
of real & imaginary part


NODES:
make a Nodes class instead of multiple single nodes.

"""
