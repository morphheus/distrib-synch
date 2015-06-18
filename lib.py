#!/usr/bin/env python
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
#import scipy.signal.fftconvolve as fftconvlve



#---------
def zadoff(u, N,oversampling=1,  q=0):
    """Returns a zadoff-chu sequence betwee with start and endpoints given by
    the list"""

    if N % 2 != 1:
        raise ValueError('N must be an odd integer for ZC sequences')

    x = np.linspace(0, N, N*oversampling+1, dtype='float64')
    return np.exp(-1j*pi*u*x*(x+1+2*q)/N)



#---------
def cplx_gaussian(shape, noise_variance):
    """Assume jointly gaussian noise. Real and Imaginary parts have
    noise_variance/2 actual variance"""
    """The magnitude of the noise will have a variance of 'noise_variance'"""
    x = np.random.normal(size=shape, scale=noise_variance) + 1j*np.random.normal(size=shape, scale=noise_variance)

    return x



#---------
def barycenter_correlation(f,g,power_weight=2, method='regular'):
    """Outputs the barycenter location of 'f' in 'g'. g is expected to be the
    longer array"""
    if len(g) < len(f):
        raise AttributeError("Expected 'g' to be longer than 'f'")
    
    if method == 'regular':
        cross_correlation = np.convolve(f.conjugate(),g,mode='full')
    elif method == 'fft':
        cross_correlation = fftconvolve(f,g,mode='full')
    else: raise ValueError("Unkwnown '" + method +"' method")

    weight = abs(cross_correlation)**power_weight
    
    lag = np.indices(weight.shape)[0]+1
    barycenter = np.sum(weight*lag)/np.sum(weight)

    return barycenter, cross_correlation



N = 1000
pulse = zadoff(1,31*3, oversampling=1,q=1)
channel = cplx_gaussian([1,N],1)
f = pulse
g = channel[0]

plt.plot(np.real(f))
plt.show()
exit()

pulse_idx = round(len(g)*0.4)
g[pulse_idx:pulse_idx+len(f)] += f

tmp, crosscorr = barycenter_correlation(f,g,power_weight=2)
print(tmp)
#plt.plot(abs(crosscorr))


"""
TODO:

2. Implement clock class and scheduler

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


"""
