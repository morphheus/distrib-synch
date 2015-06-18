#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import collections

from numpy import pi
#import scipy.signal.fftconvolve as fftconvlve




#-------------------
# SUBROUTINES
#-------------------


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


#----------------------------
# CLASSES
#----------------------------


class CustomDeque(collections.deque):
    """Just adding arbitrary insert to the deque class"""
    def insert(self, index, value):
        self.rotate(-1*index)
        self.append(value)
        self.rotate(index+1)



class Scheduler:
    """Each entry in queue is a tuple, where the first tuple entry is the event time
    and the second entry is the clock that has an event to execute"""
    
    #def __init__(self, max_frame):
    #    self.max_frame = max_frame
    
    queue = CustomDeque()
    
    def ordered_insert(self, frame, clocknum):
        """Insert from the left in descending order"""
        
        try:
            k = self.queue[0][0]
        except IndexError:
            self.queue.appen((frame,clocknum))
            return

        idx = 0
        while k < frame:
            idx += 1
            k = self.queue[idx][0]

        
        self.queue.insert(idx,(frame,clocknum))





#----------------------------
# MAIN
#----------------------------

schedule = Scheduler()
print(schedule.queue)

exit()

N = 1000
pulse = zadoff(1,31, oversampling=1,q=1)
channel = cplx_gaussian([1,N],1)
f = pulse
g = channel[0]


pulse_idx = round(len(g)*0.4)
g[pulse_idx:pulse_idx+len(f)] += f

tmp, crosscorr = barycenter_correlation(f,g,power_weight=2)
print(tmp)
plt.plot(abs(crosscorr))
plt.plot(np.real(g))
plt.show()


"""
TODO:

0 EMACS: implement shift-enter as return and remove one tab


2. Implement clock class and scheduler

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


"""
