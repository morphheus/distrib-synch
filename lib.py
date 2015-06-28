#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import collections
import bisect
import math
import warnings

from numpy import pi
from pprint import pprint
from scipy.signal import fftconvolve
#from scipy.fftpack import 



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
def barycenter_correlation(f,g, power_weight=2, method='numpy'):
    """Outputs the barycenter location of 'f' in 'g'. g is expected to be the
    longer array"""
    """Note: barycenter will correspond to the entry IN THE CROSS CORRELATION"""
    if len(g) < len(f):
        raise AttributeError("Expected 'g' to be longer than 'f'")
    
    if method == 'numpy':
        cross_correlation = np.convolve(f.conjugate(),g,mode='valid')
    elif method == 'scipy':
        cross_correlation = fftconvolve(g, f.conjugate(),mode='valid')
    else: raise ValueError("Unkwnown '" + method +"' method")

    
    weight = np.absolute(cross_correlation)**power_weight
    weightsum = np.sum(weight)
    lag = np.indices(weight.shape)[0]+1

    # If empty cross_correlation, return -1
    if not weightsum:
        barycenter = 0
    else:
        barycenter = np.sum(weight*lag)/weightsum


    return barycenter, cross_correlation



#------------------
def d_to_a(values, pulse, spacing,dtype='complex128'):
    """outputs an array with modulated pulses"""
    plen = len(pulse)
     
    output = np.zeros((len(values)-1)*(spacing)+plen,dtype=dtype)
    idx = 0;
    for val in values:
        output[idx:idx+plen] = output[idx:idx+plen] + val*pulse
        idx += spacing

    return output



#------------------
def commpyrcos(N, alpha, Ts, Fs):
    """
    IMPORTED FROM COMMPY
    Generates a raised cosine (RC) filter (FIR) impulse response.

    Parameters
    ----------
    N : int
        Length of the filter in samples.

    alpha : float
        Roll off factor (Valid values are [0, 1]).

    Ts : float
        Symbol period in seconds.

    Fs : float
        Sampling Rate in Hz.

    Returns
    -------

    h_rc : 1-D ndarray (float)
        Impulse response of the raised cosine filter.

    time_idx : 1-D ndarray (float)
        Array containing the time indices, in seconds, for the impulse response.
    """

    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2+0.5))*T_delta
    sample_num = np.arange(N)
    h_rc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rc[x] = 1.0
        elif alpha != 0 and t == Ts/(2*alpha):
            h_rc[x] = (np.pi/4)*(np.sin(np.pi*t/Ts)/(np.pi*t/Ts))
        elif alpha != 0 and t == -Ts/(2*alpha):
            h_rc[x] = (np.pi/4)*(np.sin(np.pi*t/Ts)/(np.pi*t/Ts))
        else:
            h_rc[x] = (np.sin(np.pi*t/Ts)/(np.pi*t/Ts))* \
                    (np.cos(np.pi*alpha*t/Ts)/(1-(((2*alpha*t)/Ts)*((2*alpha*t)/Ts))))

    return time_idx, h_rc



#---------------------
def commpyrootrcos(N, alpha, Ts, Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.

    Parameters
    ----------
    N : int
        Length of the filter in samples.

    alpha : float
        Roll off factor (Valid values are [0, 1]).

    Ts : float
        Symbol period in seconds.

    Fs : float
        Sampling Rate in Hz.

    Returns
    ---------

    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.

    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.
    """

    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2+0.5))*T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
        elif alpha != 0 and t == Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        elif alpha != 0 and t == -Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Ts) +  \
                    4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts))/ \
                    (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)

    return time_idx, h_rrc





#---------------------------
def rcosfilter(N, a, T, f, dtype='complex128'):
    """Raised cosine:
    N: Number of samples
    a: rolloff factor (alpha)
    T: symbol period
    f: sampling period

    t: time indexes associated with impulse response
    h: impulse response

    NOTE: this thing is slow. Faster but complexier methods exist.
    """
    time = (np.arange(N,dtype=dtype)-N/2+0.5)/float(f)
    zero_entry = math.floor(N/2) # Index of entry with a zero
    if not N % 2:
        zero_entry = -1

    warnings.filterwarnings("ignore")
    h = np.empty(N, dtype=dtype)
    for k, t in enumerate(time):
        if k == zero_entry:
            h[k] = 1
        #elif a != 0 and abs(t) == T/(2*a):
        #    h[k] = np.sin(pi*t/T) / (pi*t/T )
        else: 
            h[k] = np.sin(pi*t/T) * np.cos(pi*a*t/T) / (pi*t/T  *  (1-4*(a*t/T)**2 ))

    for k, val in enumerate(h):
        if math.isinf(val) or math.isnan(val):
            h[k] = np.sin(pi*time[k]/T) / (pi*time[k]/T )


    
    warnings.filterwarnings("always")
    return time,h



########################
# THEORY AND STATIC GRAPHS
########################




#---------------------
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



#--------------------
def test_basecase(barylist,f_samp, f_symb, CFO):

    plen = 101 # Note: must be odd
    rolloff = 0.1
    #CFO = 0
    TO = 0
    trans_delay = 0
    
    T = 1/f_samp
    G = 1/f_symb
    spacing = 2*int(G/T)
     
    zadoff_length = 53
    if not (G/T).is_integer():
        raise ValueError('The ratio between the symbol period and sampling period must be an integer')


    

    time, pulse = rcosfilter(plen, rolloff, G, f_samp)

    zpos = zadoff(1,zadoff_length)
    zneg = zpos.conjugate()
    
    # Training sequence creation and "analogification"
    repeat = 1
    training_seq = np.concatenate(tuple([zneg]*repeat +[np.array([0])] + [zpos]*repeat))
    analog_sig = d_to_a(training_seq, pulse, spacing)
    analog_zpos = d_to_a(zpos, pulse, spacing)

    # pad zeros such as to implement TO
    zerocount = round(len(analog_sig)/2)
    analog_sig = np.concatenate((np.zeros(zerocount-TO-trans_delay), \
                                 analog_sig, \
                                 np.zeros(zerocount+TO+trans_delay)))

    # Apply CFO
    for k in range(len(analog_sig)):
        analog_sig[k] = analog_sig[k] *  np.exp(2*pi*1j*CFO*(k*T-TO-trans_delay))
    

    # Zero padding the positive sequence for cross-correlation
    pad_zpos = np.insert(zpos,slice(1,None,1),0)
    for k in range(2,spacing):
        pad_zpos = np.insert(pad_zpos,slice(1,None,k),0)
    pad_zneg = pad_zpos.conjugate()


    # Taking the cross-correlation and printing the adjusted barycenter
    #barypos, crosscorrpos = barycenter_correlation(analog_zpos,analog_sig, power_weight=10) 
    barypos, crosscorrpos = barycenter_correlation(pad_zpos,analog_sig, power_weight=10) 
    baryneg, crosscorrneg = barycenter_correlation(pad_zneg,analog_sig, power_weight=10) 

    baryoffset = len(crosscorrneg)/2 + 0.5
    barylist[0].append(barypos-baryoffset)
    barylist[1].append(baryneg-baryoffset)

    
    #curve = np.abs(crosscorrpos)
    #curve = np.abs(analog_sig)
    #indep_ax = np.arange(len(curve)) - len(curve)/2 + 0.5
    #plt.plot((indep_ax, indep_ax), (curve, np.zeros(len(curve))), 'k-')
    #plt.scatter(indep_ax, curve, marker='.')
    #iplt.plot(indep_ax,curve,'k-')
    #x_lims = [min(indep_ax), max(indep_ax)]
    #plt.plot(x_lims, [0,0], 'k-')
    #plt.xlim(x_lims)

    #plt.show()


def barycenter_width_graph():

    barylist = [[],[]]
    

    f_samp = 1000 # Sampling rate
    f_symb = 100# Symbol frequency
    CFO = np.arange(-1*f_symb,f_symb,0.01*f_symb)
    for k in CFO:
        test_basecase(barylist, f_samp, f_symb, k)


    barywidth = np.array(barylist[0]) - np.array(barylist[1])

    plt.plot(CFO,barywidth)
    #plt.plot(CFO,barylist[0])
    #plt.plot(CFO,barylist[1])
    plt.show()

"""
TODO:

0 EMACS: implement shift-enter as return and remove one tab


10. Implement some sort of check for the "variance" of the barycenter. In the cross
correlation, maybe drop all entries 10% from the min?

11. Find papers/textbooks on the kind of SNR that would be observed

20. Consider using doubles everywhere?

"""



"""
OPTIMIZATION LIST:

ZADOFF:
1. If the q, u, and oversampling are not used, just make a zadoff_lite(N) function to save on computation time


BARYCENTER_CORRELATION()
1. Consider a different convolution algorithm:
http://wiki.scipy.org/Cookbook/ApplyFIRFilter

2. Use mode='valid' instead of full (to save on some computation). This will require adjusting the value of the output, as the cross-correlation matrix will not have the same size as g

3. As zadoff-chu sequences have constant amplitude, maybe you could only store the phase? instead
of real & imaginary part

4. Comsider using complex 64 or *gasp* complex32

NODES:
make a Nodes class instead of multiple single nodes.

"""
