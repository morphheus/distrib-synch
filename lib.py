#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import collections
import bisect
import math
import warnings
import time

from numpy import pi
from pprint import pprint
from scipy.signal import fftconvolve
#from scipy.fftpack import 

warnings.simplefilter('default')



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
def barycenter_correlation(f,g, power_weight=2, method='numpy', bias_thresh=0):
    """Outputs the barycenter location of 'f' in 'g'. g is expected to be the
    longer array
    Note: barycenter will correspond to the entry IN THE CROSS CORRELATION

    bias_thresh will only weight the peaks within bias_thresh of the maximum.
    
    """
    if len(g) < len(f):
        raise AttributeError("Expected 'g' to be longer than 'f'")
    
    if method == 'numpy':
        cross_correlation = np.convolve(f.conjugate(),g,mode='valid')
    elif method == 'scipy':
        cross_correlation = fftconvolve(g, f.conjugate(),mode='valid')
    else: raise ValueError("Unkwnown '" + method +"' method")


    cross_correlation = np.absolute(cross_correlation)
    if bias_thresh:
        """We calculate the bias to remove from the absolute of the crosscorr
        Note that entries smaller than 0.01 of the max are NOT included in the bias"""
        #bias = cross_correlation.max()*bias_thresh
        bias = np.sum(cross_correlation)/len(cross_correlation) * 2 
        grid = np.meshgrid(cross_correlation,bias)
        remove = (grid[0] < grid[1])[0]
        cross_correlation -= bias*0.999 # The 0.99 is to take care of rounding errors
        cross_correlation[remove] = 0


    
    weight = cross_correlation**power_weight
    weightsum = np.sum(weight)
    lag = np.indices(weight.shape)[0]

    # If empty cross_correlation, return -1
    if not weightsum:
        barycenter = -1
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








#---------------------------
def rcosfilter(N, a, T, f, dtype='complex128'):
    """Raised cosine:
    N: Number of samples
    a: rolloff factor (alpha)
    T: symbol period
    f: sampling period

    t: time indexes associated with impulse response
    h: impulse response

    NOTE: this thing far from optimized
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







#--------------------
def test_crosscorr(p):
    """This function builds the sampled analog signal from the appropriate components. It then finds the two barycenters on said built signal"""

    if not p.init_update:
        raise AttributeError("Must execute p.update() before passing the Params class to this function")

    
    # Taking the cross-correlation and printing the adjusted barycenter
    barypos, crosscorrpos =barycenter_correlation(p.pad_zpos,p.analog_sig, power_weight=p.power_weight) 
    baryneg, crosscorrneg =barycenter_correlation(p.pad_zneg,p.analog_sig, power_weight=p.power_weight) 

    baryoffset = 0#len(crosscorrneg)/2 + 0.5


    # Place all return arrays in one struct for simplicity
    output = Struct()
    output.add(barypos=barypos-baryoffset)
    output.add(baryneg=baryneg-baryoffset)

    if not p.full_sim:
        output.add(crosscorrpos=crosscorrpos)
        output.add(crosscorrneg=crosscorrneg)

    return output
    


   
# -------------------
def calc_both_barycenters(p, *args):
    """Wrapper that calculates the barycenter on the specified channel. If no channel specified,
    it uses analog_sig instead"""
    if p.full_sim:
        g = args[0]
    else:
        g = p.analog_sig

    barypos, crosscorrpos =barycenter_correlation(p.pad_zpos, g, power_weight=p.power_weight, bias_thresh=p.bias_removal) 
    baryneg, crosscorrneg =barycenter_correlation(p.pad_zneg, g, power_weight=p.power_weight, bias_thresh=p.bias_removal) 

    return barypos, baryneg, crosscorrpos, crosscorrneg



#------------------------
def build_timestamp_id():
    """Builds a timestamp, and appens a random 3 digit number after it"""
    tempo = time.localtime()
    vals = ['year', 'mon', 'mday', 'hour', 'min', 'sec']
    vals = ['tm_' + x for x in vals]

    tstr = [str(getattr(tempo,x)).zfill(2) for x in vals]

    return int(''.join(tstr) + str(np.random.randint(999)).zfill(3))








##########################
# CLASSDEFS
#########################


# ------------
class Struct: 
    """Basic structure"""
    def add(self,**kwargs):
        self.__dict__.update(kwargs)

    def __iter__(self):
        for name, val in self.__dict__.items():
            yield name, val

    def print_all_items(self):
        for name,val in self:
            print(name + '\n' + str(val)+ '\n')

#--------------
class Params(Struct):
    """Parameter struct containing all the parameters used for the simulation, from the generation of the modulated training sequence to the exponent of the cross-correlation"""
    #------------------------------------
    def __init__(self):
        self.add(plen=101) # Note: must be odd
        self.add(rolloff=0.1)
        self.add(CFO=0)
        self.add(TO=0)
        self.add(trans_delay=0)
        self.add(f_samp=1) # Sampling rate
        self.add(f_symb=1) # Symbol frequency
        self.add(zc_len=11) # Zadoff-chu length
        self.add(repeat=1) # Number of ZC sequence repeats
        self.add(spacing_factor=2) # Number of ZC sequence repeats
        self.add(power_weight=10) # Number of ZC sequence repeats
        self.add(full_sim=True)
        self.add(pulse_type='raisedcosine')
        self.add(init_update=False)
        self.add(init_basewidth=False)
        self.add(bias_removal=0)





    #------------------------------------
    def build_training_sequence(self):
        """Builds training sequence from current parameters"""
        zpos = zadoff(1,self.zc_len)
        zneg = zpos.conjugate()
        training_seq = np.concatenate(tuple([zneg]*self.repeat+[np.array([0])]+[zpos]*self.repeat))
        #training_seq = np.concatenate(tuple([zneg]*self.repeat+[zpos]*self.repeat))

        self.add(zpos=zpos)
        self.add(training_seq=training_seq)


    #------------------------------------
    def calc_base_barywidth(self):
        """Calculates the barycenter width of the given parameters"""
        """ASSUMPTION: spacing_factor = 2"""
        if not self.init_update:
            raise AttributeError("Must execute p.update() before passing the Params class to this function")
        if self.CFO != 0:
            warnings.warn("calc_base_barywidth() was called with non-zero CFO in argument parameters")

        # Finding base barywidth
        full_sim_tmp = self.full_sim
        CFO_tmp = self.CFO
        self.full_sim = False
        self.update()
        barypos, baryneg, _, _ = calc_both_barycenters(self)
        self.add(basewidth=barypos-baryneg)


    
        # Finding barywidth slope
        loc = 0.05
        
        self.CFO = -1*loc*self.f_symb
        self.update()
        barypos, baryneg, _, _ = calc_both_barycenters(self)
        lowidth = barypos-baryneg
        
        self.CFO = loc*self.f_symb
        self.update()
        barypos, baryneg, _, _ = calc_both_barycenters(self)
        hiwidth = barypos-baryneg
        
        slope = (hiwidth - lowidth)/(loc*2)

        # putting state back to proper values
        self.CFO = CFO_tmp
        self.full_sim = full_sim_tmp
        self.update()
        self.add(baryslope=slope)
        self.init_basewidth = True



    #------------------------------------
    def build_pulse(self):
        """Builds pulse from current parameters"""
        if self.pulse_type == 'raisedcosine':
            time, pulse = rcosfilter(self.plen, self.rolloff, 1/self.f_symb, self.f_samp)
        else:
            raise ValueError('The "' + pulse_type + '" pulse type is unknown')

        self.add(pulse_times=time)
        self.add(pulse=pulse)


    

    #------------------------------------
    def build_analog_sig(self):
        """Must run build_pulse and build_training_sequence first"""
        T = 1/self.f_samp
        analog_sig = d_to_a(self.training_seq, self.pulse, self.spacing)
        analog_zpos = d_to_a(self.zpos, self.pulse, self.spacing)

        # pad zeros such as to implement TO
        if not self.full_sim:
            zerocount = round(len(analog_sig))*2
            analog_sig = np.concatenate((np.zeros(zerocount- self.TO- self.trans_delay), \
                                     analog_sig, \
                                     np.zeros(zerocount + self.TO + self.trans_delay)))
        
            # Apply CFO only if not running a synchronization simulation
            time_arr = (np.arange(len(analog_sig))+np.random.rand()*1000*len(analog_sig))*T
            CFO_arr = np.exp( 2*pi*1j*self.CFO*(time_arr - self.trans_delay))
            analog_sig = analog_sig*CFO_arr

        
        # Zero padding the positive sequence for cross-correlation
        pad_zpos = np.insert(self.zpos,slice(1,None,1),0)
        for k in range(2,self.spacing):
            pad_zpos = np.insert(pad_zpos,slice(1,None,k),0)
        
        pad_zneg = pad_zpos.conjugate()

        self.add(pad_zpos=pad_zpos)
        self.add(pad_zneg=pad_zneg)
        self.add(analog_sig=analog_sig)
        self.add(analog_zpos=analog_zpos)



    #------------------------------------
    def update(self):
        """Updates dependent variables with current variables"""
        self.build_pulse()
        self.build_training_sequence()

        tmp = self.f_samp/self.f_symb
        if not float(tmp).is_integer():
            raise ValueError('The ratio between the symbol period and sampling period must be an integer')
        self.add(spacing=self.spacing_factor*int(tmp))
        
        self.build_analog_sig()
        
        self.init_update = True



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

test_crosscorr

"""
