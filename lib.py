#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import collections
import bisect
import math
import warnings
import time
from sqlite3 import OperationalError

from numpy import pi
from pprint import pprint
from scipy.signal import fftconvolve

import dumbsqlite3 as db
#from scipy.fftpack import 

warnings.simplefilter('default')




CPLX_DTYPE = 'complex128'
INT_DTYPE = 'int64'

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
def barycenter_correlation(f,g, power_weight=2, method='numpy', bias_thresh=0, mode='valid'):
    """Outputs the barycenter location of 'f' in 'g'. g is expected to be the
    longer array
    Note: barycenter will correspond to the entry IN THE CROSS CORRELATION

    bias_thresh will only weight the peaks within bias_thresh of the maximum.
    
    """
    if len(g) < len(f):
        raise AttributeError("Expected 'g' to be longer than 'f'")
    
    if method == 'numpy':
        cross_correlation = np.correlate(g, f, mode=mode)
    elif method == 'scipy':
        cross_correlation = fftconvolve(g, f.conjugate(),mode=mode)
    else: raise ValueError("Unkwnown '" + method +"' method")


    cross_correlation = np.absolute(cross_correlation)
    if bias_thresh:
        """We calculate the bias to remove from the absolute of the crosscorr"""
        
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


    if mode == 'valid':
        barycenter += math.floor(len(f)/2) # Correct for valid mode
    
    return barycenter, cross_correlation





#------------------
def d_to_a(values, pulse, spacing,dtype=CPLX_DTYPE):
    """outputs an array with modulated pulses"""
    plen = len(pulse)
     
    output = np.zeros((len(values)-1)*(spacing)+plen,dtype=dtype)
    idx = 0;
    for val in values:
        output[idx:idx+plen] = output[idx:idx+plen] + val*pulse
        idx += spacing

    return output








#---------------------------
def rcosfilter(N, a, T, f, dtype=CPLX_DTYPE):
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







# PENDING DELETION
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
def calc_both_barycenters(p, *args,mode='valid'):
    """Wrapper that calculates the barycenter on the specified channel. If no channel specified,
    it uses analog_sig instead"""
    if p.full_sim and len(args) > 0:
        g = args[0]
    else:
        g = p.analog_sig


    if p.crosscorr_fct == 'zeropadded':
        f1 = p.pad_zpos
        f2 = p.pad_zneg
    elif p.crosscorr_fct == 'analog':
        f1 = p.analog_zpos
        f2 = p.analog_zneg
    else:
        raise Exception('Invalid p.crosscorr_fct value')
    
    barypos, crosscorrpos =barycenter_correlation(f1 , g, power_weight=p.power_weight, bias_thresh=p.bias_removal, mode=mode) 
    baryneg, crosscorrneg =barycenter_correlation(f2 , g, power_weight=p.power_weight, bias_thresh=p.bias_removal, mode=mode) 

    return barypos, baryneg, crosscorrpos, crosscorrneg






#------------------------

def build_timestamp_id():
    """Builds a timestamp, and appens a random 3 digit number after it"""
    tempo = time.localtime()
    vals = ['year', 'mon', 'mday', 'hour', 'min', 'sec']
    vals = ['tm_' + x for x in vals]

    tstr = [str(getattr(tempo,x)).zfill(2) for x in vals]

    return int(''.join(tstr) + str(np.random.randint(999)).zfill(3))



#------------------------
def barywidth_map(p, reach=0.05, scaling=0.001, force_calculate=False):
    """Generates the barywidth map for a given range, given as a fraction of f_symb
    If the map already exists, it pulls it from the sql database instead"""

    """It also does a linear regression to the data"""

    if not (reach/scaling).is_integer():
        raise Exception("The ratio reach/scaling must be an integer")



    dbase_file = 'barywidths.sqlite'
    sql_table_name='barywidths'
    conn = db.connect(dbase_file)
    
    # Fetch all known barywidths
    save_skiplist = ['full_sim', 'init_update', 'init_basewidth', 'TO', 'CFO']
    query_skiplist = save_skiplist + ['basewidth', 'baryslope', 'order2fit']
    values_to_query = {key:p.__dict__[key] for key in p.__dict__.keys() if key not in query_skiplist}
    values_to_query['reach'] = reach
    values_to_query['scaling'] = scaling
    try:
        db_output = db.fetch_matching(values_to_query, tn=sql_table_name, get_data=False, conn=conn, dbase_file=dbase_file)
    except OperationalError:
        db_output = []

    
    # If we had a positive match, return the database match, or delete it if force enabled
    CFO_halflen = round(reach/scaling)
    CFO = np.arange(-CFO_halflen,CFO_halflen)*scaling*p.f_symb
    index_zero = round(len(CFO)/2)
    
    if db_output and not force_calculate:
        tmp = db.fetch_cols(db_output[0][0], ['baryslope', 'basewidth', 'barywidths', 'order2fit'], conn=conn, tn=sql_table_name)
        conn.close()
        p.add(baryslope=tmp[0])
        p.add(basewidth=tmp[1])
        p.add(order2fit=tmp[3])
        p.init_basewidth = True
        return CFO, tmp[2]
    elif db_output and force_calculate:
        db.del_row(db_output[0][0], conn=conn, tn=sql_table_name)


    # Outputs which entry isn't matching DEBUG CODE
    if False:
        db_output = db.fetchall(tn=sql_table_name)
        collist = db.fetch_collist(tn=sql_table_name)

        db_a = db_output[0][collist.index('analog_sig')]
        mem_a = p.analog_sig
        warnings.filterwarnings('ignore')
        for k,col in enumerate(collist):
            try:
                if db_output[0][k] == p.__dict__[col]:
                    string = '-'*5
                else:
                    string = 'x'*5
                print(string + ' ' + col)
            except (ValueError):
                if (db_output[0][k]==p.__dict__[col]).all():
                    string = '-'*5
                else:
                    string = 'x'*5
                print(string + ' ' + col)
            except KeyError:
                pass
    


    
    # If nothing in database, calculate a new set of bayrwidths and save it 
    print('Generating barywidth map...')
    
    initial_full_sim = p.full_sim
    p.full_sim = False

    barywidths = np.empty(len(CFO))
    for k, val in enumerate(CFO):
        p.CFO = val
        p.update()
        barypos, baryneg, _, _ = calc_both_barycenters(p)
        barywidths[k] = barypos - baryneg

    p.CFO = 0
    p.full_sim = initial_full_sim
    p.update() # Set everything back to normal
    p.add(barywidths_arr=barywidths)

    
    # FITTINGS
    p.init_basewidth = True
    basewidth = barywidths[index_zero]
    p.add(basewidth=basewidth)

    # Linear fit
    fit = np.polyfit(CFO, barywidths, 1)
    p.add(baryslope=fit[0])

    # 2nd degree fit
    fit = np.polyfit(CFO, barywidths, 2)
    p.add(order2fit=fit)

    # Save all values used to generate the map into the database for caching
    values_to_save = {key:p.__dict__[key] for key in p.__dict__.keys() if key not in save_skiplist}
    values_to_save['reach'] = reach
    values_to_save['scaling'] = scaling
    values_to_save['barywidths'] = barywidths
    values_to_save['date'] = db.build_timestamp_id()
    
    db.add(values_to_save, tn=sql_table_name, conn=conn)


    conn.close()
    return CFO, barywidths





#-------------------------
def cfo_mapper_linear(barywidth, p):
    tmp = (barywidth - p.basewidth) / (p.baryslope)
    return tmp

#-------------------------
def cfo_mapper_order2_mk0(barywidth, p):
    poly = p.order2fit
    poly[2] = p.basewidth - barywidth
    roots = np.real(np.roots(poly))
    
    # Output the CFO matching the increasing x-value of the curve
    if poly[0] > 0:
        return np.max(roots)
    else:
        return np.min(roots)
    

#-------------------------
def cfo_mapper_order2_mk1(barywidth, p):

    min_correction = 1*p.hill_width
    
    poly = p.order2fit
    poly[2] = p.basewidth - barywidth
    roots = np.real(np.roots(poly))
    
    # Output the CFO matching the increasing x-value of the curve
    if poly[0] > 0:
        CFO =  np.max(roots)
    else:
        CFO = np.min(roots)

    # don't calculate CFO that is within 0.001
    if abs(CFO) < min_correction:
        CFO = 0

    return CFO


#-------------------------
def cfo_mapper_injective(barywidth, p):
    """Does direct mapping between barywidth and CFO. Requires monotone increasing barywidth map"""
    # Works well with power_weight=8


    # Check if injective
    prev = p.barywidths_arr[0]-1
    for current in p.barywidths_arr:
        if current < prev:
            raise Exception('p.barywidths is not monotone increasing at y = ' + str(current))
        prev = current

    exit()



#------------------------
def delay_pd_gaussian():
    pass


#------------------------
def delay_pdf_static(controls):
    """Simple exponentially decaying echoes"""
    taps = controls['max_echo_taps']
   
    delay_list = [x*controls['frameunit']*0.1376732/(taps) for x in range(taps)]
    delays = np.array([round(x) for x in delay_list], dtype=INT_DTYPE)

    amp_list = np.exp([-0.5*x for x in range(taps)])


    amp = np.array(amp_list, dtype=CPLX_DTYPE)
    return delays, amp




#------------------------
def build_delay_matrix(controls, delay_fct=delay_pdf_static):
    """Insert documentation here"""
    # Note that PDF functions must be declared/imported BEFORE this function definition

    # ECHOES USAGE:
    # echo_<name>[curclk][emitclk][k] = k'th echo between the two clocks. 
    

    clkcount = controls['clkcount']
    #clkcount = 1
    array_dtype_string = INT_DTYPE+','+CPLX_DTYPE
    echoes = np.empty((clkcount, clkcount, controls['max_echo_taps']), dtype=array_dtype_string)
    echoes.dtype.names = ('delay', 'amp')


    
    for k in range(clkcount):
        for l in range(clkcount):
            echoes['delay'][k][l], echoes['amp'][k][l] = delay_fct(controls)
            


    # TODO: Pick input delay/amp from controls
    # TODO: don't use a structured array
    controls['echo_delay'] = echoes['delay']
    controls['echo_amp'] = echoes['amp']






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
        self.add(spacing_factor=2) 
        self.add(power_weight=10) 
        self.add(full_sim=True)
        self.add(pulse_type='raisedcosine')
        self.add(init_update=False)
        self.add(init_basewidth=False)
        self.add(bias_removal=0)
        self.add(crosscorr_fct='analog')





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
        
        slope = (hiwidth - lowidth)/(self.CFO*2)

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
        analog_zneg = analog_zpos.conjugate()

        # Apply CFO only if not running a synchronization simulation
        if not self.full_sim:
            zerocount = round(len(analog_sig))*2
            analog_sig = np.concatenate((np.zeros(zerocount- self.TO- self.trans_delay), \
                                     analog_sig, \
                                     np.zeros(zerocount + self.TO + self.trans_delay)))
        
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
        self.add(analog_zneg=analog_zneg)



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

        # width of each hill (units of frequencies)
        hill_width = 0.05/11*self.zc_len/51 * self.f_symb
        self.add(hill_width=hill_width)



