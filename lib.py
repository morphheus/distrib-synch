#!/usr/bin/env python
import numpy as np
import collections
import bisect
import math
import warnings
import time
import inspect
import os.path
import string
from sqlite3 import OperationalError

from numpy import pi
from numpy import log10
from pprint import pprint
from scipy import signal
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit, leastsq

import dumbsqlite3 as db



LOGFILE = 'simlog.txt'

FLOAT_DTYPE = 'float64'
CPLX_DTYPE = 'complex128'
INT_DTYPE = 'int64'

#crosscorr_fct = lambda f,g,mode: fftconvolve(g, f.conjugate(),mode=mode)
crosscorr_fct = lambda f,g,mode: np.correlate(g,f,mode=mode)

BARY_SQL_TABLE_NAME = 'barywidths'
BARY_DBASE_FILE = 'barywidths.sqlite'

LAST_PRINT_LEN = 0

BASE62_ALPHABET =  string.digits + string.ascii_uppercase + string.ascii_lowercase

SOL = 299792458 # Speed of light in m/s


#--------------------
def zadoff(u, N,oversampling=1,  q=0):
    """Returns a zadoff-chu sequence"""

    x = np.linspace(0, N-1, N*oversampling, dtype='float64')
    return np.exp(-1j*pi*u*x*(x+1+2*q)/N)

def logistic(x, coeffs):
    """Logistic function with y-intercept always  0"""
    return (1/(1+np.exp(coeffs[1]*x))-0.5)*coeffs[0] + coeffs[2]*x

def cosine_zadoff_overlap(u, N,oversampling=1,  q=0):
    """Returns a cos(un^2) sequence"""

    x = np.linspace(0, N-1, N*oversampling, dtype='float64')
    return np.cos(pi*u*x*(x+1+2*q)/N)

def step_sin(x, a, b, h, k):
    """ x + sin(x), with the 4 canonical parameters"""
    xval = b*(x-h)
    return a*(xval+np.sin(xval)) + k

def cplx_gaussian(shape, noise_variance):
    """Assume jointly gaussian noise. Real and Imaginary parts have
    noise_variance/2 actual variance"""
    """The magnitude of the noise will have a variance of 'noise_variance'"""
    # If variance is equal to zero, numpy returns an error
    if noise_variance:
        noise_std = (noise_variance/2)**2 # Due to generating a cplx numbarr
        x = (np.random.normal(size=shape, scale=noise_std) + 1j*np.random.normal(size=shape, scale=noise_std)).astype(CPLX_DTYPE)
    else:
        x = np.array([0+0j]*shape[0]*shape[1], dtype=CPLX_DTYPE).reshape(shape[0],-1)
    return x

def rcosfilter(N, a, T, f, dtype=CPLX_DTYPE, frac_TO=0):
    """Raised cosine:
    N: Number of samples
    a: rolloff factor (alpha)
    T: symbol period
    f: sampling rate

    t: time indexes associated with impulse response
    h: impulse response

    NOTE: this thing far from optimized
    """
    time = (np.arange(N,dtype=dtype)-N/2+0.5 - frac_TO)/float(f)

    if N % 2 and frac_TO % 1 == 0:
        zero_entry = math.floor(N/2 + int(frac_TO)) # Index of entry with a zero
    else:
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

def rrcosfilter(N, a, T, f, dtype=CPLX_DTYPE, frac_TO=0):
    """Root Raised cosine:
    N: Number of samples
    a: rolloff factor (alpha)
    T: symbol period
    f: sampling rate

    t: time indexes associated with impulse response
    h: impulse response

    """
    time = (np.arange(N,dtype=dtype)-N/2+0.5 - frac_TO)/float(f)
    zero_entry = math.floor(N/2) # Index of entry with a zero
    if not N % 2:
        zero_entry = -1

    warnings.filterwarnings("ignore")
    h = np.empty(N, dtype=dtype)
    for k, t in enumerate(time):
        if k == zero_entry:
            h[k] = (1-a+4*a/pi)
        else: 
            #h[k] = np.sin(pi*t/T) * np.cos(pi*a*t/T) / (pi*t/T  *  (1-4*(a*t/T)**2 ))
            h[k] =  (4*a/(pi)) * (
                np.cos((1+a)*pi*t/T) + np.sin((1-a)*pi*t/T)/(4*a*t/T)
            ) / (
                1-(4*a*t/T)**2
            )

    for k, val in enumerate(h):
        if math.isinf(val) or math.isnan(val):
            h[k] = ((1+2/pi)*np.sin(pi/(4*a)) + (1-2/pi)*np.cos(pi/(4*a))) * (a/(np.sqrt(2)))


    
    warnings.filterwarnings("always")
    return time,h

#--------------------
def calc_snr(ctrl,p):
    """Calculates the SNR of the system provided"""
    snr_db = ctrl.trans_power - ctrl.noise_power
    return(snr_db)

def db2amp(db):
    """Converts dbm to power (in miliwat)"""
    return 10**(0.05*db)

def db2pwr(db):
    """Converts db to power """
    return 10**(0.1*db)

def in_place_mov_avg(vector, wlims):
    """Runs an in-place moving average on the input vector."""
    # vector: input array
    # wlen  : list/tuple/array containing the bounds of the left and right window. Must be
    #                                      1-dimensional

    # For better efficienty, wlims should be an array
    # Make sure the input vector has a dtype of float!!

    if len(wlims) != 2:
        raise Exception('Invalid format for wlims: expected len(wlims) == 2')

    # If window too small, do nothing!
    wlen = wlims[1] + wlims[0] + 1
    if wlen < 2: return 0


    
    lb = wlims[0]
    rb = wlims[1]
    vlen = len(vector)

    # Instantiate temporary storage
    tmp = collections.deque()

    
    # Left edge, until the window can fully fit
    for k in range(lb):
        tmp.append(vector[0:k+rb+1].sum()/(k+rb+1))


    # Main moving average, where the window is fully contained in the vector
    # Replace value whenever it is no longer necessary for future averagings
    for k in range(lb, vlen-rb):
        tmp.append(vector[k-lb:k+rb+1].sum()/wlen)
        vector[k-lb] = tmp.popleft()

    # Right side MA
    for k in range(vlen-rb, vlen):
        tmp.append(vector[k-lb:].sum()/(vlen-k+lb))
        vector[k-lb] = tmp.popleft()

    # empty the deque
    for k in range(lb):
        vector[vlen-lb+k] = tmp.popleft()

def cumsum_mov_avg(vector, N):
    """Runs an in-place moving average on the input vector. Uses cumulative sums, and may run into overflow issues
    vector: input numpy array
    N     : MA window length. Should be odd
    """

    if N%2 == 0:
        raise ValueError('Moving average window should be an odd number')

    left = math.ceil(N/2)
    right = N-left
    cumsum = np.cumsum(  np.append(  np.insert(vector, 0, np.zeros(left)),  np.zeros(right))  ) 
    MA = (cumsum[N:] - cumsum[:-N])

    # Appropriate division
    MA[left-1:-right] /= N
    for k in range(right):
        MA[k] /= left+k
        MA[-(k+1)] /= left+k

    return MA

def convolve_mov_avg(vector,N):
    """Computes the running mean of width N on the vector. Uses convolution"""
    if N%2 == 0:
        raise ValueError('Moving average window should be an odd number')

    left = math.floor(N/2)
    right = N-left-1

    MA = np.convolve(vector, np.ones(N), mode='full')[left:-right]

    # fix edges
    MA[left:-right] /= N
    for k in range(right):
        tmpN = left+k+1
        MA[k] /= tmpN
        MA[-(k+1)] /= tmpN

    return MA

def d_to_a(values, pulse, spacing,dtype=CPLX_DTYPE):
    """outputs an array with modulated pulses"""
    plen = len(pulse)
     
    output = np.zeros((len(values)-1)*(spacing)+plen,dtype=dtype)
    idx = 0;
    for val in values:
        output[idx:idx+plen] = output[idx:idx+plen] + val*pulse
        idx += spacing

    return output

#--------------------
def barywidth_map(p, reach=0.05, scaling_fct=100, force_calculate=False, disp=False):
    """Generates the barywidth map for a given range, given as a fraction of f_symb
    If the map already exists, it pulls it from the sql database instead"""

    """It also does a linear regression to the data"""


    if not float(scaling_fct).is_integer():
        ValueError('Scaling must be an integer')

    scaling = reach/scaling_fct

    sql_table_name = BARY_SQL_TABLE_NAME
    dbase_file = BARY_DBASE_FILE
    if not os.path.isfile(dbase_file):
        db.init(dbase_file, sql_table_name)
    conn = db.connect(dbase_file)
    
    # Fetch all known barywidths
    save_skiplist = ['full_sim', 'init_update', 'init_basewidth', 'TO', 'CFO']
    query_skiplist = save_skiplist + ['basewidth', 'baryslope', 'order2fit', 'logisticfit']
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
    
    # Fetch data if possible/allowed. If data exist but must recalc, delete existing.
    if db_output and not force_calculate:
        to_fetch = ['baryslope', 'basewidth', 'barywidth_arr', 'order2fit', 'CFO_arr', 'logisticfit']
        tmp = db.fetch_cols(db_output[0][0], to_fetch , conn=conn, tn=sql_table_name)
        conn.close()
        p.add(baryslope=tmp[0])
        p.add(basewidth=tmp[1])
        p.add(barywidth_arr=tmp[2])
        p.add(order2fit=tmp[3])
        p.add(CFO_arr=tmp[4])
        p.add(logisticfit=tmp[5])
        p.init_basewidth = True
        if disp: db.pprint_date(db_output[0][0])
        return CFO, tmp[2]
    elif db_output and force_calculate:
        db.del_row(db_output[0][0], conn=conn, tn=sql_table_name)



    
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
    p.add(barywidth_arr=barywidths)
    p.add(CFO_arr=CFO)

    
    
    # FITTINGS
    print('Executing data fits...')
    p.init_basewidth = True
    basewidth = barywidths[index_zero]
    barywidths -= basewidth # Remove the basewidth to ease the fitting
    p.add(basewidth=basewidth)

    
    linear_erf = lambda coeffs, x, y: -1*y + coeffs[0]*x
    order2_erf = lambda coeffs, x, y: -1*y + coeffs[0]*x**2 + coeffs[1]*x
    logistic_erf = lambda coeffs, x, y: -1*y + logistic(x,coeffs)

    # Linear fit
    #fit = np.polyfit(CFO, barywidths, 1)
    fit = leastsq(linear_erf, [0], args=(CFO, barywidths))
    p.add(baryslope=fit[0][0])

    # 2nd degree fit
    fit,_ = leastsq(order2_erf, [0,0], args=(CFO, barywidths))
    fit = np.polyfit(CFO, barywidths, 2)
    p.add(order2fit=fit)

    # Logistic
    fit,_ = leastsq(logistic_erf, [1,-1,p.baryslope], args=(CFO, barywidths))
    p.add(logisticfit=fit)

    # x + sin(x) fit
    # find the initial frequency (b). The while loop spits out the index of the first half period
    #idx = index_zero + 1
    #while barywidths[idx] > ( cfo[idx]*p.baryslope + p.basewidth ):
    #    idx += 1

    #init_b = 2*pi/(CFO[idx]*2)
    #p0 = [p.baryslope/init_b, init_b, 0, p.basewidth] # Initial a,b,h,k values
    #sin_fit = curve_fit(step_sin, CFO, barywidths, p0=p0)
    #p.add(sin_fit=sin_fit)
    
    

    # Save all values used to generate the map into the database for caching
    barywidths += basewidth # Put back the original data
    values_to_save = {key:p.__dict__[key] for key in p.__dict__.keys() if key not in save_skiplist}
    values_to_save['reach'] = reach
    values_to_save['scaling'] = scaling
    values_to_save['barywidth_arr'] = barywidths
    values_to_save['CFO_arr'] = CFO
    values_to_save['date'] = db.build_timestamp_id()
    
    db.add(values_to_save, tn=sql_table_name, conn=conn)


    conn.close()
    return CFO, barywidths

def calc_both_barycenters(p, *args,mode='valid'):
    """Wrapper that calculates the barycenter on the specified channel. If no channel specified,
    it uses analog_sig instead"""
    if len(args) > 1:
        raise TypeError('Too many arguments')
    
    if p.full_sim and len(args) > 0:
        g = args[0]
    else:
        g = p.analog_sig

    # kwargs tha willl be reused
    barycorr_kwargs ={'power_weight':p.power_weight, 'bias_thresh':p.bias_removal, 'mode':mode, 'ma_window':p.ma_window}


    # Decimation handling
    if 'match_decimate' in p.crosscorr_fct:
        decimated_signal, start_index, _ = match_decimate(g, p.pulse, p.spacing)
        symbol_bary, crosscorrpos = barycenter_correlation(p.training_seq , decimated_signal,
                                                            **barycorr_kwargs) 

        if p.crosscorr_fct == 'match_decimate_argmax':
            # Redefine barypos using the argmax
            barypos = start_index + p.spacing*np.argmax(crosscorrpos)

        elif p.crosscorr_fct == 'match_decimate_wavg':
            barypos = start_index + p.spacing*symbol_bary
        else:
            raise Exception('Invalid p.crosscorr_fct string')

        return barypos, barypos, crosscorrpos, crosscorrpos

    # Non-decimation approaches
    elif p.crosscorr_fct == 'zeropadded':
        f1 = p.pad_zpos
        f2 = p.pad_zneg
    elif p.crosscorr_fct == 'analog':
        f1 = p.analog_zpos
        f2 = p.analog_zneg
    else:
        raise Exception('Invalid p.crosscorr_fct string')
    
    barypos, crosscorrpos = barycenter_correlation(f1 , g, **barycorr_kwargs) 
    if p.train_type == 'chain':
        baryneg, crosscorrneg = barycenter_correlation(f2 , g, **barycorr_kwargs) 
    elif p.train_type == 'single':
        baryneg = barypos
        crosscorrneg = crosscorrpos

    return barypos, baryneg, crosscorrpos, crosscorrneg

def barycenter_correlation(f,g, power_weight=2, method='numpy', bias_thresh=0, mode='valid', ma_window=1):
    """Outputs the barycenter location of 'f' in 'g'. g is expected to be the
    longer array
    Note: barycenter will correspond to the entry IN THE CROSS CORRELATION

    bias_thresh will only weight the peaks within bias_thresh of the maximum.

    ma_window will run a moving average on the cross-correlation before taking the weighted average
    
    """
    if len(g) < len(f):
        raise AttributeError("Expected 'g' to be longer than 'f'")
    
    cross_correlation = crosscorr_fct(f, g, mode)

    cross_correlation = np.absolute(cross_correlation)
    if bias_thresh:
        """We calculate the bias to remove from the absolute of the crosscorr"""
        maxval = np.max(cross_correlation)
        bias = maxval*bias_thresh
        cross_correlation -= bias
        np.clip(cross_correlation, 0, float('inf'), out=cross_correlation)


    # in-place MA filter on the cross correlation
    if ma_window % 2 == 0 or ma_window < 0:
        raise Exception('Moving average window should be odd and positive')

    if ma_window != 1:
        cross_correlation = convolve_mov_avg( cross_correlation, ma_window)


    
    # Generete stuff for weighted average
    weight = cross_correlation**power_weight
    
    weightsum = np.sum(weight)
    lag = np.indices(weight.shape)[0]

    # If empty cross_correlation, return -1. Otherwise, perform weighted average
    if not weightsum:
        barycenter = -1
    else:
        barycenter = np.sum(weight*lag)/weightsum


    if mode == 'valid':
        barycenter += math.floor(len(f)/2) # Correct for valid mode
    
    return barycenter, cross_correlation

def match_decimate(signal, pulse, spacing, mode='same'):
    """Cross-correlated the signal with the shaping pulse
    Then, decimate the resulting signal such that the output has the highest energy
    signal : Signal to pply matched filter on
    pulse  : Signal to match filter with
    spacing: Symbol period, in samples"""

    cross_correlation = crosscorr_fct(pulse,signal, mode)

    # Pick decimation with highest energy
    abs_crosscorr = np.abs(cross_correlation)
    max_energy = 0
    decimated_start_index = 0
    for k in range(spacing):
        energy = abs_crosscorr[k::spacing].sum()
        if energy > max_energy:
            max_energy = energy
            decimated_start_index = k

    decimated = cross_correlation[decimated_start_index::spacing]
    
    return decimated, decimated_start_index,  cross_correlation

#--------------------
def cfo_mapper_pass(barywidth, p):
    """Always returns zero"""
    return 0

def cfo_mapper_linear(barywidth, p):
    """Linear CFO estimation from p.baryslope and p.basewidth"""
    tmp = (barywidth - p.basewidth) / (p.baryslope)
    return tmp

def cfo_mapper_order2(barywidth, p):
    """2nd order CFO estimation from p.order2fit"""
    poly = p.order2fit
    poly[2] = p.basewidth - barywidth
    roots = np.real(np.roots(poly))
    
    # Output the CFO matching the good side of the curve:
    if poly[0] > 0:
        return np.max(roots)
    else:
        return np.min(roots)

def cfo_mapper_injective(barywidth, p):
    """Does direct mapping between barywidth and CFO. Requires monotone increasing barywidth map"""
    # Works well with power_weight=8


    # Check if injective

    """
    prev = p.barywidth_arr[0]-1
    for current in p.barywidth_arr:
        if current < prev:
            raise Exception('p.barywidths is not monotone increasing at y = ' + str(current))
        prev = current
    """

    
    idx = np.searchsorted(p.barywidth_arr, barywidth)
    
    if idx == 0 or idx == len(p.barywidth_arr):
        # If out of bounds, just apply linear method
        return cfo_mapper_linear(barywidth, p)

    return p.CFO_arr[idx]

def cfo_mapper_step_sin(barywidth, p):
    """Step sin function mapper"""

    return none

def delay_pdf_gaussian():
    pass

def delay_pdf_static(ctrl):
    """Simple exponentially decaying echoes"""
    taps = ctrl.max_echo_taps

   
    delay_list = [x*ctrl.basephi*0.1376732/(taps) for x in range(taps)]
    delays = np.array([round(x) for x in delay_list], dtype=INT_DTYPE)
    delays += int(ctrl.min_delay*ctrl.basephi)

    amp_list = np.exp([-0.5*x for x in range(taps)])


    amp = np.array(amp_list, dtype=CPLX_DTYPE)
    return delays, amp

def delay_pdf_exp(t, sigma, t0=0):
    """Exponentially decaying delay PDF""
    amp = (1/sigma) exp(-(1/sigma)*(t-t0))"""
    if sigma == 0:
        amp = 1 if t >= t0 else 0
    else:
        l = 1/sigma
        amp = np.exp(-l*(t - t0)) if t >= t0 else 0

    return amp

def pathloss_freespace(d, f_carr):
    """Free space path loss. Frequencies are expected in MHz, d in meters"""
    return 20*log10(d) + 46.4 + 20*log10(f_carr*1e-9/5)

def pathloss_b1(d, f_carr, los=True):
    """Path loss from table 4.1 in D5.1_v1.0, hexagonal layout. f_carr expected in Hz"""
    fc = f_carr * 1e-9 # Carrier frequency in GHz (to make the switch statement more readable)

    hbs = 1.5
    hms = 1.5
    hbsP = 0.8 # hbs'
    hmsP = 0.8 # hms'
    dbp = 3*hbsP*hmsP*f_carr/SOL

    # IF reaaallly close, model breaks down
    if 0 <= d < 3:
        return 0.2
    # yes los
    elif los:
        if d < dbp:
            pl = 22.7*log10(d) + 27 + 20*log10(fc)
        else:
            pl = 40*log10(d) + 7.56 - 17.3*log10(hbsP) - 17.3*log10(hmsP) + 2.7*log10(fc)
    # no los
    else:
        pl = (44.9 - 6.55*log10(hbs))*log10(d) + 5.83*log10(hbs) - 5 # -5 from 36843 specs
        if 0.45 <= fc < 1.5:
            pl += 16.33 + 26.16*log10(fc)
        elif 1.5 <= fc < 2:
            pl += 14.78 + 34.97*log10(fc)
        elif 2 <= fc < 6:
            pl += 18.38 + 23*log10(fc)
        else:
            raise ValueError('Expected f_carr between 0.45 and 6 GHz')
    
    return pl

@np.vectorize
def pathloss_b1_tot(x, f_samp, f_carr, out_format='amp'):
    """B1_tot, as specified in 36843, A.2.1.2"""

    if x < 0:
        raise ValueError('Expected positive delay')
    if x==0:
        return 0 if out_format=='dB' else 1.0

    # Probability of LOS
    d = samples2dist(x, f_samp, unit='m')
    prob_los = min(18/d, 1)*(1-np.exp(-d/36)) + np.exp(-d/36) # Winner-B1 from table 4.7
    los = np.random.choice([True, False], p=[prob_los, 1-prob_los]) # Roll the dice

    # Compute pathloss (in dB)
    pl = max(pathloss_freespace(d, f_carr), pathloss_b1(d, f_carr, los=los))

    # Format the pathloss
    if out_format=='amp':
        out = db2amp(pl)
    elif out_format=='pwr':
        out = db2pwr(pl)
    elif out_format=='dB':
        out = pl
    elif out_format=='prob_los':
        out = prob_los
    else:
        raise ValueError('Invalid output format: "' + out_format + '" not recognized.')

    return out

def drop_unifcircle(N, D):
    """uniform drop N nodes within a circle of diameter D. Returns x and y coords"""
    radii = np.random.rand(N)*D/2
    angle = np.random.rand(N)*2*pi
    return radii*np.cos(angle), radii*np.sin(angle)

def drop_unifsquare(N, L):
    """uniform drop N nodes within a square of side L. Returns x and y coords"""
    return tuple([(np.random.rand(N, 1)-0.5)*L for x in range(2)])

#--------------------
def buildx(TO,CFO, p):
    """Builds a new x vector with appropriate TO and CFO"""
    p.TO = int(math.floor(TO))
    p.frac_TO = TO % 1
    p.CFO = CFO
    p.update()
    return p.analog_sig.copy()

def loglikelihood_2d(p,t0,l0, theta_range, deltaf_range, var_w=1):
    """loglikelihood function over the range given by theta_range, deltaf_range, around initial values t0 and l0
    output shape: theta x deltaf array
    """

    y = buildx(t0, l0,p)

    M = len(y)

    y += cplx_gaussian([1, M], var_w).reshape(-1).copy()

    tlen = len(theta_range)
    dlen = len(deltaf_range)
    
    CFO_range = deltaf_range*p.f_samp

    diff_magnitude = np.empty([dlen, tlen], dtype=FLOAT_DTYPE)
    xy_diff = np.empty([dlen,M], dtype=CPLX_DTYPE)
    for k,theta in enumerate(theta_range):
        for l, CFO in enumerate(CFO_range):
            xy_diff[l,:] = y - buildx(theta,CFO,p)
        diff_magnitude[:,k] = np.abs(xy_diff).sum(axis=1)**2

    
    loglike = -M*np.log(pi*var_w) - 1/(2*var_w) * diff_magnitude
    return loglike

def loglikelihood_1d_TO(p,t0,l0, theta_range,  var_w=1):
    """loglikelihood function over the range given by theta_range, deltaf_range, around initial values t0 and l0
    output shape: theta x deltaf array
    """

    y = buildx(t0, l0,p)
    M = len(y)
    y += cplx_gaussian([1, M], var_w).reshape(-1).copy()

    tlen = len(theta_range)
    

    xy_diff = np.empty([tlen,M], dtype=CPLX_DTYPE)
    for k,theta in enumerate(theta_range):
        xy_diff[k,:] = y - buildx(theta,l0,p)
    diff_magnitude = np.abs(xy_diff).sum(axis=1)**2

    loglike = -M*np.log(pi*var_w) - 1/(2*var_w) * diff_magnitude
    return loglike

def loglikelihood_1d_CFO(p, t0,l0, deltaf_range, var_w=1):
    """loglikelihood function over the range deltaf_range, around initial values t0 and l0
    output shape: 1 x deltaf array
    """

    y = buildx(t0, l0,p)
    M = len(y)
    y += cplx_gaussian([1, M], var_w).reshape(-1).copy()
    dlen = len(deltaf_range)
    CFO_range = deltaf_range*p.f_samp

    xy_diff = np.zeros([dlen,M], dtype=CPLX_DTYPE)
    for l, CFO in enumerate(CFO_range):
        xy_diff[l,:] = y - buildx(t0,CFO,p)
    diff_magnitude = np.abs(xy_diff).sum(axis=1)**2


    
    loglike = -M*np.log(pi*var_w) - 1/(2*var_w) * diff_magnitude
    return loglike

def ll_redux_2d(p,t0,l0, theta_range, deltaf_range, var_w=1):
    """Function that has the same max  as the loglikelihood function
    output shape: theta x deltaf array
    """

    y = buildx(t0, l0,p)

    M = len(y)

    y += cplx_gaussian([1, M], var_w).reshape(-1).copy()

    tlen = len(theta_range)
    dlen = len(deltaf_range)
    
    CFO_range = deltaf_range*p.f_samp

    loglike = np.empty([dlen, tlen], dtype=FLOAT_DTYPE)
    tmp = np.empty([dlen,M], dtype=FLOAT_DTYPE)
    for k,theta in enumerate(theta_range):
        for l, CFO in enumerate(CFO_range):
            x = buildx(theta,CFO,p)
            tmp[l,:] = np.real(y*x.conjugate())
        loglike[:,k] = tmp.sum(axis=1).copy()

    
    return loglike

#--------------------
def hipass_avg(N):
    """filter0: regular average"""
    b = np.array([1] + list(-1*np.ones(N)/N))
    a = np.array([1])
    return b,a

def hipass_cheby(N):
    """filter1: type 1 chebychev"""
    b, a = signal.cheby2(N, 1, 0.05, btype='high')
    return b,a

def hipass_butter(N):
    """filter2: butterworth filter"""
    b, a = signal.butter(N, 0.01, btype='high')
    return b,a

def hipass_remez(N):
    """filter2: FIR highpass. Requires N Odd"""
    b = signal.remez(N, [0,0.1,0.2,0.5], [0,1])
    return b, np.array([1])

def hipass_semicirc_zeros(N, max_angle, zeros_magnitude):
    """filter2: FIR highpass. Requires N Odd"""
    if N % 2 == 0:
        raise Exception('Requires an odd order')


    angles = np.zeros(N, dtype=FLOAT_DTYPE)
    for k in range(1,N//2+1):
        angles[2*k-1] = k
        angles[2*k] = -k
    angles *= max_angle/(N//2)

    zeros = list(np.exp(1j*angles) * zeros_magnitude)

    b = np.zeros(N+1, dtype=CPLX_DTYPE)
    b[0] = 1
    b[1] = -1*zeros.pop(0)

    #np.set_printoptions(precision=4)
    for z in zeros:
        b[1:] += -1*z*b[:-1]
    
    return b.real, np.array([1])

#--------------------
def build_timestamp_id():
    """Builds a timestamp, and appens a random 3 digit number after it"""
    return db.build_timestamp_id()

def base62_encode(integer):
    """Contracts some integer in base 62"""
    alpha = BASE62_ALPHABET
    base = len(alpha)

    if integer == 0: return alpha[0]

    tmp = integer
    encoded = ''
    while tmp != 0:
        encoded += alpha[tmp % base]
        tmp = tmp // base

    return encoded[::-1]

def base62_decode(encoded):
    """Contracts some integer in base 62"""
    alpha = BASE62_ALPHABET
    base = len(alpha)

    decoded = 0
    for k, char in enumerate(encoded[::-1]):
        decoded += base**k*alpha.index(char)

    return decoded

def appendlog(logdesc):
    """Appends the text to the logfile"""
    tempo = time.localtime()
    vals = ['year', 'mon', 'mday', 'hour', 'min', 'sec']
    vals = ['tm_' + x for x in vals]
    tlst = [str(getattr(tempo,x)).zfill(2) for x in vals] # OS query of vals

    timestamp = '[' + '/'.join(tlst[0:3]) + ' ' + ':'.join(tlst[3:6]) + ' '

    callerframe = inspect.stack()[1]
    caller = str(callerframe.filename) + ':' + str(callerframe.lineno) + '] '
    logmsg = timestamp + caller + logdesc + '\n'

    with open(LOGFILE, 'a+') as f:
        f.write(logmsg)

def avg_copies(data):
    """Calculates the average and STD of all dependent values.
    data: list of tuples or array
    Expected data structure:
    [(x0, y0, z0, a0, ...), (x1, y1, z1, a1, ...), ...]"""
    if type(data).__name__ != 'ndarray':
        arr = np.array(data)
    else:
        arr = data.copy()

    # Make sure all the repeaded x values are grouped together
    #np.random.shuffle(arr)
    arr = arr[arr[:,0].argsort()]

    # Index of unique indices
    _, unique_idx = np.unique(arr[:,0], return_index=True)
    indeps = arr[unique_idx, 0].copy()
    unique_idx = np.append(unique_idx, data.shape[0])
    xcount = len(unique_idx)-1

    # Calculate average and std of all data associated with each x
    stds = np.empty([xcount, arr.shape[1]-1], dtype=arr.dtype)
    avgs = np.empty([xcount, arr.shape[1]-1], dtype=arr.dtype)
    for k in range(xcount):
        currslice = slice(unique_idx[k], unique_idx[k+1])
        stds[k,:] = np.std(arr[currslice, 1:])
        avgs[k,:] = np.mean(arr[currslice, 1:])

    # If 1-d array, flatten 
    if stds.shape[1] == 1: stds = stds.flatten()
    if avgs.shape[1] == 1: avgs = avgs.flatten()

    return indeps, avgs, stds

def samples2dist(x, f_samp, unit='m'):
    """Converts samples to a distance unit (kilometers or meters)"""
    if unit == 'km': unit_fact = SOL/f_samp/1000
    elif unit == 'm': unit_fact = SOL/f_samp
    else: raise('Unknown unit')
    return x*unit_fact

def dist2samples(d, f_samp, unit='m'):
    """Converts samples to a distance unit (kilometers or meters)"""
    return d/samples2dist(1,f_samp, unit=unit)


##########################
# CLASSDEFS
#########################
class Struct: 
    """Basic structure"""
    def add(self,**kwargs):
        self.__dict__.update(kwargs)

    def __iter__(self):
        for name, val in self.__dict__.items():
            yield name, val

class DelayParams(Struct):
    """Parameters class for the delays between nodes"""
    def __init__(self, delay_pdf,
                 taps=1,
                 max_dist_from_origin=0,
                 p_sigma=0):

        if not callable(delay_pdf):
            raise ValueError(type(self).__name__ + " must be initialized with a callable PDF function")
        self.delay_pdf = delay_pdf
        self.pathloss_fct = pathloss_b1_tot
        self.taps = taps
        self.max_dist_from_origin = max_dist_from_origin
        self.p_sigma = p_sigma

    def delay_pdf_eval(self, t, t0=0, **kwargs):
        return self.delay_pdf(t, self.p_sigma, t0=t0, **kwargs)

    def rnd_delay(self,t0, **kwargs):
        """Builds an array of delays with the associated amplitudes. Uniformly picks the delays,
        then feeds it into the PDF function. All time values in terms of basephi.
        input:
            taps  : total number of taps to generate
            delay_min  : SE
            delay_sigma: SE
        output:
            delay: np array of the delays
            amps:  np array of the amplitudes
        """
        delay_list =[t0] +  [np.random.rand()*np.sqrt(12)*self.p_sigma+t0 for x in range(self.taps-1)]
        amp_list = [self.delay_pdf_eval(t,t0=t0, **kwargs) for t in delay_list]
        delay = np.array(delay_list, FLOAT_DTYPE)
        amp = np.array(amp_list, FLOAT_DTYPE)
        return delay, amp

    def build_delay_matrix(self, nodecount, basephi, f_samp, f_carr,  **kwargs):
        """From the delay function, initiate an appropriately sized delay matrix into ctrl"""
        self.f_samp = f_samp
        self.f_carr = f_carr
        self.basephi = basephi
        array_dtype_string = INT_DTYPE+','+CPLX_DTYPE
        echoes = np.zeros((nodecount, nodecount, self.taps), dtype=array_dtype_string)
        echoes.dtype.names = ('delay', 'amp')

        # Build delay grid 
        tiled = lambda x: np.tile(x, x.shape[0])
        self.width = 2*dist2samples(self.max_dist_from_origin, f_samp, unit='m')
        self.gridx, self.gridy = drop_unifsquare(nodecount, self.width)
        tx, ty = (tiled(self.gridx), tiled(self.gridy))
        self.delay_grid = np.sqrt((tx - tx.T)**2 + (ty - ty.T)**2)

        self.pathloss_grid = self.pathloss_fct(self.delay_grid,
                                               f_samp, f_carr, out_format='dB')


        # Have to use a for loop because reasons
        for k in range(nodecount):
            for l in range(nodecount):
                if k == l:
                    continue
                t0 = self.delay_grid[k,l]
                delay, amp = self.rnd_delay(t0, **kwargs)
                echoes['delay'][k][l] = (delay).astype(INT_DTYPE)
                echoes['amp'][k][l] = amp/db2amp(self.pathloss_grid[k,l])

        self.delay_grid = (self.delay_grid*basephi).astype(INT_DTYPE)

        return echoes['delay'], echoes['amp']


class SyncParams(Struct):
    """Parameter struct containing all the parameters used for the simulation, from the generation of the modulated training sequence to the exponent of the cross-correlation"""

    def __init__(self):
        self.plen = 101 # Note: must be odd
        self.rolloff = 0.1
        self.CFO = 0
        self.TO = 0 # Must be an integer
        self.frac_TO = 0 # Ideally, between zero and 1
        self.trans_delay = 0
        self.f_samp = 1 # Sampling rate
        self.f_symb = 1 # Symbol frequency
        self.zc_len = 11 # Training sequence length
        self.train_type = 'chain' # Overlap of ZC or sequence of ZC
        self.repeat = 1 # Number of ZC sequence repeats
        self.spacing_factor = 2 
        self.power_weight = 10 
        self.full_sim = True
        self.pulse_type = 'raisedcosine'
        self.init_update = False
        self.init_basewidth = False
        self.bias_removal = False
        self.crosscorr_fct = 'analog'
        self.central_padding = 0 # As a fraction of zpos length
        self.ma_window = 1

    def build_training_sequence(self):
        """Builds training sequence from current parameters"""

        zpos = zadoff(1,self.zc_len)
        zneg = zpos.conjugate()

        # A chain of ZC
        if self.train_type == 'chain':
            zeros_count = round(self.zc_len*self.central_padding)
            if zeros_count % 2 == 0 and zeros_count > 2: zeros_count -= 1

            training_seq = np.concatenate(tuple([zneg]*self.repeat+[np.zeros(zeros_count)]+[zpos]*self.repeat))

        # A perfect overlap of ZC (essentially cos(x^2)
        elif self.train_type == 'overlap':
            training_seq = cosine_zadoff_overlap(1, self.zc_len)

        # A single ZC
        elif self.train_type == 'single':
            training_seq = zpos.copy()

        # Wrong train_type
        else:
            raise ValueError('Invalid training sequence type: ' + str(self.train_type))


        self.add(zpos=zpos)
        self.add(training_seq=training_seq)

    def calc_base_barywidth(self):
        """Calculates the barycenter width of the given parameters"""
        """ASSUMPTION: spacing_factor = 2"""
        if not self.init_update:
            raise AttributeError("Must execute p.update() before passing the SyncParams class to this function")
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

    def build_pulse(self):
        """Builds pulse from current parameters"""
        rcos_args = (self.plen, self.rolloff, 1/self.f_symb, self.f_samp, ) 
        rcos_kwargs = {'frac_TO':self.frac_TO}
        if self.pulse_type == 'raisedcosine':
            time, pulse = rcosfilter(*rcos_args, **rcos_kwargs)
        elif self.pulse_type == 'rootraisedcosine':
            time, pulse = rrcosfilter(*rcos_args, **rcos_kwargs)
        else:
            raise ValueError('The "' + self.pulse_type + '" pulse type is unknown')

        self.add(pulse_times=time)
        self.add(pulse=pulse)

    def build_analog_sig(self):
        """Pulses-shapes the training sequence. Must run build_pulse and build_training_sequence first"""
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
        
            #time_arr = (np.arange(len(analog_sig))+np.random.rand()*1000*len(analog_sig))*T
            time_arr = (np.arange(len(analog_sig)))*T
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

    def update(self):
        """Updates dependent variables with current variables"""
        self.build_pulse()
        self.build_training_sequence()

        tmp = self.f_samp/self.f_symb
        if not float(tmp).is_integer():
            raise ValueError('The ratio between the symbol period and sampling period must be an integer')
        self.add(spacing=self.spacing_factor*round(int(tmp)))


        # Find bias removal threshold if needed
        # This will set self.bias_removal to the ratio of the height of the chirp inverted
        # sequence to the peak in the chirp-like sequence.
        # For cyclical crosscorrelations, this would be sqrt(N)/N
        if self.bias_removal == True:
            tmp_full_sim = self.full_sim
            self.full_sim = False

            self.build_analog_sig()
            self.bias_removal = False
            _, _, cpos, _ = calc_both_barycenters(self)
            N = len(cpos)
            if self.train_type == 'chain':
                max1 = np.max(cpos[math.ceil(N/2):])
                max2 = np.max(cpos[:math.floor(N/2)])
            if self.train_type == 'single':
                max1 = np.max(cpos)
                max2 = np.max(cpos[:math.floor(2*N/3)])
            

            self.bias_removal = max2/max1

            #Cleanup if needed
            if tmp_full_sim:
                self.full_sim = True
        # If no bias removal, or already computed, just build  
        else:
            self.build_analog_sig()



        # Done updating
        self.init_update = True

    def change(self, var, val):
        setattr(self, var, val)
        self.update()





