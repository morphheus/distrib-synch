#!/usr/bin/env python

# My modules
from lib import *

# Python modules
import numpy as np
import matplotlib.pyplot as plt
import collections
import bisect
import math # import ALL THE MATH
import warnings

from numpy import pi
from pprint import pprint


def ordered_insert(frame, clknum):
    """Insert from the left in descending order, in a list"""
    global queue_frame, queue_clk

    idx = bisect.bisect(queue_frame, frame)
    queue_frame.insert(idx, frame)
    queue_clk.insert(idx, clknum)

###################
# VARS
###################

# Input variables
clkcount = 6
frameunit = 10000 # Unit of time. A normalized period of 1 is frameunit frames.
chansize = frameunit*50
topo_matrix = np.ones(clkcount**2).reshape(clkcount,-1)
noise_power = 1

phi_bounds = [1,1]
self_interference = 0 # When a clock emits, how much of it goes into it's own channel?


p = Params()
p.zc_len = 101
p.plen = 39
p.rolloff = 0.2
p.f_samp = 12
p.f_symb = 3
p.repeat = 1
p.power_weight = 4
p.CFO = 0
p.TO = 0
p.full_sim = True
p.bias_removal = 0


# TODO: link f_samp to frameunits


CFO_step_wait = 20

p.update()
p.calc_base_barywidth()

# Convert the baryslope from width/f_symb to width/frame
#frame_bslope = p.baryslope*p.f_symb/frameunit


#plt.plot(np.real(params.pulse))
#plt.show()


analog_pulse = p.analog_sig

# Simulation decs
global queue_frame, queue_clk
queue_frame = []
queue_clk = []

pulse_len = len(analog_pulse)
offset = int((pulse_len-1)/2)

channels = cplx_gaussian( [clkcount,chansize],noise_power) # CHANNEL MUST BE INITIALIZED AS A COMPLEX NUMBER!
max_frame = chansize-offset;


# Clock decs
phi_minmax = [round(x*frameunit) for x in phi_bounds]
deltaf_minmax = [-0.05,0.05]*p.f_symb
phi = np.random.randint(phi_minmax[0],phi_minmax[1]+1, size=clkcount)
theta = np.random.randint(frameunit, size=clkcount)
deltaf = np.random.uniform(deltaf_minmax[0],deltaf_minmax[1], size=clkcount) # In units of f_symb
clk_creation = np.random.randint(0,chansize, size=clkcount)
do_CFO_correction = np.zeros(clkcount)



# TESTING VALUES
#theta = np.arange(clkcount)/(clkcount+1) * frameunit
#theta = np.zeros(clkcount) + 0.5
#theta = theta.astype(int)
#deltaf = (np.arange(clkcount)**2 - clkcount/2)/clkcount**2 * deltaf_minmax[1]

wait_til_adjust = np.zeros(clkcount, dtype='int64')
wait_til_emit = np.zeros(clkcount, dtype='int64')
emit = np.array([True]*clkcount)

# First events happen on initial phase shift
for clknum, frame in enumerate(theta):
    ordered_insert(frame+phi_minmax[1],clknum) # the + offset is to prevent accessing negative frames


# Modifications to the topology matrix
np.fill_diagonal(topo_matrix,self_interference)
topo_matrix = topo_matrix + 0+0j # Making Topo_Matrix a complex array
deltaf_matrix = np.empty(clkcount**2, dtype='int64').reshape(clkcount,-1) + 0j




#-------------
# Exceptions
#if not len(analog_pulse) % 2:
#    raise ValueError('Length of pulse must be odd')

if len(analog_pulse) > frameunit:
    raise ValueError('Pulse is longer than a frame. Bad stuff will happen')


#-------------
# Release unused variables:
del frame, clknum, phi_minmax





####################
# MAIN SIMULATION
####################

print('Theta std init: ' + str(np.std(theta)))
print('deltaf std init: ' + str(np.std(deltaf)) + '    spread: '+ str(max(deltaf) -min(deltaf))+ '\n')


curframe = queue_frame.pop(0)
curclk = queue_clk.pop(0)

#print("Initial phi: " + str(phi))
#print("Initial theta: " + str(theta))

# These two must sum to 1!
emit_frac = 1/2;
adjust_frac = 1-emit_frac;



while curframe < max_frame:

    # ----------------
    # Emit phase
    if emit[curclk]:
        
        minframe = curframe-offset
        maxframe = curframe+offset+1
        spread = range(minframe,maxframe)
       
        deltaf_arr = np.empty(pulse_len)
        for k in range(clkcount):
            time_arr = (np.arange(minframe,maxframe) + clk_creation[curclk] )/p.f_samp 
            deltaf_arr = np.exp( 2*pi*1j* (deltaf[curclk] - deltaf[k])  *( time_arr))
            channels[k,spread] += analog_pulse*topo_matrix[curclk,k]*deltaf_arr

        # Set next event
        wait_til_adjust[curclk] = math.floor(phi[curclk]*adjust_frac)
        ordered_insert(wait_til_adjust[curclk]+curframe, curclk)
        
        emit[curclk] = False





    # ----------------
    # Adjust phase
    else:
        """Assumptions: odd-length and symmetric pulse!"""
        # ------
        # Barycenter calculation
        winmax = curframe
        winmin = winmax-(phi[curclk]+pulse_len-1)
        winlen = winmax-winmin

        #TO DO: fix adjust window to be up to last adjust!
        
        barycenter_range = range(winmin, winmax)
        barypos, baryneg, corpos, corneg = calc_both_barycenters(p, channels[curclk,barycenter_range])
    

        
        # -------
        # TO and CFO calculation
        TO = int(round((barypos+baryneg)/2))
        TO += -1*(winlen) + wait_til_adjust[curclk] + offset # adjust with respect to past pulse
        
        CFO = (barypos-baryneg - p.basewidth) / (p.baryslope)
        

        # --------
        # TO and CFO correction
        TO_correction = TO/2
        theta[curclk] += TO_correction
        theta[curclk] = theta[curclk] % phi[curclk]

        if do_CFO_correction[curclk] > CFO_step_wait:
            CFO_correction = CFO/2
            deltaf[curclk] += CFO_correction/p.f_symb
        else:
            do_CFO_correction[curclk] += 1
            #print(theta)
            
            
        
        # --------
        # Set next event
        wait_til_emit[curclk] = math.ceil(phi[curclk]*emit_frac)+curframe+TO_correction
        if wait_til_emit[curclk] < 1:
            wait_til_emit[curclk] = 1
        
        ordered_insert(wait_til_emit[curclk], curclk)
        emit[curclk] = True


    # ----------------
    # Fetch next event/clock
    curframe = queue_frame.pop(0)
    curclk = queue_clk.pop(0)




print('theta STD: ' + str(np.std(theta)))
#print(theta)

print('deltaf STD: ' + str(np.std(deltaf)) + '    spread: ' + str(max(deltaf)-min(deltaf)))
#plt.plot(np.sum(abs(channels),axis=0)/(clkcount-1))
#plt.show()



"""
Emit vs Acquire: frame round DOWN after emit and frame round UP after acquire.


OPTIMIZAION

1. Make scheduler a single list of tuples. In fact, consider a different data strucutre for schedule

2. The topo_matrix is iterated through. Try to make 3d topo_matrix to save on that iteration (will increase
memory usage)

3. Skip emitting in your own channel times zero. (very small increase)

4. Fold in "ordered insert" in the while loop to save on the function call.

"""
