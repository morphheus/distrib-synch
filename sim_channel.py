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
clkcount = 10
frameunit = 100 # Unit of time. A normalized period of 1 is frameunit frames.
chansize = frameunit*25
topo_tensor = np.ones(clkcount**2).reshape(clkcount,-1)

pulse = np.array([0,0,1,0,0])
#pulse = zadoff(1,11)


phi_bounds = [1,1]
self_interference = 0 # When a clock emits, how much of it goes into it's own channel?


# Modifications to the topology tensor
np.fill_diagonal(topo_tensor,self_interference)
topo_tensor = topo_tensor + 0+0j # Making Topo_Tensor a complex array

# Simulation decs
global queue_frame, queue_clk
queue_frame = []
queue_clk = []

pulse_len = len(pulse)
offset = int((pulse_len-1)/2)

channels = cplx_gaussian( [clkcount,chansize],0) # CHANNEL MUST BE INITIALIZED AS A COMPLEX NUMBER!
max_frame = chansize-offset;


# CLock decs
actual_bounds = [round(x*frameunit) for x in phi_bounds]
phi = np.random.randint(actual_bounds[0],actual_bounds[1]+1, size=clkcount)
theta = np.random.randint(frameunit, size=clkcount)
wait_til_adjust = np.zeros(clkcount, dtype='int64')
wait_til_emit = np.zeros(clkcount, dtype='int64')
emit = np.array([True]*clkcount)

# First events happen on initial phase shift
for clknum, frame in enumerate(theta):
    ordered_insert(frame+actual_bounds[1],clknum) # the + offset is to prevent accessing negative frames



#-------------
# Exceptions
if not len(pulse) % 2:
    raise ValueError('Length of pulse must be odd')

if len(pulse) > frameunit:
    raise ValueError('Pulse is longer than a frame. Bad stuff will happen')


#-------------
# Release unused variables:
del frame, clknum, actual_bounds





####################
# MAIN SIMULATION
####################


curframe = queue_frame.pop(0)
curclk = queue_clk.pop(0)

print("Initial phi: " + str(phi))
print("Initial theta: " + str(theta))

# These two must sum to 1!
emit_frac = 1/2;
adjust_frac = 1-emit_frac;



while curframe < max_frame:

    # ----------------
    # Emit phase
    if emit[curclk]:
        # Actually emit
        spread = range(curframe-offset,curframe+offset+1)
       
        for k in range(clkcount):
            channels[k,spread] += pulse*topo_tensor[curclk,k]

        # Set next event
        wait_til_adjust[curclk] = math.floor(phi[curclk]*emit_frac)
        ordered_insert(wait_til_adjust[curclk]+curframe, curclk)
        
        emit[curclk] = False





    # ----------------
    # Adjust phase
    else:
        """Assumption: symmetric pulse!"""
        # ------
        # Barycenter calculation
        winmax = curframe
        winmin = winmax-(phi[curclk]+pulse_len-1)
        winlen = winmax-winmin
        
        barycenter_range = range(winmin, winmax)
        lag,_ = barycenter_correlation(pulse, channels[curclk, barycenter_range], power_weight=2)
        lag = int(lag)
        if lag == -1:
            lag = 0
        lag += -1*(winlen-pulse_len+1) + wait_til_adjust[curclk]


        # --------
        # Correction algorithm
        
        correction = lag
        theta[curclk] += correction
        theta[curclk] = theta[curclk] % phi[curclk]
        
        # --------
        # Set next event
        wait_til_emit[curclk] = math.ceil(phi[curclk]*adjust_frac)+curframe+correction
        if wait_til_emit[curclk] < 1:
            wait_til_emit[curclk] = 1
        
        ordered_insert(wait_til_emit[curclk], curclk)
        emit[curclk] = True


    # ----------------
    # Fetch next event/clock
    curframe = queue_frame.pop(0)
    curclk = queue_clk.pop(0)





print(theta)
plt.plot(np.sum(abs(channels),axis=0)/(clkcount-1))
plt.show()



"""
Emit vs Acquire: frame round DOWN after emit and frame round UP after acquire.


OPTIMIZAION

1. Make scheduler a single list of tuples. In fact, consider a different data strucutre for schedule

2. The topo_tensor is iterated through. Try to make 3d topo_tensor to save on that iteration (will increase
memory usage)

3. Skip emitting in your own channel times zero. (very small increase)

4. Fold in "ordered insert" in the while loop to save on the function call.

"""
