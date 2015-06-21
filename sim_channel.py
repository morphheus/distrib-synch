#!/usr/bin/env python

# My modules
import lib

# Python modules
import numpy as np
import matplotlib.pyplot as plt
import collections
import bisect
import math # import ALL THE MATH

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
clkcount = 2
frameunit = 100 # Unit of time. A normalized period of 1 is frameunit frames.
chansize = frameunit*10
tensor = np.ones(clkcount**2).reshape(clkcount,-1)
pulse = np.array([0,1,0],dtype='float64')
phi_bounds = [0.9,1.1]


# local decs
np.fill_diagonal(tensor,0)

global queue_frame, queue_clk
queue_frame = []
queue_clk = []

pulse_len = len(pulse)
offset = int((pulse_len-1)/2)

channels = np.zeros(clkcount*chansize).reshape(clkcount,-1)
max_frame = chansize-offset;


emit = np.array([True]*clkcount)

# CLock decs
actual_bounds = [round(x*frameunit) for x in phi_bounds]
phi = np.random.randint(actual_bounds[0],actual_bounds[1]+1, size=clkcount)
theta = np.random.randint(frameunit, size=clkcount)

# First events happen on initial phase shift
for clknum, frame in enumerate(theta):
    ordered_insert(frame+offset,clknum) # the + offset is to prevent accessing negative frames



#-------------
# Exceptions
if not len(pulse) % 2:
    raise ValueError('Length of pulse must be odd')

if len(pulse) > frameunit:
    raise ValueError('Pulse is longer than a frame. Bad stuff will happen')


#-------------
# Release unused:
del frame, clknum, actual_bounds

####################
# MAIN SIMULATION
####################


curframe = queue_frame.pop(0)
curclk = queue_clk.pop(0)
print(phi)
print(theta)

while curframe < max_frame:

    
    # Emit phase
    if emit[curclk]:
        # Actually emit
        spread = range(curframe-offset,curframe+offset+1)
       
        for k in range(clkcount):
            #print(str(curclk) + ' ' + str(k) + ' ' + str( pulse*tensor[curclk,k]))
            channels[k,spread] += pulse*tensor[curclk,k]

        # Set next event
        ordered_insert(math.floor(phi[curclk]/2)+curframe, curclk)
        
        emit[curclk] = False

        
    # Adjust phase
    else:
        #lag = I
        # Set next event
        ordered_insert(math.ceil(phi[curclk]/2)+curframe, curclk)
        emit[curclk] = True
    
    # Fetch next event/clock
    curframe = queue_frame.pop(0)
    curclk = queue_clk.pop(0)





plt.plot(np.sum(channels,axis=0))
plt.show()
# Create channel array

# Create clock parameters array

# CHECK MATLAB SCRIPT! But use the same idea of the scheduler implemented.

"""
Emit vs Acquire: frame round DOWN after emit and frame round UP after acquire.


OPTIMIZAION

1. Make scheduler a single list of tuples. In fact, consider a different data strucutre for schedule

2. The tensor is iterated through. Try to make 3d tensor to save on that iteration (will increase
memory usage)

3. Skip emitting in your own channel times zero. (very small increase)

"""
