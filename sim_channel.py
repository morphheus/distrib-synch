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



#class SimControlParams(Struct):
#    def __init__(self):
#        self.add(clkcount=7)
#        self.add(frameunit=1000) # Unit of time. A normalized period of 1 is frameunit frames.
#        self.add(chansize=self.frameunit*50)
#        self.add(topo_matrix=np.ones(self.clkcount**2).reshape(self.clkcount,-1))
#        self.add(noise_power=0)
#
#        self.add(phi_bounds=[1,1])
#        self.add(self_interference=0) # When a clock emits, how much of it goes into it's own #channel?
#        self.add(CFO_step_wait=60)
#        self.add(rand_init=False)
#        self.add(display=True)
#        self.add(keep_intermediate_values=True)


def default_ctrl_dict():
    out = {}
    out['clkcount'] = 7
    out['frameunit'] = 1000
    out['chansize'] = out['frameunit']*50
    out['topo_matrix'] = np.ones(out['clkcount']**2).reshape(out['clkcount'],-1)
    out['noise_power'] = 0
    out['phi_bounds'] = [1,1]
    out['self_interference'] = 0
    out['CFO_step_wait'] = 60
    out['rand_init'] = False
    out['display'] = True
    out['keep_intermediate_values'] = True
    return out






##############################
def runsim(p,ctrl):
    """Executes a simulation with the signal parameters p and controls parameters ctrl"""

    # INPUT EXCEPTIONS
    if not p.init_update or not p.init_basewidth:
        raise AttributeError("Need to run p.update() and p.calc_base_barywidth before calling runsim()")
    if len(analog_pulse) > frameunit:
        raise ValueError('Pulse is longer than a frame. Bad stuff will happen')

    
    # Load local variables. This is done to reduce the amount of crap in the main function
    #clkcount = ctrl.clkcount
    #frameunit = ctrl.frameunit
    #chansize = ctrl.chansize
    #topo_matrix = ctrl.topo_matrix
    #noise_power = ctrl.noise_power
    #phi_bounds = ctrl.phi_bounds
    #self_interference = ctrl.self_interference
    #CFO_step_wait = ctrl.CFO_step_wait

    clkcount = ctrl['clkcount']
    frameunit = ctrl['frameunit']
    chansize = ctrl['chansize']
    topo_matrix = ctrl['topo_matrix']
    noise_power = ctrl['noise_power']
    phi_bounds = ctrl['phi_bounds']
    self_interference = ctrl['self_interference']
    CFO_step_wait = ctrl['CFO_step_wait']

    analog_pulse = p.analog_sig


    

    #----------------------
    # VARIABLE DECLARATIONS
    #----------------------

    global queue_frame, queue_clk
    queue_frame = []
    queue_clk = []
    pulse_len = len(analog_pulse)
    offset = int((pulse_len-1)/2)
    channels = cplx_gaussian( [clkcount,chansize],noise_power) 
    max_frame = chansize-offset;
    wait_til_adjust = np.zeros(clkcount, dtype='int64')
    wait_til_emit = np.zeros(clkcount, dtype='int64')
    emit = np.array([True]*clkcount)

    if ctrl['keep_intermediate_values']:
        theta_inter = [[] for k in range(clkcount)]
        phi_inter = [[] for k in range(clkcount)]
        deltaf_inter = [[] for k in range(clkcount)]


    
    # Clock initial values
    phi_minmax = [round(x*frameunit) for x in phi_bounds]
    deltaf_minmax = [-0.05,0.05]*p.f_symb
    do_CFO_correction = np.zeros(clkcount)

    if ctrl['rand_init']:
        phi = np.random.randint(phi_minmax[0],phi_minmax[1]+1, size=clkcount)
        theta = np.random.randint(frameunit, size=clkcount)
        deltaf = np.random.uniform(deltaf_minmax[0],deltaf_minmax[1], size=clkcount)
        clk_creation = np.random.randint(0,chansize, size=clkcount)

    else:
        phi = np.random.randint(phi_minmax[0],phi_minmax[1]+1, size=clkcount)
        theta = np.round(np.arange(clkcount)/(clkcount+1) * frameunit)
        theta = theta.astype(int)
        deltaf = (np.arange(clkcount)**2 - clkcount/2)/clkcount**2 * deltaf_minmax[1]
        clk_creation = np.zeros(clkcount)

    
    # Modifications to the topology matrix
    np.fill_diagonal(topo_matrix,self_interference)
    topo_matrix = topo_matrix + 0+0j # Making Topo_Matrix a complex array
    deltaf_matrix = np.empty(clkcount**2, dtype='int64').reshape(clkcount,-1) + 0j

    # First events happen on initial phase shift
    for clknum, frame in enumerate(theta):
        ordered_insert(frame+phi_minmax[1],clknum) # the + offset is to prevent accessing negative frames

    # Release unused variables:
    del frame, clknum, phi_minmax





    ####################
    # MAIN SIMULATION
    ####################

    if ctrl['display']:
        print('Theta std init: ' + str(np.std(theta)))
        print('deltaf std init: ' + str(np.std(deltaf)) + '    spread: '+ str(max(deltaf) -min(deltaf))+ '\n')

    if ctrl['keep_intermediate_values']:
        for k in range(clkcount):
            theta_inter[k].append((theta[k],theta[k]))
            phi_inter[k].append((phi[k],phi[k]))
            deltaf_inter[k].append((deltaf[k],deltaf[k]))


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

            if ctrl['keep_intermediate_values']:
                theta_inter[curclk].append((curframe,theta[curclk]))
                phi_inter[curclk].append((curframe,phi[curclk]))
                deltaf_inter[curclk].append((curframe,deltaf[curclk]))



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





    #---------------
    # Post-sim wrap up
    #---------------



    if ctrl['display']:
        print('theta STD: ' + str(np.std(theta)))
        print('deltaf STD: ' + str(np.std(deltaf)) + '    spread: ' + str(max(deltaf)-min(deltaf)))


    # Add all calculated values with the controls parameter structure
    ctrl['theta'] = theta
    ctrl['deltaf'] = deltaf
    ctrl['phi'] = phi

    if ctrl['keep_intermediate_values']:
        ctrl['theta_inter'] = theta_inter
        ctrl['deltaf_inter'] = deltaf_inter
        ctrl['phi_inter'] = phi_inter
    
    



    """
    Emit vs Acquire: frame round DOWN after emit and frame round UP after acquire.


    OPTIMIZAION

    1. Make scheduler a single list of tuples. In fact, consider a different data strucutre for schedule

    2. The topo_matrix is iterated through. Try to make 3d topo_matrix to save on that iteration (will increase
    memory usage)

    3. Skip emitting in your own channel times zero. (very small increase)

    4. Fold in "ordered insert" in the while loop to save on the function call.

    """
