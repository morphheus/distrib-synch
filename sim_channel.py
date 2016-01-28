#!/usr/bin/env python

# User modules
import lib
from lib import calc_both_barycenters, Params

# Python modules
import numpy as np
import matplotlib.pyplot as plt
import collections
import bisect
import math # import ALL THE MATH
import warnings

from numpy import pi
from pprint import pprint


#-------------------------
def ordered_insert(frame, clknum):
    """Insert from the left in descending order, in a list"""
    global queue_frame, queue_clk

    idx = bisect.bisect(queue_frame, frame)
    queue_frame.insert(idx, frame)
    queue_clk.insert(idx, clknum)





#-------------------------
def default_ctrl_dict():
    out = {}
    out['clkcount'] = 7
    out['frameunit'] = 1000
    out['chansize'] = out['frameunit']*50
    out['noise_std'] = 0
    out['phi_bounds'] = [1,1]
    out['self_emit'] = False # IF set to False, the self-emit will just be skipped.
    out['CFO_step_wait'] = 60
    out['rand_init'] = False
    out['display'] = True
    out['keep_intermediate_values'] = False
    out['saveall'] = False # This options also saves all fields in Params to the control dict
    out['cfo_mapper_fct'] = lib.cfo_mapper_linear
    out['cfo_bias'] = 0 # in terms of f_samp
    out['delay_fct'] = lib.delay_pdf_static
    out['deltaf_bound'] = 0.02 # in units of f_samp
    out['CFO_processing_avgtype'] = 'mov_avg' # 'mov_avg' or 'reg' (non-mov avg)
    out['CFO_processing_avgwindow'] = 5


    # Echo controls
    out['max_echo_taps'] = 4
    out['min_delay'] = 0

    # Correction controls
    out['epsilon_TO'] = 0.5
    out['epsilon_CFO'] = 0.25
    out['max_CFO_correction'] = 0.02 # As a factor of f_symb

    return out






#-------------------------
def runsim(p,ctrl):
    """Executes a simulation with the signal parameters p and controls parameters ctrl"""


    #----------------
    # INPUTS
    #----------------
    
    # Put important ctrl values in local namespace for faster access time
    clkcount = ctrl['clkcount']
    frameunit = ctrl['frameunit']
    chansize = ctrl['chansize']
    noise_std = ctrl['noise_std']
    phi_bounds = ctrl['phi_bounds']
    self_emit = ctrl['self_emit']
    CFO_step_wait = ctrl['CFO_step_wait']
    epsilon_TO = ctrl['epsilon_TO']
    epsilon_CFO = ctrl['epsilon_CFO']
    cfo_mapper_fct = ctrl['cfo_mapper_fct']
    CFO_processing_avgtype = ctrl['CFO_processing_avgtype']
    CFO_processing_avgwindow = ctrl['CFO_processing_avgwindow']

    # IF echoes specified, to shove in array. OW, just don't worry about it
    do_echoes = True
    try:
        echo_delay = ctrl['echo_delay']
        echo_amp = ctrl['echo_amp']
        max_echo_taps = ctrl['max_echo_taps']
    except KeyError:
        max_echo_taps = 1
        ctrl['max_taps'] = max_echo_taps
        echo_delay = np.zeros((clkcount,clkcount,1), dtype='i8')
        echo_amp = np.ones((clkcount,clkcount,1))
        ctrl['echo_delay'] = echo_delay
        ctrl['echo_amp'] = echo_amp
    
    analog_pulse = p.analog_sig

    # INPUT EXCEPTIONS
    if not p.init_update or not p.init_basewidth:
        raise AttributeError("Need to run p.update() and p.calc_base_barywidth before calling runsim()")
    if len(analog_pulse) > frameunit:
        raise ValueError('Pulse is longer than a frame. Bad stuff will happen')


    #----------------------
    # VARIABLE DECLARATIONS
    #----------------------


    global queue_frame, queue_clk
    queue_frame = []
    queue_clk = []
    pulse_len = len(analog_pulse)
    offset = int((pulse_len-1)/2)
    channels = lib.cplx_gaussian( [clkcount,chansize],noise_std) 
    max_frame = chansize-offset-np.max(echo_delay);
    wait_til_adjust = np.zeros(clkcount, dtype='int64')
    wait_til_emit = np.zeros(clkcount, dtype='int64')
    prev_adjustframe = np.zeros(clkcount, dtype='int64')
    emit = np.array([True]*clkcount)

    if ctrl['keep_intermediate_values']:
        frame_inter = [[] for k in range(clkcount)]
        theta_inter = [[] for k in range(clkcount)]
        phi_inter = [[] for k in range(clkcount)]
        deltaf_inter = [[] for k in range(clkcount)]


    
    # Clock initial values
    phi_minmax = [round(x*frameunit) for x in phi_bounds]
    deltaf_minmax = np.array([-1*ctrl['deltaf_bound'],ctrl['deltaf_bound']])*p.f_symb
    do_CFO_correction = np.array([False]*clkcount)
    wait_CFO_correction = np.zeros(clkcount)
    CFO_maxjump_direction = np.ones(clkcount)
    CFO_corr_list = [[] for x in range(clkcount)]
    TO_corr_list = [[] for x in range(clkcount)]

    if ctrl['rand_init']:
        phi = np.random.randint(phi_minmax[0],phi_minmax[1]+1, size=clkcount)
        theta = np.random.randint(frameunit, size=clkcount)
        #theta = np.zeros(clkcount).astype(int) + int(round(frameunit/2))
        deltaf = np.random.uniform(deltaf_minmax[0],deltaf_minmax[1], size=clkcount)
        clk_creation = np.random.randint(0,chansize, size=clkcount)

    else:
        phi = np.random.randint(phi_minmax[0],phi_minmax[1]+1, size=clkcount)
        theta = np.round((np.arange(clkcount)/(2*(clkcount-1))+0.25) * frameunit)
        #theta = np.zeros(clkcount) + round(frameunit/2)
        theta = theta.astype(int)
        #theta = np.zeros(clkcount).astype(int)
        #deltaf = ((np.arange(clkcount)**2 - clkcount/2)/clkcount**2) * deltaf_minmax[1]
        deltaf = (np.arange(clkcount)-clkcount/2)/clkcount * deltaf_minmax[1]
        #deltaf = np.zeros(clkcount)
        clk_creation = np.zeros(clkcount)

    

    # First events happen on initial phase shift
    for clknum, frame in enumerate(theta):
        ordered_insert(frame+phi_minmax[1],clknum) # the + offset is to prevent accessing negative frames

    # Correction algorithms variables
    max_CFO_correction = ctrl['max_CFO_correction']*p.f_symb

    # Release unused variables:
    del frame, clknum



    ####################
    # MAIN SIMULATION
    ####################

    if ctrl['display']:
        print('Theta std init: ' + str(np.std(theta)))
        print('deltaf std init: ' + str(np.std(deltaf)) + '    spread: '+ str(max(deltaf) -min(deltaf))+ '\n')

    if ctrl['keep_intermediate_values']:
        for k in range(clkcount):
            frame_inter[k].append(theta[k])
            theta_inter[k].append(theta[k])
            phi_inter[k].append(phi[k])
            deltaf_inter[k].append(deltaf[k])


    curframe = queue_frame.pop(0)
    curclk = queue_clk.pop(0)


    # These two must sum to 1!
    emit_frac = 1/2;
    adjust_frac = 1-emit_frac;


    # Set the previous adjust frame as 1 period behind
    for k in range(clkcount):
        prev_adjustframe[k] = theta[k] - round(phi[k]*emit_frac) + phi_minmax[1]
        if prev_adjustframe[k] < 0:
            prev_adjustframe[k] = 0





    # Main loop
    while curframe < max_frame:

        # ----------------
        # Emit phase
        if emit[curclk]:
            

            minframe = curframe-offset
            maxframe = curframe+offset+1
            spread = range(minframe,maxframe)

            deltaf_arr = np.empty(pulse_len)
            for emitclk in range(clkcount):
                if self_emit and emitclk == curclk: # Skip selfemission
                    continue
                time_arr = (np.arange(minframe,maxframe) + clk_creation[curclk] )/p.f_samp 
                deltaf_arr = np.exp( 2*pi*1j* (deltaf[curclk] - deltaf[emitclk])  *( time_arr))

                #Echoes management
                for k in range(max_echo_taps):
                    curr_amp = echo_amp[curclk][emitclk][k]
                    if curr_amp != 0:
                        to_emit = analog_pulse*deltaf_arr
                        channels[emitclk, spread + echo_delay[curclk][emitclk][k]] += to_emit*curr_amp

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
            winmin = prev_adjustframe[curclk]
            winlen = winmax-winmin

            prev_adjustframe[curclk] = curframe


            if winlen > pulse_len + 1:
                barycenter_range = range(winmin, winmax)
                barypos, baryneg, corpos, corneg = calc_both_barycenters(p, channels[curclk,barycenter_range])
            else:
                barypos = winlen - wait_til_adjust[curclk]
                baryneg = barypos



            # -------
            # TO and CFO calculation
            TO = int(round((barypos+baryneg)/2))
            TO += -1*winlen + wait_til_adjust[curclk]  # adjust with respect to past pulse

            CFO = cfo_mapper_fct(barypos-baryneg, p)


            # --------
            # TO correction
            TO_correction = round(TO*epsilon_TO)
            theta[curclk] += TO_correction
            theta[curclk] = theta[curclk] % phi[curclk]


            
            # CFO correction clipping
            CFO_correction = CFO*epsilon_CFO
            if CFO_correction > max_CFO_correction:
                CFO_correction = max_CFO_correction
            elif CFO_correction < -1*max_CFO_correction:
                #CFO_maxjump_direction[curclk] *= -1
                #CFO_correction = CFO_maxjump_direction[curclk]*max_CFO_correction
                CFO_correction = -1*max_CFO_correction

            # Median filtering
            #CFO_corr_list[curclk].append(CFO_correction)
            #if len(CFO_corr_list[curclk]) > 3:
            #    CFO_corr_list[curclk].pop(0)
            #deltaf[curclk] += np.median(CFO_corr_list[curclk])


            #CFO_correction += ctrl['cfo_bias']*p.f_symb
            
            if wait_CFO_correction[curclk] <= CFO_step_wait:
                wait_CFO_correction[curclk] += 1
            else:
                do_CFO_correction[curclk] = True


            # CFO correction moving average or regular average
            if do_CFO_correction[curclk]:
                CFO_corr_list[curclk].append(CFO_correction)
                if len(CFO_corr_list[curclk]) >= CFO_processing_avgwindow:
                    CFO_correction = sum(CFO_corr_list[curclk])/CFO_processing_avgwindow
                    CFO_corr_list[curclk] = []
                elif CFO_processing_avgtype == 'reg': # Moving average applies CFO at each step
                    do_CFO_correction[curclk] = False
            
            
            # apply cfo correction if needed
            if do_CFO_correction[curclk]:
                deltaf[curclk] += CFO_correction

            # -------------------
            if ctrl['keep_intermediate_values']:
                frame_inter[curclk].append(curframe)
                theta_inter[curclk].append(theta[curclk])
                phi_inter[curclk].append(phi[curclk])
                deltaf_inter[curclk].append(deltaf[curclk])



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
        ctrl['frame_inter'] = frame_inter
        ctrl['theta_inter'] = theta_inter
        ctrl['deltaf_inter'] = deltaf_inter
        ctrl['phi_inter'] = phi_inter

    if ctrl['saveall']:
        ctrl.update(p.__dict__)
    
    



    """
    Emit vs Acquire: frame round DOWN after emit and frame round UP after acquire.


    OPTIMIZAION

    1. Make scheduler a single list of tuples. In fact, consider a different data strucutre for schedule

    2. The topo_matrix is iterated through. Try to make 3d topo_matrix to save on that iteration

    3. Skip emitting in your own channel times zero. (very small increase)

    4. Fold in "ordered insert" in the while loop to save on the function call.

    """




