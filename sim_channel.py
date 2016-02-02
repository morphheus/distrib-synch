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


class SimControls(lib.Struct):
    """Container object for the control parameters of the runsim() function"""
    def __init__(self):
        """Default values"""
        self.steps = 30
        self.nodecount = 7
        self.basephi = 1000
        self.chansize = self.basephi*50
        self.noise_std = 0
        self.phi_bounds = [1,1]
        self.theta_bounds = [0,1]
        self.self_emit = False # IF set to False, the self-emit will just be skipped.
        self.CFO_step_wait = 60
        self.rand_init = False
        self.display = True # Display progress of runsim
        self.keep_intermediate_values = False
        self.saveall = False # This options also saves all fields in Params to the control dict
        self.cfo_mapper_fct = lib.cfo_mapper_linear
        self.cfo_bias = 0 # in terms of f_samp
        self.delay_fct = lib.delay_pdf_static
        self.deltaf_bound = 0.02 # in units of f_samp
        self.bmap_reach = 3e-1
        self.bmap_scaling = 100
        self.non_rand_seed = 1231231
        # Echo controls
        self.max_echo_taps = 4
        self.min_delay = 0
        # Correction controls
        self.epsilon_TO = 0.5
        self.epsilon_CFO = 0.25
        self.max_CFO_correction = 0.02 # As a factor of f_symb
        self.CFO_processing_avgtype = 'mov_avg' # 'mov_avg' or 'reg' (non-mov avg)
        self.CFO_processing_avgwindow = 5

        self.init_update = True

    def update(self):
        """Must be run before runsim can be executed"""
        self.chansize = int(self.basephi*self.steps)
        self.phi_minmax = [round(x*self.basephi) for x in self.phi_bounds]
        self.theta_minmax = [round(x*self.basephi) for x in self.theta_bounds]
        self.init_update = True

#-------------------------
def ordered_insert(sample, clknum):
    """Insert from the left in descending order, in a list"""
    global queue_sample, queue_clk

    idx = bisect.bisect(queue_sample, sample)
    queue_sample.insert(idx, sample)
    queue_clk.insert(idx, clknum)


#-------------------------
def default_ctrl_dict():
    out = {}
    out['nodecount'] = 7
    out['basephi'] = 1000
    out['chansize'] = out['basephi']*50
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

    
    # Put important ctrl values in local namespace ease of writing. These values shound not
    # change under any circumstances
    nodecount = ctrl.nodecount
    basephi = ctrl.basephi
    chansize = ctrl.chansize
    noise_std = ctrl.noise_std
    self_emit = ctrl.self_emit
    CFO_step_wait = ctrl.CFO_step_wait
    epsilon_TO = ctrl.epsilon_TO
    epsilon_CFO = ctrl.epsilon_CFO
    cfo_mapper_fct = ctrl.cfo_mapper_fct
    CFO_processing_avgtype = ctrl.CFO_processing_avgtype
    CFO_processing_avgwindow = ctrl.CFO_processing_avgwindow
    phi_minmax = ctrl.phi_minmax
    theta_minmax = ctrl.theta_minmax

    # IF echoes specified, to shove in array. OW, just don't worry about it
    try:
        echo_delay = ctrl.echo_delay
        echo_amp = ctrl.echo_amp
        max_echo_taps = ctrl.max_echo_taps
    except KeyError:
        max_echo_taps = 1
        ctrl.max_taps = max_echo_taps
        echo_delay = np.zeros((nodecount,nodecount,1), dtype='i8')
        echo_amp = np.ones((nodecount,nodecount,1))
        ctrl.echo_delay = echo_delay
        ctrl.echo_amp = echo_amp
    
    analog_pulse = p.analog_sig

    # INPUT EXCEPTIONS
    if not p.init_update or not p.init_basewidth:
        raise AttributeError("Need to run p.update() and p.calc_base_barywidth before calling runsim()")
    if not ctrl.init_update:
        raise AttributeError("Need to run ctrl.update() before calling runsim()")
    if len(analog_pulse) > basephi:
        raise ValueError('Pulse is longer than a sample. Bad stuff will happen')


    #----------------------
    # VARIABLE DECLARATIONS
    #----------------------


    global queue_sample, queue_clk
    queue_sample = []
    queue_clk = []
    pulse_len = len(analog_pulse)
    offset = int((pulse_len-1)/2)
    max_sample = chansize-offset-np.max(echo_delay);
    wait_til_adjust = np.zeros(nodecount, dtype='int64')
    wait_til_emit = np.zeros(nodecount, dtype='int64')
    prev_adjustsample = np.zeros(nodecount, dtype='int64')
    emit = np.array([True]*nodecount)

    if ctrl.keep_intermediate_values:
        sample_inter = [[] for k in range(nodecount)]
        theta_inter = [[] for k in range(nodecount)]
        phi_inter = [[] for k in range(nodecount)]
        deltaf_inter = [[] for k in range(nodecount)]


    
    # Node initial values
    deltaf_minmax = np.array([-1*ctrl.deltaf_bound,ctrl.deltaf_bound])*p.f_symb
    do_CFO_correction = np.array([False]*nodecount)
    wait_CFO_correction = np.zeros(nodecount)
    CFO_maxjump_direction = np.ones(nodecount)
    CFO_corr_list = [[] for x in range(nodecount)]
    TO_corr_list = [[] for x in range(nodecount)]

    if not ctrl.rand_init:
        np.random.seed(ctrl.non_rand_seed)
    
    phi = np.random.randint(phi_minmax[0],phi_minmax[1]+1, size=nodecount)
    theta = np.random.randint(theta_minmax[0],theta_minmax[1]+1, size=nodecount)
    deltaf = np.random.uniform(deltaf_minmax[0],deltaf_minmax[1], size=nodecount)
    clk_creation = np.random.randint(0,chansize, size=nodecount)
    channels = lib.cplx_gaussian( [nodecount,chansize],noise_std) 

    if not ctrl.rand_init:
        np.random.seed()



    # First events happen on initial phase shift
    for clknum, sample in enumerate(theta):
        ordered_insert(sample+phi_minmax[1],clknum) # the + offset is to prevent accessing negative samples

    # Correction algorithms variables
    max_CFO_correction = ctrl.max_CFO_correction*p.f_symb

    # Release unused variables:
    del sample, clknum



    ####################
    # MAIN SIMULATION
    ####################

    if ctrl.display:
        print('Theta std init: ' + str(np.std(theta)))
        print('deltaf std init: ' + str(np.std(deltaf)) + '    spread: '+ str(max(deltaf) -min(deltaf))+ '\n')

    if ctrl.keep_intermediate_values:
        for k in range(nodecount):
            sample_inter[k].append(theta[k])
            theta_inter[k].append(theta[k])
            phi_inter[k].append(phi[k])
            deltaf_inter[k].append(deltaf[k])


    cursample = queue_sample.pop(0)
    curnode = queue_clk.pop(0)


    # These two must sum to 1!
    emit_frac = 1/2;
    adjust_frac = 1-emit_frac;


    # Set the previous adjust sample as 1 period behind
    for k in range(nodecount):
        prev_adjustsample[k] = theta[k] - round(phi[k]*emit_frac) + phi_minmax[1]
        if prev_adjustsample[k] < 0:
            prev_adjustsample[k] = 0





    # Main loop
    while cursample < max_sample:

        # ----------------
        # Emit phase
        if emit[curnode]:
            

            minsample = cursample-offset
            maxsample = cursample+offset+1
            spread = range(minsample,maxsample)

            deltaf_arr = np.empty(pulse_len)
            for emitclk in range(nodecount):
                if self_emit and emitclk == curnode: # Skip selfemission
                    continue
                time_arr = (np.arange(minsample,maxsample) + clk_creation[curnode] )/p.f_samp 
                deltaf_arr = np.exp( 2*pi*1j* (deltaf[curnode] - deltaf[emitclk])  *( time_arr))

                #Echoes management
                for k in range(max_echo_taps):
                    curr_amp = echo_amp[curnode][emitclk][k]
                    if curr_amp != 0:
                        to_emit = analog_pulse*deltaf_arr
                        channels[emitclk, spread + echo_delay[curnode][emitclk][k]] += to_emit*curr_amp

            # Set next event
            wait_til_adjust[curnode] = math.floor(phi[curnode]*adjust_frac)
            ordered_insert(wait_til_adjust[curnode]+cursample, curnode)

            emit[curnode] = False





        # ----------------
        # Adjust phase
        else:
            """Assumptions: odd-length and symmetric pulse!"""
            # ------
            # Barycenter calculation
            winmax = cursample
            winmin = prev_adjustsample[curnode]
            winlen = winmax-winmin

            prev_adjustsample[curnode] = cursample


            if winlen > pulse_len + 1:
                barycenter_range = range(winmin, winmax)
                barypos, baryneg, corpos, corneg = calc_both_barycenters(p, channels[curnode,barycenter_range])
            else:
                barypos = winlen - wait_til_adjust[curnode]
                baryneg = barypos



            # -------
            # TO and CFO calculation
            TO = int(round((barypos+baryneg)/2))
            TO += -1*winlen + wait_til_adjust[curnode]  # adjust with respect to past pulse

            CFO = cfo_mapper_fct(barypos-baryneg, p)


            # --------
            # TO correction
            TO_correction = round(TO*epsilon_TO)
            theta[curnode] += TO_correction
            theta[curnode] = theta[curnode] % phi[curnode]


            
            # CFO correction clipping
            CFO_correction = CFO*epsilon_CFO
            if CFO_correction > max_CFO_correction:
                CFO_correction = max_CFO_correction
            elif CFO_correction < -1*max_CFO_correction:
                #CFO_maxjump_direction[curnode] *= -1
                #CFO_correction = CFO_maxjump_direction[curnode]*max_CFO_correction
                CFO_correction = -1*max_CFO_correction

            # Median filtering
            #CFO_corr_list[curnode].append(CFO_correction)
            #if len(CFO_corr_list[curnode]) > 3:
            #    CFO_corr_list[curnode].pop(0)
            #deltaf[curnode] += np.median(CFO_corr_list[curnode])


            #CFO_correction += ctrl.cfo_bias*p.f_symb
            
            if wait_CFO_correction[curnode] <= CFO_step_wait:
                wait_CFO_correction[curnode] += 1
            else:
                do_CFO_correction[curnode] = True


            # CFO correction moving average or regular average
            if do_CFO_correction[curnode]:
                CFO_corr_list[curnode].append(CFO_correction)
                if len(CFO_corr_list[curnode]) >= CFO_processing_avgwindow:
                    CFO_correction = sum(CFO_corr_list[curnode])/CFO_processing_avgwindow
                    CFO_corr_list[curnode] = []
                elif CFO_processing_avgtype == 'reg': # Moving average applies CFO at each step
                    do_CFO_correction[curnode] = False
            
            
            # apply cfo correction if needed
            if do_CFO_correction[curnode]:
                deltaf[curnode] += CFO_correction

            # -------------------
            if ctrl.keep_intermediate_values:
                sample_inter[curnode].append(cursample)
                theta_inter[curnode].append(theta[curnode])
                phi_inter[curnode].append(phi[curnode])
                deltaf_inter[curnode].append(deltaf[curnode])



            # --------
            # Set next event
            wait_til_emit[curnode] = math.ceil(phi[curnode]*emit_frac)+cursample+TO_correction
            if wait_til_emit[curnode] < 1:
                wait_til_emit[curnode] = 1

            ordered_insert(wait_til_emit[curnode], curnode)
            emit[curnode] = True


        # ----------------
        # Fetch next event/clock
        cursample = queue_sample.pop(0)
        curnode = queue_clk.pop(0)





    #---------------
    # Post-sim wrap up
    #---------------



    if ctrl.display:
        print('theta STD: ' + str(np.std(theta)))
        print('deltaf STD: ' + str(np.std(deltaf)) + '    spread: ' + str(max(deltaf)-min(deltaf)))


    # Add all calculated values with the controls parameter structure
    ctrl.theta = theta
    ctrl.deltaf = deltaf
    ctrl.phi = phi

    if ctrl.keep_intermediate_values:
        ctrl.sample_inter = sample_inter
        ctrl.theta_inter = theta_inter
        ctrl.deltaf_inter = deltaf_inter
        ctrl.phi_inter = phi_inter

    if ctrl.saveall:
        #ctrl.update(p.__dict__)
        warnings.warn('Not doing saveall to the ctrl struct')
        pass

    
    



    """
    Emit vs Acquire: sample round DOWN after emit and sample round UP after acquire.


    OPTIMIZAION

    1. Make scheduler a single list of tuples. In fact, consider a different data strucutre for schedule

    2. The topo_matrix is iterated through. Try to make 3d topo_matrix to save on that iteration

    3. Skip emitting in your own channel times zero. (very small increase)

    4. Fold in "ordered insert" in the while loop to save on the function call.

    """




