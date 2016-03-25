#!/usr/bin/env python

# User modules
import lib
import plotlib as graphs
from lib import calc_both_barycenters, SyncParams

# Python modules
import numpy as np
import matplotlib.pyplot as plt
import collections
import bisect
import math # import ALL THE MATH
import warnings

from numpy import pi
from pprint import pprint


NOSAVELIST = [
    'TO',
    'CFO'
    ]

#----------------------------------
class SimControls(lib.Struct):
    """Container object for the control parameters of the runsim() function"""
    need_update = ['basephi', 'chansize', 'phi_bounds', 'theta_bounds', 'echo_delay', 'echo_amp', 'nodecount', 'pdf_kwargs', 'delay_params']
    simulated = False

    def __init__(self):
        """Default values"""
        self.steps = 30
        self.nodecount = 7
        self.basephi = 2000
        self.chansize = self.basephi*self.steps
        self.noise_var = 0
        self.phi_bounds = [1,1]
        self.theta_bounds = [0,1]
        self.max_start_delay = 0 # In factor of basephi
        self.min_back_adjust = 0.1 # In terms of phi
        self.self_emit = False # IF set to False, the self-emit will just be skipped.
        self.CFO_step_wait = 60
        self.TO_step_wait = 1
        self.rand_init = False
        self.display = True # Display progress of runsim
        self.keep_intermediate_values = False
        self.delay_fct = lib.delay_pdf_static
        self.deltaf_bound = 0.02 # in units of f_samp
        self.bmap_reach = 3e-1
        self.bmap_scaling = 100
        self.non_rand_seed = 1231231
        # Node behaviours
        self.static_nodes = 0 # Static nodes do not adjust
        # Echo controls, initialized with no echoes
        self.delay_params = lib.DelayParams(lib.delay_pdf_exp)
        self.pdf_kwargs = dict()
        # Correction controls
        self.epsilon_TO = 0.5
        self.epsilon_CFO = 0.25
        self.max_CFO_correction = 0.02 # As a factor of f_symb
        self.CFO_processing_avgtype = 'mov_avg' # 'mov_avg' or 'reg' (non-mov avg)
        self.CFO_processing_avgwindow = 5
        self.cfo_mapper_fct = lib.cfo_mapper_linear
        self.cfo_bias = 0 # in terms of f_samp
        # Half-duplex constraint - all options relevant only if half_duplex is True
        self.half_duplex = False
        self.hd_slot0 = 0.3 # in terms of phi
        self.hd_slot1 = 0.7 # in terms of phi
        self.hd_block_during_emit = True
        self.hd_block_extrawidth = 0 # as a factor of offset (see runsim to know what is offset)
        # Variable adjust window length - all options relevant only if half_duplex is True
        self.var_winlen = True
        self.vw_minsize = 2 # as a factor of len(p.analog_sig)
        self.vw_lothreshold = 0.1 # winlen reduction threshold
        self.vw_hithreshold = 0.1 # winlen increase threshold
        self.vw_lofactor = 1.5 # winlen reduction factor
        self.vw_hifactor = 2 # winlen increase factor

        # Other
        self.init_update = True
        self.saveall = False # This options also saves all fields in SyncParams to the control dict

    def change(self, var, val):
        """Changes a value of ctrl. If necessary, will run update"""
        setattr(self, var, val)
        if var in self.need_update:
            self.update()

    def update(self):
        """Must be run before runsim can be executed"""
        self.chansize = int(self.basephi*self.steps)
        self.phi_minmax = [round(x*self.basephi) for x in self.phi_bounds]
        self.theta_minmax = [round(x*self.basephi) for x in self.theta_bounds]

        if not self.rand_init:
            np.random.seed(self.non_rand_seed)

        self.echo_delay, self.echo_amp = self.delay_params.build_delay_matrix(self.nodecount, self.basephi, **self.pdf_kwargs)

        if not self.rand_init:
            np.random.seed()
        
        # Input protection
        val_within_bounds(self.max_start_delay, [0,float('inf')] , 'max_start_delay')
        val_within_bounds(self.hd_slot0, [0,1] , 'hd_slot0')
        val_within_bounds(self.hd_slot1, [0,1] , 'hd_slot1')
        
        self.init_update = True


def val_within_bounds(val, bounds, name):
    if val < bounds[0] or val > bounds[1]:
        bstr = list(map(str,bounds))
        except_msg = bstr[0] + ' <= ' + name + ' <= ' + bstr[1] + ' is unsatisfied'
        raise ValueError(except_msg)

def ordered_insert(sample, clknum):
    """Insert from the left in descending order, in a list"""
    global queue_sample, queue_clk

    idx = bisect.bisect(queue_sample, sample)
    queue_sample.insert(idx, sample)
    queue_clk.insert(idx, clknum)

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
    CFO_step_wait = ctrl.CFO_step_wait
    epsilon_TO = ctrl.epsilon_TO
    epsilon_CFO = ctrl.epsilon_CFO
    cfo_mapper_fct = ctrl.cfo_mapper_fct
    CFO_processing_avgtype = ctrl.CFO_processing_avgtype
    CFO_processing_avgwindow = ctrl.CFO_processing_avgwindow
    phi_minmax = ctrl.phi_minmax
    theta_minmax = ctrl.theta_minmax


    # IF echoes specified, to shove in array. OW, just don't worry about it
    echo_delay = ctrl.echo_delay
    echo_amp = ctrl.echo_amp
    max_echo_taps = ctrl.delay_params.taps
    
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
    # Simulation initialization
    queue_sample = []
    queue_clk = []
    sync_pulse_len = len(analog_pulse)
    offset = int((sync_pulse_len-1)/2)
    max_sample = chansize-offset-np.max(echo_delay);



    # Node arrays initialization
    deltaf_minmax = np.array([-1*ctrl.deltaf_bound,ctrl.deltaf_bound])*p.f_symb
    do_CFO_correction = np.array([False]*nodecount)
    wait_CFO_correction = np.zeros(nodecount)
    wait_emit = np.zeros(nodecount)
    CFO_maxjump_direction = np.ones(nodecount)
    CFO_corr_list = [[] for x in range(nodecount)]
    TO_corr_list = [[] for x in range(nodecount)]
    wait_til_adjust = np.zeros(nodecount, dtype=lib.INT_DTYPE)
    prev_adjustsample = np.zeros(nodecount, dtype=lib.INT_DTYPE)
    prev_emit_range = [None for x in range(nodecount)]
    prev_TO = np.array([float('inf')]*nodecount, dtype=lib.FLOAT_DTYPE)
    nodes_winlen = np.array([ctrl.chansize]*nodecount, dtype=lib.INT_DTYPE)
    hd_sync_slot = np.array([0 if k%2 else 1 for k in range(nodecount)])
    nodetype = np.array(['static']*ctrl.static_nodes + ['variable']*(nodecount-ctrl.static_nodes))
    next_event = np.array(['emit' if x=='static' else 'adjust' for x in nodetype])


    if ctrl.keep_intermediate_values:
        sample_inter = [[] for k in range(nodecount)]
        theta_inter = [[] for k in range(nodecount)]
        phi_inter = [[] for k in range(nodecount)]
        deltaf_inter = [[] for k in range(nodecount)]

    if ctrl.half_duplex:
        emit_frac = np.array([ctrl.hd_slot0 if k==0 else ctrl.hd_slot1 for k in hd_sync_slot], dtype=lib.FLOAT_DTYPE)
        adjust_frac = np.array([ctrl.hd_slot0 if k==1 else ctrl.hd_slot1 for k in hd_sync_slot], dtype=lib.FLOAT_DTYPE)
    else:
        emit_frac = np.array([1/2]*nodecount, dtype=lib.FLOAT_DTYPE)
        adjust_frac = 1-emit_frac;
    
    hd_correction = np.round((emit_frac-adjust_frac)*basephi).astype(dtype=lib.INT_DTYPE)
    
    # Node initial values (drop)
    if not ctrl.rand_init:
        np.random.seed(ctrl.non_rand_seed)
    
    phi = np.random.randint(phi_minmax[0],phi_minmax[1]+1, size=nodecount)
    theta = np.random.randint(theta_minmax[0],theta_minmax[1]+1, size=nodecount)
    deltaf = np.random.uniform(deltaf_minmax[0],deltaf_minmax[1], size=nodecount)
    clk_creation = np.random.randint(0,chansize, size=nodecount)
    channels = lib.cplx_gaussian( [nodecount,chansize], ctrl.noise_var) 

    if ctrl.max_start_delay:
        start_delay = np.random.randint(0, ctrl.max_start_delay, size=nodecount)*basephi
    else:
        start_delay = np.zeros(nodecount, dtype=lib.INT_DTYPE)

    # Make sure static nodes are always start by 
    start_delay[nodetype=='static'] = 0
    

    if not ctrl.rand_init:
        np.random.seed()



    # Correction algorithms variables
    max_CFO_correction = ctrl.max_CFO_correction*p.f_symb




    ####################
    # MAIN SIMULATION
    ####################

    if ctrl.display:
        print('Theta std init: ' + str(np.std(theta)))
        print('deltaf std init: ' + str(np.std(deltaf)) + '    spread: '+ str(max(deltaf) -min(deltaf))+ '\n')


    # Local fct to add intermediate values, if necessary
    def add_inter(sample, node):
        if ctrl.keep_intermediate_values:
            sample_inter[node].append(sample)
            theta_inter[node].append(theta[node])
            phi_inter[node].append(phi[node])
            deltaf_inter[node].append(deltaf[node])

    # First event happens based on initial phase shift
    for clk, sample in enumerate(theta):
        prev_adjustsample[clk] = sample - phi[clk] + phi_minmax[1] + start_delay[clk]
        wait_til_adjust[clk] = math.floor(phi[clk]*adjust_frac[clk])
        if next_event[clk] == 'emit':
            first_event = int(round(emit_frac[clk]*phi[clk]))
        else:
            first_event = phi[clk]

        add_inter(prev_adjustsample[clk], clk)
        ordered_insert(prev_adjustsample[clk]+first_event,clk) 







    # Main loop
    del clk, sample
    cursample = queue_sample.pop(0)
    node = queue_clk.pop(0)
    while cursample < max_sample:

        # ----------------
        # Emit phase
        if next_event[node] == 'emit':
            minsample = cursample-offset
            maxsample = cursample+offset+1
            spread = range(minsample,maxsample)

            # Store appropriate blackout range
            tmp = int(round(offset*ctrl.hd_block_extrawidth))
            prev_emit_range[node] = range(minsample-tmp, maxsample+tmp)

            # Emit across all channels
            deltaf_arr = np.empty(sync_pulse_len)
            for emitclk in range(nodecount):
                if ctrl.self_emit and emitclk == node: # Skip selfemission
                    continue
                time_arr = (np.arange(minsample,maxsample) + clk_creation[node] )/p.f_samp 
                deltaf_arr = np.exp( 2*pi*1j* (deltaf[node] - deltaf[emitclk])  *( time_arr))

                #Echoes management
                for k in range(max_echo_taps):
                    curr_amp = echo_amp[node][emitclk][k]
                    if curr_amp != 0:
                        to_emit = analog_pulse*deltaf_arr
                        channels[emitclk, spread + echo_delay[node][emitclk][k]] += to_emit*curr_amp

            # Set next event
            if nodetype[node] == 'variable':
                wait_til_adjust[node] = math.floor(phi[node]*adjust_frac[node])
                ordered_insert(wait_til_adjust[node]+cursample, node)

                next_event[node] = 'adjust'
            elif nodetype[node] == 'static':
                add_inter(cursample, node)
                ordered_insert(phi[node]+cursample, node)
                next_event[node] = 'emit'







        # ----------------
        # Adjust phase
        elif next_event[node] == 'adjust':
            # Variable window length
            if ctrl.var_winlen:
                # Calculate winlen
                winlen = nodes_winlen[node]
                lothresh = int(round(ctrl.vw_lofactor*winlen))
                hithresh = int(round(ctrl.vw_hifactor*winlen))
                minlen = int(round(ctrl.vw_minsize*sync_pulse_len))
                maxlen = int(round(cursample - prev_adjustsample[node]))
                if prev_TO[node] < lothresh:
                    winlen = max(int(round(winlen/ctrl.vw_lofactor)), minlen)
                elif prev_TO[node] > hithresh:
                    winlen = min(int(round(winlen*ctrl.vw_hifactor)), maxlen)
                nodes_winlen[node] = winlen # Store computed winlen for later use
                

                expected_loc = cursample - wait_til_adjust[node]
                expected_loc += hd_correction[node] if ctrl.half_duplex else 0
                winmax = int(round(expected_loc + math.ceil(winlen/2)))
                winmin = int(round(expected_loc - math.floor(winlen/2)))
                # If out of bounds, go back to full adjust length
                if winmax > cursample or winmin < prev_adjustsample[node]:
                    winmax = cursample
                    winmin = prev_adjustsample[node]

                winlen = winmax - winmin # Use real winlen for calculations if there was oob issue

            else:
                winmax = cursample
                winmin = prev_adjustsample[node]
                winlen = winmax-winmin

            prev_adjustsample[node] = cursample

            # Block channel values when node emitted
            if ctrl.half_duplex and ctrl.hd_block_during_emit and prev_emit_range[node] is not None:
                channels[node, prev_emit_range[node]] = 0


            # Obtain barycenters
            if winlen > sync_pulse_len + 1:
                barycenter_range = range(winmin, winmax)
                barypos, baryneg, corpos, corneg = calc_both_barycenters(p, channels[node,barycenter_range])
            else:
                barypos = winlen - wait_til_adjust[node]
                baryneg = barypos


            
            #print(corpos.shape); exit()
            bary_avg = int(round((barypos+baryneg)/2)) 

            # Offset with respect to emit time
            TO = winmax - winlen + bary_avg - cursample + wait_til_adjust[node] 
            #if node==2:
            #    print(wait_emit[node])
            #    graphs.continuous(corpos)
            #    graphs.show()
            
            # Fix slot offset for 
            if ctrl.half_duplex:
                TO += hd_correction[node]

            
            TO_correction = round(TO*epsilon_TO)
            theta[node] += TO_correction
            theta[node] = theta[node] % phi[node]

            prev_TO[node] = TO



            # CFO correction
            CFO = cfo_mapper_fct(barypos-baryneg, p)
            
            # CFO correction clipping
            CFO_correction = CFO*epsilon_CFO
            if CFO_correction > max_CFO_correction:
                CFO_correction = max_CFO_correction
            elif CFO_correction < -1*max_CFO_correction:
                CFO_correction = -1*max_CFO_correction

            # Median filtering
            #CFO_corr_list[node].append(CFO_correction)
            #if len(CFO_corr_list[node]) > 3:
            #    CFO_corr_list[node].pop(0)
            #deltaf[node] += np.median(CFO_corr_list[node])
            
            if wait_CFO_correction[node] <= CFO_step_wait:
                wait_CFO_correction[node] += 1
            else:
                do_CFO_correction[node] = True


            # CFO correction moving average or regular average
            if do_CFO_correction[node]:
                CFO_corr_list[node].append(CFO_correction)
                if len(CFO_corr_list[node]) >= CFO_processing_avgwindow:
                    CFO_correction = sum(CFO_corr_list[node])/CFO_processing_avgwindow
                    CFO_corr_list[node] = []
                elif CFO_processing_avgtype == 'reg': # Moving average applies CFO at each step
                    do_CFO_correction[node] = False
            
            
            # apply cfo correction if needed
            if do_CFO_correction[node]:
                deltaf[node] += CFO_correction

            # Bookeeping
            add_inter(cursample, node)



            # --------
            # Set next event
            if wait_emit[node] < ctrl.TO_step_wait:
                next_event_sample = (math.ceil(phi[node])\
                                          +cursample+TO_correction).astype(lib.INT_DTYPE)
                next_event[node] = 'adjust'
                wait_emit[node] += 1
            else:
                next_event_sample = (math.ceil(phi[node]*emit_frac[node])\
                                          +cursample+TO_correction).astype(lib.INT_DTYPE)
                next_event[node] = 'emit'

            if next_event_sample <= cursample + phi[node]*ctrl.min_back_adjust:
                next_event_sample += phi[node]
            
            ordered_insert(next_event_sample, node)


        # ----------------
        # Fetch next event/node
        prev_sample = cursample
        cursample = queue_sample.pop(0)
        if prev_sample > cursample: raise Exception("Cursample was lowered; time travel isn't allowed")
        node = queue_clk.pop(0)





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
    ctrl.theta_ssstd = np.std(theta)
    ctrl.deltaf_ssstd = np.std(deltaf)
    ctrl.phi_ssstd = np.std(phi)
    ctrl.simulated = True

    if ctrl.keep_intermediate_values:
        ctrl.sample_inter = sample_inter
        ctrl.theta_inter = theta_inter
        ctrl.deltaf_inter = deltaf_inter
        ctrl.phi_inter = phi_inter

    if ctrl.saveall:
        ctrl.add(**p.__dict__)
        for var in NOSAVELIST:
            del ctrl.__dict__[var]






