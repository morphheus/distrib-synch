#!/usr/bin/env python
"""Channel simulation wrapper, to be executed directly in a terminal"""

import dumbsqlite3 as db
import plotlib as graphs
import os
import lib
import numpy as np
import warnings


from sim_channel import runsim, SimControls




#-------------------------
def dec_wrap1():
    """Chained ZC sequences"""
    p = lib.SyncParams()
    p.zc_len = 101
    p.plen = 31
    p.rolloff = 0.2
    p.f_samp = 4e6
    p.f_symb = 1e6
    p.repeat = 1
    p.spacing_factor = 2 # CHANGE TO TWO!
    p.power_weight = 4
    p.full_sim = True
    p.bias_removal = True
    p.ma_window = 1 # number of samples i.e. after analog modulation
    p.crosscorr_fct = 'analog'
    p.train_type = 'chain'
    p.pulse_type = 'raisedcosine'
    p.central_padding = 0 # As a fraction of zpos length
    p.update()

    ctrl = SimControls()
    ctrl.steps = 40
    ctrl.basephi = 4000
    ctrl.max_start_delay = 10 # In factor of basephi
    ctrl.display = True
    ctrl.saveall = True
    ctrl.keep_intermediate_values = True
    ctrl.nodecount = 5
    ctrl.CFO_step_wait = 10
    ctrl.theta_bounds = [0,1]
    #ctrl.cfo_bias = 0.0008 # in terms of f_symb
    ctrl.deltaf_bound = 3e-6
    ctrl.noise_std = 0
    ctrl.rand_init = True
    ctrl.non_rand_seed = 112312341 # Only used if rand_init is False
    ctrl.max_echo_taps = 1
    ctrl.cfo_mapper_fct = lib.cfo_mapper_order2
    ctrl.bmap_reach = 1e-6
    ctrl.bmap_scaling = 100
    ctrl.CFO_processing_avgtype = 'reg'
    ctrl.CFO_processing_avgwindow = 1
    #ctrl.min_delay = 0.02 # in terms of basephi
    #ctrl.delay_sigma = 0.001 # Standard deviation used for the generator delay function
    #ctrl.delay_fct = delay_pdf_exp
    ctrl.max_CFO_correction = 1e-6 # As a factor of f_symb
    ctrl.update()

    return p, ctrl

def dec_wrap2():
    """Single ZC sequence with Decimation"""
    p = lib.SyncParams()
    p.zc_len = 101
    p.plen = 31

    p.rolloff = 0.2
    p.f_samp = 4e6
    p.f_symb = 1e6
    p.repeat = 1
    p.spacing_factor = 1 # CHANGE TO TWO!

    p.power_weight = 4
    p.full_sim = True
    p.bias_removal = True
    p.ma_window = 1 # number of samples to average in the crosscorr i.e. after analog modulation
    p.train_type = 'single' # Type of training sequence
    p.crosscorr_fct = 'match_decimate' 
    #p.pulse_type = 'raisedcosine'
    p.pulse_type = 'rootraisedcosine'
    p.central_padding = 0 # As a fraction of zpos length
    p.update()


    ctrl = SimControls()
    ctrl.steps = 60 # Approx number of emissions per node
    ctrl.basephi = 6000 # How many samples between emission
    ctrl.display = True # Show stuff in the console
    ctrl.keep_intermediate_values = True # Needed to draw graphs
    ctrl.nodecount = 3 # Number of nodes
    ctrl.static_nodes = 1
    ctrl.CFO_step_wait = float('inf') # Use float('inf') to never correct for CFO
    ctrl.max_start_delay = 10 # In factor of basephi

    ctrl.theta_bounds = [0.3,0.7] # In units of phi
    #ctrl.theta_bounds = [0.5,0.5] # In units of phi
    #ctrl.deltaf_bound = 3e-6
    ctrl.deltaf_bound = 0
    ctrl.noise_var = 0
    ctrl.rand_init = False
    ctrl.non_rand_seed = 192912341 # Only used if rand_init is False
    ctrl.max_echo_taps = 1 

    ctrl.bmap_reach = 3e-6
    ctrl.bmap_scaling = 100

    ctrl.cfo_mapper_fct = lib.cfo_mapper_pass
    ctrl.CFO_processing_avgtype = 'reg'
    ctrl.CFO_processing_avgwindow = 1
    ctrl.max_CFO_correction = 1e-6 # As a factor of f_symb
    #ctrl.min_delay = 0.02 # in terms of basephi
    #ctrl.delay_sigma = 0.001 # Standard deviation used for the generator delay function
    #ctrl.delay_fct = lib.delay_pdf_exp

    ctrl.half_duplex = False
    ctrl.hd_slot0 = 0.3 # in terms of phi
    ctrl.hd_slot1 = 0.7 # in terms of phi
    ctrl.hd_block_during_emit = True
    ctrl.hd_block_extrawidth = 2 # as a factor of offset (see runsim to know what is offset)

    ctrl.var_winlen = False
    ctrl.vw_minsize = 5 # as a factor of len(p.analog_sig)
    ctrl.vw_lothreshold = 0.1 # winlen reduction threshold
    ctrl.vw_hithreshold = 0.1 # winlen increase threshold
    ctrl.vw_lofactor = 1.5 # winlen reduction factor
    ctrl.vw_hifactor = 2 # winlen increase factor
    
    ctrl.saveall = True

    ctrl.update()

    return p, ctrl

#------------------------
def main_thesis():

    p, ctrl = dec_wrap1()
    #graphs.barywidth_wrap(p,ctrl, force_calculate=True); graphs.show(); exit()
    
    #graphs.crosscorr(p); graphs.show(); exit()
    #graphs.analog(p); graphs.show(); exit()
    #graphs.pulse(p); graphs.show(); exit()

    sim_object = SimWrap(p, ctrl)
    sim_object.show_plots = True
    sim_object.simulate()

def main_interd():

    p, ctrl = dec_wrap2()
    sim_object = SimWrap(p, ctrl)
    sim_object.show_CFO = False
    sim_object.simulate()

#-----------------------
class SimWrap(lib.Struct):
    """Simulation object for ease of use"""

    force_calculate = False # Forces the update of all values before starting the simulation
    make_plots = True
    show_CFO = True # only works if make_plots is True
    show_TO = True # only works if make_plots is True
    show_SNR = True
    show_siglen = True
    

    def __init__(self, p, ctrl):
        """Prepares ctrl & p for simulation"""
        self.add(p=p)
        self.add(ctrl=ctrl)

        lib.barywidth_map(self.p, reach=self.ctrl.bmap_reach , scaling_fct=self.ctrl.bmap_scaling , force_calculate=self.force_calculate, disp=True)
        if self.force_calculate:
            self.p.update()
            self.ctrl.update()

    def simulate(self):
        """Simulate and run post-sim stuff, such as graphs or output saving"""

        # Display SNR
        msg = ''
        siglen_value = "{:.2f}".format(len(self.p.analog_sig)/self.ctrl.basephi)
        if self.show_siglen: msg = 'Sync signal length: ' + siglen_value + ' basephi' + '    '
        if self.show_SNR: msg += "SNR : " + str(lib.calc_snr(self.ctrl,self.p)) + " dB"

        if msg!='': print(msg)
        

        # Exec the simulation and save the output
        runsim(self.p, self.ctrl)
        self.ctrl.date = lib.build_timestamp_id();
        #db.add(self.ctrl)

        
        # Plot pretty graphs
        if self.make_plots:
            graphs.post_sim_graphs(self)



if __name__ == '__main__':
    main_interd()
    #main_thesis()





















    #  LocalWords:  avgwindow
