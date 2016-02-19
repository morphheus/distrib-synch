#!/usr/bin/env python
"""Channel simulation wrapper, to be executed directly in a terminal"""

import dumbsqlite3 as db
import plotlib as graphs
import os
import lib
import numpy as np

from sim_channel import default_ctrl_dict, runsim, SimControls




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
    ctrl.display = True
    ctrl.saveall = True
    ctrl.keep_intermediate_values = True
    ctrl.nodecount = 3
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
    p.train_type = 'singledecimate' # Type of training sequence
    p.crosscorr_fct = 'analog' # Only important when not using 'singledecimate' train type
    #p.pulse_type = 'raisedcosine'
    p.pulse_type = 'rootraisedcosine'
    p.central_padding = 0 # As a fraction of zpos length
    p.update()


    ctrl = SimControls()
    ctrl.steps = 40 # Approx number of emissions per node
    ctrl.basephi = 4000 # How many samples between emission
    ctrl.display = True # Show stuff in the console
    ctrl.keep_intermediate_values = True # Needed to draw graphs
    ctrl.nodecount = 20 # Number of nodes
    ctrl.CFO_step_wait = float('inf') # Use float('inf') to never correct for CFO

    ctrl.theta_bounds = [0,1] # In units of phi
    ctrl.deltaf_bound = 3e-6
    ctrl.noise_std = 0
    ctrl.rand_init = True
    ctrl.non_rand_seed = 112312341 # Only used if rand_init is False
    ctrl.max_echo_taps = 1 

    ctrl.bmap_reach = 3e-6
    ctrl.bmap_scaling = 100

    ctrl.cfo_mapper_fct = lib.cfo_mapper_pass
    ctrl.CFO_processing_avgtype = 'reg'
    ctrl.CFO_processing_avgwindow = 1
    ctrl.max_CFO_correction = 1e-6 # As a factor of f_symb
    #ctrl.min_delay = 0.02 # in terms of basephi
    #ctrl.delay_sigma = 0.001 # Standard deviation used for the generator delay function
    #ctrl.delay_fct = delay_pdf_exp

    ctrl.saveall = True

    ctrl.update()

    return p, ctrl

#------------------------
def main_thesis(p,ctrl):
    graphs.barywidth(p, fit_type='linear', reach=ctrl.bmap_reach , scaling_fct=ctrl.bmap_scaling, residuals=True, force_calculate=False ); graphs.show(); exit()


    sim_object = SimWrap(p, ctrl)
    sim_object.show_plots = True
    sim_object.simulate()

def main_interd(p,ctrl):
    #graphs.crosscorr(p); graphs.show(); exit()
    #graphs.analog(p); graphs.show(); exit()
    #graphs.pulse(p); graphs.show(); exit()

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
        self.add(p=p)
        self.add(ctrl=ctrl)

    def simulate(self):
        """Prepare the ctrl & p for simulation, then execute simulation"""
        lib.barywidth_map(p, reach=ctrl.bmap_reach , scaling_fct=ctrl.bmap_scaling , force_calculate=self.force_calculate, disp=True)
        if self.force_calculate:
            self.p.update()
            self.ctrl.update()

        # Display SNR
        msg = ''
        if self.show_siglen: msg = 'Sync signal length: ' + str(len(p.analog_sig)) + '    '
        if self.show_SNR: msg += "SNR : " + str(lib.calc_snr(self.ctrl,self.p)) + " dB"

        if msg!='': print(msg)
        

        # Exec the simulation and save the output
        runsim(self.p, self.ctrl)
        self.ctrl.date = lib.build_timestamp_id();
        #db.add(self.ctrl)

        
        # Plot pretty graphs
        if self.make_plots:
            graphs.post_sim_graphs(self)



p, ctrl = dec_wrap2()
main_interd(p,ctrl)
#main_thesis(p,ctrl)





















#  LocalWords:  avgwindow
