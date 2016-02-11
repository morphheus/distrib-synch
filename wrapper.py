#!/usr/bin/env python
"""Channel simulation wrapper, to be executed directly in a terminal"""

import dumbsqlite3 as db
import plotlib as graphs
import os
import lib
import numpy as np

from sim_channel import default_ctrl_dict, runsim, SimControls


#x = np.array([4]*15)
#print(lib.convolve_mov_avg(x,5))





p = lib.Params()
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
p.central_padding = 0 # As a fraction of zpos length
p.update()


#graphs.pulse(p); graphs.show(); exit()

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
ctrl.bmap_reach = 3e-6
ctrl.bmap_scaling = 100
ctrl.CFO_processing_avgtype = 'reg'
ctrl.CFO_processing_avgwindow = 1
#ctrl.min_delay = 0.02 # in terms of basephi
#ctrl.delay_sigma = 0.001 # Standard deviation used for the generator delay function
#ctrl.delay_fct = delay_pdf_exp
ctrl.max_CFO_correction = 1e-6 # As a factor of f_symb
ctrl.update()

graphs.barywidth(p, fit_type='logistic', reach=ctrl.bmap_reach , scaling_fct=ctrl.bmap_scaling, residuals=True, force_calculate=True ); graphs.show(); exit()


# Prepare the sync pulse
print(len(p.analog_sig))



print("SNR : " + str(lib.calc_snr(ctrl,p)) + " dB")
lib.barywidth_map(p, reach=ctrl.bmap_reach , scaling_fct=ctrl.bmap_scaling , force_calculate=False, disp=True)

# Run the simulation
lib.build_delay_matrix(ctrl, delay_fct = ctrl.delay_fct);
runsim(p, ctrl); ctrl.date = lib.build_timestamp_id(); #db.add(ctrl)


# Plot pretty graphs
graphs.hair(ctrl.sample_inter , ctrl.deltaf_inter , y_label='CFO (\Delta\lambda)', savename='lastCFO'); graphs.show()
graphs.hair(ctrl.sample_inter , ctrl.theta_inter , y_label='TO', savename='lastTO'); graphs.show()

#graphs.barywidth(p, savename='short_barywidth', reach=0.05, scaling=0.0001)


#graphs.analog_zpos(p) # TO DO WITH ZCLEN 
#graphs.show()



#graphs.show()


# TODO:
# Barywidth: base barywidth with a line fitting instead of dumb values
# CFO correction: put in a max values





