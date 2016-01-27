#!/usr/bin/env python
"""Channel simulation wrapper, to be executed directly in a terminal"""

from sim_channel import *
import dumbsqlite3 as db
import plotlib as graphs





p = Params()
p.zc_len = 101
p.plen = 31
p.rolloff = 0.2
p.f_samp = 4e6
p.f_symb = 1e6
p.repeat = 1
p.spacing_factor = 2
p.power_weight = 4
p.full_sim = True
p.bias_removal = True
p.ma_window = 21 # number of samples i.e. after analog modulation
p.crosscorr_fct = 'analog'
p.train_type = 'chain'
p.update()


steps = 200
controls = default_ctrl_dict()
controls['frameunit'] = 4000
controls['chansize'] = int(controls['frameunit']*steps)
controls['display'] = True
controls['saveall'] = True
controls['keep_intermediate_values'] = True
controls['clkcount'] = 11
controls['CFO_step_wait'] = 10
#controls['cfo_bias'] = 0.0008 # in terms of f_symb
controls['deltaf_bound'] = 3e-6
controls['noise_std'] = 0.1
controls['rand_init'] = True
controls['max_echo_taps'] = 1
controls['cfo_mapper_fct'] = cfo_mapper_order2
controls['bmap_reach'] = 3e-6
controls['bmap_scaling'] = 3e-8
controls['CFO_processing_avgtype'] = 'reg'
controls['CFO_processing_avgwindow'] = 6
#controls['min_delay'] = 0.02 # in terms of frameunit
#controls['delay_sigma'] = 0.001 # Standard deviation used for the generator delay function
#controls['delay_fct'] = delay_pdf_exp
controls['max_CFO_correction'] = 3e-6 # As a factor of f_symb

#graphs.barywidth(p, fit_type='order2', reach=controls['bmap_reach'], scaling=controls['bmap_scaling'] ); graphs.show(); exit()


# Prepare the sync pulse
print(len(p.analog_sig))
print("SNR : " + str(calc_snr(controls,p)) + " dB")
barywidth_map(p, reach=controls['bmap_reach'], scaling=controls['bmap_scaling'], force_calculate=False, disp=True)

# Run the simulation
build_delay_matrix(controls, delay_fct = controls['delay_fct']);
runsim(p, controls); controls['date'] = build_timestamp_id(); #db.add(controls)


# Plot pretty graphs
graphs.hair(controls['frame_inter'], controls['deltaf_inter'], y_label='CFO (\Delta\lambda)', savename='lastCFO'); graphs.show()
graphs.hair(controls['frame_inter'], controls['theta_inter'], y_label='TO', savename='lastTO'); graphs.show()

#graphs.barywidth(p, savename='short_barywidth', reach=0.05, scaling=0.0001)


#graphs.analog_zpos(p) # TO DO WITH ZCLEN 
#graphs.show()



#graphs.show()


# TODO:
# Barywidth: base barywidth with a line fitting instead of dumb values
# CFO correction: put in a max values



