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
p.central_padding = 0
p.ma_window = 11
p.crosscorr_fct = 'analog'
p.update()



steps = 70
controls = default_ctrl_dict()
controls['frameunit'] = 5000
controls['chansize'] = int(controls['frameunit']*steps)
controls['display'] = True
controls['saveall'] = True
controls['keep_intermediate_values'] = True
controls['clkcount'] = 11
controls['CFO_step_wait'] = 15
#controls['cfo_bias'] = 0.0008 # in terms of f_symb
controls['deltaf_bound'] = 3e-5
controls['noise_std'] = 0.5
controls['rand_init'] = True
controls['max_echo_taps'] = 1
controls['cfo_mapper_fct'] = cfo_mapper_order2
controls['bmap_reach'] = 1e-4
controls['bmap_scaling'] = 5e-7
#controls['min_delay'] = 0.02 # in terms of frameunit
#controls['delay_sigma'] = 0.001 # Standard deviation used for the generator delay function
#controls['delay_fct'] = delay_pdf_exp

controls['max_CFO_correction'] = 5e-6 # As a factor of f_symb
print(len(p.analog_sig))

#graphs.barywidth(p, fit_type='linear', reach=controls['bmap_reach'], scaling=controls['bmap_scaling'] ); graphs.show(); exit()
#graphs.pulse(p); graphs.show(); exit()
graphs.crosscorr(p); graphs.show(); exit()
#graphs.analog(p); graphs.show(); exit()


print("SNR : " + str(calc_snr(controls,p)) + " dB")
barywidth_map(p, reach=controls['bmap_reach'], scaling=controls['bmap_scaling'], force_calculate=False)

build_delay_matrix(controls, delay_fct = controls['delay_fct']);
#print(controls['echo_delay']); exit()
runsim(p, controls); controls['date'] = build_timestamp_id(); #db.add(controls)

graphs.hair(controls['frame_inter'], controls['deltaf_inter'], y_label='CFO'); graphs.show()
graphs.hair(controls['frame_inter'], controls['theta_inter'], y_label='TO'); graphs.show()

#graphs.barywidth(p, savename='short_barywidth', reach=0.05, scaling=0.0001)

#graphs.crosscorr(p)
#graphs.show()

#graphs.analog_zpos(p) # TO DO WITH ZCLEN 
#graphs.show()



#graphs.show()


# TODO:
# Barywidth: base barywidth with a line fitting instead of dumb values
# CFO correction: put in a max values



