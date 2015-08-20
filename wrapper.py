#!/usr/bin/env python
"""Channel simulation wrapper, to be executed directly in a terminal"""

from sim_channel import *
import dumbsqlite3 as db
import plotlib as graphs


#controls = SimControlParams()
steps = 90
controls = default_ctrl_dict()
controls['frameunit'] = 5000
controls['display'] = True
controls['saveall'] = True
controls['keep_intermediate_values'] = True
controls['clkcount'] = 20
controls['CFO_step_wait'] = 100
controls['noise_power'] = 0
controls['rand_init'] = False
controls['max_echo_taps'] = 1

controls['chansize'] = int(controls['frameunit']*steps)


p = Params()
p.zc_len = 121
p.plen = 15
p.rolloff = 0.2
p.f_samp = 4
p.f_symb = 1
p.repeat = 1
p.spacing_factor = 2
p.power_weight = 4
p.full_sim = True
p.bias_removal = 0
p.crosscorr_fct = 'analog'

p.update()
p.calc_base_barywidth()

build_delay_matrix(controls)
runsim(p, controls); #controls['date'] = build_timestamp_id(); db.add(controls)

#graphs.hair(controls['frame_inter'], controls['deltaf_inter'])
#graphs.save('deltaf_hair')

graphs.hair(controls['frame_inter'], controls['theta_inter'])
graphs.show()

#graphs.barywidth(p, savename='short_barywidth', reach=0.05, scaling=0.0001)

#graphs.crosscorr(p)
#graphs.save('crosscorr')

#graphs.analog_zpos(p) # TO DO WITH ZCLEN 
#graphs.save('modulated_zpos')

#graphs.pulse(p)
#graphs.save('pulse')


#graphs.show()


# TODO:
# Barywidth: base barywidth with a line fitting instead of dumb values
# CFO correction: put in a max values



