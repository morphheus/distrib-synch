#!/usr/bin/env python
"""Channel simulation wrapper, to be executed directly in a terminal"""

from sim_channel import *
import dumbsqlite3 as db
import plotlib as graphs


#controls = SimControlParams()
steps = 40
controls = default_ctrl_dict()
controls['frameunit'] = 5000
controls['display'] = True
controls['saveall'] = True
controls['keep_intermediate_values'] = True
controls['clkcount'] = 11
controls['CFO_step_wait'] = 5
controls['noise_power'] = 1
controls['rand_init'] = False
controls['max_echo_taps'] = 1
controls['cfo_mapper_fct'] = cfo_mapper_order2

controls['chansize'] = int(controls['frameunit']*steps)


p = Params()
p.zc_len = 121
p.plen = 31
p.rolloff = 0.2
p.f_samp = 400
p.f_symb = 100
p.repeat = 1
p.spacing_factor = 2
p.power_weight = 4
p.full_sim = True
p.bias_removal = 0
p.crosscorr_fct = 'analog'



p.update()
barywidth_map(p, reach=0.05, scaling=0.001, force_calculate=False)



#graphs.barywidth(p, reach=0.05, scaling=0.001); graphs.show(); exit()
#graphs.pulse(p); graphs.show(); exit()

print(len(p.analog_sig)); build_delay_matrix(controls)
runsim(p, controls); #controls['date'] = build_timestamp_id(); db.add(controls)

graphs.hair(controls['frame_inter'], controls['deltaf_inter']); graphs.show()
#graphs.hair(controls['frame_inter'], controls['theta_inter']); graphs.show()

#graphs.barywidth(p, savename='short_barywidth', reach=0.05, scaling=0.0001)

#graphs.crosscorr(p)
#graphs.show()

#graphs.analog_zpos(p) # TO DO WITH ZCLEN 
#graphs.show()



#graphs.show()


# TODO:
# Barywidth: base barywidth with a line fitting instead of dumb values
# CFO correction: put in a max values



