#!/usr/bin/env python
"""Channel simulation wrapper, to be executed directly in a terminal"""

from sim_channel import *
import dumbsqlite3 as db
import plotlib as graphs



#controls = SimControlParams()
steps = 20
controls = default_ctrl_dict()
controls['frameunit'] = 10000
controls['display'] = False
controls['saveall'] = True
controls['keep_intermediate_values'] = True
controls['clkcount'] = 9
controls['CFO_step_wait'] = 2
controls['noise_power'] = 0
controls['rand_init'] = False

controls['chansize'] = int(controls['frameunit']*steps)


p = Params()
p.zc_len = 121
p.plen = 5
p.rolloff = 0.2
p.f_samp = 4
p.f_symb = 1
p.repeat = 1
p.power_weight = 4
p.CFO = 0
p.TO = 0
p.full_sim = True
p.bias_removal = 0

p.update()
p.calc_base_barywidth()


#runsim(p, controls)

graphs.barywidth(p, reach=0.5, scaling=0.01)







