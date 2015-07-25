#!/usr/bin/env python
"""Channel simulation wrapper, to be executed directly in a terminal"""

from sim_channel import *
import dumbsqlite3 as db



#controls = SimControlParams()
controls = default_ctrl_dict()


p = Params()
p.zc_len = 10
p.plen = 39
p.rolloff = 0.2
p.f_samp = 12
p.f_symb = 3
p.repeat = 1
p.power_weight = 4
p.CFO = 0
p.TO = 0
p.full_sim = True
p.bias_removal = 0

#p.update()
#p.calc_base_barywidth()

#runsim(p, controls)


arr = np.random.randint(3,size=3)

data = {'date':build_timestamp_id(), 'f_samp':3, 'full_sim':True, 'CFO':0, 'pulse':arr}



dblist = db.fetchall()

print(db.fetchall())
