#!/usr/bin/env python
"""Channel simulation wrapper, to be executed directly in a terminal"""

import dumbsqlite3 as db
import plotlib as graphs
import os
import lib
import numpy as np
import warnings
import inspect
import time

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

    p.power_weight = 2
    p.full_sim = True
    p.bias_removal = False
    p.ma_window = 1 # number of samples to average in the crosscorr i.e. after analog modulation
    p.train_type = 'single' # Type of training sequence
    p.crosscorr_fct = 'match_decimate' 
    #p.pulse_type = 'raisedcosine'
    p.pulse_type = 'rootraisedcosine'
    p.central_padding = 0 # As a fraction of zpos length
    p.update()


    ctrl = SimControls()
    ctrl.steps = 40 # Approx number of emissions per node
    ctrl.basephi = 5000 # How many samples between emission
    ctrl.display = True # Show stuff in the console
    ctrl.keep_intermediate_values = False # Needed to draw graphs
    ctrl.nodecount = 5 # Number of nodes
    ctrl.static_nodes = 0
    ctrl.CFO_step_wait = float('inf') # Use float('inf') to never correct for CFO
    ctrl.TO_step_wait = 0
    ctrl.max_start_delay = 0 # In factor of basephi

    ctrl.theta_bounds = [0.3,0.7] # In units of phi
    ctrl.theta_bounds = [0.48,0.52] # In units of phi
    #ctrl.theta_bounds = [0.5,0.5] # In units of phi
    #ctrl.deltaf_bound = 3e-6
    ctrl.deltaf_bound = 0
    ctrl.noise_var = 0.01
    ctrl.rand_init = False
    ctrl.non_rand_seed = 11231231 # Only used if rand_init is False

    ctrl.bmap_reach = 3e-6
    ctrl.bmap_scaling = 100

    ctrl.cfo_mapper_fct = lib.cfo_mapper_pass
    ctrl.CFO_processing_avgtype = 'reg'
    ctrl.CFO_processing_avgwindow = 1
    ctrl.max_CFO_correction = 1e-6 # As a factor of f_symb

    ctrl.delay_params = lib.DelayParams(lib.delay_pdf_exp)
    ctrl.delay_params.taps = 1
    ctrl.delay_params.sigma = 0.02

    ctrl.half_duplex = False
    ctrl.hd_slot0 = 0.3 # in terms of phi
    ctrl.hd_slot1 = 0.7 # in terms of phi
    ctrl.hd_block_during_emit = True
    ctrl.hd_block_extrawidth = 0 # as a factor of offset (see runsim to know what is offset)

    ctrl.var_winlen = False
    ctrl.vw_minsize = 5 # as a factor of len(p.analog_sig)
    ctrl.vw_lothreshold = 0.1 # winlen reduction threshold
    ctrl.vw_hithreshold = 0.1 # winlen increase threshold
    ctrl.vw_lofactor = 1.5 # winlen reduction factor
    ctrl.vw_hifactor = 2 # winlen increase factor
    
    ctrl.saveall = True

    ctrl.update()

    cdict = {
        'nodecount':[x for x in range(5,15)]
        }

    pdict = {}

    return p, ctrl, cdict, pdict

#------------------------
def main_thesis():

    p, ctrl = dec_wrap1()
    #graphs.barywidth_wrap(p,ctrl, force_calculate=True); graphs.show(); exit()
    
    #graphs.crosscorr(p); graphs.show(); exit()
    #graphs.analog(p); graphs.show(); exit()
    graphs.pulse(p); graphs.show(); exit()

    sim_object = SimWrap(p, ctrl)
    sim_object.show_plots = True
    sim_object.simulate()

def main_interd():


    p, ctrl, cdict, pdict = dec_wrap2()
    sim = SimWrap(ctrl, p, cdict, pdict)
    #p, ctrl = dec_wrap1()


    #graphs.delay_pdf(ctrl); graphs.show(); exit()
    #graphs.delay_grid(ctrl); graphs.show(); exit()
    sim.ctrl.keep_intermediate_values = True
    sim.show_CFO = False
    sim.make_plots = False 
    sim.simulate()
    sim.eval_convergence()
    exit()

    sim.set_all_nodisp()
    sim.make_plots = False
    sim.repeat = 4




    tsims = sim.total_sims('all')
    dates = sim.simmany('all'); dates = [dates[0], dates[-1]]
    #dates = [20160316135854599,  20160316135906136]
    collist = ['nodecount', 'theta_ssstd']
    graphs.scatter_range(dates, collist)
    graphs.show()


    exit()
    #db_out = np.array(db.fetch_last_n(tsims, ['nodecount', 'theta_ssstd'], dateid=True))
    #dates, data = np.split(db_out, [1,], axis=1); dates = dates.astype(int)
    data = np.array(db.fetch_range([dates[-1], dates[0]], collist))

    x, y, ystd = lib.avg_copies(data)
    
    savename = 'graphdump/'
    savename += 'stuff'
    #savename += '_' + str(dates[0,0])[4:]
    savename += '_' + str(dates[0])[4:]
    graphs.scatter(x, y, ystd, 'nodecount', 'theta_std (samples)', savename=savename)
    graphs.show()


#-----------------------



class SimWrap(lib.Struct):
    """Simulation object for ease of use"""

    force_calculate = False # Forces the update of all values before starting the simulation
    make_plots = True
    show_CFO = True # only works if make_plots is True
    show_TO = True # only works if make_plots is True
    show_SNR = True
    show_siglen = True
    show_bary = False
    repeat = 1
    last_msg_len = 0

    cdict = dict()
    pdict = dict()

    #Convergence criterions
    conv_eval_cfo = False
    conv_min_slope_samples = 5 # Minimum # of samples to take for slope eval
    


    def __init__(self, ctrl, p, cdict=None, pdict=None):
        """Prepares ctrl & p for simulation"""
        self.add(p=p)
        self.add(ctrl=ctrl)
        lib.barywidth_map(self.p, reach=self.ctrl.bmap_reach , scaling_fct=self.ctrl.bmap_scaling , force_calculate=self.force_calculate, disp=self.show_bary)
        if self.force_calculate:
            self.p.update()
            self.ctrl.update()
        if cdict is not None:
            self.cdict=cdict
        if pdict is not None:
            self.pdict=pdict

    def update_params(self,  ctrl=None, p=None):
        """Updates the params and runs barywidth_map if needed"""
        # Check if barywidth update
        if ctrl is None: ctrl = self.ctrl
        if p is None: ctrl = self.p

        do_bary = p != self.p or ctrl.bmap_reach != self.ctrl.bmap_reach or ctrl.bmap_scaling != self.ctrl.bmap_scaling
        self.add(p=p)
        self.add(ctrl=ctrl)

        if do_bary: #Update barymap?
            self.update_bary()

    def update_bary(self):
        """Triggers the update of the barywidth map"""
        lib.barywidth_map(self.p, reach=self.ctrl.bmap_reach , scaling_fct=self.ctrl.bmap_scaling , force_calculate=self.force_calculate, disp=self.show_bary)

    def set_all_nodisp(self):
        """All display values are set to false"""
        self.show_CFO = False
        self.show_TO = False
        self.show_SNR = False
        self.show_siglen = False
        self.show_bary = False
        self.ctrl.display = False

    def simulate(self):
        """Simulate and run post-sim stuff, such as graphs or output saving"""
        msg = ''
        siglen_value = "{:.2f}".format(len(self.p.analog_sig)/self.ctrl.basephi)
        if self.show_siglen: msg = 'Sync signal length: ' + siglen_value + ' basephi' + '    '
        if self.show_SNR: msg += "SNR : " + str(lib.calc_snr(self.ctrl,self.p)) + " dB"

        if msg!='': print(msg)
        

        # Exec the simulation and save the output
        runsim(self.p, self.ctrl)
        self.ctrl.date = lib.build_timestamp_id();
        db.add(self.ctrl.__dict__)

        
        # Plot pretty graphs
        if self.make_plots:
            graphs.post_sim_graphs(self)

    def simmany(self, assign_method='all'):
        """Simulates many simulations according to the simdicts"""
        if not self.ctrl.rand_init:
            warnings.warn('Not using random initialization')

        notsaveall_string = "Some simulations have the parameter set saveall parameter set to False"
        if 'saveall' in self.cdict:
            if False in self.cdict['saveall']:
                warnings.warn(notsaveall_string)
        elif not self.ctrl.saveall:
            warnings.warn(notsaveall_string)

        if assign_method == 'all':
            assign_fct = self.assign_next_all
        elif assign_method == 'one':
            assign_fct = self.assign_next_one
        else:
            raise ValueError('Invalid assign method')

        oprint = lambda x: print(x, end='\r')
        tf = time.clock
        tsims = self.total_sims(assign_method)
        count = 0
        t0 = tf()

        # Main loop
        while next(assign_fct()):
            for k in range(self.repeat):
                msg = "Iteration " + str(count).zfill(len(str(tsims))) + " of " + str(tsims) + '. ' 
                msg += "%2.2f"%(count*100/tsims) + "% done"
                oprint(msg)
                count += 1
                self.simulate()
        oprint(' '*len(msg))
        print('Done simmany in ' + "%2.2f"%(tf()-t0) + ' seconds.')

        # Logging the dateid set
        dates = np.array(db.fetch_last_n(tsims, ['date'], dateid=False)).flatten()
        lib.appendlog('Done ' + str(tsims) + ' simulations: ' +\
                      str(dates[-1]) + ' to ' + str(dates[0]))

        return dates

    def assign_next_all(self):
        """Makes a generator of both ctrl and p. It iterates through cdict and pdict over all
        lists at the same time

        Note: Shorter lists are deleted when done. THe concerned variable will keep that final value
        for future iteration"""
        while self.cdict or self.pdict:
            for d, obj in zip([self.cdict, self.pdict], [self.ctrl, self.p]):
                todel = []
                for key, lst in d.items():
                    if lst:
                        obj.change(key, lst.pop())
                    if not lst: 
                        todel.append(key)
                for key in todel: 
                    del d[key]
            
            self.update_bary()
            yield True

        yield False # When both dicts are empty,  return false.

    def assign_next_one(self):
        """Makes a generator of both ctrl and p. It iterates through cdict and pdict over all
        lists at the same time

        Note: Shorter lists are deleted when done. THe concerned variable will keep that final value
        for future iteration"""
        while self.cdict or self.pdict:
            for d, obj in zip([self.cdict, self.pdict], [self.ctrl, self.p]):
                todel = []
                for key, lst in d.items():
                    while lst: 
                        obj.change(key, lst.pop())
                        self.update_bary()
                        yield True
                    if not lst: 
                        todel.append(key)
                for key in todel: 
                    del d[key]
            

        yield False # When both dicts are empty,  return false.

    def total_sims(self, assign_method='all'):
        """Returns the total number of simulations that will be executed based on the # of elements in cdict/pdict"""
        count = 0
        if assign_method=='all':
            max_len = 0
            for d in [self.pdict, self.cdict]:
                for key, lst in d.items():
                    if len(lst) > max_len:
                        max_len = len(lst)
            count = max_len

        elif assign_method=='one':
            for d in [self.pdict, self.cdict]:
                for key, lst in d.items():
                    count += len(lst)
        
        return count*self.repeat

    def eval_convergence(self, disp=True):
        """Evaluates if convergence has been achieved"""
        if not self.ctrl.simulated:
            raise Exception("Function must be executed after a simulation has been done")

        output = {}

        def drift_eval(lst):
            """Calculates the average slope for the last self.min_slope_samples"""
            min_len = min([len(x) for x in lst])
            datacount = self.conv_min_slope_samples
            if datacount > min_len:
                datacount = min_len
                warnings.warn('slope domain bigger than minimum domain; not enough intermediate samples')
            #extract the relevant samples
            data = np.zeros([len(lst), datacount])
            for k,sublist in enumerate(lst):
                data[k,:] = np.array(sublist[-datacount:])

            # Calculate slope
            slopes = np.ediff1d(np.mean(data, axis=0))
            return np.mean(slopes), np.std(slopes)

        # Evaluate drift slope over the last domain% or 5 intermediate vals
        tlist = self.ctrl.theta_inter
        clist = self.ctrl.deltaf_inter

        output['theta_drift'] = drift_eval(tlist)
        if self.conv_eval_cfo:
            output['deltaf_drift'] = drift_eval(dlist)

        # Evaluate communication capabilites between all nodes 
        theta = self.ctrl.theta
        tmp_del_grid = (self.ctrl.delay_params.delay_grid*self.ctrl.basephi).astype(theta.dtype)
        offset_grid = np.tile(theta.reshape(-1,1), theta.shape[0])
        offset_grid +=  -1*offset_grid.T - tmp_del_grid


if __name__ == '__main__':
    main_interd()
    #main_thesis()





















    #  LocalWords:  avgwindow
