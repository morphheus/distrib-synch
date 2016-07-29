#!/usr/bin/env python
"""Channel simulation wrapper. contains the SimWrap object to handle multi-iretation of potentially different starting values."""

import dumbsqlite3 as db
import plotlib as graphs
import thygraphs
import lib

import os
import numpy as np
import warnings
import inspect
import time
import math
import copy

from numpy import pi


from sim_channel import runsim, SimControls



NOSAVELIST = [
    'delay_params',
    'TO',
    'CFO'
    ]

#-------------------------
def dec_wrap2():
    """Single ZC sequence with Decimation"""
    p = lib.SyncParams()
    p.zc_len = 31
    p.plen = 31

    p.rolloff = 0.2
    p.f_symb = 30.72e6
    p.f_samp = p.f_symb*4
    p.repeat = 1
    p.spacing_factor = 1

    p.power_weight = 2
    p.full_sim = True
    p.bias_removal = False
    p.ma_window = 1 # number of samples to average in the crosscorr i.e. after analog modulation
    p.train_type = 'chain' # Type of training sequence
    p.crosscorr_type = 'match_decimate' 
    p.match_decimate_fct = lib.downsample
    p.peak_detect = 'wavg' 
    p.pulse_type = 'rootraisedcosine'
    p.central_padding = 0 # As a fraction of zpos length
    p.scfdma_precode = True
    p.scfdma_L = 4 
    p.scfdma_sinc_len_factor = p.scfdma_L

    ctrl = SimControls()
    ctrl.steps = 35 # BULK FOR THESIS
    #ctrl.steps = 50 # Approx number of emissions per node
    ctrl.basephi = 6000 # BULK FOR THESIS
    #ctrl.basephi = 12000 # How many samples between emission
    ctrl.nodecount = 15 # BULK FOR THESUS
    #ctrl.nodecount = 10 # Number of nodes
    ctrl.display = True # Show stuff in the console
    ctrl.static_nodes = 0
    ctrl.quiet_nodes = 0
    ctrl.CFO_step_wait = float('inf') # Use float('inf') to never correct for CFO
    ctrl.TO_step_wait = 4
    ctrl.max_start_delay = 7 # In factor of basephi

    ctrl.use_ringarr = True

    ctrl.theta_bounds = [0,1] # In units of phi
    ctrl.deltaf_bound = 3e-2
    #ctrl.deltaf_bound = 0
    ctrl.rand_init = False
    ctrl.epsilon_TO = 0.5
    ctrl.non_rand_seed = 11232819 # Only used if rand_init is False
    #ctrl.noise_power = float('-inf')
    ctrl.noise_power = -101 + 9 # in dbm

    ctrl.delay_params = lib.DelayParams(lib.delay_pdf_3gpp_exp)
    ctrl.delay_params.taps = 5
    ctrl.delay_params.max_dist_from_origin = 500 # (in meters)
    ctrl.delay_params.max_dist_from_origin = 1000 # (in meters)
    ctrl.delay_params.max_dist_from_origin = 2000 # (in meters)

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

    ctrl.outage_detect = False
    ctrl.outage_threshold_noisefactor = 1/(p.zc_len)
    

    ctrl.prop_correction = True
    ctrl.pc_step_wait = 0
    ctrl.pc_b, ctrl.pc_a = lib.hipass_avg(6)
    ctrl.pc_avg_thresh = float('inf') # If std of N previous TOx samples is above this value, then\
    ctrl.pc_std_thresh = float(80) # If std of N previous TOx samples is above this value, then\
                     # no PC is applied (but TOy is still calculated)
    
    ctrl.saveall = True


    ncount_lo = 12
    ncount_hi = ctrl.nodecount-2+1
    step = 2
    ntot = math.floor((ncount_hi - ncount_lo - 1)/abs(step))

    #cdict = {
    #    'quiet_nodes':[x for x in range(ncount_lo, ncount_hi,step)]
    #    }
    #pdict = {}
    
    cdict = {
        'prop_correction':[False, True , True ],
    }
    pdict = {
        'scfdma_precode':[False, False, True ]
    }

    return ctrl, p, cdict, pdict

#------------------------
def main():


    #data = np.array([65,70,71,74,122,128,281,292,297,3,6,10])
    #data = lib.minimize_distance(data, 300)
    #np.random.seed(2044239)
    #np.random.shuffle(data)
    #lib.cluster_1d(data)


    #x = np.arange(25).reshape(-1,5)
    #indexes = np.array([0,1,3])
    #lib.reduce_square_matrix(x, indexes)
    #exit()


    ctrl, p, cdict, pdict = dec_wrap2()
    init_cdict = {**cdict, **pdict}
    sim = SimWrap(ctrl, p, cdict, pdict)

    #sim.conv_offset_limits = [x*4 for x in sim.conv_offset_limits]
    sim.ctrl.saveall = False
    sim.simulate()
    sim.post_sim_plots()
    exit()

    sim.set_all_nodisp()
    sim.make_plots = False
    sim.repeat = 500
    sim.ctrl.rand_init = True

    simstr = 'all'
    tsims = sim.total_sims(simstr)
    #sim.simmany(simstr);# dates = [dates[0], dates[-1]]
    #alldates = db.fetch_last_n_dates(tsims);

    
    alldates = db.fetch_dates([20160723012902720, 20160724030205810]) # 4.5k sims july 24 
    #alldates = db.fetch_dates([20160716122731390, 20160715204752242]) # 900sim quiet 1km
    #alldates = db.fetch_dates([20160718215702723, 20160718114958378]) # 540sim quiet 1.414 km
    #dates = [alldates[0], alldates[-1]]

    lib.options_convergence_analysis(alldates, init_cdict, write=True)


    #graphs.scatter_range(dates, ['quiet_nodes', 'good_link_ratio']); graphs.show()
    #graphs.scatter_range(dates, ['quiet_nodes', 'theta_drift_slope_std']); graphs.show()
    #graphs.scatter_range(dates, ['quiet_nodes', 'theta_ssstd']); graphs.show()


class SimWrap(lib.Struct):
    """Simulation object that handles multi-iteration of runsim(). Also dumps results to disc if needed."""

    force_calculate = False # Forces the update of all values before starting the simulation
    make_CFO = False
    show_CFO = False 
    make_TO = True
    show_TO = False 
    make_grid = True
    show_grid = False
    make_cat = True
    show_cat = True

    show_SNR = True
    show_siglen = True
    show_bary = False
    show_elapsed = True
    show_conv = True
    repeat = 1
    last_msg_len = 0
    conv_min_slope_samples = 20
    conv_offset_limits = [-3.4, 1.8]

    cdict = dict()
    pdict = dict()

    def __init__(self, ctrl, p, cdict=None, pdict=None, prep_bary=False):
        """Prepares ctrl & p for simulation"""
        self.add(p=p)
        self.add(ctrl=ctrl)

        self.ctrl.f_samp = p.f_samp
        self.p.update()
        self.ctrl.update()
        self.prep_bary = prep_bary

        if prep_bary:
            lib.barywidth_map(self.p, reach=self.ctrl.bmap_reach , scaling_fct=self.ctrl.bmap_scaling , force_calculate=self.force_calculate, disp=self.show_bary)
        if self.force_calculate:
            self.p.update()
            self.ctrl.update()
        if cdict is not None:
            self.cdict=cdict
        if pdict is not None:
            self.pdict=pdict
            self.ctrl.f_samp = self.p.f_samp

        base_avg_amp = np.float64((np.sum(np.abs(self.p.analog_sig)))/len(self.p.analog_sig))
        self.ctrl.trans_amp = lib.db2amp(ctrl.trans_power)*base_avg_amp

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

    def snr_stats(self):
        """Returns the SNR stats (max, med, min), snr_grid"""
        Prx = self.ctrl.trans_power - self.ctrl.delay_params.pathloss_grid
        snr_grid = Prx - self.ctrl.noise_power

        # Grab the upper triangular entries in snrgrid
        x = snr_grid[np.triu_indices(snr_grid.shape[0], m=snr_grid.shape[1], k=1)]
        return (max(x), np.median(x), min(x)) , snr_grid
  
    def set_all_nodisp(self):
        """All display values are set to false"""
        self.show_CFO = False
        self.show_TO = False
        self.show_grid = False
        self.show_cat = False
        self.show_SNR = False
        self.show_siglen = False
        self.show_bary = False
        self.show_elapsed = False
        self.ctrl.display = False
        self.show_conv = False

    def simulate(self):
        """Simulate and run post-sim stuff, such as graphs or output saving"""

        # Some display stuff
        msg = ''
        if self.show_siglen:
            siglen_value = "{:.2f}".format(len(self.p.analog_sig)/self.ctrl.basephi)
            msg = 'Sync signal length: ' + siglen_value + ' basephi' + '    '

        if self.show_SNR:
            stats = self.snr_stats()[0]
            stats_str = '(' + ', '.join(['{0:.2f}'.format(k) for k in stats]) + ') dB'
            msg += 'SNR (max, median, min): ' + stats_str

        if self.show_elapsed:
            tf = time.clock
            t0 = tf()

        if msg!='': print(msg)

        # Exec the simulation and save the output
        runsim(self.p, self.ctrl)

        # Fix bias removal opt (it contains numbers, should be a bool)
        if self.ctrl.saveall and self.ctrl.bias_removal != False:
            self.ctrl.bias_removal = True
        
        self.ctrl.add(**lib.eval_convergence(self, show_eval_convergence=self.show_conv, conv_min_slope_samples=self.conv_min_slope_samples, conv_offset_limits=self.conv_offset_limits))
        self.ctrl.date = lib.build_timestamp_id();

        savedict = self.ctrl.__dict__.copy()
        savedict['delay_grid'] = self.ctrl.delay_params.delay_grid
        try:
            for var in NOSAVELIST:
                del savedict[var]
        except KeyError:
            pass
        
        db.add(savedict)

        if self.show_elapsed:
            print('Elapsed: ' + "%2.2f"%(tf()-t0) + ' seconds.')

        return self.ctrl.date

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
        dates = []

        # Main sim loop
        while next(assign_fct()):
            for k in range(self.repeat):
                avg_time = (tf()-t0)/count if count!=0 else 0
                msg = "Iteration " + str(count).zfill(len(str(tsims))) + " of " + str(tsims)+ '. ' 
                msg += "%2.2f"%(count*100/tsims) + "% done -- avg time:" +  "%2.2f"%avg_time + ' sec/iteration'
                oprint(msg)
                self.ctrl.update()
                curdate = self.simulate()
                dates.append(curdate)
                count += 1
        oprint(' '*len(msg))
        print('Done simmany in ' + "%2.2f"%(tf()-t0) + ' seconds.')

        # Logging the dateid set
        dates = np.array(dates)
        lib.appendlog('Done ' + str(tsims) + ' simulations: ' +\
                      str(dates[0]) + ' to ' + str(dates[-1]))

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
            
            if self.prep_bary:
                self.update_bary()

            yield True

        yield False # When both dicts are empty,  return false.

    def assign_next_one(self):
        """Makes a generator of both ctrl and p. It iterates through cdict and pdict
        one item at a time

        Note: Shorter lists are deleted when done. Concerned variables will keep that final value
        for future iteration"""
        while self.cdict or self.pdict:
            for d, obj in zip([self.cdict, self.pdict], [self.ctrl, self.p]):
                todel = []
                for key, lst in d.items():
                    while lst: 
                        obj.change(key, lst.pop())
                        if self.prep_bary:
                            self.update_bary()
                        yield True
                    if not lst: 
                        todel.append(key)
                for key in todel: 
                    del d[key]

        yield False # When both dicts are empty,  return false.

    def post_sim_plots(self, **kwargs):
        """Wrapper for the plotlib fct"""
        return graphs.post_sim_graphs(self, **kwargs)

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



if __name__ == '__main__':
    main()

