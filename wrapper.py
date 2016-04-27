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
import thygraphs

from numpy import pi


from sim_channel import runsim, SimControls




#-------------------------
def dec_wrap1():
    """Chained ZC sequences"""
    p = lib.SyncParams()
    p.zc_len = 64
    p.plen = 31

    p.rolloff = 0.2
    p.f_samp = 1e6
    p.f_symb = 0.25e6
    p.repeat = 1
    p.spacing_factor = 2 # CHANGE TO TWO!

    p.power_weight = 4
    p.full_sim = True
    p.bias_removal = True
    p.ma_window = 1 # number of samples to average in the crosscorr i.e. after analog modulation
    p.train_type = 'chain' # Type of training sequence
    p.crosscorr_fct = 'analog' 
    p.pulse_type = 'raisedcosine'
    p.central_padding = 0 # As a fraction of zpos length


    ctrl = SimControls()
    ctrl.steps = 60 # Approx number of emissions per node
    ctrl.basephi = 6000 # How many samples between emission
    ctrl.display = True # Show stuff in the console
    ctrl.keep_intermediate_values = False # Needed to draw graphs
    ctrl.nodecount = 15 # Number of nodes
    ctrl.static_nodes = 0
    ctrl.CFO_step_wait = float('inf') # Use float('inf') to never correct for CFO
    ctrl.TO_step_wait = 0
    ctrl.max_start_delay = 0 # In factor of basephi

    ctrl.theta_bounds = [0.3,0.7] # In units of phi
    #ctrl.theta_bounds = [0.48,0.52] # In units of phi
    #ctrl.theta_bounds = [0.5,0.5] # In units of phi
    #ctrl.deltaf_bound = 3e-6
    ctrl.deltaf_bound = 3e-2
    ctrl.noise_var = 0
    ctrl.rand_init = False
    ctrl.non_rand_seed = 11231231 # Only used if rand_init is False

    ctrl.bmap_reach = 3e-6
    ctrl.bmap_scaling = 100

    ctrl.cfo_mapper_fct = lib.cfo_mapper_order2
    ctrl.CFO_processing_avgtype = 'reg'
    ctrl.CFO_processing_avgwindow = 4
    ctrl.max_CFO_correction = 1e-6 # As a factor of f_symb

    ctrl.delay_params = lib.DelayParams(lib.delay_pdf_exp)
    ctrl.delay_params.taps = 1
    ctrl.delay_params.max_dist_from_origin = 500 # in meters
    ctrl.delay_params.p_sigma = 0.2 # Paths sigma

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
    

    ctrl.prop_correction = True
    ctrl.pc_step_wait = 0
    #ctrl.pc_b, ctrl.pc_a = lib.hipass_filter4(11, pi/1000, 0.5)
    #ctrl.pc_b, ctrl.pc_a = lib.hipass_filter4(11, pi/4, 0.1)
    #ctrl.pc_b, ctrl.pc_a = lib.hipass_filter4(11, pi/4, 1)
    #ctrl.pc_b, ctrl.pc_a = lib.hipass_filter3(20)
    #ctrl.pc_b, ctrl.pc_a = lib.hipass_filter2(8)
    #ctrl.pc_b, ctrl.pc_a = lib.hipass_filter1(8)
    ctrl.pc_b, ctrl.pc_a = lib.hipass_avg(20)
    ctrl.pc_std_thresh = float('inf')
    
    ctrl.saveall = True


    cdict = {
        'nodecount':[x for x in range(5,15)]
        }

    pdict = {}

   

    return ctrl, p, cdict, pdict

def dec_wrap2():
    """Single ZC sequence with Decimation"""
    p = lib.SyncParams()
    p.zc_len = 73
    p.plen = 61

    p.rolloff = 0.2
    #p.f_samp = 4e6
    #p.f_symb = 1e6
    p.f_symb = 30.72e6
    p.f_samp = p.f_symb*8
    p.repeat = 1
    p.spacing_factor = 1 # CHANGE TO TWO!

    p.power_weight = 2
    p.full_sim = True
    p.bias_removal = True
    p.ma_window = 1 # number of samples to average in the crosscorr i.e. after analog modulation
    p.train_type = 'single' # Type of training sequence
    p.crosscorr_type = 'match_decimate' 
    p.match_decimate_fct = lib.md_clkphase
    p.peak_detect = 'wavg' 
    p.pulse_type = 'rootraisedcosine'
    p.central_padding = 0 # As a fraction of zpos length


    ctrl = SimControls()
    ctrl.steps = 80 # Approx number of emissions per node
    ctrl.basephi = 6000 # How many samples between emission
    ctrl.display = True # Show stuff in the console
    ctrl.keep_intermediate_values = False # Needed to draw graphs
    ctrl.nodecount = 15 # Number of nodes
    ctrl.static_nodes = 0
    ctrl.CFO_step_wait = float('inf') # Use float('inf') to never correct for CFO
    ctrl.TO_step_wait = 4
    ctrl.max_start_delay = 30 # In factor of basephi

    ctrl.theta_bounds = [0.3,0.7] # In units of phi
    #ctrl.theta_bounds = [0.48,0.52] # In units of phi
    #ctrl.theta_bounds = [0.5,0.5] # In units of phi
    ctrl.theta_bounds = [0,1] # In units of phi
    ctrl.deltaf_bound = 3e-2
    #ctrl.deltaf_bound = 0
    ctrl.rand_init = False
    ctrl.epsilon_TO = 0.5
    ctrl.non_rand_seed = 11231231 # Only used if rand_init is False
    #ctrl.noise_power = float('-inf')
    ctrl.noise_power = -101 + 9 # in dbm

    ctrl.delay_params = lib.DelayParams(lib.delay_pdf_exp)
    ctrl.delay_params.taps = 4
    ctrl.delay_params.max_dist_from_origin = 250 # (in meters)
    ctrl.delay_params.p_sigma = 0.02 # Paths sigma

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
    

    ctrl.prop_correction = False
    ctrl.pc_step_wait = 0
    #ctrl.pc_b, ctrl.pc_a = lib.hipass_filter4(11, pi/1000, 0.5)
    #ctrl.pc_b, ctrl.pc_a = lib.hipass_semicirc_zeros(11, pi/4, 0.1)
    #ctrl.pc_b, ctrl.pc_a = lib.hipass_filter4(11, pi/4, 1)
    #ctrl.pc_b, ctrl.pc_a = lib.hipass_remez(20)
    #ctrl.pc_b, ctrl.pc_a = lib.hipass_butter(8)
    #ctrl.pc_b, ctrl.pc_a = lib.hipass_cheby(8)
    ctrl.pc_b, ctrl.pc_a = lib.hipass_avg(15)
    ctrl.pc_avg_thresh = float('inf') # If std of N previous TOx samples is above this value, then\
    ctrl.pc_std_thresh = float(50) # If std of N previous TOx samples is above this value, then\
                     # no PC is applied (but TOy is still calculated)
    
    ctrl.saveall = True


    cdict = {
        'nodecount':[x for x in range(10,121,10)]
        #'nodecount':[x for x in range(3,8)]
        }

    pdict = {}
    #pdict = {'match_decimate_fct':[lib.md_clkphase, lib.md_energy]}

    return ctrl, p, cdict, pdict

#------------------------
def main_thesis():


    ctrl, p, cdict, pdict = dec_wrap1()
    sim = SimWrap(ctrl, p, cdict, pdict)


    graphs.crosscorr(p); graphs.show(); exit()

    #graphs.freq_response(ctrl.pc_b, ctrl.pc_a); graphs.show(); exit()
    #graphs.delay_pdf(ctrl); graphs.show(); exit()
    #graphs.delay_grid(ctrl); graphs.show(); exit()
    #graphs.node_multitaps(ctrl); graphs.show(); exit()
    sim.ctrl.keep_intermediate_values = True
    sim.show_CFO = False
    sim.simulate()
    sim.eval_convergence()
    sim.post_sim_plots()
    exit()


    # Figures and output names!
    thygraphs.zero_padded_crosscorr(); exit()

def main_interd():


    ctrl, p, cdict, pdict = dec_wrap2()
    sim = SimWrap(ctrl, p, cdict, pdict)


    #print(lib.thy_ssstd(ctrl))
    #graphs.freq_response(ctrl.pc_b, ctrl.pc_a); graphs.show(); exit()
    #graphs.delay_pdf(ctrl); graphs.show(); exit()
    #graphs.delay_grid(ctrl); graphs.show(); exit()
    #sim.ctrl.keep_intermediate_values = True
    #sim.show_CFO = False
    #sim.simulate()
    #sim.eval_convergence()
    #sim.post_sim_plots()
    #exit()

    sim.set_all_nodisp()
    sim.ctrl.keep_intermediate_values = True
    sim.make_plots = False
    sim.repeat = 10
    sim.ctrl.rand_init = True




    simstr = 'one'
    tsims = sim.total_sims(simstr)
    dates = sim.simmany(simstr); #dates = [dates[0], dates[-1]]
    #dates = [20160316135854599,  20160316135906136]

    dates = db.fetch_last_n_dates(tsims); dates = [dates[0], dates[-1]]
    collist = ['nodecount', 'good_link_ratio']
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
    #make_plots = True
    show_CFO = True # only works if make_plots is True
    show_TO = True # only works if make_plots is True
    show_SNR = True
    show_siglen = True
    show_bary = False
    show_elapsed = True
    show_eval_convergence = True
    repeat = 1
    last_msg_len = 0

    cdict = dict()
    pdict = dict()

    #Convergence criterions
    conv_eval_cfo = False
    conv_min_slope_samples = 5 # Minimum # of samples to take for slope eval
    conv_offset_limits = [-3.4, 1.8] # In micro seconds
    


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

    def update_conv_criterions(self):
        """Updates the time offset limits from the p object"""
        pass

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
        self.show_SNR = False
        self.show_siglen = False
        self.show_bary = False
        self.show_elapsed = False
        self.ctrl.display = False
        self.show_eval_convergence = False

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
        self.ctrl.add(**self.eval_convergence())
        self.ctrl.date = lib.build_timestamp_id();
        db.add(self.ctrl.__dict__)

        
        if self.show_elapsed:
            print('Elapsed: ' + "%2.2f"%(tf()-t0) + ' seconds.')

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
                print(self.ctrl.nodecount)
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
            
            if self.prep_bary:
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
                        if self.prep_bary:
                            self.update_bary()
                        yield True
                    if not lst: 
                        todel.append(key)
                for key in todel: 
                    del d[key]
            

        yield False # When both dicts are empty,  return false.

    def post_sim_plots(self):
        """Wrapper for the plotlib fct"""
        graphs.post_sim_graphs(self)

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

    def eval_convergence(self):
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
        flist = self.ctrl.deltaf_inter

        output['theta_drift_slope_avg'], output['theta_drift_slope_std'] = drift_eval(tlist)
        if self.conv_eval_cfo:
            output['deltaf_drift_slope_avg'], output['deltaf_drift_slope_std'] = drift_eval(flist)

        # Evaluate communication capabilites between all nodes 
        theta = self.ctrl.theta
        prop_delay_grid = self.ctrl.delay_params.delay_grid
        offset_grid = np.tile(theta.reshape(-1,1), theta.shape[0])
        offset_grid +=  -1*offset_grid.T
        offset_grid +=  -1*prop_delay_grid
        offsets = offset_grid[~np.eye(offset_grid.shape[0], dtype=bool)]
        linkcount = len(offsets)

        lo, hi = [k*1e-6*self.p.f_samp for k in self.conv_offset_limits]
        good_links = ((offsets>lo) & (offsets<hi)).sum()

        output['good_link_ratio'] = good_links/linkcount

        
        if self.show_eval_convergence:
            for key, item in sorted(output.items()):
                print(key + ": " + str(item))

        return output

    

if __name__ == '__main__':
    main_interd()
    #main_thesis()





















    #  LocalWords:  avgwindow
