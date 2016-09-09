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
import traceback

from numpy import pi


from sim_channel import runsim, SimControls



NOSAVELIST = [
    'delay_params',
    'TO',
    'CFO'
    ]

#-------------------------
def dec_wrap2():
    """Declaration function. It initializes all the relevant parameters to the simulatiions """

    # Broadcast parameters
    p = lib.SyncParams()
    p.zc_len = 31 # Length of the ZC sequence to use
    p.f_symb = 30.72e6 # "true" sampling rate
    p.f_samp = p.f_symb*4 # oversampling rate. SHould be an integer factor of p.f_samp. This factor is the oversampling factor
    p.repeat = 1 # Repeat the synchronization signal
    p.spacing_factor = 1 # Spacing between the raisedcosines. A factor of 1 uses the standard definition, where 
                         # one symbol is placed every 
    p.power_weight = 2 # Exponent \gamma 
    p.full_sim = True # Leave true
    p.bias_removal = False # Utilization of the bias removal algorithm
    p.ma_window = 1 # If bigger than 1, applies a moving average on the crosscorrelation R_yz
    p.train_type = 'chain' # 'single': only one ZC sequence makes up the synchronization systemm
                           # 'chain' : pair of ZC sequence with opposite root parameter
    p.crosscorr_type = 'match_decimate' # don't touch
    p.match_decimate_fct = lib.downsample # lib.downsample corresponds to a standard idealized analog-digital sampler
    p.peak_detect = 'wavg'  # 'wavg':   Performs a weighted average on the crosscorrelation to determine the location of the
                            #           synchronization signal
                            # 'argmax': Takes the max of the crosscorrelation instead of a weighted average
    p.pulse_type = 'rootraisedcosine' # Analog shaping pulse
    p.plen = 31   # Length of the raisedcosine pulseshape
    p.rolloff = 0.2 # Raisedcosine rolloff factor. A value of 0 corresponds to a normalized sinc
    p.central_padding = 0 # Pad zeros between the pair of ZC sequences. As a fraction of zpos length
    p.scfdma_precode = True # Apply SCFDMA?
    p.scfdma_L = 4  # SC-FDMA factor
    p.scfdma_sinc_len_factor = p.scfdma_L # Used for the demodulation of SC-FDMA in a decentralized setting

    #------------------------
    # Simulation parameters
    ctrl = SimControls()
    ctrl.steps = 40 # Approximately how many T_0 synchronization steps will be executed
    ctrl.basephi = 40000 # How many oversamples per period T_0? T_0 = basephi*p.f_samp 
    ctrl.nodecount = 35 # Number of nodes to simulate
    ctrl.display = True # Display stuff for a single simulation
    ctrl.static_nodes = 0 # Nodes that broadcast but do not synchronize
    ctrl.quiet_nodes = 0 # Nodes that synchronize but do not broadcast
    ctrl.quiet_selection = 'random' # How are quiet nodes assigned
    #ctrl.quiet_selection = 'kmeans' 
    #ctrl.quiet_selection = 'contention' # Note this renders ctrl.quiet_nodes uiseless, and requires the use of outage detectection

    # Parameters specific to the contention method (sensing)
    ctrl.qc_threshold = 5 # As a factor of the outage threshold
    ctrl.qc_steps = 3

    # Initialization parameters
    ctrl.CFO_step_wait = float('inf') # Use float('inf') to never correct for CFO. CFO CORRECTION DEPRECATED!
    ctrl.TO_step_wait = 4 # How many synchronization steps to wait before broadcasting. In factors of T_0
    ctrl.max_start_delay = 7 # nodes are onlined between t \in [0, max_start_delay]. In factor of T_0
    ctrl.use_ringarr = True # Use True unless you want to blow up your RAM
    ctrl.theta_bounds = [0,1] # bounds for the uniform distribution of \theta
    ctrl.deltaf_bound = 3e-2 # bounds for the uniform distribution of \delta f between the nodes. 
    ctrl.rand_init = False # Randomly initiate values?
    ctrl.non_rand_seed = 1238819 # If rand_init = False, the 'random' values in the simulation will be initiated with this seed
    ctrl.epsilon_TO = 0.5
    ctrl.noise_power = -101 + 9 # Reception thermal noise. -101 + 9 : thermal noise + receiver noise amplification.
    #ctrl.noise_power = float('-inf') # Use if you want a noiseless simulation

    # Multipath parameters
    ctrl.delay_params = lib.DelayParams(lib.delay_pdf_3gpp_exp) # corresponds to 3gpp specifications
    ctrl.delay_params.taps = 50 # How many multipath taps?

    # The next variable is badly named. it correponds to half the side of a square area
    # ctrl.max_dist_from_origin = 500 corresponds to an area of (1000m)^2
    ctrl.max_dist_from_origin = 500 # (in meters)

    # Half-duplexing method. Use at your own risk
    ctrl.half_duplex = False
    ctrl.hd_slot0 = 0.3 # in terms of phi
    ctrl.hd_slot1 = 0.7 # in terms of phi
    ctrl.hd_block_during_emit = True
    ctrl.hd_block_extrawidth = 0 # as a factor of offset (see runsim to know what is offset)

    # Variable adjustement window size. Use at your own risk
    ctrl.var_winlen = False
    ctrl.vw_minsize = 5 # as a factor of len(p.analog_sig)
    ctrl.vw_lothreshold = 0.1 # winlen reduction threshold
    ctrl.vw_hithreshold = 0.1 # winlen increase threshold
    ctrl.vw_lofactor = 1.5 # winlen reduction factor
    ctrl.vw_hifactor = 2 # winlen increase factor

    # Outage detection. Only broadcast of "near" outage. E-mail david.tetreault-laroche@mail.mcgill.ca for more 
    # explanation on how it works
    ctrl.outage_detect = False # thesis
    #ctrl.outage_detect = True # INTERD
    ctrl.outage_threshold_noisefactor = 1/(p.zc_len)*2

    # COrrect propagation delay by applying some filter.
    # lib.highpass_avg(6) corresponds to a filter of length Q=6, as described in my thesis
    ctrl.prop_correction = True
    ctrl.pc_step_wait = 0
    ctrl.pc_b, ctrl.pc_a = lib.hipass_avg(6) #THESIS
    ctrl.pc_avg_thresh = float('inf') # If std of N previous TOx samples is above this value, then
    ctrl.pc_std_thresh = float(80) # If std of N previous TOx samples is above this value, then
                                   # no PC is applied (but TOy is still calculated)
    
    # Save values to disk?
    ctrl.saveall = True

    #-------------
    # To run multiple simulations in bulk, a dictionary of modification to ctrl (cdict) and modifications to p
    # (pdict) can be used to iterate over multiple configurations
    # Note that any parameter to p or ctrl can be iterated this way

    # For example, this will iterate over 5 configurations, from right to left. Each iteration assigns 
    # the value to the property and runs the simulations. See SimWrap for more details
    cdict = {
        'prop_correction':[False, True , True , True , True ],
    }
    pdict = {
        'scfdma_precode': [False, False, False, True , True ],
        'bias_removal':   [False, False, True , False, True]
    }


    #cdict = {
    #    'noise_power':[x for x in range(-120,-91,2)]
    #    }
    #pdict = {}

    #cdict = {
    #}
    #pdict = {}

    return ctrl, p, cdict, pdict

def dec_r12():
    """Declaration function for the Release 12 method. Specific to word done at InterDigital""" 

    ctrl, p, cdict, pdict = dec_wrap2()
    p.train_type = 'single' # Type of training sequence
    p.peak_detect = 'argmax' 

    ctrl.quiet_selection = 'contention' # Note this renders ctrl.quiet_nodes uiseless, and requires the use of outage detectection
    ctrl.qc_threshold = 1 # As a factor of the outage threshold
    ctrl.qc_steps = 1
    ctrl.TO_step_wait = 1
    ctrl.epsilon_TO = 1
    ctrl.prop_correction = False
    ctrl.outage_threshold_noisefactor = 1/(p.zc_len)

    return ctrl, p, cdict, pdict


#------------------------
def main():
    """This will be executed if you run "python wrapper.py" in a terminal"""


    # Declare the variable and build the simulaton object
    ctrl, p, cdict, pdict = dec_wrap2()
    init_cdict = {**cdict, **pdict}
    sim = SimWrap(ctrl, p, cdict, pdict)

    
    # Uncomment this if you want to do only ONE simulation
    #sim.ctrl.saveall = False
    #sim.simulate()
    #sim.post_sim_plots()
    #exit()

    # Utilize this if you want to do MANY simulations.
    # see SimWrap for details
    sim.set_all_nodisp() # kill off display
    sim.make_plots = False # Don't make a plot for a single simulation
    sim.repeat = 30 # Repeat each "column" in cdict/pdict 
    sim.ctrl.rand_init = True # Force the use of random initialization
    simstr = 'all' # Don't touch
    tsims = sim.total_sims(simstr) # Calculate the total number of simulations to be done
    sim.simmany(simstr) # actually run the many simulations

    #alldates = db.fetch_last_n_dates(tsims); # Fetch the last batch of simulations

    #lib.options_convergence_analysis(alldates, init_cdict, write=True) # Redo the covergence analysis


    # Graphs produced for master's thesis.
    #thygraphs.highlited_regimes();
    #thygraphs.thesis_cavg_vs_distance();
    #thygraphs.thesis_cavg_vs_zc();
    #thygraphs.thesis_cavg_vs_nodecount(); 
    #thygraphs.thesis_cavg_vs_noise(); 
    #thygraphs.interd_quiet_grids(); 
    #thygraphs.interd_compare_quiet(); 
    #thygraphs.interd_cavg_vs_distance();
    #thygraphs.interd_cavg_vs_nodecount(); 
    #thygraphs.interd_dpll_vs_r12(); 
    #exit()

    #--------------------------
    # Data for different simulations. They require to be connected to a specific database, and should produce an error

    #THESIS
    #alldates = db.fetch_dates([20160723012902720, 20160724030205810]) # 4.5k sims july 24 
    #alldates += db.fetch_dates([20160803172306291, 20160804023828227 ]) # 3k sims aug3
    #alldates = db.fetch_dates([20160823191214935, 20160823194541680]) # 360k sims nodecount
    #alldates = db.fetch_dates([20160824001859759, 20160824034043274]) # nodecount vs cavg
    #alldates = db.fetch_dates([20160824125623027, 20160824143207623]) # dist vs cavg
    #alldates += db.fetch_dates([20160824155429863, 20160824170025866]) # dist vs cavg (extra)
    #alldates = db.fetch_dates([20160825010048334, 20160825040232662]) # zc vs cavg 
    #alldates = db.fetch_dates([20160830143035701, 20160830162947428]) # zc vs noise_power 
    
    #INTERD
    #alldates = db.fetch_dates([20160716122731390, 20160715204752242]) # 900sim quiet 1km
    #alldates = db.fetch_dates([20160718215702723, 20160718114958378]) # 540sim quiet 1.414 km
    #alldates = db.fetch_dates([20160807005022420, 20160807101308389]) # 1200sim quiet 2 km
    #alldates = db.fetch_dates([20160812174745548, 20160813032432642]) # 1200sim kmeans
    #alldates = db.fetch_dates([20160822172521531, 20160822215536865]) # 960sims quiet compare
    #alldates = db.fetch_dates([20160825101515414, 20160825135628487]) # cavg vs nodecount
    #alldates = db.fetch_dates([20160825141108531, 20160825183253474]) # cavg vs dist
    #alldates = db.fetch_dates([20160825232333852, 20160826024432633]) # cavg vs zclen
    #alldates = db.fetch_dates([20160828200017740, 20160828205450849]) # r12 vs dpll
    #alldates = db.fetch_dates([20160829122809568, 20160829152128098]) # plain dpll vs contention

    #dates = [alldates[0], alldates[-1]]
    #graphs.scatter_range(dates, ['max_dist_from_origin', 'good_link_ratio'], multiplot='quiet_selection'); graphs.show()
    #graphs.scatter_range(dates, ['nodecount', 'good_link_ratio']); graphs.show()
    #graphs.scatter_range(dates, ['nodecount', 'good_link_ratio'], multiplot='peak_detect'); graphs.show()
    #print(db.fetchone(alldates[0], ''))
    #lib.update_db_conv_metrics(alldates, conv_offset_limits=sim.conv_offset_limits); exit()
    #graphs.scatter_range(dates, ['zc_len', 'good_link_ratio']); graphs.show(); exit()
    #graphs.scatter_range(dates, ['quiet_nodes', 'good_link_ratio']); graphs.show()
    #graphs.scatter_range(dates, ['quiet_nodes', 'cluster_count_single']); graphs.show()
    #graphs.scatter_range(dates, ['quiet_nodes', 'cluster_count']); graphs.show()


class SimWrap(lib.Struct):
    """Simulation object that helps manage the execution of sim_channel.runsim()
    """

    force_calculate = False # Forces the update of all values before starting the simulation
    make_CFO = False # Make the CFO evolution graph
    show_CFO = False 
    make_TO = True # Make the TO evolution graph
    show_TO = False 
    TO_show_clusters = True # Show the result of cluster estimations in the TO graph

    make_grid = True # Make the grid display graph
    show_grid = False 

    make_cat = True # Make the concatenated graph of TO, CFO and GRID, if they are enabled
    show_cat = True

    show_SNR = True # Display statistics on received SNR across the network
    show_siglen = True # Display the length of the sync signal in oversamples
    show_bary = False # DEPRECATED. leave false
    show_elapsed = True # Show elapsed time
    show_conv = True # Show convergence analysis after simulation
    repeat = 1 # repeat each configuration in simmmany
    last_msg_len = 0 # Leave at 0
    conv_min_slope_samples = 20 # How many samples to do the convergence analyssi on?
    conv_offset_limits = lib.DEFAULT_OFFSET_LIMITS.copy() # defaults for the calculation for C

    cdict = dict() # Empty defaults
    pdict = dict()

    def __init__(self, ctrl, p, cdict=None, pdict=None, prep_bary=False):
        """Prepares ctrl & p for simulation by running the parameter updates when appropriate"""
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
        """All display values are set to false. Useful for simmany"""
        self.show_CFO = False
        self.show_TO = False
        self.show_grid = False
        self.show_cat = False
        self.show_SNR = False
        self.show_siglen = False
        self.show_bary = False
        self.show_elapsed = False
        self.show_conv = False
        self.ctrl.display = False

    def simulate(self):
        """Executed a single simulation with current p and ctrl values.
        Also runs run post-sim stuff, such as graphs or output saving, if enabled"""

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

        # Revert bias removal to a boolean (it contains numbers, should be a bool)
        if self.ctrl.saveall and self.ctrl.bias_removal != False:
            self.ctrl.bias_removal = True
        
        # Add convergence analysis to the saving dictionary
        self.ctrl.add(**lib.eval_convergence(self, show_eval_convergence=self.show_conv, conv_min_slope_samples=self.conv_min_slope_samples, conv_offset_limits=self.conv_offset_limits))
        self.ctrl.date = lib.build_timestamp_id();

        # Save p and ctrl to DB. Note that p is already in ctrl at this point
        savedict = self.ctrl.__dict__.copy()
        savedict['delay_grid'] = self.ctrl.delay_params.delay_grid
        try:
            for var in NOSAVELIST:
                del savedict[var]
        except KeyError:
            pass
        db.add(savedict)

        # Display stuff
        if self.show_elapsed:
            print('Elapsed: ' + "%2.2f"%(tf()-t0) + ' seconds.')

        return self.ctrl.date

    def simmany(self, assign_method='all'):
        """Simulates many simulations according to the dictionaries cdict and ctrl
        Assign method: 'all' or 'any'. See SimWrap.assign_next_all and SimWrap.assign_next_any for description"""
        # Check if using random initialization
        if not self.ctrl.rand_init:
            warnings.warn('Not using random initialization')

        # Make sure 'saveall' option is enabled
        notsaveall_string = "Some simulations have the parameter set saveall parameter set to False"
        if 'saveall' in self.cdict:
            if False in self.cdict['saveall']:
                warnings.warn(notsaveall_string)
        elif not self.ctrl.saveall:
            warnings.warn(notsaveall_string)

        # Select assign method
        if assign_method == 'all':
            assign_fct = self.assign_next_all
        elif assign_method == 'one':
            assign_fct = self.assign_next_one
        else:
            raise ValueError('Invalid assign method')

        # Print some time update stuff
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
                count += 1
                try:
                    curdate = self.simulate()
                # If an exception, print it but continue executing
                except Exception:
                    print('\n')
                    traceback.print_exc()
                    print('\n')
                    continue
                dates.append(curdate)
        oprint(' '*len(msg))
        print('Done simmany in ' + "%2.2f"%(tf()-t0) + ' seconds.')

        # Logging the dateid set
        dates = np.array(dates)
        lib.appendlog('Done ' + str(tsims) + ' simulations: ' +\
                      str(dates[0]) + ' to ' + str(dates[-1]))
        return dates

    def assign_next_all(self):
        """Makes a generator of both ctrl and p. It iterates through cdict and pdict over all
        lists at the same time, from right to left.

        Note: Shorter lists are deleted when done. The concerned variable will keep that final value
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
        one item at a time.j

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

