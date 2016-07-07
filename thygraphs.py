#!/usr/bin/env python

import lib
import plotlib as graphs
import numpy as np
from wrapper import SimWrap
from sim_channel import SimControls

from numpy import pi



OUTPUT_FOLDER = 'graphs/latex_figures/' # Don't forget the theta_rangeailing slash


def ml_pinit():
    """Discrete non pulse-shaped sequence"""
    p = lib.SyncParams()
    p.zc_len = 100
    p.plen = 31
    p.rolloff = 0.2
    p.f_samp = 4e6
    p.f_symb = 1e6
    p.repeat = 1
    p.spacing_factor = 2
    p.power_weight = 4
    p.full_sim = False
    p.bias_removal = False
    p.ma_window = 1 # number of samples i.e. after analog modulation
    p.crosscorr_fct = 'analog'
    p.train_type = 'single'
    p.update()
    return p 

def ml_pinit_no_pulse_shape():
    """Discrete non pulse-shaped sequence"""
    p = lib.SyncParams()
    p.zc_len = 100
    p.plen = 1
    p.rolloff = 0.2
    p.f_samp = 1e6
    p.f_symb = 1e6
    p.repeat = 1
    p.spacing_factor = 1
    p.power_weight = 4
    p.full_sim = False
    p.bias_removal = False
    p.ma_window = 1 # number of samples i.e. after analog modulation
    p.crosscorr_fct = 'analog'
    p.theta_rangeain_type = 'single'
    p.update()
    return p 

def pinit64_no_pulse_shape():
    p = lib.SyncParams()
    p.zc_len = 63
    p.plen = 1
    p.rolloff = 0.2
    p.f_samp = 1e6
    p.f_symb = 1e6
    #p.f_symb = 0.25e6
    p.repeat = 1
    p.spacing_factor = 1 # CHANGE TO TWO!
    p.power_weight = 4
    p.full_sim = True
    p.bias_removal = False
    p.ma_window = 1 # number of samples to average in the crosscorr i.e. after analog modulation
    p.train_type = 'chain' # Type of training sequence
    p.crosscorr_fct = 'analog' 
    p.pulse_type = 'raisedcosine'
    p.central_padding = 0 # As a fraction of zpos length
    p.CFO = 1
    p.update()
    return p

def dec_regimes():
    """Single ZC sequence with Decimation"""
    p = lib.SyncParams()
    p.zc_len = 73
    p.plen = 61

    p.rolloff = 0.2
    #p.f_samp = 4e6
    #p.f_symb = 1e6
    p.f_symb = 30.72e6
    p.f_samp = p.f_symb*4
    p.repeat = 1
    p.spacing_factor = 1 # CHANGE TO TWO!

    p.power_weight = 2
    p.full_sim = True
    p.bias_removal = True
    p.ma_window = 1 # number of samples to average in the crosscorr i.e. after analog modulation
    p.train_type = 'chain' # Type of training sequence
    p.crosscorr_type = 'match_decimate' 
    p.match_decimate_fct = lib.md_clkphase
    p.peak_detect = 'wavg' 
    p.pulse_type = 'rootraisedcosine'
    p.central_padding = 0 # As a fraction of zpos length


    ctrl = SimControls()
    ctrl.steps = 40 # Approx number of emissions per node
    ctrl.basephi = 6000 # How many samples between emission
    ctrl.display = True # Show stuff in the console
    ctrl.keep_intermediate_values = False # Needed to draw graphs
    ctrl.nodecount = 20 # Number of nodes
    ctrl.static_nodes = 0
    ctrl.CFO_step_wait = float('inf') # Use float('inf') to never correct for CFO
    ctrl.TO_step_wait = 3
    ctrl.max_start_delay = 8 # In factor of basephi

    ctrl.theta_bounds = [0.2,0.8] # In units of phi
    #ctrl.theta_bounds = [0.48,0.52] # In units of phi
    #ctrl.theta_bounds = [0.5,0.5] # In units of phi
    #ctrl.theta_bounds = [0,1] # In units of phi
    ctrl.deltaf_bound = 3e-2
    #ctrl.deltaf_bound = 0
    ctrl.rand_init = False
    ctrl.epsilon_TO = 0.5
    ctrl.non_rand_seed = 11231231 # Only used if rand_init is False
    ctrl.noise_power = float('-inf')
    #ctrl.noise_power = -101 + 9 # in dbm

    ctrl.delay_params = lib.DelayParams(lib.delay_pdf_3gpp_exp)
    ctrl.delay_params.taps = 5
    ctrl.delay_params.max_dist_from_origin = 250 # (in meters)
    ctrl.delay_params.p_sigma = 500 # Paths sigma

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
    ctrl.pc_b, ctrl.pc_a = lib.hipass_avg(10)
    ctrl.pc_avg_thresh = float('inf') # If std of N previous TOx samples is above this value, then\
    ctrl.pc_std_thresh = float(30) # If std of N previous TOx samples is above this value, then\
                     # no PC is applied (but TOy is still calculated)
    
    ctrl.saveall = True


    cdict = {
        'nodecount':[x for x in range(10,61,1)]
        #'nodecou
        }

    pdict = {}
    #pdict = {'match_decimate_fct':[lib.md_clkphase, lib.md_energy]}

    return ctrl, p, cdict, pdict

def dec_sample_theta():
    p = lib.SyncParams()
    p.zc_len = 32
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
    p.train_type = 'single' # Type of training sequence
    p.crosscorr_type = 'match_decimate' 
    p.match_decimate_fct = lib.downsample
    p.peak_detect = 'wavg' 
    p.pulse_type = 'rootraisedcosine'
    p.central_padding = 0 # As a fraction of zpos length
    p.scfdma_precode = True
    p.scfdma_L = 8
    p.scfdma_M = p.zc_len*p.scfdma_L
    p.scfdma_sinc_len_factor = p.scfdma_L


    ctrl = SimControls()
    ctrl.steps = 40 # Approx number of emissions per node
    ctrl.basephi = 6000 # How many samples between emission
    ctrl.display = True # Show stuff in the console
    ctrl.keep_intermediate_values = False # Needed to draw graphs
    ctrl.nodecount = 15 # Number of nodes
    ctrl.static_nodes = 0
    ctrl.CFO_step_wait = float('inf') # Use float('inf') to never correct for CFO
    ctrl.TO_step_wait = 5
    ctrl.max_start_delay = 10 # In factor of basephi

    #ctrl.theta_bounds = [0.3,0.7] # In units of phi
    #ctrl.theta_bounds = [0.48,0.52] # In units of phi
    #ctrl.theta_bounds = [0.5,0.5] # In units of phi
    #ctrl.theta_bounds = [0,1] # In units of phi
    ctrl.theta_bounds = [0,0.6] # In units of phi
    ctrl.deltaf_bound = 3e-2
    #ctrl.deltaf_bound = 0
    ctrl.rand_init = False
    ctrl.epsilon_TO = 0.5

    #seed = int(np.random.rand()*1e8)
    seed = 5845527
    ctrl.non_rand_seed = seed # Only used if rand_init is False

    #ctrl.non_rand_seed = 57276545 # Only used if rand_init is False
    #ctrl.noise_power = float('-inf')
    ctrl.noise_power = -101 + 9 # in dbm

    ctrl.delay_params = lib.DelayParams(lib.delay_pdf_3gpp_exp)
    ctrl.delay_params.taps = 5
    ctrl.delay_params.max_dist_from_origin = 250 # (in meters)

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
    ctrl.pc_b, ctrl.pc_a = lib.hipass_avg(7)
    ctrl.pc_avg_thresh = float('inf') # If std of N previous TOx samples is above this value, then\
    ctrl.pc_std_thresh = float(80) # If std of N previous TOx samples is above this value, then\
                     # no PC is applied (but TOy is still calculated)
    
    ctrl.saveall = True


    cdict = {'rand_init':[True]}
    #cdict = {
    #    'nodecount':[x for x in range(ncount_lo, ncount_hi,step)]*3
    #    }

    pdict = {}
    #pdict = {'match_decimate_fct':[lib.md_clkphase]*ntot+[lib.md_energy]*ntot+[lib.md_static]*ntot}

    return ctrl, p, cdict, pdict




def ml_full_3d(noise_var=1, fct=lib.ll_redux_2d):
    """Graphs a 3d surface of the specified ML fct"""
    p = ml_pinit_no_pulse_shape()

    points = 10
    t0 = 0
    d0 = 0
    t_halfwidth = 10
    d_halfwidth = 10

    t_min = t0-t_halfwidth
    t_max = t0+t_halfwidth
    d_min = d0-d_halfwidth
    d_max = d0+d_halfwidth

    t_step = t_halfwidth*2/points # int(round()) because integer samples
    theta_range = np.arange(t_min, t_max, t_step) 
    deltaf_range = np.arange(d_min, d_max,d_halfwidth*2/points)*1e-6 # Deltaf in ppm
    #deltaf_range = np.zeros(len(deltaf_range))
    
    loglike = fct(p,0,0,theta_range,deltaf_range, var_w=noise_var)

    #deltaf_range = np.arange(-1,1.1, 0.1)*1e-6
    # Plot preparations
    x = theta_range
    y = deltaf_range*1e6
    z = loglike*1e-3

    # Plot prameters
    fig, ax = graphs.surface3d(x, y, z, density=40)
    ax.set_xlabel('Time offset (samples)')
    ax.set_ylabel('CFO (ppm)')
    ax.set_zlabel('Log likelihood (1e3)')
    graphs.plt.tight_layout()
    
    graphs.show()

def ml_full_one(variable='CFO',noise_var=1):
    """Graphs only 1 axis of the log ML"""
    p = ml_pinit()

    points = 1000
    t0 = 10
    d0 = 0
    t_halfwidth = 100
    d_halfwidth = 1

    t_min = t0-t_halfwidth
    t_max = t0+t_halfwidth
    d_min = d0-d_halfwidth
    d_max = d0+d_halfwidth

    t_step = max(int(round(t_halfwidth*2/points)),1) # int(round()) because integer samples
    theta_range = np.arange(t_min, t_max, t_step) 
    deltaf_range = np.arange(d_min, d_max,d_halfwidth*2/points)*1e-6 # Deltaf in ppm

    
    # x & z preparation
    if variable == 'CFO':
        loglike = lib.loglikelihood_1d_CFO(p,t0,d0,deltaf_range, var_w=noise_var)
        x = deltaf_range*1e6
        xlabel = 'CFO (ppm)'
    elif variable == 'TO':
        loglike = lib.loglikelihood_1d_TO(p,t0,d0,theta_range, var_w=noise_var)
        x = theta_range
        xlabel = 'TO (samples)'
    z = loglike*1e-3

    # Plot prametersZ
    ax = graphs.continuous(x, z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Log likelihood (1e3)')
    graphs.plt.tight_layout()

    
    graphs.show()


    p = ml_pinit()

    points = 40
    t0 = 0
    d0 = 0
    t_halfwidth = 10
    d_halfwidth = 1

    t_min = t0-t_halfwidth
    t_max = t0+t_halfwidth
    d_min = d0-d_halfwidth
    d_max = d0+d_halfwidth

    t_step = t_halfwidth*2/points # int(round()) because integer samples
    theta_range = np.arange(t_min, t_max, t_step) 
    deltaf_range = np.arange(d_min, d_max,d_halfwidth*2/points)*1e-6 # Deltaf in ppm
    #deltaf_range = np.zeros(len(deltaf_range))
    
    loglike = lib.ll_redux_2d(p,0,0,theta_range,deltaf_range, var_w=noise_var)

    #deltaf_range = np.arange(-1,1.1, 0.1)*1e-6
    # Plot preparations
    x = theta_range
    y = deltaf_range*1e6
    z = loglike*1e-3

    # Plot prameters
    fig, ax = graphs.surface3d(x, y, z, density=40)
    ax.set_xlabel('Time offset (samples)')
    ax.set_ylabel('CFO (ppm)')
    ax.set_zlabel('Log likelihood (1e3)')
    graphs.plt.tight_layout()
    
    graphs.show()

def zero_padded_crosscorr():
    p = pinit64_no_pulse_shape()
    ax = graphs.crosscorr_both(p, savename='latex_figures/discrete_crosscorr');
    graphs.show()
    return ax

def highlited_regimes():
    """Highlights the converging vs the drift regime"""


    # DRIFT REGIME
    ctrl, p, cdict, pdict = dec_sample_theta()
    sim = SimWrap(ctrl, p, cdict, pdict)

    sim.conv_min_slope_samples = 15 
    sim.ctrl.keep_intermediate_values = True
    sim.show_CFO = False
    sim.show_TO = False
    sim.simulate()
    ax, _ = sim.post_sim_plots(save_TO='', save_CFO='')

    # Draw biarrows
    xmid = 12
    xmax = ctrl.steps-5
    xmin = 0
    ymin, ymax = ax.get_ylim()
    ya = ymin-0.03
    ax.annotate('', xy=(xmid, ya), xycoords='data',
                xytext=(xmin, ya), textcoords='data',
                arrowprops=dict(arrowstyle="<->"))
    ax.annotate('', xy=(xmax, ya), xycoords='data',
                xytext=(xmid, ya), textcoords='data',
                arrowprops=dict(arrowstyle="<->"))
    ax.plot([xmid, xmid], [ya-1, ymax], 'k--')

    # text
    ax.text(xmid/2, ya, 'Converging', ha='center', va='bottom' )
    ax.text((xmax-xmid)/2 +xmid, ya, 'Drifting', ha='center', va='bottom' )
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ya-0.05,ymax])


    #graphs.show()
    fname = 'latex_figures/highlighted_regimes'
    graphs.save(fname)
    graphs.show()

    # Save 



    
    # THETA EXAMPLE
    ctrl, p, cdict, pdict = dec_sample_theta()
    ctrl.prop_correction = True
    sim = SimWrap(ctrl, p, cdict, pdict)

    sim.conv_min_slope_samples = 15 
    sim.ctrl.keep_intermediate_values = True
    sim.show_CFO = False
    sim.show_TO = False
    sim.simulate()

    ax, _ = sim.post_sim_plots(save_TO='', save_CFO='')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ya-0.05,ymax])

    
    fname = 'latex_figures/example_theta'
    graphs.save(fname)
    graphs.show()

def sample_theta():
    """Sample N=20 theta evolution"""
    ctrl, p, cdict, pdict = dec_regimes()
    sim = SimWrap(ctrl, p, cdict, pdict)

    sim.ctrl.keep_intermediate_values = True
    sim.show_CFO = False
    sim.set_all_nodisp()
    sim.show_TO = False
    sim.simulate()
    ax, _ = sim.post_sim_plots(save_TO='', save_CFO='')
    



# MAIN
if __name__ == '__main__':
    noise_variance = 0.00
    #ml_full_one('CFO', noise_var=noise_variance)
    ml_full_3d(noise_var=noise_variance)

