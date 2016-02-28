#!/usr/bin/env python

import lib
import plotlib as graphs
import numpy as np

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
    p.theta_rangeain_type = 'chain'
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
    p.theta_rangeain_type = 'chain'
    p.update()
    return p 


def ml_thy_3d(noise_var=1):

    p = ml_pinit()

    points = 40
    t0 = 0
    d0 = 0
    t_halfwidth = 100
    d_halfwidth = 1

    t_min = t0-t_halfwidth
    t_max = t0+t_halfwidth
    d_min = d0-d_halfwidth
    d_max = d0+d_halfwidth

    t_step = int(round(t_halfwidth*2/points)) # int(round()) because integer samples
    theta_range = np.arange(t_min, t_max, t_step) 
    deltaf_range = np.arange(d_min, d_max,d_halfwidth*2/points)*1e-6 # Deltaf in ppm
    #deltaf_range = np.zeros(len(deltaf_range))
    
    loglike = lib.loglikelihood_fct(p,0,0,theta_range,deltaf_range, var_w=noise_var)

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

def ml_thy_one(variable='CFO',noise_var=1):
    p = ml_pinit()

    points = 1000
    t0 = 0
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
        loglike = lib.loglikelihood_fct_CFO(p,t0,d0,deltaf_range, var_w=noise_var)
        x = deltaf_range*1e6
        xlabel = 'CFO (ppm)'
    elif variable == 'TO':
        loglike = lib.loglikelihood_fct_TO(p,t0,d0,theta_range, var_w=noise_var)
        x = theta_range
        xlabel = 'TO (samples)'
    z = loglike*1e-3

    # Plot prametersZ
    ax = graphs.continuous(x, z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Log likelihood (1e3)')
    graphs.plt.tight_layout()

    
    graphs.show()




# MAIN
if __name__ == '__main__':
    noise_variance = 0.01
    ml_thy_one('TO', noise_var=noise_variance)
    #ml_thy_3d(noise_var=noise_variance)

