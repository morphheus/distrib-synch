#!/usr/bin/env python

import lib
import plotlib as graphs
import numpy as np

from numpy import pi



OUTPUT_FOLDER = 'graphs/latex_figures/' # Don't forget the theta_rangeailing slash


def ml_pinit():
    """Discrete non pulse-shaped sequence"""
    p = lib.SyncParams()
    p.zc_len = 10
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

def buildx(TO,CFO, p):
    """Builds a new x vector with appropriate TO and CFO"""
    p.TO = TO
    p.CFO = CFO
    p.update()
    return p.analog_sig.copy()

def loglikelihood_fct(p,t0,l0, theta_range, deltaf_range, var_w=1):
    """loglikelihood function over the range given by theta_range, deltaf_range, around initial values t0 and l0
    t0 : TO for the y vector  (units of samples)
    l0 : CFO for the y vector (units of f_samp)
    t0 : TO for the y vector  (units of samples)
    t0 : TO for the y vector  (units of f_samp)

    output: theta x deltaf array
    """

    y = buildx(t0, l0,p)

    M = len(y)
    tlen = len(theta_range)
    dlen = len(deltaf_range)
    
    CFO_range = deltaf_range*p.f_samp

    diff_magnitude = np.empty([dlen, tlen], dtype=lib.FLOAT_DTYPE)
    xy_diff = np.empty([dlen,M], dtype=lib.CPLX_DTYPE)
    for k,theta in enumerate(theta_range):
        for l, CFO in enumerate(CFO_range):
            xy_diff[l,:] = y - buildx(theta,CFO,p)
        diff_magnitude[:,k] = np.abs(xy_diff).sum(axis=1)**2

    
    loglike = -M*np.log(pi*var_w) - 1/(2*var_w) * diff_magnitude
    return loglike



def ml_thy_3d():

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
    
    loglike = loglikelihood_fct(p,0,0,theta_range,deltaf_range)

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
    
    graphs.show()




# MAIN
if __name__ == '__main__':
    ml_thy_3d()

