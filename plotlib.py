#!/usr/bin/env python

from lib import barywidth_map, calc_both_barycenters

import math
import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=PendingDeprecationWarning)
    import matplotlib
    import matplotlib.pyplot as plt

GRAPH_OUTPUT_LOCATION = 'graphs/' # don't forget the trailing slash
GRAPH_OUTPUT_FORMAT = 'eps'

matplotlib.rcParams.update({'font.size': 14})


###################
# Helper functions
###################
#--------------------
def remove_zeropad(x,y,repad_ratio):
    """Returns the truncated x and y arrays with the zero padding removed"""
    if len(x) > len(y):
        raise ValueError('Expected x and y to have same length')
    
    first, last = y.nonzero()[0][[0,-1]]

    padding = int(round(repad_ratio*(last-first)))
    lo = max(0, first - padding)
    hi = min(len(y), last + padding)

    outx = x[lo:(hi+1)]
    outy = y[lo:(hi+1)]
    return outx,outy

#---------------------
def discrete(*args, repad_ratio=0.1):
    """Plots a discrete graph with vertical bars"""
    if len(args) < 1 and len(args) > 2:
        raise Exception("Too many or too little inputs")
    elif len(args) == 2:
        y = args[1]
        x = args[0]
    else:
        y = args[0]
        x = np.arange(len(y))

    fh = plt.figure()

    plt.plot((x, x) , (y, np.zeros(len(y))), 'k-')

    # Pad the limits so we always see the leftmost-rightmost point
    x_pad = (x[1]-x[0])
    x_lims = [min(x)-x_pad, max(x)+x_pad]
    plt.plot(x_lims, [0,0], 'k-')
    plt.xlim(x_lims)
    return fh


#---------------------
def continuous(*args, repad_ratio=0.1, label='curve0'):
    """Plots a continuous graph"""
    if len(args) < 1 and len(args) > 2:
        raise Exception("Too many or too little inputs")
    elif len(args) == 2:
        y = args[1]
        x = args[0]
    else:
        y = args[0]
        x = np.arange(len(y))

    

    fh = plt.figure()
    lh = plt.plot(x, y, 'k-', label=label)
    #plt.scatter(x, y, marker='.')
    #plt.plot(indep_ax,curve,'k-')
    x_lims = [min(x), max(x)]
    plt.xlim(x_lims)

    return fh


#---------------------
def save(name, **kwargs):
    """Saves the current figure to """
    if save != '':
        fname = GRAPH_OUTPUT_LOCATION + name + '.' + GRAPH_OUTPUT_FORMAT
        plt.savefig(fname, bbox_inches='tight', format=GRAPH_OUTPUT_FORMAT)


#---------------------
def show():
    plt.show()



###################
# SPECIFIC GRAPHS
###################

#-------------------------
def hair(samples,param, y_label='Parameter', savename=''):
    """Plots an evolution graph of the parameter of"""
    # samples: sample_inter output from the simulation
    # Param:  <param>_inter output from the simulation

    fh = plt.figure()
    for flist, plist in zip(samples,param):
        plt.plot(flist,plist, 'k-')

    xmin = max([x[0] for x in samples])
    xmax = max([x[-1] for x in samples])
    plt.xlim([xmin,xmax])
    plt.xlabel('Sample')
    plt.ylabel(y_label)

    save(savename)
    return fh


#---------------
def barywidth(*args, savename='', fit_type='linear', residuals=False, **kwargs):
    """Accepts either a Params() object or two iterables representing the CFO and the barywidth"""


    if len(args) == 1 and type(args[0]).__name__ == 'Params':
        CFO, barywidths = barywidth_map(args[0], **kwargs)
        f_symb = args[0].f_symb
    elif len(args) == 2:
        CFO = args[0]
        barywidths = args[1]
        f_symb = 1
        fit_type = 'none'
    else:
        print("Invalid input")

    x = CFO/f_symb
    y = barywidths
    
    fh = plt.figure()
    main_axes_loc = [.13,.3,.8,.6] 
    frame1 = fh.add_axes(main_axes_loc)

    lh = plt.plot(x, y, 'k-', label='Barycenter width')
    x_lims = [min(x), max(x)]
    plt.plot((0,0), (min(y),max(y)))

    fit_curve = None
    if fit_type == 'linear':
    # Linear fit display
        fit = np.empty(2)
        fit[0] = args[0].baryslope*f_symb
        fit[1] = args[0].basewidth
        fit_curve = fit[0]*x + fit[1]
        plt.plot(x, fit_curve)
    elif fit_type == 'order2':
        # parabola fit display
        fit = args[0].order2fit
        fit_curve = fit[0]*CFO**2 + fit[1]*CFO**1 + args[0].basewidth#fit[2]
        plt.plot(x, fit_curve)
    elif fit_type == 'none':
        pass
    else:
        print('Unknown fit option. No fit displayed')
    

    xlabel = 'CFO (f_symb)'
    plt.xlabel(xlabel)
    plt.ylabel('Barycenter width')
    plt.xlim(x_lims)

    # Plotting residuals
    if fit_curve is not None and residuals:
        residuals_vals = fit_curve - barywidths
        residuals_curve = residuals_vals/np.std(residuals_vals)
        frame1.set_xticklabels([]) 

        residuals_loc = [main_axes_loc[0],.1,.8,0]
        residuals_loc[3] = main_axes_loc[1]-residuals_loc[1]-0.005
        frame2 = fh.add_axes(residuals_loc) 
        plt.plot((x[0],x[-1]),(0,0), 'k-')
        plt.plot(x,residuals_curve)
        frame2.yaxis.set_ticks((-1,1))
        plt.ylabel('\sigma')
        plt.xlabel(xlabel)
        plt.xlim(x_lims)
    

    save(savename)


#----------------------------
def crosscorr(p, savename='', is_zpos=True):
    p.full_sim = tmp # Restore entrance value

    y = rpos if is_zpos else rneg
    label = 'Leading' if is_zpos else 'Trailing'

    
    x = np.arange(len(y))
    # Only plot the non-zero interval
    _ , y = remove_zeropad(x,y, repad_ratio=0.05)

    x = np.arange(len(y)) - int(math.floor(len(y)/2))
    continuous(x,y, label=label)
    plt.xlabel('l')
    plt.ylabel('|r[l]|')
    plt.legend()

    save(savename)
    return x, y, rpos, rneg


#-------------------
def crosscorr_zneg(p, savename=''):
    """Same as crosscorr, but with zneg instead"""
    crosscorr(p, savename=savename, is_zpos=False)

#-------------------
def crosscorr_both(p, savename=''):
    """Builds a crosscorrelation graph from the crosscorrelation of both zpos and zneg
    Accepted input:
    (<class 'Params'>)      will plot the crosscorrelation with the analog signal 
    """
    tmp = p.full_sim # Save temporary fullsim value
    p.full_sim = False

    _, _, rpos, rneg = calc_both_barycenters(p,mode='same')
    p.full_sim = tmp # Restore entrance value

    x = np.arange(len(rpos))
    # Only plot the non-zero interval
    _ , y0 = remove_zeropad(x,rpos, repad_ratio=0.05)
    _ , y1 = remove_zeropad(x,rneg, repad_ratio=0.05)

    x = np.arange(len(y0)) - int(math.floor(len(y0)/2))

    continuous(x,y0, label='Leading')
    plt.plot(x, y1, label='Trailing')

    plt.xlabel('l')
    plt.ylabel('|r[l]|')

    plt.legend()

    save(savename)


#----------------
def analog(p, savename=''):
    y = abs(p.analog_sig)
    x = np.arange(len(y)) - len(y)/2 + 0.5

    fh = discrete(x,y)
    
    plt.xlabel('n')
    plt.ylabel('|x(n)|')
    
    save(savename)
    return fh


#------------------
def analog_zpos(p, savename=''):
    
    
    y = abs(p.analog_zpos)
    x = np.arange(len(y)) - len(y)/2 + 0.5
    x = x/p.f_samp
    
    fh = discrete(x,y)

    plt.xlabel('t')
    plt.ylabel('|z(t)|')

    
    
    return fh


#----------------
def pulse(p,savename=''):
    
    y = np.real(p.pulse)
    x = np.arange(len(y)) - len(y)/2 + 0.5


    discrete(x,y)

    plt.xlabel('n (samples)')
    plt.ylabel('p[n])')
    
    save(savename)
    return x,y



