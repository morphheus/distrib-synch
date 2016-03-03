#!/usr/bin/env python

from sim_channel import SimControls
from lib import barywidth_map, calc_both_barycenters
import lib

import math
import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=PendingDeprecationWarning)
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

GRAPH_OUTPUT_LOCATION = 'graphs/' # don't forget the trailing slash
GRAPH_OUTPUT_FORMAT = 'eps'

matplotlib.rcParams.update({'font.size': 14})


#----- HELPER FCTS
def remove_zeropad(x,y,repad_ratio):
    """Returns the truncated x and y arrays with the zero padding removed"""
    if len(x) > len(y):
        raise ValueError('Expected x and y to have same length')
    
    first, last = y.nonzero()[0][[0,-1]]

    padding = int(round(repad_ratio*(last-first+1)))
    lo = max(0, first - padding)
    hi = min(len(y), last + padding)

    outx = x[lo:(hi+1)]
    outy = y[lo:(hi+1)]
    return outx,outy

def discrete(*args, axes=None):
    """Plots a discrete graph with vertical bars"""
    if len(args) < 1 and len(args) > 2:
        raise Exception("Too many or too little inputs")
    elif len(args) == 2:
        y = args[1]
        x = args[0]
    else:
        y = args[0]
        x = np.arange(len(y))

    if axes == None:
        ax = plt.axes()
    else:
        ax = axes

    ax.plot((x, x) , (y, np.zeros(len(y))), 'k-')

    # Pad the limits so we always see the leftmost-rightmost point
    x_pad = (x[1]-x[0])
    x_lims = [min(x)-x_pad, max(x)+x_pad]
    ax.plot(x_lims, [0,0], 'k-')
    ax.set_xlim(x_lims)
    return ax

def continuous(*args, label='curve0', axes=None):
    """Plots a continuous graph"""
    if len(args) < 1 and len(args) > 2:
        raise Exception("Too many or too little inputs")
    elif len(args) == 2:
        y = args[1]
        x = args[0]
    else:
        y = args[0]
        x = np.arange(len(y))

    if axes == None:
        ax = plt.axes()
    else:
        ax = axes

    lh = ax.plot(x, y, 'k-', label=label)
    #plt.scatter(x, y, marker='.')
    #plt.plot(indep_ax,curve,'k-')
    x_lims = [min(x), max(x)]
    ax.set_xlim(x_lims)

    return ax

def surface3d(x,y,z, density=20, **kwargs):
    """3d plot of the x, y vectors and z 2d array"""

    xstride = max(int(round(len(x)/density)),1)
    ystride = max(int(round(len(y)/density)),1)
    
    X, Y = np.meshgrid(x,y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,z,cstride=xstride, rstride=ystride, cmap=matplotlib.cm.coolwarm, **kwargs)
    return fig, ax

def save(name, **kwargs):
    """Saves the current figure to """
    if save != '':
        fname = GRAPH_OUTPUT_LOCATION + name + '.' + GRAPH_OUTPUT_FORMAT
        plt.savefig(fname, bbox_inches='tight', format=GRAPH_OUTPUT_FORMAT)

def show():
    plt.show()

#----- GRAPHS
def hair(samples,param, y_label='Parameter', savename=''):
    """Plots an evolution graph of the parameter of"""
    # samples: sample_inter output from the simulation
    # Param:  <param>_inter output from the simulation

    fh = plt.figure()
    ax = plt.axes()
    for slist, plist in zip(samples,param):
        ax.plot(np.array(slist),np.array(plist), 'k-')

    xmin = 0
    xmax = max([x[-1] for x in samples])
    ax.set_xlim([xmin,xmax])
    ax.set_xlabel('Sample')
    ax.set_ylabel(y_label)

    save(savename)
    return fh

def post_sim_graphs(simwrap):
    """Graphs to output at the end of a simulation"""
    ctrl = simwrap.ctrl
    # CFO graph
    hair(ctrl.sample_inter , ctrl.deltaf_inter , y_label='CFO (\Delta\lambda)', savename='lastCFO');
    if simwrap.show_CFO: show()
    else: plt.close(plt.gcf())

    # TO graphs
    hair(ctrl.sample_inter , ctrl.theta_inter , y_label='TO', savename='lastTO'); 
    if simwrap.show_TO: show()

def barywidth(*args, axes=None, savename='', fit_type='order2', residuals=True, disp=True, **kwargs):
    """Accepts either a SyncParams() object or two iterables representing the CFO and the barywidth"""
    if len(args) == 1 and type(args[0]).__name__ == 'SyncParams':
        CFO, barywidths = barywidth_map(args[0], **kwargs)
        f_symb = args[0].f_symb
    elif len(args) == 2:
        CFO = args[0]
        barywidths = args[1]
        f_symb = 1
        fit_type = 'none'
    else:
        print("Invalid input")

    # Axis specification shenanigans
    if axes is None:
        fh = plt.figure()
        if residuals:
            main_axes_loc = [.13,.3,.8,.6] 
        else:
            main_axes_loc = [.13,.11,.8,.8] 
        ax = fh.add_axes(main_axes_loc)
        make_rax = True if residuals else False
    else:
        make_rax = False
        typename = type(axes).__name__
        # Only one axes object given
        if typename in ['Axes', 'AxesSubplot']:
            ax = axes
            if residuals:
                warnings.warn('No residuals plotted; not enough axes')
                residuals = False
        # Input checking
        elif len(axes) > 2:
            raise('Too many axes specified')
        # If only 1 ax is given
        elif len(axes) == 1:
            ax = axes
            if residuals:
                warnings.warn('No residuals plotted; not enough axes')
                residuals = False
        # If two axes given
        elif len(axes) == 2:
            ax = axes[0]
            rax = axes[1]
            if not residuals: warnings.warn('Specified a residuals axis, but residuals option False')

    
    x = CFO/f_symb
    y = barywidths

    lh = ax.plot(x, y, 'k-', label='Barycenter width')
    x_lims = [min(x), max(x)]
    ax.plot((0,0), (min(y),max(y)))

    fit_curve = None
    if fit_type == 'linear':
    # Linear fit display
        fit = np.empty(2)
        fit[0] = args[0].baryslope*f_symb
        fit[1] = args[0].basewidth
        fit_curve = fit[0]*x + fit[1]
        ax.plot(x, fit_curve, label='Linear fit')
    elif fit_type == 'order2':
        # parabola fit display
        fit = args[0].order2fit
        fit_curve = fit[0]*CFO**2 + fit[1]*CFO**1 + args[0].basewidth#fit[2]
        ax.plot(x, fit_curve, label='Order2 fit')
    elif fit_type == 'logistic':
        coeffs = args[0].logisticfit
        fit_curve = lib.logistic(CFO, coeffs) + args[0].basewidth
        ax.plot(x, fit_curve, label='Logistic fit')
    elif fit_type == 'none':
        pass
    else:
        print('Unknown fit option. No fit displayed')
    

    xlabel = 'CFO (f_symb)'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Barycenter width')
    ax.set_xlim(x_lims)
    ax.legend()

    # Plotting residuals
    if make_rax:
        residuals_loc = [main_axes_loc[0],.1,.8,0]
        residuals_loc[3] = main_axes_loc[1]-residuals_loc[1]-0.005
        rax = fh.add_axes(residuals_loc) 
        ax.set_xticklabels([]) 

    if fit_curve is not None and residuals:
        residuals_vals = fit_curve - barywidths
        residuals_curve = residuals_vals/np.std(residuals_vals)

        rax.plot((x[0],x[-1]),(0,0), 'k-')
        rax.plot(x,residuals_curve)
        rax.yaxis.set_ticks((-1,1))
        rax.set_ylabel('\sigma')
        rax.set_xlabel(xlabel)
        rax.set_xlim(x_lims)
        rax.set_ylim([-1.7,1.7])
        msg = "Total error: " + str(np.abs(residuals_vals).sum())
        if disp: print(msg)
    

    save(savename)
    if axes is None:
        return fh

def barywidth_wrap(p,ctrl, **kwargs):
    return barywidth(p, reach=ctrl.bmap_reach , scaling_fct=ctrl.bmap_scaling, **kwargs)

def crosscorr(p, axes=None, savename='', is_zpos=True):
    """Builds a crosscorrelation graph from the crosscorrelation with zpos (default)
    Accepted input:
    (<class 'Params'>)      will plot the crosscorrelation with the analog signal 
    """
    tmp_full_sim = p.full_sim # Save temporary fullsim value
    tmp_bias = p.bias_removal # Save temporary fullsim value
    p.full_sim = False
    p.bias_removal = False

    _, _, rpos, rneg = calc_both_barycenters(p, mode='full')
    p.full_sim = tmp_full_sim # Restore entrance value
    p.bias_removal = tmp_bias # Restore entrance value


    y = rpos if is_zpos else rneg

    #label = 'Leading' if is_zpos else 'Trailing'
    label = None
    
    x = np.arange(len(y))
    # Only plot the non-zero interval
    _ , y = remove_zeropad(x,y, repad_ratio=0.05)

    x = np.arange(len(y)) - int(math.floor(len(y)/2))
    ax = continuous(x,y, axes=axes, label=label)
    ax.set_xlabel('l')
    ax.set_ylabel('|r[l]|')
    if label is not None: ax.legend()

    save(savename)
    return x, y, rpos, rneg

def crosscorr_zneg(p, axes=None, savename=''):
    """Same as crosscorr, but with zneg instead"""
    crosscorr(p, axes=axes, savename=savename, is_zpos=False)

def crosscorr_both(p, axes=None, savename=''):
    """Builds a crosscorrelation graph from the crosscorrelation of both zpos and zneg"""
    tmp = p.full_sim # Save temporary fullsim value
    tmp_bias = p.bias_removal # Save temporary biasremoval value
    p.full_sim = False
    p.bias_removal = False

    _, _, rpos, rneg = calc_both_barycenters(p,mode='full')
    p.full_sim = tmp # Restore entrance value
    p.bias_removal = tmp_bias

    x = np.arange(len(rpos))
    # Only plot the non-zero interval
    #_ , y0 = remove_zeropad(x,rpos, repad_ratio=0.05)
    #_ , y1 = remove_zeropad(x,rneg, repad_ratio=0.05)
    y0 = rpos
    y1 = rneg
    x = np.arange(len(y0)) - int(math.floor(len(y0)/2))

    ax = continuous(x,y0, axes=axes, label='Leading')
    ax.plot(x, y1, label='Trailing')

    ax.set_xlabel('l')
    ax.set_ylabel('|r[l]|')

    ax.legend()
    save(savename)
    return ax

def analog(p, axes=None, savename=''):
    """Absolute value of the pulse-shaped training sequence"""
    y = abs(p.analog_sig)
    x = np.arange(len(y)) - math.floor(len(y)/2)

    ax = discrete(x,y, axes=axes)
    
    ax.set_xlabel('n')
    ax.set_ylabel('|x(n)|')
    
    save(savename)
    return ax

def analog_zpos(p,axes=None, savename=''):
    """Absolute value of the pulse-shaped zpos sequence"""
    y = abs(p.analog_zpos)
    x = np.arange(len(y)) 
    
    fh = discrete(x,y)

    plt.xlabel('n')
    plt.ylabel('|z(t)|')

    
    
    return fh

def pulse(p, axes=None, savename=''):
    """Shaping pulse: p.pulse"""
    
    y = np.real(p.pulse)
    x = np.arange(len(y)) - len(y)/2 + 0.5


    ax = discrete(x,y, axes=axes)

    ax.set_xlabel('n (samples)')
    ax.set_ylabel('p[n])')
    
    save(savename)
    return ax

def delay(ctrl, axes=None, savename=''):
    """Plots the PDF of the delay params"""
    obj = ctrl.delay_params
    fct = obj.delay_pdf_eval

    tmp_sigma = obj.sigma
    obj.sigma *= ctrl.basephi

    xmin = obj.t0
    xmax = 4 + xmin
    x = np.arange(xmin, xmax, 0.01)*obj.sigma
    y = np.array(list(map(fct, x)))

    obj.sigma = tmp_sigma

    ax = continuous(x,y,axes=axes)
    ax.set_xlabel('Delay (Samples)')
    ax.set_ylabel('Amplitude')
    return ax

#----------------
def cat_graphs(graphs, rows=2,subplotsize=(9,5), savename=''):
    """Concatenate the figures together together
    graphlist: list of tuples of (fct name, args, kwarg)"""
    
    # Make sure lst is a list, indep of input.
    if type(graphs) is tuple:
        lst = [graphs.copy]
    elif type(graphs) is list:
        lst = graphs.copy()
    else:
        raise ValueError("Expected first argument to be a tuple or list. Currently is: " + str(type(graphs)))

    spcount = len(lst) # Subplotcount
    spargs = (rows, (spcount+1)//2) # Premade argument for rows-cols in add_subplots
    figsize = (spargs[1]*subplotsize[0], spargs[0]*subplotsize[1])
    fig = plt.figure(figsize=figsize)
    

    # Build axes and draw in them
    for k, tpl in enumerate(lst):
        # Break down the tuple if needed
        fct = tpl[0]
        fargs = tpl[1] if len(tpl) > 1 else tuple()
        fkwargs = tpl[2] if len(tpl) > 2 else dict()
        if len(tpl) > 3: raise ValueError('Input list element is a length ' + len(tpl) + 'iterable')
        # Build and populate axes
        ax = fig.add_subplot(*spargs, k+1)
        fkwargs['axes'] = ax
        fct(*fargs, **fkwargs)

        #Make axes title
        try:
            ax.set_title(fkwargs['sb_title'])
        except KeyError:
            ax.set_title(fct.__name__)

    # Finalize figure
    fig.tight_layout()
    save(savename)

def all_graphs(p,ctrl=None):
    
    glist = []
    glist.append((crosscorr_zneg, [p]))
    glist.append((analog, [p]))
    glist.append((pulse, [p]))

    # CTRL dependent graphs
    if ctrl is not None:
        glist.append((barywidth,
                        [p],
                        dict(axes=None, fit_type='linear', reach=ctrl.bmap_reach , scaling_fct=ctrl.bmap_scaling, residuals=False, force_calculate=False, disp=False)
                        ))

    cat_graphs(glist)






