#!/usr/bin/env python

from lib import *


###################
# BASE GRAPHS
###################

#---------------------
def discrete(*args):
    """Plots a discrete graph with vertical bars"""
    if len(args) < 1 and len(args) > 2:
        raise Exception("Too many or too little inputs")
    elif len(args) == 2:
        y = args[1]
        x = args[0]
    else:
        y = args[0]
        x = np.arange(len(y))

    plt.plot((x, x) , (y, np.zeros(len(y))), 'k-')
    #plt.scatter(x, y, marker='.')
    #plt.plot(indep_ax,curve,'k-')
    x_lims = [min(x), max(x)]
    plt.plot(x_lims, [0,0], 'k-')
    plt.xlim(x_lims)
    return plt


#---------------------
def continuous(*args):
    """Plots a continuous graph"""
    if len(args) < 1 and len(args) > 2:
        raise Exception("Too many or too little inputs")
    elif len(args) == 2:
        y = args[1]
        x = args[0]
    else:
        y = args[0]
        x = np.arange(len(y))

    plt.plot(x, y, 'k-')
    #plt.scatter(x, y, marker='.')
    #plt.plot(indep_ax,curve,'k-')
    x_lims = [min(x), max(x)]
    plt.xlim(x_lims)
    







###################
# SPECIFIC GRAPHS
###################


#-------------------------
def hair(frames,param):
    """Plots an evolution graph of the parameter of"""
    # Frames: frame_inter output from the simulation
    # Param:  <param>_inter output from the simulation
    for flist, plist in zip(frames,param):
        continuous(flist,plist)

    xmin = max([x[0] for x in frames])
    xmax = max([x[-1] for x in frames])
    plt.xlim([xmin,xmax])



#---------------
def barywidth(*args, **kwargs):
    """Accepts either a Params() object or two iterables representing the CFO and the barywidth"""


    if len(args) == 1 and type(args[0]).__name__ == 'Params':
        CFO, barywidths = barywidth_map(args[0], **kwargs)
        CFO = CFO/args[0].f_symb
    elif len(args) == 2:
        CFO = args[0]
        barywidths = args[1]
    else:
        print("Invalid input")


    plt.plot(CFO,barywidths)
    plt.plot((0,0), (min(barywidths),max(barywidths)))
    plt.show()



#----------------
def crosscorr(*args):
    """Builds a crosscorrelation graph.
    Accepted inputs:
    (<class 'Params'>)      will plot the crosscorrelation with the analog signal 
    (x,y)                   where x and y are same length numpy arrays
    """
    if len(args) == 1 and type(args[0]).__name__ == 'Params':
        _, _, y, _ = calc_both_barycenters(args[0],mode='same')
        x = np.arange(len(y))# - len(y)/2 + 0.5
    elif len(args) == 2:
        x = args[0]
        y = args[1]
    else:
        raise Exception('Wrong number of arguments')

    discrete(x,y)



#----------------
def analog_graph():
    params = Params()
    params.zc_len = 11
    params.f_samp = 2
    params.f_symb = 1
    params.repeat = 1
    params.full_sim = False
    params.update()
    
    tmp = test_crosscorr(params)
    y = tmp.analog_sig
    x = np.arange(len(y)) - len(y)/2 + 0.5
    plt.plot((x, x) , (y, np.zeros(len(y))), 'k-')
    #plt.scatter(x, y, marker='.')
    #plt.plot(indep_ax,curve,'k-')
    x_lims = [min(x), max(x)]
    plt.plot(x_lims, [0,0], 'k-')
    plt.xlim(x_lims)

    plt.show()




#------------------
def modulated_zpos_graph():
    params = Params()
    params.zc_len = 11
    params.f_samp = 10
    params.f_symb = 1
    params.spacing = 2
    
    tmp = test_crosscorr(params)
    y = abs(tmp.analog_zpos)
    x = np.arange(len(y)) - len(y)/2 + 0.5
    plt.plot((x, x) , (y, np.zeros(len(y))), 'k-')
    #plt.scatter(x, y, marker='.')
    #plt.plot(indep_ax,curve,'k-')
    x_lims = [min(x), max(x)]
    plt.plot(x_lims, [0,0], 'k-')
    plt.xlim(x_lims)

    plt.show()



#----------------
def pulse_graph():
    params = Params()
    params.zc_len = 151
    params.plen = 19
    params.rolloff = 0.2
    params.f_samp = 12
    params.f_symb = 3
    params.repeat = 1
    params.power_weight = 4
    params.CFO = 0
    params.TO = 0
    params.full_sim = False
    params.bias_removal = 0
    
    params.update()
    
    y = np.real(params.pulse)
    x = np.arange(len(y)) - len(y)/2 + 0.5
    plt.plot((x, x) , (y, np.zeros(len(y))), 'k-')
    #plt.scatter(x, y, marker='.')
    #plt.plot(indep_ax,curve,'k-')
    x_lims = [min(x), max(x)]
    plt.plot(x_lims, [0,0], 'k-')
    plt.xlim(x_lims)

    plt.show()
