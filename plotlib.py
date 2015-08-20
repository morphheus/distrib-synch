#!/usr/bin/env python

from lib import *

GRAPH_OUTPUT_LOCATION = 'graphs/' # don't forget the trailing slash
GRAPH_OUTPUT_FORMAT = 'eps'

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

    fh = plt.figure()

    plt.plot((x, x) , (y, np.zeros(len(y))), 'k-')
    #plt.scatter(x, y, marker='.')
    #plt.plot(indep_ax,curve,'k-')
    x_lims = [min(x), max(x)]
    plt.plot(x_lims, [0,0], 'k-')
    plt.xlim(x_lims)
    return fh


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

    fh = plt.figure()
    plt.plot(x, y, 'k-')
    #plt.scatter(x, y, marker='.')
    #plt.plot(indep_ax,curve,'k-')
    x_lims = [min(x), max(x)]
    plt.xlim(x_lims)

    return fh







###################
# SPECIFIC GRAPHS
###################


#-------------------------
def hair(frames,param, y_label='Parameter', savename=''):
    """Plots an evolution graph of the parameter of"""
    # Frames: frame_inter output from the simulation
    # Param:  <param>_inter output from the simulation

    fh = plt.figure()
    for flist, plist in zip(frames,param):
        plt.plot(flist,plist, 'k-')

    xmin = max([x[0] for x in frames])
    xmax = max([x[-1] for x in frames])
    plt.xlim([xmin,xmax])
    plt.xlabel('Frame')
    plt.ylabel(y_label)


    save(savename)
    return fh



#---------------
def barywidth(*args, savename='', **kwargs):
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

    plt.xlabel('CFO (f_samp)')
    plt.ylabel('Barycenter width')
    

    save(savename)


#----------------
def crosscorr(*args, savename=''):
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
    save(savename)



#----------------
def analog_graph(p, savename=''):
    y = abs(p.analog_hair)
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
def pulse(p):
    
    y = np.real(p.pulse)
    x = np.arange(len(y)) - len(y)/2 + 0.5
    x = x/p.f_samp

    discrete(x,y)

    plt.xlabel('t')
    plt.ylabel('p(t)')
    
    return x,y



#------------------
def save(name, **kwargs):
    """Saves the current figure to """
    if save != '':
        fname = GRAPH_OUTPUT_LOCATION + name + '.' + GRAPH_OUTPUT_FORMAT
        plt.savefig(fname, bbox_inches='tight', format=GRAPH_OUTPUT_FORMAT)
    


#------------------
def show():
    plt.show()
