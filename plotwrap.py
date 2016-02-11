#!/usr/bin/env python

import lib
import plotlib as graphs



OUTPUT_FOLDER = 'graphs/latex_figures/' # Don't forget the trailing slash


#------------------------------
def discrete0_pinit():
    """Discrete non pulse-shaped sequence"""
    p = lib.SyncParams()
    p.zc_len = 101
    p.plen = 1
    p.rolloff = 0.2
    p.f_samp = 1
    p.f_symb = 1
    p.repeat = 1
    p.spacing_factor = 1
    p.power_weight = 4
    p.full_sim = False
    p.bias_removal = False
    p.ma_window = 1 # number of samples i.e. after analog modulation
    p.crosscorr_fct = 'analog'
    p.train_type = 'chain'
    p.update()

    name='discrete'

    graphlist= [ [graphs.crosscorr, {}],
                 [graphs.crosscorr_zneg, {}],
                 [graphs.crosscorr_both, {}],
                 [graphs.barywidth, {'reach':3e-5, 'scaling':3e-7}],
                 [graphs.analog, {}]
    ]
    
    
    
    return p, name, graphlist


#---------------------------
def print_all(p, name, graphlist):
    """Print to file all the graphs listed at the top of this function"""

    for fct in graphlist:
        savename = name + '_' + fct[0].__name__
        fct[0](p, savename=savename, **fct[1])



# MAIN
if __name__ == '__main__':

    plist = []
    plist.append(discrete0_pinit())

    graphs.GRAPH_OUTPUT_LOCATION = OUTPUT_FOLDER

    for p, name, graphlist in plist:
        print_all(p, name, graphlist)

