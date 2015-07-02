#!/usr/bin/env python

from lib import *


#---------------
def barycenter_width_graph():

    params = Params()
    params.zc_len = 21
    params.plen = 21
    params.repeat = 1 # REPEAT MUST BE SET TO 1!!!
    params.f_samp = 10
    params.f_symb = 1
    params.update()
    barylist = [[],[]]
    

    CFO = np.arange(-1*params.f_symb, params.f_symb, 0.01*params.f_symb)
    for k in CFO:
        params.CFO=k
        tmp = analog_crosscorr(params)
        barylist[0].append(tmp.barypos)
        barylist[1].append(tmp.baryneg)


    barywidth = np.array(barylist[0]) - np.array(barylist[1])

    plt.plot(CFO,barywidth)
    #plt.plot(CFO,barylist[0])
    #plt.plot(CFO,barylist[1])
    plt.show()



#----------------
def crosscorr_graph():
    params = Params()
    params.zc_len = 11
    params.f_samp = 10
    params.f_symb = 1
    params.repeat = 1
    params.build_training_sequence()
    
    tmp = analog_crosscorr(params)
    y = abs(tmp.crosscorrpos)
    x = np.arange(len(y)) - len(y)/2 + 0.5
    plt.plot((x, x) , (y, np.zeros(len(y))), 'k-')
    #plt.scatter(x, y, marker='.')
    #plt.plot(indep_ax,curve,'k-')
    x_lims = [min(x), max(x)]
    plt.plot(x_lims, [0,0], 'k-')
    plt.xlim(x_lims)

    plt.show()


#----------------
def analog_graph():
    params = Params()
    params.zc_len = 11
    params.f_samp = 2
    params.f_symb = 1
    params.repeat = 1
    params.build_training_sequence()
    
    tmp = test_crosscorr(params)
    y = abs(tmp.analog_sig)
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
    params.plen = 101
    params.f_samp = 10
    params.f_symb = 1
    
    tmp = test_crosscorr(params)
    y = np.real(tmp.pulse)
    x = np.arange(len(y)) - len(y)/2 + 0.5
    plt.plot((x, x) , (y, np.zeros(len(y))), 'k-')
    #plt.scatter(x, y, marker='.')
    #plt.plot(indep_ax,curve,'k-')
    x_lims = [min(x), max(x)]
    plt.plot(x_lims, [0,0], 'k-')
    plt.xlim(x_lims)

    plt.show()
