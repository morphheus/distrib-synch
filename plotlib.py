#!/usr/bin/env python

from lib import *


#---------------
def barycenter_width_graph():

    params = Params()
    params.zc_len = 3
    params.plen = 17
    params.repeat = 10 # REPEAT MUST BE SET TO 1 (most of the time)
    params.f_samp = 10
    params.f_symb = 1
    params.power_weight = 2
    params.full_sim = False
    params.bias_removal = 0
    params.update()
    barylist = [[],[]]
    

    CFO = np.arange(-0.5*params.f_symb, 0.5*params.f_symb, 0.001*params.f_symb)
    for k in CFO:
        params.CFO = k
        params.update()
        barypos, baryneg, _, _ = calc_both_barycenters(params)
        barylist[0].append(barypos)
        barylist[1].append(baryneg)


    barywidth = np.array(barylist[0]) - np.array(barylist[1])

    plt.plot(CFO/params.f_symb,barywidth)
    plt.plot((0,0), (min(barywidth),max(barywidth)))
    #plt.plot(CFO,barylist[0])
    #plt.plot(CFO,barylist[1])
    plt.show()



#----------------
def crosscorr_graph():
    params = Params()
    params.zc_len = 201
    params.plen = 1
    params.rolloff = 0.2
    params.f_samp = 12
    params.f_symb = 3
    params.repeat = 1
    params.power_weight = 4
    params.CFO = -0.0125
    params.TO = 0
    params.full_sim = False
    params.bias_removal = 0


    
    params.update()
    
    
    barypos, baryneg, crosscorrpos, crosscorrneg = calc_both_barycenters(params)
    print('Barypos: ' + str(barypos))
    y = crosscorrneg
    x = np.arange(len(y))# - len(y)/2 + 0.5
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
