#!/usr/bin/env python
"""Writing numbers/values to a latex \variable """

import numpy as np
import io


OFILE = 'graphs/latex_figures/results.tex'

PRE = '\\def' # What prefaces each variables




def get_vars(fname):
    """Gets all the \def\vname{val} from the file"""
    with open(fname, 'r') as f:
        lines = f.read().splitlines()

    # Only get important lines. Remove the PRE
    vlist = [(x[len(PRE):], k) for k,x in enumerate(lines) if x.startswith(PRE)]
    
    # Break down the lines in (var, val, linenum)
    pairlist = []
    for x,k in vlist:
        brakl = x.find('{') # Index of bracket
        brakr = x.find('}') # Index of bracket
        if brakl == -1: raise Exception('No { found in entry: ' + x)
        if brakr == -1: raise Exception('No } found in entry: ' + x)
        pairlist.append((x[1:brakl], x[brakl+1:brakr], k ))

    return pairlist

def check_pairlist_integrity(pairlist):
    """Checks if the first entry of each item is a valid latex name"""
    for x in pairlist:
        if not x[0].isalpha():
            raise Exception("Invalid variable name: " + x[0] + " ; expected only alphabet")

def write(pairs, fname=OFILE):
    """Writes the pairlist to fname. Overwrites all instances if necessary; otherwise appends
    at the EOF
    Pairs: list of length 2 tuples/lists"""

    current_pairs = get_vars(fname)
    check_pairlist_integrity(pairs)

    with open(fname, 'r') as f:
        flines = f.readlines()

    for name, val in pairs:
        append = True
        to_write = PRE + '\\' + name + '{' + str(val) + '}'
        for cname, cval, linenum in current_pairs:
            if name==cname:
                brakr = flines[linenum].find('}') # Index of closing bracket
                flines[linenum] = to_write + flines[linenum][brakr+1:]
                append = False

        if append:
            flines.append(to_write + '\n')

    with open(fname, 'w') as f:
        f.writelines(flines)


def build_conv_pairs(conv, opts):
    """Builds the convergence metric pairs for writing"""
    out = []
    for ckey, cval in conv.items():
        s = ''
        for okey, oval in sorted(opts):
            if okey == 'bias_removal': tmp = 'b'
            elif okey == 'prop_correction': tmp = 'p'
            elif okey == 'scfdma_precode': tmp = 's'
            else: raise Exception('Unrecognized opts: ' + okey)

            s += tmp.upper() if oval else tmp
        
        if ckey=='tot':
            s += 'T'
            cstr = str(cval)
        elif ckey=='gl_min':
            s += 'LM'
            cstr = "%.3f"%cval
        elif ckey=='gl_avg':
            s += 'LA'
            cstr = "%.3f"%cval
        elif ckey=='gl_std':
            s += 'LS'
            cstr = "%.3f"%cval
        elif ckey=='beta_avg':
            s += 'BA'
            cstr = "%.3g"%cval
        elif ckey=='beta_var':
            s += 'BS'
            cstr = "%.3g"%cval
        else: raise Exception('Unreconized conv key: ' + ckey)

        cstr = "%.3g"%cval
        out.append((s, cstr))
    
    return out


if __name__ == '__main__':
    pass

