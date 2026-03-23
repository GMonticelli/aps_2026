# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:07:46 2026

@author: gabri
"""

import numpy as np

def mi_senoidal(vmax, dc, f0, n, fs, ts, ph=0):
    
    tt = np.linspace(0, (n-1)*ts, n)
    xx = dc + vmax*np.sin(2*np.pi*f0*tt+ph)
    
    return tt, xx