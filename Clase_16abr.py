# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:24:13 2026

@author: gabri
"""

# Inicialización e importación de módulos

# Módulos externos
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

this_order = 2

z,p,k = sig.buttap(this_order)

# eps = np.sqrt( 10**(this_ripple/10) - 1 ) --> epsilon --> como es butter, el epsilon vale 1
eps = 1

num, den = sig.zpk2tf(z,p,k)

#Armado de filtros de Butterword y derivados con TP de Mariano