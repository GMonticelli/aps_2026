# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 21:17:29 2026

@author: gabri
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sig

#%% ITEM a 
# T(z) = (z^3 + z^2 + z + 1) / (z^3)

b_coeff = np.array([1, 1, 1, 1])
a_coeff = np.array([1, 0, 0, 0]) # Si disminuyo el ultimo coeficiente, bajo el Q, el grafico del modulo es cada vez mas lineal, y la respuesta de fase queda casi lineal 

taps = b_coeff.shape[0]

omega, resp_freq = sig.freqz(b_coeff, a = a_coeff, worN = 1024)

fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, tight_layout = True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap FIR Filter") # taps es la cantidad de coeficientes --> me va a dar el orden del filtro 
ax1.plot(omega, abs(resp_freq), 'C0')
# ax1.plot(omega, abs(resp_freq), 'C0')
ax1.set_ylabel("Amplitude in dB", color = 'C0')
ax1.grid(True)
ax1.axis('tight')
ax1.set(xlabel="Frequency in rad/sample", xlim = (0, np.pi))

# ax2 = ax1.twinx()

phase = np.unwrap(np.angle(resp_freq)) # En este caso si me conviene usar unwrap, porque evito la discontinuidad en la fase
# phase = np.angle(resp_freq)

ax2.plot(omega, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color = 'C1')
ax2.grid(True)
ax2.axis('tight')
plt.show()

#%% ITEM b
# T(z) = (z^4 + z^3 + z^2 + z + 1) / (z^4)

b_coeff = np.array([1, 1, 1, 1, 1])
a_coeff = np.array([1, 0, 0, 0, 0]) # Si disminuyo el ultimo coeficiente, bajo el Q, el grafico del modulo es cada vez mas lineal, y la respuesta de fase queda casi lineal 

taps = b_coeff.shape[0]

omega, resp_freq = sig.freqz(b_coeff, a = a_coeff, worN = 1024)

fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, tight_layout = True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap FIR Filter") # taps es la cantidad de coeficientes --> me va a dar el orden del filtro 
ax1.plot(omega, abs(resp_freq), 'C0')
# ax1.plot(omega, abs(resp_freq), 'C0')
ax1.set_ylabel("Amplitude in dB", color = 'C0')
ax1.grid(True)
ax1.axis('tight')
ax1.set(xlabel="Frequency in rad/sample", xlim = (0, np.pi))

# ax2 = ax1.twinx()

phase = np.unwrap(np.angle(resp_freq)) # En este caso si me conviene usar unwrap, porque evito la discontinuidad en la fase
# phase = np.angle(resp_freq)

ax2.plot(omega, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color = 'C1')
ax2.grid(True)
ax2.axis('tight')
plt.show() 

#%% ITEM c
# T(z) = (z - 1) / (z)

b_coeff = np.array([1, -1])
a_coeff = np.array([1, 0]) # Si disminuyo el ultimo coeficiente, bajo el Q, el grafico del modulo es cada vez mas lineal, y la respuesta de fase queda casi lineal 

taps = b_coeff.shape[0]

omega, resp_freq = sig.freqz(b_coeff, a = a_coeff, worN = 1024)

fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, tight_layout = True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap FIR Filter") # taps es la cantidad de coeficientes --> me va a dar el orden del filtro 
ax1.plot(omega, abs(resp_freq), 'C0')
# ax1.plot(omega, abs(resp_freq), 'C0')
ax1.set_ylabel("Amplitude in dB", color = 'C0')
ax1.grid(True)
ax1.axis('tight')
ax1.set(xlabel="Frequency in rad/sample", xlim = (0, np.pi))

# ax2 = ax1.twinx()

phase = np.unwrap(np.angle(resp_freq)) # En este caso si me conviene usar unwrap, porque evito la discontinuidad en la fase
# phase = np.angle(resp_freq)

ax2.plot(omega, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color = 'C1')
ax2.grid(True)
ax2.axis('tight')
plt.show()

#%% ITEM d
# T(z) = (z^2 - 1) / (z^2)

b_coeff = np.array([1, 0, -1])
a_coeff = np.array([1, 0, 0]) # Si disminuyo el ultimo coeficiente, bajo el Q, el grafico del modulo es cada vez mas lineal, y la respuesta de fase queda casi lineal 

taps = b_coeff.shape[0]

omega, resp_freq = sig.freqz(b_coeff, a = a_coeff, worN = 1024)

fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True, tight_layout = True)
ax1, ax2 = axs
ax1.set_title(f"Frequency Response of {taps} tap FIR Filter") # taps es la cantidad de coeficientes --> me va a dar el orden del filtro 
ax1.plot(omega, abs(resp_freq), 'C0')
# ax1.plot(omega, abs(resp_freq), 'C0')
ax1.set_ylabel("Amplitude in dB", color = 'C0')
ax1.grid(True)
ax1.axis('tight')
ax1.set(xlabel="Frequency in rad/sample", xlim = (0, np.pi))

# ax2 = ax1.twinx()

phase = np.unwrap(np.angle(resp_freq)) # En este caso si me conviene usar unwrap, porque evito la discontinuidad en la fase
# phase = np.angle(resp_freq)

ax2.plot(omega, phase, 'C1')
ax2.set_ylabel('Phase [rad]', color = 'C1')
ax2.grid(True)
ax2.axis('tight')
plt.show()