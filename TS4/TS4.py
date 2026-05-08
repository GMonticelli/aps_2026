# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:57:52 2026

@author: gabri
"""

# Módulos externos
import matplotlib.pyplot as plt
import numpy as np

N = 1000
fs = 1000
ts = 1 / fs
R = 200
a0 = np.sqrt(2)
omega0 = N/4

SNR = 10
PS = (a0)**2/2
PR = PS / 10**(SNR/10)
desvio = np.sqrt(PR)

n = np.arange(N)
n = n.reshape(1, N)

t = np.arange(N)/fs
tt = (t).reshape(1, -1)

na = np.random.normal(0, desvio, N)
fr = np.random.uniform(-2, 2, R)
df = np.pi*2/N

omega1 = omega0 + fr*df
omega1 = omega1.reshape(R,1)
arg = omega1*n

#omega1 = np.tile(omega1.reshape(1,R), (N,1))
#tt_mat = np.tile(tt, (R, 1))

señal = a0*np.sin(2*np.pi*arg*ts) + na
señal_t = np.transpose(señal)

#%%
plt.figure()
plt.plot(t, señal_t)
plt.grid()
plt.show()

#%% FFT

eps = 1e-12
 
señal_X = np.fft.fft(señal, axis = 1) / N
psd_señal = np.abs(señal_X)
señal_x_db = 10*np.log10(2*(psd_señal)**2 + eps)

frec = np.arange(N) * fs / N
frec = frec.reshape(1000, 1)


#%% Grafico FFT

plt.figure()
plt.title('PSD')
plt.plot(frec, señal_x_db.T)
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid()
plt.show()

#%% Estimador a1

