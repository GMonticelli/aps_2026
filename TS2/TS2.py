# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:34:08 2026

@author: gabri
"""

import numpy as np
import matplotlib.pyplot as plt

fs = 2 #Hz, frecuencia de muestreo
f0 = 20 #Hz, frecuencia de la senoidal
n = 1000 #muestras por ciclo
vmax = 1 #volts
dc = 0 #valor medio, volts
ts = 1/fs
df = fs/n

def mi_senoidal(vmax, dc, f0, n, fs, ph=0):
    
    tt = np.linspace(0, (n-1)*ts, n)
    xx = dc + vmax*np.sin(2*np.pi*f0*tt+ph)
    
    return tt, xx

tt, xx = mi_senoidal(vmax, dc, f0, n, fs, ph=0)
plt.plot(tt, xx)

#%% Potencia de señal
#Fijamos en 1 Watt el valor de potencia de la señal para despejar Pr
#Queda SNR = 10* Log(1/Pr) y con eso regulo cuanto necesito de Pr para el
#SNR que busco

mu = 0
vmax = np.sqrt(2)
tt, xx = mi_senoidal(vmax, dc, f0, n, fs, ph=0)
Px = np.var(xx)
SNR = 15
Pr = 10**(-SNR/10)

U_n = np.random.normal(mu,np.sqrt(Pr),n)

xxn = xx + U_n

plt.figure()
plt.plot(tt, xxn, label =f'Senoidal con Ruido - SNR = {SNR}')
plt.plot(tt, xx, 'r', label = 'Senoidal pura')
plt.grid()
plt.legend()
plt.show()

#%% Cuantización

B = 4       # bits
Vfs = 2     # volts
qq = Vfs / (2**B)
#Pr = qq**2/12

# Cuantización
xxq_idx = np.round(xxn / qq)
xxq = xxq_idx * qq

# Error
#error = xxn - xxq


#%%

XXQ = np.fft.fft(xxq)/n
XXQ = XXQ[:n//2]
XXQ_db = 20 * np.log(np.abs(XXQ)*2)

freq = np.fft.fftfreq(n, ts)
freq = freq[:n//2]

plt.figure()
plt.title(f'Densidad espectral de potencia - SNR = {SNR}')
plt.plot(freq, XXQ_db, label = 'Señal cuantizada con ruido')
plt.xlabel('Frecuencia')
plt.ylabel('dB')
plt.grid()
plt.show()


