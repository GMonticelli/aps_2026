# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:54:00 2026

@author: gabri
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Agrego la ruta a la carpeta lib
ruta_lib = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(ruta_lib)

from mis_funciones import mi_senoidal

fs = 1000 #Hz, frecuencia de muestreo
n = 1000 #muestras por ciclo
f0 = n/4 #Hz, frecuencia de la senoidal
vmax = 1 #volts
dc = 0 #valor medio, volts
ts = 1/fs
df = fs/n

tt, xx = mi_senoidal(vmax, dc, f0, n, fs, ts, ph=0)

plt.figure(1)
plt.plot(tt, xx, label = 'Senoidal pura')
plt.grid()
plt.legend()
plt.show()

#%% Ruido analógico y ADC

mu = 0
vmax = np.sqrt(2)
tt, xx = mi_senoidal(vmax, dc, f0, n, fs, ts, ph=0)

#ADC
B = 4  # bits
Vfs = 2  # volts
qq = (2*Vfs) / (2**B)
Pq = qq**2/12

kn = 1

Pn = kn * Pq
sigma = np.sqrt(Pn)

U_n = np.random.normal(0, sigma, n)

xxn = xx + U_n

plt.figure(2)
plt.plot(tt, xxn, label ='Senoidal con Ruido')
plt.plot(tt, xx, 'r', label = 'Senoidal pura')
plt.grid()
plt.legend()
plt.show()

#%% Cuantización
eps = 1e-12

xxq_idx = np.round(xxn / qq)
xxq = xxq_idx * qq
xxq = np.clip(xxq, -Vfs, Vfs)

#%% Espectro

XX = np.fft.fft(xx)/n
XX = XX[:n//2]
XX_db = 10 * np.log10(np.abs(XX)*2 + eps)

NA = np.fft.fft(U_n)/n
NA = NA[:n//2]
NA_db = 10 * np.log10(np.abs(NA)*2 + eps)

error_cuan = xxn - xxq

ND = np.fft.fft(error_cuan)/n
ND = ND[:n//2]
ND_db = 10 * np.log10(np.abs(ND)*2 + eps)

freq = np.fft.fftfreq(n, ts)
freq = freq[:n//2]

piso_analogico = 10 * np.log10(np.var(U_n))
piso_digital = 10 * np.log10(np.var(error_cuan))

plt.figure(3)
plt.title(f'Densidad espectral de potencia - Kn = {kn}')
plt.plot(freq, XX_db, label = 'Señal cuantizada')
plt.plot(freq, NA_db, label = 'Ruido analogico')
plt.plot(freq, ND_db, label = 'Ruido Digital')
plt.axhline(piso_analogico, linestyle='--', color='red', label='Piso ruido analógico')
plt.axhline(piso_digital, linestyle='--', label='Piso ruido digital')
plt.ylim(-50, None)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid()
plt.legend()
plt.show()

#%%Histograma

plt.figure(4)
plt.hist(error_cuan, bins=10, density=True, edgecolor='black')
plt.title('Histograma del error de cuantización')
plt.xlabel('Error')
plt.ylabel('Densidad')
plt.grid()
plt.show()

#%% Kn = 10

kn = 10
Pn = kn * Pq
sigma = np.sqrt(Pn)

U_n = np.random.normal(0, sigma, n)
xxn = xx + U_n

xxq_idx = np.round(xxn / qq)
xxq = xxq_idx * qq
xxq = np.clip(xxq, -Vfs, Vfs)

XX = np.fft.fft(xx)/n
XX = XX[:n//2]
XX_db = 10 * np.log10(np.abs(XX)*2 + eps)

NA = np.fft.fft(U_n)/n
NA = NA[:n//2]
NA_db = 10 * np.log10(np.abs(NA)*2 + eps)

error_cuan = xxn - xxq

ND = np.fft.fft(error_cuan)/n
ND = ND[:n//2]
ND_db = 10 * np.log10(np.abs(ND)*2 + eps)

freq = np.fft.fftfreq(n, ts)
freq = freq[:n//2]

piso_analogico = 10 * np.log10(np.var(U_n))
piso_digital = 10 * np.log10(np.var(error_cuan))

plt.figure(5)
plt.title(f'Densidad espectral de potencia - Kn = {kn}')
plt.plot(freq, XX_db, label = 'Señal cuantizada')
plt.plot(freq, NA_db, label = 'Ruido analogico')
plt.plot(freq, ND_db, label = 'Ruido Digital')
plt.axhline(piso_analogico, linestyle='--', color='red', label='Piso ruido analógico')
plt.axhline(piso_digital, linestyle='--', label='Piso ruido digital')
plt.ylim(-50, None)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid()
plt.legend()
plt.show()

#%% Kn = 0.1

kn = 0.1
Pn = kn * Pq
sigma = np.sqrt(Pn)

U_n = np.random.normal(0, sigma, n)
xxn = xx + U_n

xxq_idx = np.round(xxn / qq)
xxq = xxq_idx * qq
xxq = np.clip(xxq, -Vfs, Vfs)

XX = np.fft.fft(xx)/n
XX = XX[:n//2]
XX_db = 10 * np.log10(np.abs(XX)*2 + eps)

NA = np.fft.fft(U_n)/n
NA = NA[:n//2]
NA_db = 10 * np.log10(np.abs(NA)*2 + eps)

error_cuan = xxn - xxq

ND = np.fft.fft(error_cuan)/n
ND = ND[:n//2]
ND_db = 10 * np.log10(np.abs(ND)*2 + eps)

freq = np.fft.fftfreq(n, ts)
freq = freq[:n//2]

piso_analogico = 10 * np.log10(np.var(U_n))
piso_digital = 10 * np.log10(np.var(error_cuan))

plt.figure(6)
plt.title(f'Densidad espectral de potencia - Kn = {kn}')
plt.plot(freq, XX_db, label = 'Señal cuantizada')
plt.plot(freq, NA_db, label = 'Ruido analogico')
plt.plot(freq, ND_db, label = 'Ruido Digital')
plt.axhline(piso_analogico, linestyle='--', color='red', label='Piso ruido analógico')
plt.axhline(piso_digital, linestyle='--', label='Piso ruido digital')
plt.ylim(-50, None)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid()
plt.legend()
plt.show()