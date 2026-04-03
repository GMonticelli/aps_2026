# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:34:08 2026

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
f0 = fs/n #Hz, frecuencia de la senoidal
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

#%% Potencia de señal
#Fijamos en 1 Watt el valor de potencia de la señal para despejar Pr
#Queda SNR = 10* Log(1/Pr) y con eso regulo cuanto necesito de Pr para el
#SNR que busco

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

#%% SNR

xxq_idx = np.round(xxn / qq)
xxq = xxq_idx * qq
error = xx - xxq

SNR = 10 * np.log10(np.var(xx) / np.var(error))

#%% Espectro e Histograma

XX = np.fft.fft(xxq)/n
XX = XX[:n//2]
XX_db = 20 * np.log10(np.abs(XX)*2)

freq = np.fft.fftfreq(n, ts)
freq = freq[:n//2]

plt.figure(3)
plt.title('Densidad espectral de potencia')
plt.plot(freq, XX_db, label = 'Señal cuantizada con ruido')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid()
plt.legend()
plt.show()

#Histograma
plt.figure(4)
plt.hist(error, bins=20, density=True, edgecolor='black')
plt.title('Histograma del error total')
plt.xlabel('Error')
plt.ylabel('Densidad')
plt.grid()
plt.show()

error_ruido = xx - xxn
error_cuant = xxn - xxq

plt.figure(5)
plt.hist(error_ruido, bins=20, alpha=0.5, label='Ruido (gaussiano)', density=True)
plt.hist(error_cuant, bins=20, alpha=0.5, label='Cuantización (uniforme)', density=True)
plt.hist(error, bins=20, alpha=0.5, label='Total', density=True)
plt.legend()
plt.grid()
plt.title('Comparación de errores')
plt.show()

#%% Punto b: B = 4, Kn = 10

B = 4
kn = 10

qq = (2*Vfs) / (2**B)
Pq = qq**2 / 12

Pn = kn * Pq
sigma = np.sqrt(Pn)

ruido = np.random.normal(0, sigma, n)
xxn_b = xx + ruido

xxq_b = np.round(xxn_b / qq) * qq

error_b = xx - xxq_b

SNR_b = 10 * np.log10(np.var(xx) / np.var(error_b))

# Espectro
XX_b = np.fft.fft(xxq_b)/n
XX_b = XX_b[:n//2]
XX_b_db = 20 * np.log10(np.abs(XX_b)*2)

plt.figure(6)
plt.plot(freq, XX_b_db, label='Caso b')
plt.plot(freq, XX_db, label='Caso a', alpha=0.7)
plt.title('Comparación espectral')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid()
plt.legend()
plt.show()

# Histograma
plt.figure(7)
plt.hist(error_b, bins=20, density=True, alpha=0.7, label='Caso b')
plt.hist(error, bins=20, density=True, alpha=0.5, label='Caso a')
plt.legend()
plt.title('Comparación de error')
plt.grid()
plt.show()

#%% Aliasing

f0 = 600  # > fs/2

tt, xx = mi_senoidal(vmax, dc, f0, n, fs, ts)

plt.figure(8)
plt.plot(tt, xx)
plt.title('Aliasing: f0 = 600 Hz → señal observada ≈ 400 Hz')
plt.grid()
