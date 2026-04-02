# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:05:13 2026

@author: gabri
"""

import numpy as np
import matplotlib.pyplot as plt

N = 8
n = np.arange(N)
xn = 4 + 3*np.sin((np.pi/2)*n)

Xk = np.fft.fft(xn)

xk_mod = np.abs(Xk)
xk_ph = np.angle(Xk)

plt.figure(1)
plt.title('DFT x[n]')
plt.stem(n, xk_mod, basefmt = " ")
plt.xlabel('K')
plt.ylabel('|X[k]')
plt.grid()
plt.show()

plt.figure(2)
plt.title('Fase de X[k]')
plt.stem(n, xk_ph, label = 'Senoidal de frecuencia pi/2', basefmt = " ")
plt.xlabel('K')
plt.ylabel('<X[k]')
plt.grid()
plt.legend()
plt.show()

#%% Fase de 3/2pi

N = 8
n = np.arange(N)
xn = 4 + 3*np.sin((np.pi*3/2)*n)

Xk = np.fft.fft(xn)

xk_mod = np.abs(Xk)
xk_ph = np.angle(Xk)

plt.figure(3)
plt.title('DFT x[n]')
plt.stem(n, xk_mod, basefmt = " ")
plt.xlabel('K')
plt.ylabel('|X[k]')
plt.grid()
plt.show()

plt.figure(4)
plt.title('Fase de X[k]')
plt.stem(n, xk_ph, label = 'Senoidal de fase pi*3/2', basefmt = " ")
plt.xlabel('K')
plt.ylabel('<X[k]')
plt.grid()
plt.legend()
plt.show()

#%% Espectro de potencia

x_pot = (xk_mod**2)/(N**2)

plt.figure(5)
plt.title('Espectro de Potencia - Frecuencia: pi/2')
plt.stem(n, x_pot, basefmt = " ")
plt.xlabel('K')
plt.ylabel('|X[k]^2')
plt.grid()
plt.show()

#Conclusión : En el gráfico se visualiza la amplitud de los deltas correspondiente a 4.5/2
#(la potencia dividida entre las dos deltas) en los valores 2 y 6. Esas muestras son 
#las correspondientes a las dos deltas de la senoidal, mientras que la energía que se
#ve en 0 corresponde al aporte de la constante de la señal.

#%%Fase de 1/3pi

xn = 4 + 3*np.sin((np.pi*1/3)*n)

Xk = np.fft.fft(xn)

xk_mod = np.abs(Xk)
xk_ph = np.angle(Xk)

x_pot = (xk_mod**2)/(N**2)

plt.figure(6)
plt.title('Espectro de Potencia - Freciencia: pi/3')
plt.stem(n, x_pot, basefmt = " ")
plt.xlabel('K')
plt.ylabel('|X[k]^2')
plt.grid()
plt.show()

plt.figure(7)
plt.title('Fase de X[k]')
plt.stem(n, xk_ph, label = 'Senoidal de Frecuencia pi/3', basefmt = " ")
plt.xlabel('K')
plt.ylabel('<X[k]')
plt.grid()
plt.legend()
plt.show()

#Se visualiza el leakage en la energía de la señal.