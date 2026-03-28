# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:09:34 2026

@author: gabri
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Agrego la ruta a la carpeta lib
ruta_lib = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(ruta_lib)

from mis_funciones import mi_senoidal

#%% Señal de 2KHz

fs = 40000 #Hz, frecuencia de muestreo
f0 = 2000 #Hz, frecuencia de la senoidal
n = 200 #muestras - 10 períodos
vmax = 1 #volts
dc = 0 #valor medio, volts
ts = 1/fs #Tiempo entre muestras = 1/ts = 0.000025 s
P = (vmax**2)/2

tt, xx = mi_senoidal(vmax, dc, f0, n, fs, ts, ph=0)

plt.figure(1)
plt.title('Señal 2Khz')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.plot(tt, xx)
plt.grid()
plt.show()

#%% Señal desfasada 

k = 2 #Factor de amplificacion
P = ((k * vmax)**2)/2 # La potencia escala al cuadrado

tt, xx = mi_senoidal(k * vmax, dc, f0, n, fs, ts, ph=np.pi/2)

plt.figure(2)
plt.title('Señal amplificada y desfasada')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.plot(tt, xx)
plt.grid()
plt.show()

#%% Señal modulada por senoidal

m = 0.5 #factor de modulación

tt, xxm = mi_senoidal(vmax, dc, f0/2, n, fs, ts, ph=0) #Senoidal moduladora
xx_mod = xx * ( 1 + m * xxm)

plt.figure(3)
plt.title('Señal modulada por senoidal')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.plot(tt, xx_mod)
plt.grid()
plt.show()

#%% Señal recortada en aplitud (clipping)

a_lim = 0.75

xx_clip = np.copy(xx_mod)

xx_clip[xx_clip > a_lim] = a_lim
xx_clip[xx_clip < -a_lim] = -a_lim

plt.figure(4)
plt.title('Señal con clipping')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.plot(tt, xx_clip)
plt.grid()
plt.show()

#%% Señal cuadrada

from scipy import signal

f0c = 4000 #Frecuencia de onda cuadrada 4KHz

tt = np.linspace(0, (n-1)/fs, n)
xx_sq = signal.square(2*np.pi*f0c*tt)

plt.figure(5)
plt.title('Señal cuadrada')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.plot(tt, xx_sq)
plt.grid()
plt.show()

#%% Pulso de 10ms

fs = 10000
ts = 1/fs

t_total = 0.05  # 50 ms
n = int(t_total * fs)

tt = np.linspace(0, (n-1)*ts, n)

T = 0.01  # 10 ms
A = 1

xx = np.zeros(n)
xx = A * (tt < T)

plt.figure(6)
plt.title('Señal de pulso')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.plot(tt, xx)
plt.grid()
plt.show()

#%% Chequeo de ejercicios de convolución

from scipy import signal as sig

# Eje temporal
n = np.arange(-10, 20)

# Respuesta al impulso
h = np.zeros_like(n)
h[n == 0] = 1
h[n == 4] = -1

#%% A) x(n) = cos(w0*n*Ts)
# Parámetros
w0 = 0.2 * np.pi
Ts = 1

x = np.cos(w0 * n * Ts)

# Convolución
y = sig.convolve(x, h, mode='same')

# Teórico
y_teo = x - np.roll(x, 4)

# Gráfico
plt.figure()
plt.plot(n, y, label='Convolución')
plt.plot(n, y_teo, '--', label='x[n] - x[n-4]')
plt.title('Punto a)')
plt.xlabel('n')
plt.grid()
plt.legend()
plt.show()

#%% B) x(n) =(1/2)^n * u(n)

# Escalón unitario
u = (n >= 0).astype(int)

x = (0.5)**n * u

# Convolución
y = sig.convolve(x, h, mode='same')

# Teórico
x_shift = np.roll(x, 4)
x_shift[n < 4] = 0   # corregir el corrimiento

y_teo = x - x_shift

# Gráfico
plt.figure()
plt.stem(n, y, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.stem(n, y_teo, linefmt='r--', markerfmt='ro', basefmt=' ')
plt.title('Punto b)')
plt.xlabel('n')
plt.grid()
plt.show() 

#%% C) x(n) = u(n+1) - u(n-2)

u1 = (n >= -1).astype(int)
u2 = (n >= 2).astype(int)

x = u1 - u2

# Convolución
y = sig.convolve(x, h, mode='same')

# Teórico
x_shift = np.roll(x, 4)
y_teo = x - x_shift

# Gráfico
plt.figure()
plt.stem(n, y, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.stem(n, y_teo, linefmt='r--', markerfmt='ro', basefmt=' ')
plt.title('Punto c)')
plt.xlabel('n')
plt.grid()
plt.show()