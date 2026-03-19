# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:28:38 2026

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt

fs = 1000 #Hz, frecuencia de muestreo
f0 = 1 #Hz, frecuencia de la senoidal
n = 1000 #muestras por ciclo
vmax = 1 #volts
dc = 0 #valor medio, volts
ts = 1/fs

def mi_senoidal(vmax, dc, f0, n, fs, ph=0):
    
    tt = np.linspace(0, (n-1)*ts, n)
    xx = dc + vmax*np.sin(2*np.pi*f0*tt+ph)
    
    return tt, xx

tt, xx = mi_senoidal(vmax, dc, f0, n, fs, ph=0)
plt.plot(tt, xx)

#%% frecuencia de senoidal de 500 Hz

tt, xx = mi_senoidal(vmax, dc, 500, n, fs, ph=0)

plt.figure()
plt.plot(tt,xx,'o-')
plt.title("Señal de 500 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.show()

#%% frecuencia de senoidal de 999 Hz

tt, xx = mi_senoidal(vmax, dc, 999, n, fs, ph=0)

plt.figure()
plt.plot(tt,xx)
plt.title("Señal de 999 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.show()

#%% frecuencia de senoidal de 1001 Hz

tt, xx = mi_senoidal(vmax, dc, 1001, n, fs, ph=0)

plt.figure()
plt.plot(tt,xx)
plt.title("Señal de 1001 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.show()

#%% frecuencia de senoidal de 2001 Hz

tt, xx = mi_senoidal(vmax, dc, 2001, n, fs, ph=0)

plt.figure()
plt.plot(tt,xx)
plt.title("Señal de 2001 Hz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid()
plt.show()

#%% Señal particular - Diente de sierra

from scipy import signal

f0 = 1 #Hz

tt = np.linspace(0,1,n)
xx = signal.sawtooth(2*np.pi*5*tt)

plt.figure()
plt.plot(tt,xx)
plt.title("Onda diente de sierra")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid()
plt.show()

#%% Señal con ruido
#Agregamos una señal aleatoria (ruido) a la señal pura

sigma = 1
mu = 0
U_n = np.random.normal(mu,sigma,n)

xxn = xx + U_n

plt.title('Señal con ruido')
plt.xlabel('Tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.plot(tt, xxn, label ='Senoidal con Ruido')
plt.plot(tt, xx, label = 'Senoidal pura')
plt.grid()
plt.legend()
plt.show()

#%% Potencia de señal
#Fijamos en 1 Watt el valor de potencia de la señal para despejar Pr
#Queda SNR = 10* Log(1/Pr) y con eso regulo cuanto necesito de Pr para el
#SNR que busco

vmax = np.sqrt(2)
tt, xx = mi_senoidal(vmax, dc, f0, n, fs, ph=0)
Px = np.var(xx)
SNR = -20
Pr = 10**(SNR/10)

U_n = np.random.normal(mu,sigma,n)

xxn = xx + U_n

plt.plot(tt, xxn, label ='Senoidal con Ruido')
plt.plot(tt, xx, 'r', label = 'Senoidal pura')
plt.grid()
plt.legend()
plt.show()

#%%

from scipy import signal as sig

n= 1000 #cantidad de muestras

#Armo la delta
n0 = 300
dd = np.zeros(n)
dd[n0] = 1

yy = (1/n)*sig.convolve(xx, dd)
plt.title('Convolucion de senoidal con delta')
plt.plot(yy)
plt.grid()
plt.show()

#%%

yyf = (1/n)*sig.convolve(U_n, np.flip(U_n))
plt.title('Correlación de ruido consigo mismo')
plt.plot(yyf)
plt.grid()
plt.show()

#%%Cuantizacion

B = 3 #Bits
Vfs = 3 #Volts
qq = Vfs / 2**B

xxq = np.round(xx / qq)

plt.plot(xxq, label ='Senoidal cuantizada')
plt.grid()
plt.show()
