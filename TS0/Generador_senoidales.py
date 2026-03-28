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
SNR = 15
Pr = 10**(-SNR/10)

U_n = np.random.normal(mu,sigma,n)

xxn = xx + U_n

plt.plot(tt, xxn, label ='Senoidal con Ruido')
plt.plot(tt, xx, 'r', label = 'Senoidal pura')
plt.grid()
plt.legend()
plt.show()

#%% Convoluciones

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

#%% Correlación (Conv + flip)

yyf = (1/n)*sig.convolve(U_n, np.flip(U_n))
plt.title('Correlación de ruido consigo mismo')
plt.plot(yyf)
plt.grid()
plt.show()

#%% Cuantización y error

B = 4       # bits
Vfs = 2     # volts
qq = Vfs / (2**B)
Pr = qq**2/12

# Cuantización
xxq_idx = np.round(xx / qq)
xxq = xxq_idx * qq

# Error
error = xx - xxq


#%% Figura con subplots

fig, axs = plt.subplots(2, 1, figsize=(10,7), sharex=True)

# --- Señal original + cuantizada ---
axs[0].plot(tt, xx, color='blue', label='Señal original')
axs[0].step(tt, xxq, where='mid', color='orange', label='Señal cuantizada')
axs[0].set_title('Señal original vs cuantizada')
axs[0].set_ylabel('Amplitud [V]')
axs[0].grid()
axs[0].legend()

# --- Error de cuantización ---
axs[1].plot(tt, error, color='red', label='Error de cuantización')
axs[1].set_title('Error de cuantización')
axs[1].set_xlabel('Tiempo [s]')
axs[1].set_ylabel('Error [V]')
axs[1].grid()
axs[1].legend()

plt.tight_layout()
plt.show()
#%% Usamos la señal con ruido

B = 3       # bits
Vfs = 3     # volts
qq = Vfs / (2**B)

# Cuantización
xxq_idx = np.round(xxn / qq)
xxq = xxq_idx * qq

# Error
error = xxn - xxq
#%% Figura con subplots con ruido

fig, axs = plt.subplots(2, 1, figsize=(10,7), sharex=True)

# --- Señal original + cuantizada ---
axs[0].plot(tt, xxn, color='blue', label='Señal original')
axs[0].step(tt, xxq, where='mid', color='orange', label='Señal cuantizada')
axs[0].set_title('Señal original vs cuantizada')
axs[0].set_ylabel('Amplitud [V]')
axs[0].grid()
axs[0].legend()

# --- Error de cuantización ---
axs[1].plot(tt, error, color='red', label='Error de cuantización')
axs[1].set_title('Error de cuantización')
axs[1].set_xlabel('Tiempo [s]')
axs[1].set_ylabel('Error [V]')
axs[1].grid()
axs[1].legend()

plt.tight_layout()
plt.show()

#%% Histograma del error de cuantización

plt.figure(figsize=(8,4))
plt.hist(error, bins=20, density=True, edgecolor='black')

plt.title('Histograma del error de cuantización')
plt.xlabel('Error [V]')
plt.ylabel('Densidad de probabilidad')
plt.grid()

plt.show()

#%% Autocorrelación del error

autocorr = (1/len(error)) * sig.convolve(error, np.flip(error))

lags = np.arange(-len(error)+1, len(error))

plt.figure(figsize=(8,4))
plt.plot(lags, autocorr, color='purple')

plt.title('Autocorrelación del error de cuantización')
plt.xlabel('Retardo (lags)')
plt.ylabel('Autocorrelación')
plt.grid()

plt.show()

#%% FFT

N = len(xx)

XX = np.fft.fft(xx)

XXmod = np.abs(XX)
XXph = np.angle(XX)

# Eje de frecuencias
freq = np.fft.fftfreq(n, ts)


#%% Gráficos

fig, axs = plt.subplots(2, 1, figsize=(10,7), sharex=True)

# --- Módulo ---
axs[0].plot(freq, XXmod[:n//2], color='blue')
axs[0].set_title('Módulo de la FFT')
axs[0].set_ylabel('|X(f)|')
axs[0].grid()

# --- Fase ---
axs[1].plot(freq, XXph[:n//2], color='green')
axs[1].set_title('Fase de la FFT')
axs[1].set_xlabel('Frecuencia [Hz]')
axs[1].set_ylabel('Fase [rad]')
axs[1].grid()

plt.tight_layout()
plt.show()

#%%

SNR = 15
Pr = 10**(-SNR/10)

#XX = XX[:n//2]
#XX_db = 20 * np.log(np.abs(XX))

#U_N = np.fft.fft(U_n)
#U_N = U_N[:n//2]
#UN_db = 20 * np.log(np.abs(U_N))

XXQ = np.fft.fft(xxq)
XXQ = XXQ[:n//2]
freq = freq[:n//2]

plt.title('Densidad espectral de potencia')
plt.plot(freq, XXQ, label = 'Potencia de Senoidal pura')
#plt.plot(freq, UN_db, label = 'Potencioa de ruido')
plt.grid()
plt.show()

