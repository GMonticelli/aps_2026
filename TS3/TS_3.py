# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:32:20 2026

@author: gabri
"""

# Inicialización e importación de módulos

# Módulos externos
import matplotlib.pyplot as plt
import numpy as np

fs = 1000
N = 1000
df = fs/N
ts = 1/fs
dc = 0

f0 = N/4 * df
f01 = (N/4 + 1/4)* df
f02 = (N/4 + 1/2)* df
vmax = np.sqrt(2)

tt = np.arange(N) * ts

fx = dc + vmax * np.sin(2*np.pi*f0*tt)
fx1 = dc + vmax * np.sin(2*np.pi*f01*tt)
fx2 = dc + vmax * np.sin(2*np.pi*f02*tt)


frec = np.arange(N) * fs / N

#%%FFT f0 = N/4
eps = 1e-12

FX = np.fft.fft(fx) / N
FX_mod = np.abs(FX)**2
FX_db = 10 * np.log10(FX_mod*2 + eps)

plt.figure()
plt.title('Espectro de frecuencia')
plt.plot(frec, FX_db, marker='o', linestyle='None', label = 'Senoidal de N/4')
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid()
plt.legend()
plt.show()

#%%FFT f0 = N/4 + 1/4

FX1 = np.fft.fft(fx1) / N
FX1_mod = np.abs(FX1)**2
FX1_db = 10 * np.log10(FX1_mod*2 + eps)

plt.figure()
plt.title('Espectro de frecuencia')
plt.plot(frec, FX1_db, marker='o', linestyle='None', label = 'Senoidal de N/4 + 1/4')
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid()
plt.legend()
plt.show()

#%%FFT f0 = N/4 + 1/2

FX2 = np.fft.fft(fx2) / N
FX2_mod = np.abs(FX2)**2
FX2_db = 10 * np.log10(FX2_mod*2 + eps)

plt.figure()
plt.title('Espectro de frecuencia')
plt.plot(frec, FX2_db, marker='o', linestyle='None', label = 'Senoidal de N/4 + 1/2')
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid()
plt.legend()
plt.show()

#%% Ventana rectangular - zero padding
# Frecuencia = N/4

N0 = N*10
frecz = np.arange(N0) * fs / N0

zz = np.zeros(9*N)
fxzz = np.concatenate((fx,zz), axis = None)
fxzz_sinc_FFT = np.fft.fft(fxzz) / N
fxzz_sinc_mod = np.abs(fxzz_sinc_FFT)**2
fxzz_sinc_db = 10 * np.log10(fxzz_sinc_mod *2 + eps)

plt.figure()
plt.title('Zero padding sobre la señal N/4')

plt.plot(frec, FX_db, 'o', linestyle='None', label='Sin zero padding')
plt.plot(frecz, fxzz_sinc_db, '.', linestyle='None', label='Con zero padding')

plt.xlim([100, 400])
plt.ylim([-60, 2])
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid()
plt.legend()
plt.show()

#%% Frecuencia = N/4 + 1/4

fxzz1 = np.concatenate((fx1,zz), axis = None)
fxzz1_sinc_FFT = np.fft.fft(fxzz1) / N
fxzz1_sinc_mod = np.abs(fxzz1_sinc_FFT)**2
fxzz1_sinc_db = 10 * np.log10(fxzz1_sinc_mod *2 + eps)

plt.figure()
plt.title('Zero padding sobre la señal N/4 + 1/4')

plt.plot(frec, FX1_db, 'o', linestyle='None', label='Sin zero padding')
plt.plot(frecz, fxzz1_sinc_db, '.', linestyle='None', label='Con zero padding')

plt.xlim([100, 400])
plt.ylim([-60, 2])
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid()
plt.legend()
plt.show()

#%% Frecuencia = N/4 + 1/2

fxzz2 = np.concatenate((fx2,zz), axis = None)
fxzz2_sinc_FFT = np.fft.fft(fxzz2) / N
fxzz2_sinc_mod = np.abs(fxzz2_sinc_FFT)**2
fxzz2_sinc_db = 10 * np.log10(fxzz2_sinc_mod *2 + eps)

plt.figure()
plt.title('Zero padding sobre la señal N/4 + 1/2')

plt.plot(frec, FX2_db, 'o', linestyle='None', label='Sin zero padding')
plt.plot(frecz, fxzz2_sinc_db, '.', linestyle='None', label='Con zero padding')

plt.xlim([100, 400])
plt.ylim([-60, 2])
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid()
plt.legend()
plt.show()

#%% Potencia

P_fx = np.mean(fx**2)
P_fx1 = np.mean(fx1**2)
P_fx2 = np.mean(fx2**2)

P_fxzz = np.sum(fxzz**2) / N
P_fxzz1 = np.sum(fxzz1**2) / N
P_fxzz2 = np.sum(fxzz2**2) / N

print("Potencia fx =", P_fx)
print("Potencia fx1 =", P_fx1)
print("Potencia fx2 =", P_fx2)

print("Potencia fxzz escalada por N =", P_fxzz)
print("Potencia fxzz1 escalada por N =", P_fxzz1)
print("Potencia fxzz2 escalada por N =", P_fxzz2)

#%% Magnitud y fase - caso N/4 + 1/2 (con zero padding) - Trabajo en clase

# Shift
FX2_zp_shift = np.fft.fftshift(fxzz2_sinc_FFT)

# Eje de frecuencia centrado
frec_zp = np.fft.fftshift(np.fft.fftfreq(N0, d=ts))

# Magnitud
FX2_mod = np.abs(FX2_zp_shift)**2
FX2_db = 10 * np.log10(2 * FX2_mod + eps)

# Fase
FX2_fase = np.angle(FX2_zp_shift)

umbral_db = -50
zona = (frecz >= 240) & (frecz <= 261)

plt.figure(figsize=(11,6))

plt.subplot(2,1,1)
plt.title('Magnitud y fase - pico positivo N/4 + 1/2')
plt.plot(frecz[zona], FX2_db[zona], '.-', label='Módulo')
plt.ylabel('Magnitud [dB]')
plt.grid()
plt.legend()

plt.subplot(2,1,2)
plt.plot(frecz[zona], FX2_fase[zona], '.-', label='Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.ylim([-np.pi, np.pi])
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()