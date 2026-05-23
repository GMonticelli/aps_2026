# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:57:52 2026

@author: gabri
"""

# Módulos externos
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import windows
import pandas as pd
from IPython.display import display

#%% Parámetros

N = 1000
fs = 1000
ts = 1 / fs
R = 200

a0 = np.sqrt(2)
omega0 = N/4
df = fs/N

PS = (a0)**2/2

n = np.arange(N)
n = n.reshape(1, N)

t = np.arange(N)/fs
tt = (t).reshape(1, -1)

fr = np.random.uniform(-2, 2, R)

omega1 = omega0 + fr*df
omega1 = omega1.reshape(R,1)

arg = omega1*n

eps = 1e-12

#%% Generación señal SNR = 10 dB

SNR = 10

PR = PS / 10**(SNR/10)
desvio = np.sqrt(PR)

na = np.random.normal(0, desvio, size = (R, N))

señal = a0*np.sin(2*np.pi*arg*ts) + na
señal_t = np.transpose(señal)

#%% Grafico señal

plt.figure()
plt.title('Grafico de Señal - SNR = 10 dB')
plt.plot(t, señal_t)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()
plt.show()

#%% FFT

señal_X = np.fft.fft(señal, axis = 1) / N
psd_señal = np.abs(señal_X)
señal_x_db = 10*np.log10(2*(psd_señal)**2 + eps)

frec = np.arange(N) * fs / N
frec = frec.reshape(1000, 1)

#%% Grafico FFT

plt.figure()
plt.title('PSD - SNR = 10 dB')
plt.plot(frec, señal_x_db.T)
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid()
plt.show()

#%% Ventanas

# Ganancia coherente
Gc_rec = np.mean(windows.boxcar(N))
Gc_flat = np.mean(windows.flattop(N))
Gc_blk = np.mean(windows.blackmanharris(N))
Gc_ham = np.mean(windows.hamming(N))

# Rectangular
w_rec = señal * windows.boxcar(N)

W_REC = np.fft.fft(w_rec, axis = 1) / N
W_REC_Mod = np.abs(W_REC)
W_REC_db = 10*np.log10(2*(W_REC_Mod)**2 + eps)
W_REC_db_t = np.transpose(W_REC_db)

# Flattop
w_flat = señal * windows.flattop(N)

W_FLAT = np.fft.fft(w_flat, axis = 1) / N
W_FLAT_Mod = np.abs(W_FLAT)
W_FLAT_db = 10*np.log10(2*(W_FLAT_Mod)**2 + eps)
W_FLAT_db_t = np.transpose(W_FLAT_db)

# Blackman-Harris
w_blk = señal * windows.blackmanharris(N)

W_BLK = np.fft.fft(w_blk, axis = 1) / N
W_BLK_Mod = np.abs(W_BLK)
W_BLK_db = 10*np.log10(2*(W_BLK_Mod)**2 + eps)
W_BLK_db_t = np.transpose(W_BLK_db)

# Hamming
w_ham = señal * windows.hamming(N)

W_HAM = np.fft.fft(w_ham, axis = 1) / N
W_HAM_Mod = np.abs(W_HAM)
W_HAM_db = 10*np.log10(2*(W_HAM_Mod)**2 + eps)
W_HAM_db_t = np.transpose(W_HAM_db)

#%% Grafico ventanas SNR = 10 dB

plt.figure()
plt.suptitle('Ventanas - SNR = 10 dB')

plt.subplot(2, 2, 1)
plt.title('Rectangular')
plt.plot(frec, W_REC_db_t)
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid()

plt.subplot(2, 2, 2)
plt.title('Flattop')
plt.plot(frec, W_FLAT_db_t)
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid()

plt.subplot(2, 2, 3)
plt.title('Blackman-Harris')
plt.plot(frec, W_BLK_db_t)
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid()

plt.subplot(2, 2, 4)
plt.title('Hamming')
plt.plot(frec, W_HAM_db_t)
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid()

plt.tight_layout()
plt.show()

#%% Estimador amplitud SNR = 10 dB

a1_rec_10 = np.abs(W_REC[:,N//4])*2/Gc_rec
a1_flat_10 = np.abs(W_FLAT[:,N//4])*2/Gc_flat
a1_blk_10 = np.abs(W_BLK[:,N//4])*2/Gc_blk
a1_ham_10 = np.abs(W_HAM[:,N//4])*2/Gc_ham

bins = 10

plt.figure()
plt.title('Histograma - a1 - SNR = 10 dB')

plt.hist(a1_rec_10, label = 'Rectangular', alpha = 0.5, bins = bins)
plt.hist(a1_flat_10, label = 'Flattop', alpha = 0.5, bins = bins)
plt.hist(a1_blk_10, label = 'Blackman-Harris', alpha = 0.5, bins = bins)
plt.hist(a1_ham_10, label = 'Hamming', alpha = 0.5, bins = bins)

plt.xlabel('Amplitud estimada')
plt.ylabel('Cantidad de realizaciones')

plt.grid()
plt.legend()
plt.show()

#%% Estimador frecuencia SNR = 10 dB

omegai_rec_10 = np.argmax(W_REC_Mod[:, :N//2], axis = 1) * df
omegai_flat_10 = np.argmax(W_FLAT_Mod[:, :N//2], axis = 1) * df
omegai_blk_10 = np.argmax(W_BLK_Mod[:, :N//2], axis = 1) * df
omegai_ham_10 = np.argmax(W_HAM_Mod[:, :N//2], axis = 1) * df

plt.figure()
plt.title('Histograma - omega1 - SNR = 10 dB')

plt.hist(omegai_rec_10, label = 'Rectangular', alpha = 0.5, bins = bins)
plt.hist(omegai_flat_10, label = 'Flattop', alpha = 0.5, bins = bins)
plt.hist(omegai_blk_10, label = 'Blackman-Harris', alpha = 0.5, bins = bins)
plt.hist(omegai_ham_10, label = 'Hamming', alpha = 0.5, bins = bins)

plt.xlabel('Frecuencia estimada [Hz]')
plt.ylabel('Cantidad de realizaciones')

plt.grid()
plt.legend()
plt.show()

#%% Sesgo y varianza SNR = 10 dB

# Sesgo amplitud
s_a1_rec_10 = np.mean(a1_rec_10) - a0
s_a1_flat_10 = np.mean(a1_flat_10) - a0
s_a1_blk_10 = np.mean(a1_blk_10) - a0
s_a1_ham_10 = np.mean(a1_ham_10) - a0

# Sesgo frecuencia
s_omegai_rec_10 = np.mean(omegai_rec_10) - np.mean(omega1)
s_omegai_flat_10 = np.mean(omegai_flat_10) - np.mean(omega1)
s_omegai_blk_10 = np.mean(omegai_blk_10) - np.mean(omega1)
s_omegai_ham_10 = np.mean(omegai_ham_10) - np.mean(omega1)

# Varianza amplitud
v_a1_rec_10 = np.var(a1_rec_10)
v_a1_flat_10 = np.var(a1_flat_10)
v_a1_blk_10 = np.var(a1_blk_10)
v_a1_ham_10 = np.var(a1_ham_10)

# Varianza frecuencia
v_omegai_rec_10 = np.var(omegai_rec_10)
v_omegai_flat_10 = np.var(omegai_flat_10)
v_omega1_blk_10 = np.var(omegai_blk_10)
v_omegai_ham_10 = np.var(omegai_ham_10)

# Tabla amplitud
tabla_amp_10 = pd.DataFrame({
    'Ventana': ['Rectangular', 'Flat-top', 'Blackman-Harris', 'Hamming'],
    's_a': [s_a1_rec_10, s_a1_flat_10, s_a1_blk_10, s_a1_ham_10],
    'v_a': [v_a1_rec_10, v_a1_flat_10, v_a1_blk_10, v_a1_ham_10]
})

# Tabla frecuencia
tabla_frec_10 = pd.DataFrame({
    'Ventana': ['Rectangular', 'Flat-top', 'Blackman-Harris', 'Hamming'],
    's_f': [s_omegai_rec_10, s_omegai_flat_10, s_omegai_blk_10, s_omegai_ham_10],
    'v_f': [v_omegai_rec_10, v_omegai_flat_10, v_omega1_blk_10, v_omegai_ham_10]
})

print('Estimación de Amplitud - SNR = 10 dB')
display(tabla_amp_10)

print('Estimación de Frecuencia - SNR = 10 dB')
display(tabla_frec_10)

#%% Señal SNR = 3 dB

SNR = 3

PR = PS / 10**(SNR/10)
desvio = np.sqrt(PR)

na = np.random.normal(0, desvio, size = (R, N))

señal = a0*np.sin(2*np.pi*arg*ts) + na

#%% Ventanas SNR = 3 dB

# Rectangular
w_rec = señal * windows.boxcar(N)

W_REC = np.fft.fft(w_rec, axis = 1) / N
W_REC_Mod = np.abs(W_REC)
W_REC_db = 10*np.log10(2*(W_REC_Mod)**2 + eps)
W_REC_db_t = np.transpose(W_REC_db)

# Flattop
w_flat = señal * windows.flattop(N)

W_FLAT = np.fft.fft(w_flat, axis = 1) / N
W_FLAT_Mod = np.abs(W_FLAT)
W_FLAT_db = 10*np.log10(2*(W_FLAT_Mod)**2 + eps)
W_FLAT_db_t = np.transpose(W_FLAT_db)

# Blackman-Harris
w_blk = señal * windows.blackmanharris(N)

W_BLK = np.fft.fft(w_blk, axis = 1) / N
W_BLK_Mod = np.abs(W_BLK)
W_BLK_db = 10*np.log10(2*(W_BLK_Mod)**2 + eps)
W_BLK_db_t = np.transpose(W_BLK_db)

# Hamming
w_ham = señal * windows.hamming(N)

W_HAM = np.fft.fft(w_ham, axis = 1) / N
W_HAM_Mod = np.abs(W_HAM)
W_HAM_db = 10*np.log10(2*(W_HAM_Mod)**2 + eps)
W_HAM_db_t = np.transpose(W_HAM_db)

#%% Graficos ventanas SNR = 3 dB

plt.figure()
plt.suptitle('Ventanas - SNR = 3 dB')

plt.subplot(2, 2, 1)
plt.title('Rectangular')
plt.plot(frec, W_REC_db_t)
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid()

plt.subplot(2, 2, 2)
plt.title('Flattop')
plt.plot(frec, W_FLAT_db_t)
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid()

plt.subplot(2, 2, 3)
plt.title('Blackman-Harris')
plt.plot(frec, W_BLK_db_t)
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid()

plt.subplot(2, 2, 4)
plt.title('Hamming')
plt.plot(frec, W_HAM_db_t)
plt.xlim(0, fs/2)
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.grid()

plt.tight_layout()
plt.show()

#%% Estimador amplitud SNR = 3 dB

a1_rec_3 = np.abs(W_REC[:,N//4])*2/Gc_rec
a1_flat_3 = np.abs(W_FLAT[:,N//4])*2/Gc_flat
a1_blk_3 = np.abs(W_BLK[:,N//4])*2/Gc_blk
a1_ham_3 = np.abs(W_HAM[:,N//4])*2/Gc_ham

plt.figure()
plt.title('Histograma - a1 - SNR = 3 dB')

plt.hist(a1_rec_3, label = 'Rectangular', alpha = 0.5, bins = bins)
plt.hist(a1_flat_3, label = 'Flattop', alpha = 0.5, bins = bins)
plt.hist(a1_blk_3, label = 'Blackman-Harris', alpha = 0.5, bins = bins)
plt.hist(a1_ham_3, label = 'Hamming', alpha = 0.5, bins = bins)

plt.xlabel('Amplitud estimada')
plt.ylabel('Cantidad de realizaciones')

plt.grid()
plt.legend()
plt.show()

#%% Estimador frecuencia SNR = 3 dB

omegai_rec_3 = np.argmax(W_REC_Mod[:, :N//2], axis = 1) * df
omegai_flat_3 = np.argmax(W_FLAT_Mod[:, :N//2], axis = 1) * df
omegai_blk_3 = np.argmax(W_BLK_Mod[:, :N//2], axis = 1) * df
omegai_ham_3 = np.argmax(W_HAM_Mod[:, :N//2], axis = 1) * df

plt.figure()
plt.title('Histograma - omega1 - SNR = 3 dB')

plt.hist(omegai_rec_3, label = 'Rectangular', alpha = 0.5, bins = bins)
plt.hist(omegai_flat_3, label = 'Flattop', alpha = 0.5, bins = bins)
plt.hist(omegai_blk_3, label = 'Blackman-Harris', alpha = 0.5, bins = bins)
plt.hist(omegai_ham_3, label = 'Hamming', alpha = 0.5, bins = bins)

plt.xlabel('Frecuencia estimada [Hz]')
plt.ylabel('Cantidad de realizaciones')

plt.grid()
plt.legend()
plt.show()

#%% Sesgo y varianza SNR = 3 dB

# Sesgo amplitud
s_a1_rec_3 = np.mean(a1_rec_3) - a0
s_a1_flat_3 = np.mean(a1_flat_3) - a0
s_a1_blk_3 = np.mean(a1_blk_3) - a0
s_a1_ham_3 = np.mean(a1_ham_3) - a0

# Sesgo frecuencia
s_omegai_rec_3 = np.mean(omegai_rec_3) - np.mean(omega1)
s_omegai_flat_3 = np.mean(omegai_flat_3) - np.mean(omega1)
s_omegai_blk_3 = np.mean(omegai_blk_3) - np.mean(omega1)
s_omegai_ham_3 = np.mean(omegai_ham_3) - np.mean(omega1)

# Varianza amplitud
v_a1_rec_3 = np.var(a1_rec_3)
v_a1_flat_3 = np.var(a1_flat_3)
v_a1_blk_3 = np.var(a1_blk_3)
v_a1_ham_3 = np.var(a1_ham_3)

# Varianza frecuencia
v_omegai_rec_3 = np.var(omegai_rec_3)
v_omegai_flat_3 = np.var(omegai_flat_3)
v_omega1_blk_3 = np.var(omegai_blk_3)
v_omegai_ham_3 = np.var(omegai_ham_3)

# Tabla amplitud
tabla_amp_3 = pd.DataFrame({
    'Ventana': ['Rectangular', 'Flat-top', 'Blackman-Harris', 'Hamming'],
    's_a': [s_a1_rec_3, s_a1_flat_3, s_a1_blk_3, s_a1_ham_3],
    'v_a': [v_a1_rec_3, v_a1_flat_3, v_a1_blk_3, v_a1_ham_3]
})

# Tabla frecuencia
tabla_frec_3 = pd.DataFrame({
    'Ventana': ['Rectangular', 'Flat-top', 'Blackman-Harris', 'Hamming'],
    's_f': [s_omegai_rec_3, s_omegai_flat_3, s_omegai_blk_3, s_omegai_ham_3],
    'v_f': [v_omegai_rec_3, v_omegai_flat_3, v_omega1_blk_3, v_omegai_ham_3]
})

print('Estimación de Amplitud - SNR = 3 dB')
display(tabla_amp_3)

print('Estimación de Frecuencia - SNR = 3 dB')
display(tabla_frec_3)

#%% Zero-padding

N_zp = 2000
df_zp = fs / N_zp

zp_W_REC = np.fft.fft(w_rec, n = N_zp, axis = 1)
zp_W_REC_Mod = np.abs(zp_W_REC)

zp_W_FLAT = np.fft.fft(w_flat, n = N_zp, axis = 1)
zp_W_FLAT_Mod = np.abs(zp_W_FLAT)

zp_W_BLK = np.fft.fft(w_blk, n = N_zp, axis = 1)
zp_W_BLK_Mod = np.abs(zp_W_BLK)

zp_W_HAM = np.fft.fft(w_ham, n = N_zp, axis = 1)
zp_W_HAM_Mod = np.abs(zp_W_HAM)

zp_omegai_rec_3 = np.argmax(zp_W_REC_Mod[:, :N_zp//2], axis = 1) * df_zp
zp_omegai_flat_3 = np.argmax(zp_W_FLAT_Mod[:, :N_zp//2], axis = 1) * df_zp
zp_omegai_blk_3 = np.argmax(zp_W_BLK_Mod[:, :N_zp//2], axis = 1) * df_zp
zp_omegai_ham_3 = np.argmax(zp_W_HAM_Mod[:, :N_zp//2], axis = 1) * df_zp

plt.figure()
plt.title('Histograma - omega1 - SNR = 3 dB - zero-padding')

plt.hist(zp_omegai_rec_3, label = 'Rectangular', alpha = 0.5, bins = bins)
plt.hist(zp_omegai_flat_3, label = 'Flattop', alpha = 0.5, bins = bins)
plt.hist(zp_omegai_blk_3, label = 'Blackman-Harris', alpha = 0.5, bins = bins)
plt.hist(zp_omegai_ham_3, label = 'Hamming', alpha = 0.5, bins = bins)

plt.xlabel('Frecuencia estimada [Hz]')
plt.ylabel('Cantidad de realizaciones')

plt.grid()
plt.legend()
plt.show()

#%% Tabla zero-padding

s_zp_omegai_rec_3 = np.mean(zp_omegai_rec_3) - np.mean(omega1)
s_zp_omegai_flat_3 = np.mean(zp_omegai_flat_3) - np.mean(omega1)
s_zp_omegai_blk_3 = np.mean(zp_omegai_blk_3) - np.mean(omega1)
s_zp_omegai_ham_3 = np.mean(zp_omegai_ham_3) - np.mean(omega1)

v_zp_omegai_rec_3 = np.var(zp_omegai_rec_3)
v_zp_omegai_flat_3 = np.var(zp_omegai_flat_3)
v_zp_omegai_blk_3 = np.var(zp_omegai_blk_3)
v_zp_omegai_ham_3 = np.var(zp_omegai_ham_3)

tabla_frec_zp_3 = pd.DataFrame({
    'Ventana': ['Rectangular', 'Flat-top', 'Blackman-Harris', 'Hamming'],
    's_f': [s_zp_omegai_rec_3, s_zp_omegai_flat_3, s_zp_omegai_blk_3, s_zp_omegai_ham_3],
    'v_f': [v_zp_omegai_rec_3, v_zp_omegai_flat_3, v_zp_omegai_blk_3, v_zp_omegai_ham_3]
})

print('Estimación de Frecuencia con zero-padding - SNR = 3 dB')
display(tabla_frec_zp_3)