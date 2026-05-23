# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:01:58 2026

@author: gabri
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy.io as sio
from scipy.io.wavfile import write

# Lectura de señales

##################
## ECG sin ruido
##################

fs_ecg = 1000 # Hz

ecg_one_lead = np.load('ecg_sin_ruido.npy')

plt.figure(1)
plt.plot(ecg_one_lead)

##################
## PPG sin ruido
##################

fs_ppg = 400 # Hz

ppg = np.load('ppg_sin_ruido.npy')

plt.figure(2)
plt.plot(ppg)

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio_cuca, wav_data_cuca = sio.wavfile.read('la cucaracha.wav')
fs_audio_prueba, wav_data_prueba = sio.wavfile.read('prueba psd.wav')
fs_audio_sil, wav_data_sil = sio.wavfile.read('silbido.wav')

plt.figure(3)
plt.subplot(3,1,1)
plt.title('La Cucaracha')
plt.plot(wav_data_cuca)
plt.subplot(3,1,2)
plt.title('Prueba')
plt.plot(wav_data_prueba)
plt.subplot(3,1,3)
plt.title('Silbido')
plt.plot(wav_data_sil)

#%% Estimación de PSD - Welch

# ECG
tamaño = ecg_one_lead.shape[0]
promedios = 30
nperseg = tamaño // promedios

welch_ecg_fr, welch_ecg = sig.welch(ecg_one_lead, fs = fs_ecg, window = 'hamming',nperseg = nperseg, noverlap = 15)

plt.figure(4)
plt.title('Welch - ECG')
plt.xlim(-1,40)
plt.plot(welch_ecg_fr, welch_ecg)

#%% PPD
tamaño = ppg.shape[0]
promedios = 10
nperseg = tamaño // promedios

welch_ppg_fr, welch_ppg = sig.welch(ppg, fs = fs_ecg, window = 'hamming',nperseg = nperseg, noverlap = 5)

plt.figure(5)
plt.title('Welch - PPG')
plt.xlim(-1,30)
plt.plot(welch_ppg_fr, welch_ppg)

# Audio
# La cucaracha
tamaño = wav_data_cuca.shape[0]
promedios = 100
nperseg = tamaño // promedios

welch_cuca_fr, welch_cuca = sig.welch(wav_data_cuca, fs = fs_audio_cuca, window = 'hamming',nperseg = nperseg, noverlap = 50)

plt.figure(6)
plt.title('Welch - La cucaracha')
plt.xlim(500,2500)
plt.plot(welch_cuca_fr, welch_cuca)

#%%

Pot_acum = np.cumsum(welch_ecg)