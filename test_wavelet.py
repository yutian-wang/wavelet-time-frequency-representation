# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:27:29 2019

@author: playfish
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
import librosa
import librosa.display


sampling_rate=16000
frame_size = 640
frame_shift = 320
audio, sr = librosa.load('./abjones_1_01.wav', sr=sampling_rate, mono=False)
audio = audio[1]
data=audio[50000:60000]

t=np.arange(0, len(data)) / sampling_rate

#wavename = "cgau8"
wavename = "shan"
totalscal = 322
fc = pywt.central_frequency(wavename)#中心频率
cparam = 2 * fc * totalscal
scales = cparam/np.arange(totalscal,1,-1)
[cwtmatr, frequencies] = pywt.cwt(data,scales,wavename,1.0/sampling_rate)#连续小波变换
cwtmatr_db = librosa.amplitude_to_db(np.abs(cwtmatr), ref=np.max)

spectra = librosa.stft(data, n_fft = frame_size, hop_length = frame_shift)
spectra_db = librosa.amplitude_to_db(np.abs(spectra), ref=np.max)

plt.figure(figsize=(10, 6))
plt.subplot(311)
plt.plot(t, data)
plt.xlabel(u"time(s)")
plt.title(u"Time-Frequency spectrum")
plt.subplot(312)
plt.pcolormesh(t, frequencies, cwtmatr_db)
plt.ylabel(u"freq(Hz)")
plt.xlabel(u"time(s)")
plt.subplots_adjust(hspace=0.4)
plt.subplot(313)
librosa.display.specshow(spectra_db, y_axis='linear',x_axis='s',sr=sampling_rate,hop_length=frame_shift)

plt.show()