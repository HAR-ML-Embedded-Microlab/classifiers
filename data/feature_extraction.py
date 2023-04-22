import random
import json

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import scipy as sp
from scipy import stats
from scipy import signal

train_windows = np.load("svm/train_windows.npy", mmap_mode='r')
test_windows = np.load("svm/test_windows.npy", mmap_mode='r')

# GENERATE TRAIN FEATURES

train_features = []

def corr_grav(x,y):
    corr = np.corrcoef(x,y)[0][1]
    if np.isnan(corr):
          corr = 0
    return corr

def cv_gravity(data):
        cv = stats.variation(data)
        if np.isnan(cv):
          cv = 0  # some might return NaN bc mean and std are 0
        return cv

def freq_cent(fft_ampl, fft_freq):
        sums = np.sum(fft_ampl)
        sums = np.where(sums, sums, 1.)  # Avoid dividing by zero
        return np.sum(fft_freq * fft_ampl) / sums

for window in train_windows:

    window_features = []

    b_mag = (np.linalg.norm(window[:,:3],axis=1))
    t_mag = (np.linalg.norm(window[:,3:],axis=1))

    b_mag = b_mag.reshape(list(b_mag.shape)+[1])
    t_mag = t_mag.reshape(list(t_mag.shape)+[1])

    windows_withMag = np.concatenate([window,b_mag,t_mag], axis=1)

    window = np.transpose(window)
    windows_withMag = np.transpose(windows_withMag)

    #  Filtering
    # create the LP filter
    # 1Hz 4th order LP Butterworth filter
    b, a  = signal.butter(4, Wn=1, fs=50, btype='lowpass')

    # create the HP filter
    # 1Hz 4th order HP Butterworth filter
    d, c = signal.butter(4, Wn=1, fs=50, btype='highpass')

    train_grav = signal.filtfilt(b, a, windows_withMag, axis=1)
    train_mot = signal.filtfilt(d, c, windows_withMag, axis=1)

    #print(len(window_features))

    window_features.append(corr_grav(windows_withMag[0],windows_withMag[1]))
    #print(corr_grav(windows_withMag[0],windows_withMag[1]))
    window_features.append(corr_grav(windows_withMag[0],windows_withMag[2]))
    window_features.append(corr_grav(windows_withMag[0],windows_withMag[3]))
    window_features.append(corr_grav(windows_withMag[0],windows_withMag[4]))
    window_features.append(corr_grav(windows_withMag[0],windows_withMag[5]))

    window_features.append(corr_grav(windows_withMag[1],windows_withMag[2]))
    window_features.append(corr_grav(windows_withMag[1],windows_withMag[3]))
    window_features.append(corr_grav(windows_withMag[1],windows_withMag[4]))
    window_features.append(corr_grav(windows_withMag[1],windows_withMag[5]))

    window_features.append(corr_grav(windows_withMag[2],windows_withMag[3]))
    window_features.append(corr_grav(windows_withMag[2],windows_withMag[4]))
    window_features.append(corr_grav(windows_withMag[2],windows_withMag[5]))

    window_features.append(corr_grav(windows_withMag[3],windows_withMag[4]))
    window_features.append(corr_grav(windows_withMag[3],windows_withMag[5]))
    window_features.append(corr_grav(windows_withMag[4],windows_withMag[5]))

    window_features.append(corr_grav(windows_withMag[6],windows_withMag[7]))

    #print(len(window_features))


    # GRAVITY:
    #  Mean over all axes
    #print(np.concatenate(train_grav[0]).mean())
    window_features+=[np.mean(train_grav)]
    #print(len(window_features))

    # Mean
    window_features+=[np.mean(x) for x in train_grav]

    # Median
    window_features+=[np.median(x) for x in train_grav]

    # Standard Deviation
    window_features+=[np.std(x) for x in train_grav]

    #  Coefficient of Variation
    window_features+=[cv_gravity(x) for x in train_grav]
    
    #  Min Value
    window_features+=[np.min(x) for x in train_grav]

    # 25th Percentile
    window_features+=[np.percentile(x,25) for x in train_grav]

    # 50th Percentile
    window_features+=[np.percentile(x,50) for x in train_grav]

    # 75th Percentile
    window_features+=[np.percentile(x,75) for x in train_grav]

    # Max Value
    window_features+=[np.max(x) for x in train_grav]

    # ACCELERATION:
    # Skew
    window_features+=[stats.skew(x) for x in train_mot]

    # Kurtosis
    window_features+=[stats.kurtosis(x) for x in train_mot]

    # Signal Energy
    window_features+=[np.sum(x**2) for x in train_mot]

    # Frequency-Domain Mean
    # FFT
    train_fft = np.abs(np.fft.fft(windows_withMag, axis=0))

    window_features+=[np.mean(x) for x in train_fft]

    # Frequency-Domain Standard Deviation
    window_features+=[np.std(x) for x in train_fft]

    # Dominant Frequency
    window_features+=[np.fft.fftfreq(window.shape[1])[x.argmax(axis=0)] for x in train_fft]

    # Dominant Frequency Magnitude
    window_features+=[np.max(x) for x in train_fft]

    # Spectral Centroid
    window_features+=[freq_cent(x, np.fft.fftfreq(window.shape[1])) for x in train_fft]
    
    # Total Signal Power
    window_features+=[np.sum(x**2) for x in train_fft]

    #print(len(window_features))
    train_features.append(window_features)

# SAVE TRAIN FEATURES
train_features = np.array(train_features,dtype=np.float32)
np.save("svm/train_features.npy", train_features)

# GENERATE TEST FEATURES

test_features = []

for window in test_windows:

    window_features = []

    b_mag = (np.linalg.norm(window[:,:3],axis=1))
    t_mag = (np.linalg.norm(window[:,3:],axis=1))

    b_mag = b_mag.reshape(list(b_mag.shape)+[1])
    t_mag = t_mag.reshape(list(t_mag.shape)+[1])

    windows_withMag = np.concatenate([window,b_mag,t_mag], axis=1)

    window = np.transpose(window)
    windows_withMag = np.transpose(windows_withMag)

    #  Filtering
    # create the LP filter
    # 1Hz 4th order LP Butterworth filter
    b, a  = signal.butter(4, Wn=1, fs=50, btype='lowpass')

    # create the HP filter
    # 1Hz 4th order HP Butterworth filter
    d, c = signal.butter(4, Wn=1, fs=50, btype='highpass')

    test_grav = signal.filtfilt(b, a, windows_withMag, axis=1)
    test_mot = signal.filtfilt(d, c, windows_withMag, axis=1)

    #print(len(window_features))

    window_features.append(corr_grav(windows_withMag[0],windows_withMag[1]))
    #print(corr_grav(windows_withMag[0],windows_withMag[1]))
    window_features.append(corr_grav(windows_withMag[0],windows_withMag[2]))
    window_features.append(corr_grav(windows_withMag[0],windows_withMag[3]))
    window_features.append(corr_grav(windows_withMag[0],windows_withMag[4]))
    window_features.append(corr_grav(windows_withMag[0],windows_withMag[5]))

    window_features.append(corr_grav(windows_withMag[1],windows_withMag[2]))
    window_features.append(corr_grav(windows_withMag[1],windows_withMag[3]))
    window_features.append(corr_grav(windows_withMag[1],windows_withMag[4]))
    window_features.append(corr_grav(windows_withMag[1],windows_withMag[5]))

    window_features.append(corr_grav(windows_withMag[2],windows_withMag[3]))
    window_features.append(corr_grav(windows_withMag[2],windows_withMag[4]))
    window_features.append(corr_grav(windows_withMag[2],windows_withMag[5]))

    window_features.append(corr_grav(windows_withMag[3],windows_withMag[4]))
    window_features.append(corr_grav(windows_withMag[3],windows_withMag[5]))
    window_features.append(corr_grav(windows_withMag[4],windows_withMag[5]))

    window_features.append(corr_grav(windows_withMag[6],windows_withMag[7]))

    #print(len(window_features))


    # GRAVITY:
    #  Mean over all axes
    #print(np.concatenate(test_grav[0]).mean())
    window_features+=[np.mean(test_grav)]
    #print(len(window_features))

    # Mean
    window_features+=[np.mean(x) for x in test_grav]

    # Median
    window_features+=[np.median(x) for x in test_grav]

    # Standard Deviation
    window_features+=[np.std(x) for x in test_grav]

    #  Coefficient of Variation
    window_features+=[cv_gravity(x) for x in test_grav]
    
    #  Min Value
    window_features+=[np.min(x) for x in test_grav]

    # 25th Percentile
    window_features+=[np.percentile(x,25) for x in test_grav]

    # 50th Percentile
    window_features+=[np.percentile(x,50) for x in test_grav]

    # 75th Percentile
    window_features+=[np.percentile(x,75) for x in test_grav]

    # Max Value
    window_features+=[np.max(x) for x in test_grav]

    # ACCELERATION:
    # Skew
    window_features+=[stats.skew(x) for x in test_mot]

    # Kurtosis
    window_features+=[stats.kurtosis(x) for x in test_mot]

    # Signal Energy
    window_features+=[np.sum(x**2) for x in test_mot]

    # Frequency-Domain Mean
    # FFT
    test_fft = np.abs(np.fft.fft(windows_withMag, axis=0))

    window_features+=[np.mean(x) for x in test_fft]

    # Frequency-Domain Standard Deviation
    window_features+=[np.std(x) for x in test_fft]

    # Dominant Frequency
    window_features+=[np.fft.fftfreq(window.shape[1])[x.argmax(axis=0)] for x in test_fft]

    # Dominant Frequency Magnitude
    window_features+=[np.max(x) for x in test_fft]

    # Spectral Centroid
    window_features+=[freq_cent(x, np.fft.fftfreq(window.shape[1])) for x in test_fft]
    
    # Total Signal Power
    window_features+=[np.sum(x**2) for x in test_fft]

    #print(len(window_features))
    test_features.append(window_features)


# SAVE TEST FEATURES
test_features = np.array(test_features,dtype=np.float32)
np.save("svm/test_features.npy", test_features)
